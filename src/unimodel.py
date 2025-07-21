from __future__ import annotations
from typing import Callable
from random import random

import contextlib

from torch import nn
import torch 
import copy
import os

from f5_tts.model import DiT, UNetT
from pathlib import Path
from guidance_model import Guidance
from f5_tts.model.utils import (
    default,
    exists,
    list_str_to_idx,
    list_str_to_tensor,
    lens_to_mask,
    mask_from_frac_lengths,
    sample_consecutive_steps,
    sample_from_list,
)

class UniModel(nn.Module):
    def __init__(self, 
                 model: DiT, # teacher model (dit model)
                 checkpoint_path: str = "",
                 second_time: bool = True,
                 use_fp16: bool = True,
                 real_guidance_scale: float = 2.0, 
                 fake_guidance_scale: float = 0.0, 
                 gen_cls_loss: bool = False,
                 sway_coeff: float = -1.0,
                 vocab_char_map: dict[str, int] | None = None,
                 frac_lengths_mask: tuple[float, float] = (0.7, 1.0)):
        
        super().__init__()
        
        if checkpoint_path != "":
            if "model_last.pt" in os.listdir(checkpoint_path):
                latest_checkpoint = "model_last.pt"
            else:
                latest_checkpoint = sorted(
                    [f for f in os.listdir(checkpoint_path) if f.endswith(".pt")],
                    key=lambda x: int("".join(filter(str.isdigit, x))),
                )[-1]
            checkpoint = torch.load(f"{checkpoint_path}/{latest_checkpoint}", weights_only=True, map_location="cpu")

            if "scale" in checkpoint:
                self.scale = checkpoint["scale"]
            else:
                self.scale = 1.0
            print(f"Loaded teacher model with scale: {self.scale}")

            if "step" in checkpoint:
                state = checkpoint["model_state_dict"]
            else:
                checkpoint["model_state_dict"] = {
                    k.replace("ema_model.", ""): v
                    for k, v in checkpoint["ema_model_state_dict"].items()
                    if k not in ["initted", "step"]
                }
                state = checkpoint["model_state_dict"]

            # only load the DiT module
            filtered_state_dict = {
                k.replace("transformer.", ""): v
                for k, v in state.items()
                if k.startswith("transformer.")
            }

            model.load_state_dict(filtered_state_dict, strict=False)
        else:
            self.scale = 1.0
        
        real_unet = copy.deepcopy(model)
        real_unet.time_embed2 = None
        
        fake_unet = copy.deepcopy(model)
        
        # Instantiate Guidance, which internally uses real_unet and fake_unet initialized from the teacher
        self.guidance_model = Guidance(
            real_unet=real_unet,
            fake_unet=fake_unet,
            use_fp16=use_fp16,
            real_guidance_scale=real_guidance_scale,
            fake_guidance_scale=fake_guidance_scale,
            gen_cls_loss=gen_cls_loss,
            sway_coeff=sway_coeff,
        )
        
        self.feedforward_model = copy.deepcopy(model) # initialize the student model
        self.feedforward_model.requires_grad_(True)
        self.feedforward_model.time_embed2 = None

        self.vocab_char_map = vocab_char_map
        self.frac_lengths_mask = frac_lengths_mask
        
        self.second_time = second_time # fake_unet.time_embed2 is not None

    def forward(self,
                inp: float["b n d"],  # mel
                text: int["b nt"] | list[str],
                *,
                lens: int["b"] | None = None,
                student_steps: list[int] = [0, 0.25, 0.5, 0.75],
                update_generator: bool = False,
    ):
        """
        Forward pass that routes to either generator_forward or guidance_forward
        in the Guidance class, depending on the arguments.

        Parameters:
        -----------
        generator_turn: bool
            If True, run the generator forward pass (distribution matching loss, etc.)
        guidance_turn: bool
            If True, run the guidance forward pass (fake loss, cls loss, etc.)
        data_dict: dict
            Input dictionary containing the necessary keys for the forward passes.
            Expected keys may include:
                "inp": Tensor (B, N, D) - input mel or latent
                "text": Tensor or list[str] - text input
                "rand_span_mask": Tensor (B, N) - boolean mask
                "real_data": dict with keys like:
                     "inp", "text", "rand_span_mask"
                     
        Returns:
        --------
        loss_dict: dict[str, Tensor]
            Dictionary of losses.
        log_dict: dict[str, Tensor or float]
            Dictionary of logging tensors or values.
        """
        
        batch, seq_len, dtype, device = *inp.shape[:2], inp.dtype, inp.device

        # handle text as string
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # lens and mask
        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)

        mask = lens_to_mask(lens, length=seq_len)  # useless here, as collate_fn will pad to max length in batch

        # sample from the list of student steps
        time = sample_from_list(student_steps, batch).to(device)
        c_time, p_time = sample_consecutive_steps(student_steps)
        time = torch.ones_like(time) * c_time
        p_time = torch.ones_like(time) * p_time

        frac_lengths = torch.zeros((batch,), device=device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)
        
        if exists(mask):
            rand_span_mask &= mask
            

        # # use generated output from previous step as input
        with torch.no_grad():
            x1 = inp
            x0 = torch.randn_like(x1)
            t = p_time.unsqueeze(-1).unsqueeze(-1)
            phi = (1 - t) * x0 + t * x1
            cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)
            
            pred = self.feedforward_model(
                x=phi, 
                cond=cond,
                text=text, 
                time=p_time, 
                drop_audio_cond=False, 
                drop_text=False # make sure the cfg=1
            ) # flow prediction
            
            # predicted mel spectrogram
            output = phi + (1 - t) * pred 
            output[~rand_span_mask] = inp[~rand_span_mask]
        
        # forward diffusion
        x1 = output
        x0 = torch.randn_like(x1)
        t = time.unsqueeze(-1).unsqueeze(-1)
        phi = (1 - t) * x0 + t * x1
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)
        
        with torch.no_grad() if not update_generator else contextlib.nullcontext():
            pred = self.feedforward_model(
                x=phi, 
                cond=cond,
                text=text, 
                time=time, 
                drop_audio_cond=False, 
                drop_text=False # make sure no cfg is used  
            )
            
            # predicted mel spectrogram
            output = phi + (1 - t) * pred
            output[~rand_span_mask] = inp[~rand_span_mask]
        
        if update_generator:
            generator_data_dict = {
                "inp": output,
                "text": text,
                "rand_span_mask": rand_span_mask,
                "second_time": time if self.second_time else None,
                "mse_loss": time.mean() == student_steps[-1].mean(),
                "real_data": {
                    "inp": inp,
                    "text": text,
                    "rand_span_mask": rand_span_mask
                }
            }
            
            # avoid any side effects of gradient accumulation
            # self.guidance_model.requires_grad_(False)
            # self.feedforward_model.requires_grad_(True)
            generator_loss_dict, generator_log_dict = self.guidance_model(
                generator_turn=True,
                guidance_turn=False,
                generator_data_dict=generator_data_dict,
                guidance_data_dict=None
            )
                
            generator_log_dict['ground_truth'] = x1
            generator_log_dict['generator_input'] = phi
            generator_log_dict['generator_output'] = output
            generator_log_dict['generator_cond'] = cond
            generator_log_dict['time'] = time
            
            return generator_loss_dict, generator_log_dict
        else:
            guidance_data_dict = {
                "inp": output.detach(),
                "text": text,
                "rand_span_mask": rand_span_mask,
                "second_time": time if self.second_time else None,
                "real_data": {
                    "inp": inp,
                    "text": text,
                    "rand_span_mask": rand_span_mask
                }
            }
            
            # avoid any side effects of gradient accumulation
            # self.feedforward_model.requires_grad_(False)
            # self.guidance_model.requires_grad_(True)
            guidance_loss_dict, guidance_log_dict = self.guidance_model(
                generator_turn=False,
                guidance_turn=True,
                generator_data_dict=None,
                guidance_data_dict=guidance_data_dict
            )
            # self.feedforward_model.requires_grad_(True)
            
            return guidance_loss_dict, guidance_log_dict
            
        # return guidance_loss_dict, guidance_log_dict, generator_loss_dict, generator_log_dict
    

if __name__ == "__main__":
    
    from f5_tts.model.utils import get_tokenizer
    from torch.utils.data import DataLoader, Dataset, SequentialSampler
    from f5_tts.model.dataset import load_dataset    
    from f5_tts.model.dataset import DynamicBatchSampler, collate_fn

    bsz = 16
    
    tokenizer = "pinyin"  # 'pinyin', 'char', or 'custom'
    tokenizer_path = None  # if tokenizer = 'custom', define the path to the tokenizer you want to use (should be vocab.txt)
    dataset_name = "Emilia_ZH_EN"
    if tokenizer == "custom":
        tokenizer_path = tokenizer_path
    else:
        tokenizer_path = dataset_name
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    dit = DiT(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4, text_num_embeds=vocab_size, mel_dim=100)
    
    model = UniModel(dit, 
                     checkpoint_path="/data4/F5TTS/ckpts/F5TTS_Base_norm_flow_8GPU_vocos_pinyin_Emilia_ZH_EN",
                     gen_cls_loss=True,
                     vocab_char_map=vocab_char_map,
                     frac_lengths_mask=(0.7, 1.0)
                     ).cuda()
    
    # batch = next(iter(train_dataloader))
    # torch.save(batch, "batch.pt")
    batch = torch.load("batch.pt")
    inp, text, lens = batch["mel"].permute(0, 2, 1).cuda(), batch["text"], batch["mel_lengths"].cuda()    

    
    # text = ["hello world"] * bsz
    # lens = torch.randint(1, 1000, (bsz,)).cuda()
    # inp = torch.randn(bsz, lens.max(), 100).cuda()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        num_student_step = 4

        guidance_loss_dict, guidance_log_dict = model(inp, text, lens=lens, update_generator=False, student_steps=(torch.linspace(0.0, 1.0, num_student_step + 1)[:-1]))

        generator_loss_dict, generator_log_dict = model(inp, text, lens=lens, update_generator=True, student_steps=(torch.linspace(0.0, 1.0, num_student_step + 1)[:-1]))
                
        print(guidance_loss_dict)
        print(generator_loss_dict)
        
        guidance_loss = 0
        guidance_loss += guidance_loss_dict["loss_fake_mean"]
        guidance_loss += guidance_loss_dict["guidance_cls_loss"]

        generator_loss = 0
        generator_loss += generator_loss_dict["loss_dm"]
        generator_loss += generator_loss_dict["loss_ctc"]
        generator_loss += generator_loss_dict["loss_sim"]
        generator_loss += generator_loss_dict["gen_cls_loss"]
        generator_loss += generator_loss_dict["loss_mse"]

        guidance_loss.backward()
        generator_loss.backward()