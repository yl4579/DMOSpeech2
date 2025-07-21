"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations
from typing import Callable
from random import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from f5_tts.model import DiT

from f5_tts.model.utils import (
    default,
    exists,
    list_str_to_idx,
    list_str_to_tensor,
    lens_to_mask,
    mask_from_frac_lengths,
)

from discriminator_conformer import ConformerDiscirminator
from ctcmodel import ConformerCTC
from ecapa_tdnn import ECAPA_TDNN

class NoOpContext:
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

def predict_flow(transformer, # flow model
                  x, # noisy input
                  cond, # mask (prompt mask + length mask)
                  text, # text input
                  time, # time step
                  second_time=None,
                  cfg_strength=1.0
):
    pred = transformer(
        x=x, 
        cond=cond, 
        text=text, time=time, 
        second_time=second_time,
        drop_audio_cond=False, 
        drop_text=False
    )
    
    if cfg_strength < 1e-5:
        return pred
        
    null_pred = transformer(
            x=x, 
            cond=cond, 
            text=text, time=time, 
            second_time=second_time,
            drop_audio_cond=True, 
            drop_text=True
    )

    return pred + (pred - null_pred) * cfg_strength

def _kl_dist_func(x, y):
    log_probs = F.log_softmax(x, dim=2)
    target_probs  = F.log_softmax(y, dim=2)
    return torch.nn.functional.kl_div(log_probs, target_probs, reduction="batchmean", log_target=True)


class Guidance(nn.Module):
    def __init__(self, 
                real_unet: DiT, # teacher flow model
                fake_unet: DiT, # student flow model

                use_fp16: bool = True,
                real_guidance_scale: float = 0.0, 
                fake_guidance_scale: float = 0.0, 
                gen_cls_loss: bool = False,
                
                sv_path_en: str = "",
                sv_path_zh: str = "",
                ctc_path: str = "",
                sway_coeff: float = 0.0,
                scale: float = 1.0,
                ):
        super().__init__()
        self.vocab_size = real_unet.vocab_size
        
        if ctc_path != "":
            model = ConformerCTC(vocab_size=real_unet.vocab_size, mel_dim=real_unet.mel_dim, num_heads=8, d_hid=512, nlayers=6)
            self.ctc_model = model.eval()
            self.ctc_model.requires_grad_(False)
            self.ctc_model.load_state_dict(torch.load(ctc_path, weights_only=True, map_location='cpu')['model_state_dict'])

        if sv_path_en != "":
            model = ECAPA_TDNN()
            self.sv_model_en = model.eval()
            self.sv_model_en.requires_grad_(False)
            self.sv_model_en.load_state_dict(torch.load(sv_path, weights_only=True, map_location='cpu')['model_state_dict'])

        if sv_path_zh != "":
            model = ECAPA_TDNN()
            self.sv_model_zh = model.eval()
            self.sv_model_zh.requires_grad_(False)
            self.sv_model_zh.load_state_dict(torch.load(sv_path_zh, weights_only=True, map_location='cpu')['model_state_dict'])

        self.scale = scale
        
        self.real_unet = real_unet
        self.real_unet.requires_grad_(False) # no update on the teacher model

        self.fake_unet = fake_unet
        self.fake_unet.requires_grad_(True) # update the student model
        
        self.real_guidance_scale = real_guidance_scale 
        self.fake_guidance_scale = fake_guidance_scale
        
        assert self.fake_guidance_scale == 0, "no guidance for fake"

        self.use_fp16 = use_fp16

        self.gen_cls_loss = gen_cls_loss 
        
        self.sway_coeff = sway_coeff
        
        if self.gen_cls_loss:
            self.cls_pred_branch = ConformerDiscirminator(
                input_dim=(self.fake_unet.depth + 1) * self.fake_unet.dim + 3 * 512, # 3 is the number of layers from the CTC model
                num_layers=3,
                channels=self.fake_unet.dim // 2,
            )

            self.cls_pred_branch.requires_grad_(True)
        
        self.network_context_manager = torch.autocast(device_type="cuda", dtype=torch.float16) if self.use_fp16 else NoOpContext()


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

        self.vocab_char_map = vocab_char_map


             
    def compute_distribution_matching_loss(
        self, 

        inp: float["b n d"] | float["b nw"],  # mel or raw wave, ground truth latent
        text: int["b nt"] | list[str],  # text input
        *,
        second_time: torch.Tensor | None = None,  # second time step for flow prediction
        rand_span_mask: bool["b n d"] | bool["b nw"] | None = None,  # combined mask (prompt mask + padding mask)
    ):
        """
        Compute DMD loss (L_DMD) between the student distribution and teacher distribution.
        Following the DMDSpeech logic:
        - Sample time t
        - Construct noisy input phi = (1 - t)*x0 + t*x1, where x0 is noise and x1 is inp
        - Predict flows with teacher (f_phi) and student (G_theta)
        - Compute gradient that aligns student distribution with teacher distribution

        The code is adapted from F5-TTS but conceptualized per DMD:
        L_DMD encourages p_theta to match p_data via the difference between teacher and student predictions.
        """
        
        original_inp = inp
        
        with torch.no_grad():
            batch, seq_len, dtype, device = *inp.shape[:2], inp.dtype, inp.device
            
            # mel is x1
            x1 = inp

            # x0 is gaussian noise
            x0 = torch.randn_like(x1)

            # time step
            time = torch.rand((batch,), dtype=dtype, device=device)
                        
            # get flow
            t = time.unsqueeze(-1).unsqueeze(-1)
            # t = t + self.sway_coeff * (torch.cos(torch.pi / 2 * t) - 1 + t)
            sigma_t, alpha_t = (1 - t), t

            phi = (1 - t) * x0 + t * x1 # noisy x
            flow = x1 - x0 # flow target
            
            # only predict what is within the random mask span for infilling
            cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)
            
            # run at full precision as autocast and no_grad doesn't work well together 
            with self.network_context_manager:
                pred_fake = predict_flow(
                    self.fake_unet, 
                    phi, 
                    cond, # mask (prompt mask + length mask)
                    text, # text input
                    time, # time step
                    second_time=second_time,
                    cfg_strength=self.fake_guidance_scale
                )

            # pred = (x1 - x0), thus phi + (1-t) * pred = (1 - t) * x0 + t * x1 + (1 - t) * (x1 - x0) = (1 - t) * x1 + t * x1 = x1 
            pred_fake_image = phi + (1 - t) * pred_fake
            pred_fake_image[~rand_span_mask] = inp[~rand_span_mask]
                        
            with self.network_context_manager:
                pred_real = predict_flow(
                    self.real_unet, phi, cond, text, time, cfg_strength=self.real_guidance_scale
                )
                    
            pred_real_image = phi + (1 - t) * pred_real
            pred_real_image[~rand_span_mask] = inp[~rand_span_mask]

            p_real = (inp - pred_real_image)
            p_fake = (inp - pred_fake_image)
            
            grad = (p_real - p_fake) / torch.abs(p_real).mean(dim=[1, 2], keepdim=True)            
            grad = torch.nan_to_num(grad)
            
            # grad  = grad / sigma_t # pred_fake - pred_real
            # grad = grad * (1 + sigma_t / alpha_t)
            
            # grad = grad / (1 + sigma_t / alpha_t) # noise
            # grad = grad / sigma_t # score difference
            # grad = grad * alpha_t
            # grad = grad * (sigma_t ** 2 / alpha_t)
            
            # grad = grad * (alpha_t + sigma_t ** 2 / alpha_t)
        
        # The DMD loss: MSE to move student distribution closer to teacher distribution
        # Only optimize over the masked region
        loss = 0.5 * F.mse_loss(original_inp.float(), (original_inp-grad).detach().float(), reduction="none") * rand_span_mask.unsqueeze(
                -1
            )
        loss = loss.sum() / (rand_span_mask.sum() * grad.size(-1))
        
        loss_dict = {
            "loss_dm": loss 
        }

        dm_log_dict = {
            "dmtrain_time": time.detach().float(),
            "dmtrain_noisy_inp": phi.detach().float(),
            "dmtrain_pred_real_image": pred_real_image.detach().float(),
            "dmtrain_pred_fake_image": pred_fake_image.detach().float(),
            "dmtrain_grad": grad.detach().float(),
            "dmtrain_gradient_norm": torch.norm(grad).item()
        }

        return loss_dict, dm_log_dict
    
    
    def compute_ctc_sv_loss(
        self,
        real_inp: torch.Tensor,   # real data latent
        fake_inp: torch.Tensor,   # student-generated data latent
        text: torch.Tensor,
        text_lens: torch.Tensor,
        rand_span_mask: torch.Tensor,
        second_time: torch.Tensor | None = None,
    ):
        """
        Compute CTC + SV loss for direct metric optimization, as described in DMDSpeech.
        - CTC loss reduces WER
        - SV loss improves speaker similarity

        Both CTC and SV models operate on latents.
        """

        # compute CTC loss
        out, layer, ctc_loss = self.ctc_model(fake_inp * self.scale, text, text_lens)  # lengths from rand_span_mask or known

        with torch.no_grad():
            real_out, real_layers, ctc_loss_test = self.ctc_model(real_inp * self.scale, text, text_lens)
            real_logits = real_out.log_softmax(dim=2)
            # emb_real = self.sv_model(real_inp * self.scale) # snippet from prompt region            
        
        fake_logits = out.log_softmax(dim=2)
        kl_loss = F.kl_div(fake_logits, real_logits, reduction="mean", log_target=True)
        
        # For SV:
        # Extract speaker embeddings from real (prompt) and fake:
        # emb_fake = self.sv_model(fake_inp * self.scale)
        # sv_loss = 1 - F.cosine_similarity(emb_real, emb_fake, dim=-1).mean()

        input_lengths = rand_span_mask.sum(axis=-1).cpu().numpy()
        prompt_lengths = real_inp.size(1) - rand_span_mask.sum(axis=-1).cpu().numpy()

        chunks_real = []
        chunks_fake = []
        mel_len = min([int(input_lengths.min().item() - 1), 300])

        for bib in range(len(input_lengths)):
            prompt_length = int(prompt_lengths[bib].item())
            mel_length = int(input_lengths[bib].item())
            mask = rand_span_mask[bib]
            mask = torch.where(mask)[0]

            prompt_start = mask[0].cpu().numpy()
            prompt_end = mask[-1].cpu().numpy()

            if prompt_end - mel_len <= prompt_start:
                random_start = np.random.randint(0, mel_length - mel_len)
            else:
                random_start = np.random.randint(prompt_start, prompt_end - mel_len)
            
            chunks_fake.append(fake_inp[bib, random_start:random_start + mel_len, :])
            chunks_real.append(real_inp[bib, :mel_len, :])

        chunks_real = torch.stack(chunks_real, dim=0)
        chunks_fake = torch.stack(chunks_fake, dim=0)

        with torch.no_grad():
            emb_real_en = self.sv_model_en(chunks_real * self.scale)
        emb_fake_en = self.sv_model_en(chunks_fake * self.scale)

        sv_loss_en = 1 - F.cosine_similarity(emb_real_en, emb_fake_en, dim=-1).mean()

        with torch.no_grad():
            emb_real_zh = self.sv_model_zh(chunks_real * self.scale)
        emb_fake_zh = self.sv_model_zh(chunks_fake * self.scale)

        sv_loss_zh = 1 - F.cosine_similarity(emb_real_zh, emb_fake_zh, dim=-1).mean()

        sv_loss = (sv_loss_en + sv_loss_zh) / 2

        return {
            "loss_ctc": ctc_loss,
            'loss_kl': kl_loss,
            "loss_sim": sv_loss
        }, layer, real_layers

        
    
    def compute_loss_fake(
        self,
        inp: torch.Tensor, # student generator output 
        text: torch.Tensor | list[str],
        rand_span_mask: torch.Tensor,
        second_time: torch.Tensor | None = None,
    ):
        """
        Compute flow loss for the fake flow model, which is trained to estimate the flow (score) of the student distribution.
        
        This is the same as L_diff in the paper. 
        """
        
        # Similar to distribution matching, but only train fake to predict flow directly
        batch, seq_len, dtype, device = *inp.shape[:2], inp.dtype, inp.device

        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # Sample a time
        time = torch.rand((batch,), dtype=dtype, device=device)

        x1 = inp
        x0 = torch.randn_like(x1)
        t = time.unsqueeze(-1).unsqueeze(-1)
        
        phi = (1 - t) * x0 + t * x1
        flow = x1 - x0
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        with self.network_context_manager:
            pred = self.fake_unet(
                x=phi, 
                cond=cond,
                text=text, 
                time=time, 
                second_time=second_time,
                drop_audio_cond=False, 
                drop_text=False # make sure the cfg=1
            )

        # Compute MSE between predicted flow and actual flow, masked by rand_span_mask
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask].mean()
        
        loss_dict = {
            "loss_fake_mean": loss
        }
        log_dict = {
            "faketrain_noisy_inp": phi.detach().float(),
            "faketrain_x1": x1.detach().float(),
            "faketrain_pred_flow": pred.detach().float(),
        }

        return loss_dict, log_dict

    def compute_cls_logits(
        self,
        inp: torch.Tensor, # student generator output
        layer: torch.Tensor,
        text: torch.Tensor,
        rand_span_mask: torch.Tensor,
        second_time: torch.Tensor | None = None,
        guidance: bool = False,
    ):
        '''
        Compute adversarial loss logits for the generator.
        
        This is used to compute L_adv in the paper.
        
        '''
        context_no_grad = torch.no_grad if guidance else NoOpContext

        with context_no_grad():
            # If we are not doing generator classification loss, return zeros
            if not self.gen_cls_loss:
                return torch.zeros_like(inp[..., 0])  # shape (b, n)

            # For classification, we need some representation:
            # We'll mimic the logic from compute_loss_fake
            
            batch, seq_len, dtype, device = *inp.shape[:2], inp.dtype, inp.device
            if isinstance(text, list):
                if exists(self.vocab_char_map):
                    text = list_str_to_idx(text, self.vocab_char_map).to(device)
                else:
                    text = list_str_to_tensor(text).to(device)
                assert text.shape[0] == batch

            # Sample a time
            time = torch.rand((batch,), dtype=dtype, device=device)

            x1 = inp
            x0 = torch.randn_like(x1)
            t = time.unsqueeze(-1).unsqueeze(-1)
            
            phi = (1 - t) * x0 + t * x1
            cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

            with self.network_context_manager:
                layers = self.fake_unet(
                    x=phi, 
                    cond=cond,
                    text=text, 
                    time=time, 
                    second_time=second_time,
                    drop_audio_cond=False, 
                    drop_text=False, # make sure the cfg=1
                    classify_mode=True
                )
                # layers = torch.stack(layers, dim=0)

        if guidance:
            layers = [layer.detach() for layer in layers]
        layer = layer[-3:] # only use the last 3 layers
        layer = [l.transpose(-1, -2) for l in layer]
        # layer = [F.interpolate(l, mode='nearest', scale_factor=4).transpose(-1, -2) for l in layer]
        if layer[0].size(1) < layers[0].size(1):
            layer = [F.pad(l, (0, 0, 0, layers[0].size(1) - l.size(1))) for l in layer]

        layers = layer + layers
        # logits: (b, 1)
        logits = self.cls_pred_branch(layers)

        return logits, layers


    def compute_generator_cls_loss(
        self,
        inp: torch.Tensor, # student generator output
        layer: torch.Tensor,
        real_layers: torch.Tensor,
        text: torch.Tensor,
        rand_span_mask: torch.Tensor,
        second_time: torch.Tensor | None = None,
        mse_loss: bool = False,
        mse_inp: torch.Tensor | None = None,
    ):
        '''
        Compute the adversarial loss for the generator. 
        '''
        
        # Compute classification loss for generator:
        if not self.gen_cls_loss:
            return {"gen_cls_loss": 0}

        logits, fake_layers = self.compute_cls_logits(inp, layer, text, rand_span_mask, second_time, guidance=False)

        loss = ((1 - logits) ** 2).mean()

        return {"gen_cls_loss": loss, "loss_mse": 0}
    
    def compute_guidance_cls_loss(
        self,
        fake_inp: torch.Tensor,
        text: torch.Tensor,
        rand_span_mask: torch.Tensor,
        real_data: dict,
        second_time: torch.Tensor | None = None,
    ):
        '''
        This function computes the adversarial loss for the discirminator.

        The discriminator is trained to classify the generator output as real or fake.
        '''

        with torch.no_grad():
            # get layers from CTC model
            _, layer = self.ctc_model(fake_inp * self.scale)

        logits_fake, _ = self.compute_cls_logits(fake_inp.detach(), layer, text, rand_span_mask, second_time, guidance=True)
        loss_fake = (logits_fake**2).mean()

        real_inp = real_data["inp"]

        with torch.no_grad():
            # get layers from CTC model
            _, layer = self.ctc_model(real_inp * self.scale)
        
        logits_real, _ = self.compute_cls_logits(real_inp.detach(), layer, text, rand_span_mask, second_time, guidance=True)
        loss_real = ((1 - logits_real)**2).mean()

        classification_loss = loss_real + loss_fake

        loss_dict = {
            "guidance_cls_loss": classification_loss
        }
        log_dict = {
            "pred_realism_on_real": loss_real.detach().item(),
            "pred_realism_on_fake": loss_fake.detach().item()
        }

        return loss_dict, log_dict

    def generator_forward(
        self,
        inp: torch.Tensor,
        text: torch.Tensor,
        text_lens: torch.Tensor,
        text_normalized: torch.Tensor,
        text_normalized_lens: torch.Tensor,
        rand_span_mask: torch.Tensor,
        real_data: dict | None = None, # ground truth data (primarily prompt) to compute SV loss
        second_time: torch.Tensor | None = None,
        mse_loss: bool = False,
    ):
        '''
        Forward pass for the generator.
        
        This function computes the loss for the generator, which includes:
        - Distribution matching loss (L_DMD)
        - Adversarial generator loss (L_adv(G; D))
        - CTC/SV loss (L_ctc + L_sv)
        '''
        
        # 1. Compute DM loss
        dm_loss_dict, dm_log_dict = self.compute_distribution_matching_loss(inp, text, rand_span_mask=rand_span_mask, second_time=second_time)

        ctc_sv_loss_dict = {}
        cls_loss_dict = {}

        # 2. Compute optional CTC/SV loss if real_data provided
        if real_data is not None:
            real_inp = real_data["inp"]
            ctc_sv_loss_dict, layer, real_layers = self.compute_ctc_sv_loss(real_inp, inp, text_normalized, text_normalized_lens, rand_span_mask, second_time=second_time)

            # 3. Compute optional classification loss
            if self.gen_cls_loss:
                cls_loss_dict = self.compute_generator_cls_loss(inp, layer, real_layers, text,
                                                                    rand_span_mask=rand_span_mask, 
                                                                    second_time=second_time,
                                                                    mse_inp = real_data["inp"] if real_data is not None else None,
                                                                    mse_loss = mse_loss,
                                                                    )


        loss_dict = {**dm_loss_dict, **cls_loss_dict, **ctc_sv_loss_dict}
        log_dict = {**dm_log_dict}

        return loss_dict, log_dict

    def guidance_forward(
        self,
        fake_inp: torch.Tensor,
        text: torch.Tensor,
        text_lens: torch.Tensor,
        rand_span_mask: torch.Tensor,
        real_data: dict | None = None,
        second_time: torch.Tensor | None = None,
    ):
        '''
        Forward pass for the guidnce module (discriminator + fake flow function).
        
        This function computes the loss for the guidance module, which includes:
        - Flow matching loss (L_diff)
        - Adversarial discrminator loss (L_adv(D; G))
        
        '''
        
        # Compute fake loss (like epsilon prediction loss in Guidance)
        fake_loss_dict, fake_log_dict = self.compute_loss_fake(fake_inp, text, rand_span_mask=rand_span_mask, second_time=second_time)

        # If gen_cls_loss, compute guidance cls loss
        cls_loss_dict = {}
        cls_log_dict = {}
        if self.gen_cls_loss and real_data is not None:
            cls_loss_dict, cls_log_dict = self.compute_guidance_cls_loss(fake_inp, text, rand_span_mask, real_data, second_time=second_time)

        loss_dict = {**fake_loss_dict, **cls_loss_dict}
        log_dict = {**fake_log_dict, **cls_log_dict}

        return loss_dict, log_dict
    
    def forward(
        self,
        generator_turn=False,
        guidance_turn=False,
        generator_data_dict=None,
        guidance_data_dict=None
    ):
        if generator_turn:
            loss_dict, log_dict = self.generator_forward(
                inp=generator_data_dict["inp"],
                text=generator_data_dict["text"],
                text_lens=generator_data_dict["text_lens"],
                text_normalized=generator_data_dict["text_normalized"],
                text_normalized_lens=generator_data_dict["text_normalized_lens"],
                rand_span_mask=generator_data_dict["rand_span_mask"],
                real_data=generator_data_dict.get("real_data", None),
                second_time=generator_data_dict.get("second_time", None),
                mse_loss=generator_data_dict.get("mse_loss", False),
            )
        elif guidance_turn:
            loss_dict, log_dict = self.guidance_forward(
                fake_inp=guidance_data_dict["inp"],
                text=guidance_data_dict["text"],
                text_lens=guidance_data_dict["text_lens"],
                rand_span_mask=guidance_data_dict["rand_span_mask"],
                real_data=guidance_data_dict.get("real_data", None),
                second_time=guidance_data_dict.get("second_time", None),
            )
        else:
            raise NotImplementedError("Must specify either generator_turn or guidance_turn")

        return loss_dict, log_dict



    
if __name__ == "__main__":
    from f5_tts.model.utils import get_tokenizer


    bsz = 16
    
    tokenizer = "pinyin"  # 'pinyin', 'char', or 'custom'
    tokenizer_path = None  # if tokenizer = 'custom', define the path to the tokenizer you want to use (should be vocab.txt)
    dataset_name = "Emilia_ZH_EN"
    if tokenizer == "custom":
        tokenizer_path = tokenizer_path
    else:
        tokenizer_path = dataset_name
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)
    
    
    real_unet = DiT(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4, text_num_embeds=vocab_size, mel_dim=100)
    fake_unet = DiT(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4, text_num_embeds=vocab_size, mel_dim=100)
    
    guidance = Guidance(real_unet, 
                        fake_unet,
                        real_guidance_scale=1.0,
                        fake_guidance_scale=0.0,
                        use_fp16=True,
                        gen_cls_loss=True, 
                        ).cuda()
        
    text = ["hello world"] * bsz
    lens = torch.randint(1, 1000, (bsz,)).cuda()
    inp = torch.randn(bsz, lens.max(), 80).cuda()
    
    batch, seq_len, dtype, device = *inp.shape[:2], inp.dtype, inp.device
    
    # handle text as string
    if isinstance(text, list):
        if exists(vocab_char_map):
            text = list_str_to_idx(text, vocab_char_map).to(device)
        else:
            text = list_str_to_tensor(text).to(device)
        assert text.shape[0] == batch

    # lens and mask
    if not exists(lens):
        lens = torch.full((batch,), seq_len, device=device)

    mask = lens_to_mask(lens, length=seq_len)  # useless here, as collate_fn will pad to max length in batch
    frac_lengths_mask = (0.7, 1.0)
    
    # get a random span to mask out for training conditionally
    frac_lengths = torch.zeros((batch,), device=device).float().uniform_(*frac_lengths_mask)
    rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)
                
    if exists(mask):
        rand_span_mask &= mask
        
    # Construct data dicts for generator and guidance phases
    # For flow, `real_data` can just be the ground truth if available; here we simulate it
    real_data_dict = {
        "inp": torch.zeros_like(inp),  # simulating real data
    }

    generator_data_dict = {
        "inp": inp,
        "text": text,
        "rand_span_mask": rand_span_mask,
        "real_data": real_data_dict
    }

    guidance_data_dict = {
        "inp": inp,
        "text": text,
        "rand_span_mask": rand_span_mask,
        "real_data": real_data_dict
    }

        
    # Generator forward pass
    loss_dict, log_dict = guidance(generator_turn=True, generator_data_dict=generator_data_dict)
    print("Generator turn losses:", loss_dict)

    # Guidance forward pass
    loss_dict, log_dict = guidance(guidance_turn=True, guidance_data_dict=guidance_data_dict)
    print("Guidance turn losses:", loss_dict)
