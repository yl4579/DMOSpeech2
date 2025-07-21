from __future__ import annotations

import gc
import os

import math

import torch
import torchaudio
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, SequentialSampler, Subset  # <-- Added Subset import
from tqdm import tqdm

import torch.nn.functional as F

from f5_tts.model import CFM
from f5_tts.model.dataset import collate_fn, DynamicBatchSampler
from f5_tts.model.utils import default, exists

from duration_predictor import calculate_remaining_lengths

# trainer

from f5_tts.model.utils import (
    default,
    exists,
    list_str_to_idx,
    list_str_to_tensor,
    lens_to_mask,
    mask_from_frac_lengths,
)

SAMPLE_RATE = 24_000


def masked_l1_loss(est_lengths, tar_lengths):
    first_zero_idx = (tar_lengths == 0).int().argmax(dim=1)  
    B, L = tar_lengths.shape
    range_tensor = torch.arange(L, device=tar_lengths.device).expand(B, L)  
    mask = range_tensor <= first_zero_idx[:, None]  # Include the first 0
    loss = F.l1_loss(est_lengths, tar_lengths, reduction='none')  # (B, L)
    loss = loss * mask  # Zero out ignored positions
    loss = loss.sum() / mask.sum()  # Normalize by valid elements
    return loss


def masked_cross_entropy_loss(est_length_logits, tar_length_labels):
    first_zero_idx = (tar_length_labels == 0).int().argmax(dim=1)
    B, L = tar_length_labels.shape
    range_tensor = torch.arange(L, device=tar_length_labels.device).expand(B, L)
    mask = range_tensor <= first_zero_idx[:, None]  # Include the first 0
    loss = F.cross_entropy(
        est_length_logits.reshape(-1, est_length_logits.size(-1)), 
        tar_length_labels.reshape(-1), 
        reduction='none'
    ).reshape(B, L)
    loss = loss * mask
    loss = loss.sum() / mask.sum()
    return loss


class Trainer:
    def __init__(
        self,
        model,
        vocab_size,
        vocab_char_map,
        process_token_to_id=True,
        loss_fn='L1',
        lambda_L1=1,
        gumbel_tau=0.5,
        n_class=301,
        n_frame_per_class=10,
        epochs=15,
        learning_rate=1e-4,
        num_warmup_updates=20000,
        save_per_updates=1000,
        checkpoint_path=None,
        batch_size=32,
        batch_size_type: str = "sample",
        max_samples=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        logger: str | None = "wandb",  # "wandb" | "tensorboard" | None
        wandb_project="test_e2-tts",
        wandb_run_name="test_run",
        wandb_resume_id: str = None,
        last_per_steps=None,
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

        if logger == "wandb" and not wandb.api.api_key:
            logger = None
        print(f"Using logger: {logger}")

        self.accelerator = Accelerator(
            log_with=logger if logger == "wandb" else None,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs,
        )

        self.logger = logger
        if self.logger == "wandb":
            if exists(wandb_resume_id):
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, "id": wandb_resume_id}}
            else:
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name}}

            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config={
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "num_warmup_updates": num_warmup_updates,
                    "batch_size": batch_size,
                    "batch_size_type": batch_size_type,
                    "max_samples": max_samples,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "gpus": self.accelerator.num_processes,
                },
            )

        elif self.logger == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=f"runs/{wandb_run_name}")

        self.model = model
        self.vocab_size = vocab_size
        self.vocab_char_map = vocab_char_map
        self.process_token_to_id = process_token_to_id
        assert loss_fn in ['L1', 'CE', 'L1_and_CE']
        self.loss_fn = loss_fn
        self.lambda_L1 = lambda_L1
        self.n_class = n_class
        self.n_frame_per_class = n_frame_per_class
        self.gumbel_tau = gumbel_tau

        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.last_per_steps = default(last_per_steps, save_per_updates * grad_accumulation_steps)
        self.checkpoint_path = default(checkpoint_path, "ckpts/test_e2-tts")

        self.batch_size = batch_size
        self.batch_size_type = batch_size_type
        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        if bnb_optimizer:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        
    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_checkpoint(self, step, last=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:  
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.accelerator.unwrap_model(self.optimizer).state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                step=step,
            )
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if last:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_last.pt")
            else:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_{step}.pt")

    def load_checkpoint(self):
        if (
            not exists(self.checkpoint_path)
            or not os.path.exists(self.checkpoint_path)
            or not any(filename.endswith(".pt") for filename in os.listdir(self.checkpoint_path))
        ):
            return 0

        self.accelerator.wait_for_everyone()
        if "model_last.pt" in os.listdir(self.checkpoint_path):
            latest_checkpoint = "model_last.pt"
        else:
            latest_checkpoint = sorted(
                [f for f in os.listdir(self.checkpoint_path) if f.endswith(".pt")],
                key=lambda x: int("".join(filter(str.isdigit, x))),
            )[-1]

        print(f'To load from {latest_checkpoint}.')

        # checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", map_location=self.accelerator.device)  # rather use accelerator.load_state ಥ_ಥ
        checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", weights_only=True, map_location="cpu")

        print(f'Loaded from {latest_checkpoint}.')

        if "step" in checkpoint:
            # patch for backward compatibility, 305e3ea
            for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
                if key in checkpoint["model_state_dict"]:
                    del checkpoint["model_state_dict"][key]

            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            self.accelerator.unwrap_model(self.optimizer).load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            step = checkpoint["step"]
        else:
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "step"]
            }
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            step = 0
        
        del checkpoint
        gc.collect()

        print(f'Exit load_checkpoint.')

        return step


    def validate(self, valid_dataloader, global_step):
        """
        Runs evaluation on the validation set, computes the average loss,
        and logs the average validation loss along with the CTC decoded strings.
        """
        self.model.eval()
        total_valid_loss = 0.0
        total_sec_error = 0.0
        count = 0
        # Iterate over the validation dataloader
        with torch.no_grad():
            for batch in valid_dataloader:
                # Inputs
                mel = batch['mel'].permute(0, 2, 1) # (B, L_mel, D)
                text = batch['text']

                if self.process_token_to_id:
                    text_ids = list_str_to_idx(text, self.vocab_char_map).to(mel.device)
                    text_ids = text_ids.masked_fill(text_ids==-1, self.vocab_size)
                else:
                    text_ids = text

                # Targets
                mel_lengths = batch['mel_lengths']
                tar_lengths = calculate_remaining_lengths(mel_lengths)
                predictions = self.model(text_ids=text_ids, mel=mel)

                if self.loss_fn == 'L1':
                    est_lengths = predictions
                    loss = masked_l1_loss(
                        est_lengths=est_lengths, tar_lengths=tar_lengths
                    )
                    frame_error = loss

                elif self.loss_fn == 'CE':
                    tar_length_labels = (tar_lengths // self.n_frame_per_class) \
                        .clamp(min=0, max=self.n_class-1) # [0, 1, ..., n_class-1]
                    est_length_logtis = predictions
                    est_length_labels = torch.argmax(est_length_logtis, dim=-1)
                    loss = masked_cross_entropy_loss(
                        est_length_logits=est_length_logtis, tar_length_labels=tar_length_labels
                    )
                    est_lengths = est_length_labels * self.n_frame_per_class
                    frame_error = masked_l1_loss(
                        est_lengths=est_lengths, tar_lengths=tar_lengths
                    )

                elif self.loss_fn == 'L1_and_CE':
                    tar_length_labels = (tar_lengths // self.n_frame_per_class) \
                        .clamp(min=0, max=self.n_class-1) # [0, 1, ..., n_class-1]
                    est_length_logtis = predictions
                    est_length_1hots = F.gumbel_softmax(
                        est_length_logtis, tau=self.gumbel_tau, hard=True, dim=-1
                    )
                    length_values = torch.arange(
                        self.n_class, device=est_length_1hots.device
                    ).float() * self.n_frame_per_class
                    est_lengths = (est_length_1hots * length_values).sum(-1)

                    loss_CE = masked_cross_entropy_loss(
                        est_length_logits=est_length_logtis, tar_length_labels=tar_length_labels
                    )

                    loss_L1 = masked_l1_loss(
                        est_lengths=est_lengths, tar_lengths=tar_lengths
                    )

                    loss = loss_CE + self.lambda_L1 * loss_L1

                    frame_error = loss_L1

                else:
                    raise NotImplementedError(self.loss_fn)

                sec_error = frame_error * 256 / 24000
                total_sec_error += sec_error.item()
                total_valid_loss += loss.item()
                count += 1

        avg_valid_loss = total_valid_loss / count if count > 0 else 0.0
        avg_valid_sec_error = total_sec_error / count if count > 0 else 0.0
        # Log validation metrics
        self.accelerator.log(
            {
                f"valid_loss": avg_valid_loss,
                f"valid_sec_error": avg_valid_sec_error
            }, 
            step=global_step
        )
 
        self.model.train()


    def train(self, train_dataset: Dataset, valid_dataset: Dataset,
        num_workers=64, resumable_with_seed: int = None):
        if exists(resumable_with_seed):
            generator = torch.Generator()
            generator.manual_seed(resumable_with_seed)
        else:
            generator = None

        # Create training dataloader using the appropriate batching strategy
        if self.batch_size_type == "sample":
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_size=self.batch_size,
                shuffle=True,
                generator=generator,
            )
            # Create validation dataloader (always sequential, no shuffling)
            valid_dataloader = DataLoader(
                valid_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                batch_size=self.batch_size,
                shuffle=False,
            )

        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False

            sampler = SequentialSampler(train_dataset)
            batch_sampler = DynamicBatchSampler(
                sampler, self.batch_size, max_samples=self.max_samples, random_seed=resumable_with_seed, drop_last=False
            )
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_sampler=batch_sampler,
            )

            sampler = SequentialSampler(valid_dataset)
            batch_sampler = DynamicBatchSampler(
                sampler, self.batch_size, max_samples=self.max_samples, random_seed=resumable_with_seed, drop_last=False
            )
            # Create validation dataloader (always sequential, no shuffling)
            valid_dataloader = DataLoader(
                valid_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,  
                persistent_workers=True,
                batch_sampler=batch_sampler,
            )
        else:
            raise ValueError(f"batch_size_type must be either 'sample' or 'frame', but received {self.batch_size_type}")
        
        #  accelerator.prepare() dispatches batches to devices;
        #  which means the length of dataloader calculated before, should consider the number of devices
        warmup_steps = (
            self.num_warmup_updates * self.accelerator.num_processes
        )  # consider a fixed warmup steps while using accelerate multi-gpu ddp
        # otherwise by default with split_batches=False, warmup steps change with num_processes
        total_steps = len(train_dataloader) * self.epochs / self.grad_accumulation_steps
        decay_steps = total_steps - warmup_steps
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_steps)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_steps]
        )
        train_dataloader, self.scheduler = self.accelerator.prepare(
            train_dataloader, self.scheduler
        )  # actual steps = 1 gpu steps / gpus
        start_step = self.load_checkpoint()
        global_step = start_step

        valid_dataloader = self.accelerator.prepare(valid_dataloader)

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0

        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            if exists(resumable_with_seed) and epoch == skipped_epoch:
                progress_bar = tqdm(
                    skipped_dataloader,
                    desc=f"Epoch {epoch+1}/{self.epochs}",
                    unit="step",
                    disable=not self.accelerator.is_local_main_process,
                    initial=skipped_batch,
                    total=orig_epoch_step,
                )
            else:
                progress_bar = tqdm(
                    train_dataloader,
                    desc=f"Epoch {epoch+1}/{self.epochs}",
                    unit="step",
                    disable=not self.accelerator.is_local_main_process,
                )

            for batch in progress_bar:
                with self.accelerator.accumulate(self.model):
                    # Inputs
                    mel = batch['mel'].permute(0, 2, 1) # (B, L_mel, D)
                    text = batch['text']

                    if self.process_token_to_id:
                        text_ids = list_str_to_idx(text, self.vocab_char_map).to(mel.device)
                        text_ids = text_ids.masked_fill(text_ids==-1, self.vocab_size)
                    else:
                        text_ids = text

                    # Targets
                    mel_lengths = batch['mel_lengths']
                    tar_lengths = calculate_remaining_lengths(mel_lengths)
                    predictions = self.model(text_ids=text_ids, mel=mel)

                    if self.loss_fn == 'L1':
                        est_lengths = predictions
                        loss = masked_l1_loss(
                            est_lengths=est_lengths, tar_lengths=tar_lengths
                        )

                        with torch.no_grad():
                            frame_error = loss
                            sec_error = frame_error * 256 / 24000

                        log_dict = {
                            'loss': loss.item(), 
                            'loss_L1': loss.item(), 
                            'sec_error': sec_error.item(),
                            'lr': self.scheduler.get_last_lr()[0]
                         }

                    elif self.loss_fn == 'CE':
                        tar_length_labels = (tar_lengths // self.n_frame_per_class) \
                            .clamp(min=0, max=self.n_class-1) # [0, 1, ..., n_class-1]
                        est_length_logtis = predictions
                        est_length_labels = torch.argmax(est_length_logtis, dim=-1)
                        loss = masked_cross_entropy_loss(
                            est_length_logits=est_length_logtis, tar_length_labels=tar_length_labels
                        )
                        with torch.no_grad():
                            est_lengths = est_length_labels * self.n_frame_per_class
                            frame_error = masked_l1_loss(
                                est_lengths=est_lengths, tar_lengths=tar_lengths
                            )
                            sec_error = frame_error * 256 / 24000

                        log_dict = {
                            'loss': loss.item(), 
                            'loss_CE': loss.item(), 
                            'sec_error': sec_error.item(),
                            'lr': self.scheduler.get_last_lr()[0]
                         }

                    elif self.loss_fn == 'L1_and_CE':
                        tar_length_labels = (tar_lengths // self.n_frame_per_class) \
                            .clamp(min=0, max=self.n_class-1) # [0, 1, ..., n_class-1]
                        est_length_logtis = predictions
                        est_length_1hots = F.gumbel_softmax(
                            est_length_logtis, tau=self.gumbel_tau, hard=True, dim=-1
                        )
                        length_values = torch.arange(
                            self.n_class, device=est_length_1hots.device
                        ).float() * self.n_frame_per_class
                        est_lengths = (est_length_1hots * length_values).sum(-1)

                        loss_CE = masked_cross_entropy_loss(
                            est_length_logits=est_length_logtis, tar_length_labels=tar_length_labels
                        )

                        loss_L1 =  masked_l1_loss(
                            est_lengths=est_lengths, tar_lengths=tar_lengths
                        )

                        loss = loss_CE + self.lambda_L1 * loss_L1

                        with torch.no_grad():
                            frame_error = loss_L1
                            sec_error = frame_error * 256 / 24000

                        log_dict = {
                            'loss': loss.item(), 
                            'loss_L1': loss_L1.item(), 
                            'loss_CE': loss_CE.item(), 
                            'sec_error': sec_error.item(),
                            'lr': self.scheduler.get_last_lr()[0]
                        }

                    else:
                        raise NotImplementedError(self.loss_fn)

        
                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                global_step += 1

                if self.accelerator.is_local_main_process:
                    self.accelerator.log(log_dict, step=global_step)
                progress_bar.set_postfix(step=str(global_step), loss=loss.item())

                if global_step % (self.save_per_updates * self.grad_accumulation_steps) == 0:
                    self.save_checkpoint(global_step)
                    # if self.log_samples and self.accelerator.is_local_main_process:
                    # Run validation at the end of each epoch (only on the main process)
                    if self.accelerator.is_local_main_process:
                        self.validate(valid_dataloader, global_step)
                # if global_step % self.last_per_steps == 0:
                #     self.save_checkpoint(global_step, last=True)

        self.save_checkpoint(global_step, last=True)
        self.accelerator.end_training()
