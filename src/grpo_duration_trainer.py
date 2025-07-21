import os
import gc
import json
import random
import time
import io
import copy
from typing import List, Dict, Any, Optional, Callable, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, SequentialSampler, Subset
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import wandb

from f5_tts.model.dataset import collate_fn, DynamicBatchSampler
from f5_tts.model.utils import list_str_to_idx

# torch.autograd.set_detect_anomaly(True)
# os.environ['HYDRA_FULL_ERROR'] = 'True'


def safe_sample(logits, temperature=1.0):
    """
    logits: Tensor of shape (B, n_class)
    temperature: Sampling temperature (higher => more random)
    """
    # Apply temperature scaling
    scaled_logits = logits / temperature
    
    # Compute categorical distribution
    probs = F.softmax(scaled_logits, dim=-1)
    
    # Sample from the distribution once per batch element
    samples = torch.multinomial(probs, num_samples=1)  # (B, 1)
    
    # Convert to one-hot encoding
    one_hot_samples = torch.zeros_like(probs).scatter_(1, samples, 1)
    
    return one_hot_samples


class GRPODurationTrainer:
    """
    Trainer class that implements GRPO (Generative Reinforcement Learning from Preference Optimization)
    for a duration predictor in text-to-speech synthesis.
    """
    def __init__(
        self,
        model,                      # Duration predictor model
        inference_fn,               # Function to generate speech
        reward_fn,                  # Function to compute rewards from generated speech
        
        vocab_size: int,            # Size of the vocabulary
        vocab_char_map: dict,       # Mapping from characters to token IDs

        # Duration model parameters
        n_class: int = 301,         # Number of duration classes
        n_frame_per_class: int = 10, # Number of frames per class
        gumbel_tau: int = 0.7,
        
        # GRPO parameters
        beta: float = 0.04,         # KL regularization weight
        clip_param: float = 0.2,    # PPO clip parameter
        num_pre_samples: int = 8,   # Number of samples per prompt
        compute_gen_logps: bool = True, # Whether to compute generation log probabilities
        
        # Training parameters
        learning_rate: float = 5e-6,
        num_warmup_updates: int = 10000,
        save_per_updates: int = 10000,
        checkpoint_path: Optional[str] = None,
        all_steps: int = 100000,     # Total training steps
        
        # Batch parameters
        batch_size: int = 8,
        batch_size_type: str = "sample",
        max_samples: int = 16,
        grad_accumulation_steps: int = 2,
        max_grad_norm: float = 1.0,
        
        # Logging parameters
        logger: Optional[str] = "wandb",
        wandb_project: str = "tts-duration-grpo",
        wandb_run_name: str = "grpo_run",
        wandb_resume_id: Optional[str] = None,
        
        accelerate_kwargs: dict = dict(),
    ):
        # Initialize accelerator for distributed training
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
            if wandb_resume_id:
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, "id": wandb_resume_id}}
            else:
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name}}

            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config={
                    "learning_rate": learning_rate,
                    "num_warmup_updates": num_warmup_updates,
                    "batch_size": batch_size,
                    "beta": beta,
                    "clip_param": clip_param,
                    "num_pre_samples": num_pre_samples,
                    "n_class": n_class,
                    "n_frame_per_class": n_frame_per_class,
                    "all_steps": all_steps,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "gpus": self.accelerator.num_processes,
                },
            )
        elif self.logger == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=f"runs/{wandb_run_name}")

        # Store model, inference function, and reward function
        self.model = model
        
        # Create reference model (frozen clone of the initial model)
        self.ref_model = copy.deepcopy(model)
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()
        
        # prepare inference_fn
        self.inference_fn = inference_fn
        self.inference_fn.scale = self.inference_fn.scale.to(self.accelerator.device)
        self.inference_fn.tts_model = self.inference_fn.tts_model.to(self.accelerator.device)
        # prepare reward_fn
        self.reward_fn = reward_fn
        
        # Store vocabulary and mapping
        self.vocab_size = vocab_size
        self.vocab_char_map = vocab_char_map

        # Store duration model parameters
        self.n_class = n_class
        self.n_frame_per_class = n_frame_per_class
        self.gumbel_tau = gumbel_tau
        
        # Store GRPO parameters
        self.beta = beta
        self.clip_param = clip_param
        self.num_pre_samples = num_pre_samples
        self.compute_gen_logps = compute_gen_logps
        
        # Store training parameters
        self.learning_rate = learning_rate
        self.num_warmup_updates: int = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.checkpoint_path = checkpoint_path or f"ckpts/{wandb_run_name}"
        self.all_steps = all_steps
        
        # Store batch parameters
        self.batch_size = batch_size
        self.batch_size_type = batch_size_type
        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # Initialize optimizer
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        
        # Prepare model and optimizer with accelerator
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        self.ref_model = self.accelerator.prepare(self.ref_model)
        self.reward_fn, self.inference_fn = self.accelerator.prepare(self.reward_fn, self.inference_fn)        
        
        # GRPO batch queue
        self.batch_queue = []
        
        # Store distributed rank
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        self.device = f'cuda:{self.rank}'
        
    @property
    def is_main(self):
        return self.accelerator.is_main_process
    
    def save_checkpoint(self, step, last=False):
        """Save model and optimizer state"""
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.accelerator.unwrap_model(self.optimizer).state_dict(),
                scheduler_state_dict=self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
                step=step,
            )
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if last:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_last.pt")
            else:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_{step}.pt")
    
    def load_checkpoint(self):
        """Load latest checkpoint if available"""
        if (
            not self.checkpoint_path
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

        print(f'Loading checkpoint: {latest_checkpoint}')
        checkpoint = torch.load(
            f"{self.checkpoint_path}/{latest_checkpoint}", 
            weights_only=True, 
            map_location="cpu"
        )

        if "step" in checkpoint:
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            self.accelerator.unwrap_model(self.optimizer).load_state_dict(checkpoint["optimizer_state_dict"])
            if hasattr(self, 'scheduler') and checkpoint["scheduler_state_dict"]:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            step = checkpoint["step"]
        else:
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            step = 0
        
        del checkpoint
        gc.collect()
        
        print(f'Successfully loaded checkpoint at step {step}')
        return step
    
    @torch.no_grad()
    def get_ref_logps(self, text_ids, mel, sampled_classes):
        """
        Get log probabilities from the reference model for the sampled classes
        """
        B = text_ids.shape[0]
        K = self.num_pre_samples
        with torch.no_grad():
            ref_logits = self.ref_model(text_ids=text_ids, mel=mel)[:, -1, :]
            ref_logits = ref_logits.unsqueeze(1).repeat(1, K, 1).view(B*K, -1)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            ref_logps = torch.gather(
                ref_log_probs, 
                dim=-1, 
                index=sampled_classes.unsqueeze(-1)
            ).squeeze(-1)
        return ref_logps
    
    @torch.no_grad()
    def generate_duration_samples(self, batch_inputs):
        """
        Generate multiple duration predictions from the model for each input
        and evaluate them using the inference function and reward model
        
        Args:
            batch_inputs: Dictionary with text, prompt audio, etc.
            
        Returns:
            Dictionary with duration samples, rewards, and reference logits
        """

        if self.rank == 0:
            print("Generating duration samples...")
        
        # all_logits = []
        all_text_ids = []
        all_mels = []
        all_sampled_classes = []
        all_durations = []
        all_rewards = []
        all_gen_logps = []

        all_ctc_loss = []
        all_sv_loss = []

        # Fetch batch inputs
        # prompt_mel = batch_inputs['mel'].permute(0, 2, 1).to(self.device)
        prompt_mel = batch_inputs['mel'].permute(0, 2, 1) # (B, T, 100)
        prompt_text = batch_inputs['text']

        batch_size = prompt_mel.shape[0]

        # Shift text to unpair 'mel' and 'text'; The shifted text will be synthesized
        target_text = batch_inputs['target_text']
        target_text_lengths = torch.LongTensor([len(t) for t in target_text]).to(prompt_mel.device)
        try:
            full_text = [prompt+[' ']+target for prompt, target in zip(prompt_text, target_text)]
        except:
            target_text = [batch_inputs['text'][-1]] + batch_inputs['text'][:-1]
            target_text_lengths = batch_inputs['text_lengths'].clone().roll(1, 0)
            full_text = [prompt+[' ']+target for prompt, target in zip(prompt_text, target_text)]

        # Goes to reward model
        target_text_ids = list_str_to_idx(target_text, self.vocab_char_map).to(self.accelerator.device)    # to device, the dataloader only gives list

        # Goes to duration model and TTS
        full_text_ids = list_str_to_idx(full_text, self.vocab_char_map).to(self.accelerator.device)

        # Deepcopy to separate text_ids for SLP and TTS
        slp_text_ids = full_text_ids.detach().clone()
        slp_text_ids = slp_text_ids.masked_fill(slp_text_ids==-1, self.vocab_size) # (B, L)

        # Pre-compute duration logits
        K = self.num_pre_samples
        B, T, _ = prompt_mel.shape
        _, L = slp_text_ids.shape
        # prompt_mel_k_repeats = prompt_mel.unsqueeze(1).repeat(1, K, 1, 1)  # (B, K, T, 100)
        # slp_text_ids_k_repeats = slp_text_ids.unsqueeze(1).repeat(1, K, 1)  # (B, K, L)

        # Run model once for B inputs
        old_logits = self.model(
            text_ids=slp_text_ids, # (B, L)
            mel=prompt_mel         # (B, T, 100)
        )[:, -1, :]  # (B, n_class)

        # Repeat each result K times along batch dimension
        old_logits = old_logits.unsqueeze(1).repeat(1, K, 1) # (B, K, n_class)
        # logits_nograd = logits_grad.detach().clone().view(B, K, -1) # (B, K, n_class)

        for _full_text_ids, _target_text_ids, _target_text_lengths, \
            _prompt_mel, _old_logits in zip(
                full_text_ids, target_text_ids, target_text_lengths, 
                prompt_mel, old_logits
            ):

            duration_sample = F.gumbel_softmax(_old_logits, tau=self.gumbel_tau, hard=True, dim=-1)
            duration2frames = torch.arange(self.n_class).float().to(self.accelerator.device) * self.n_frame_per_class
            est_frames = (duration_sample * duration2frames).sum(-1) # (K, )

            # Compute log probabilities of the samples
            sampled_classes = duration_sample.argmax(dim=-1)
            log_probs = F.log_softmax(_old_logits, dim=-1)
            gen_logps = torch.gather(
                log_probs, 
                dim=-1, 
                index=sampled_classes.unsqueeze(-1)
            ).squeeze(-1)  # Shape: [K, n_class]
            
            # Generate speech using the sampled durations
            sampled_rewards = []

            for i in range(K):
                cur_duration = est_frames[i]
                if cur_duration == 0:
                    cur_duration = cur_duration + 50 # prevent 0 duration
                infer_full_text_ids = _full_text_ids.unsqueeze(0)
                infer_prompt_mel = _prompt_mel.unsqueeze(0)
                cur_duration = cur_duration.unsqueeze(0)
                infer_target_text_ids = _target_text_ids.unsqueeze(0)
                infer_target_text_lengths = _target_text_lengths.unsqueeze(0)
                with torch.inference_mode():
                    try:
                        _est_mel = self.inference_fn(
                            full_text_ids=infer_full_text_ids, 
                            prompt_mel=infer_prompt_mel, 
                            target_duration=cur_duration, 
                            teacher_steps=0
                        )
                        _est_mel = _est_mel.permute(0, 2, 1) # (1, T, 100)
                        
                        loss_dict = self.reward_fn(
                            prompt_mel=infer_prompt_mel,
                            est_mel=_est_mel,
                            target_text_id=infer_target_text_ids,
                            target_text_length=infer_target_text_lengths
                        )
                        # #TODO reweight the loss for reward
                        reward_sim = loss_dict['loss_sim'] # 0 to 1
                        reward_ctc = loss_dict['loss_ctc']
                        reward = -(reward_ctc + reward_sim * 3)
                        all_ctc_loss.append(reward_ctc)
                        all_sv_loss.append(reward_sim)
                    except Exception as e:
                        if self.rank == 0:
                            print(f"Error in speech synthesis: {e}")
                        reward = torch.tensor(-1.0).to(cur_duration.device)
                        
                    sampled_rewards.append(reward)
                        # list with length of K
            sampled_rewards = torch.stack(sampled_rewards)  # (K, )
            # Normalize rewards
            if (sampled_rewards.max() - sampled_rewards.min()).item() > 1e-6:
                sampled_rewards = (sampled_rewards - sampled_rewards.mean()) / (sampled_rewards.std() + 1e-8)

            # Store all data
            # all_logits.append(duration_logits)
            # all_text_ids.append(duration_input_expanded["text_ids"])
            # all_mels.append(duration_input_expanded["mel"])
            all_sampled_classes.append(sampled_classes)
            all_durations.append(est_frames)
            all_gen_logps.append(gen_logps)
            all_rewards.extend(sampled_rewards)  # list with length of B*K
        
        # Concatenate all data
        # logits = torch.cat(all_logits, dim=0)
        # text_ids = torch.cat(all_text_ids, dim=0)
        # mels = torch.cat(all_mels, dim=0)
        sampled_classes = torch.cat(all_sampled_classes, dim=0)
        durations = torch.cat(all_durations, dim=0)
        rewards = torch.stack(all_rewards)    # use stack to keep the same device of elements
        gen_logps = torch.cat(all_gen_logps, dim=0)

        ctc_losses = torch.stack(all_ctc_loss)
        sv_losses = torch.stack(all_sv_loss)
        
        if self.is_main:
            self.accelerator.log({
                "ctc_loss": ctc_losses.mean().item(),
                "sv_loss": sv_losses.mean().item(),
                "reward": rewards.mean().item(),
                "reward_min": rewards.min().item(),
                "reward_max": rewards.max().item(),
            }, step=self.global_step)

        # # Normalize rewards
        # if (rewards.max() - rewards.min()).item() > 1e-6:
        #     rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        ref_logps = self.get_ref_logps(slp_text_ids, prompt_mel, sampled_classes)

        # Create batch dict similar to Qwen2.5 implementation
        batch_outputs = {
            # "logits": logits_grad,
            "text_ids": slp_text_ids,
            "prompt_mel": prompt_mel,
            "rewards": rewards,
            "refs": ref_logps,
            "sampled_classes": sampled_classes,
            "durations": durations,
        }
        
        if self.compute_gen_logps:
            batch_outputs["gen_logps"] = gen_logps
        
        if self.rank == 0:
            print(f"Generated {len(rewards)} samples with reward min/mean/max: {rewards.min().item():.4f}/{rewards.mean().item():.4f}/{rewards.max().item():.4f}")
        
        return batch_outputs
    
    def GRPO_step(self, batch):
        """
        Perform a GRPO update step
        
        Args:
            batch: Dictionary with inputs, rewards, reference logits, etc.
            
        Returns:
            Loss value
        """
        # Extract batch data
        # NOTE: why .unsqueeze(1) ???
        rewards = batch['rewards'] #.unsqueeze(1)
        ref_logps = batch['refs']  # (B)
        sampled_classes = batch['sampled_classes'] # (B)
        prompt_mel = batch['prompt_mel']
        text_ids = batch['text_ids']

        # Forward pass to get current model logits
        K = self.num_pre_samples
        B, T, _ = prompt_mel.shape
        _, L = text_ids.shape
        cur_logits = self.model(
            text_ids=text_ids, # (B, L)
            mel=prompt_mel         # (B, T, 100)
        )[:, -1, :]
        cur_logits = cur_logits.unsqueeze(1).repeat(1, K, 1).view(B*K, -1) 

        # Compute current log probabilities for sampled actions
        log_probs = F.log_softmax(cur_logits, dim=-1)
        cur_logps = torch.gather(
            log_probs, 
            dim=-1, 
            index=sampled_classes.unsqueeze(-1)
        ).squeeze(-1)  # (B)

        # KL divergence computation (same as in Qwen2.5 code)
        # KL = exp(ref - cur) - (ref - cur) - 1
        kl_div = torch.exp(ref_logps - cur_logps) - (ref_logps - cur_logps) - 1 # (B)
        
        # Compute probability ratio for PPO
        if "gen_logps" in batch:
            gen_logps = batch['gen_logps']
            ratio = torch.exp(cur_logps - gen_logps)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)
            loss = torch.min(ratio * rewards, clipped_ratio * rewards)
        else:
            # Simplification if gen_logps not available
            loss = torch.exp(cur_logps - cur_logps.detach()) * rewards
        
        # Final GRPO loss with KL regularization
        loss = -(loss - self.beta * kl_div) # (B)
        loss = loss.mean()
        
        return loss
    
    def get_batch(self):
        """Get a batch from the queue or return None if empty"""
        if not self.batch_queue:
            return None
        return self.batch_queue.pop(0)
    
    def generate_mode(self, num_batches=5):
        """
        Generate samples and add them to the batch queue
        
        Args:
            dataset: Dataset to sample from
            num_batches: Number of batches to generate
        """
        if self.rank == 0:
            print("Entering generate mode...")
        
        tic = time.time()
        for _ in range(num_batches):
            try:
                batch_inputs = next(self.train_iterator)
            except StopIteration:
                self.train_iterator = iter(self.train_dataloader)
                batch_inputs = next(self.train_iterator)

            # Generate samples and compute rewards
            batch_outputs = self.generate_duration_samples(batch_inputs)
            # Check if batch has sufficient reward diversity
            rewards = batch_outputs["rewards"]
            if (rewards.max() - rewards.min()).item() < 0.01:
                if self.rank == 0:
                    print("Skipping batch with low reward diversity")
                continue
            # Add batch to queue
            self.batch_queue.append(batch_outputs)
        
        if self.rank == 0:
            print(f"Exiting generate mode: {time.time() - tic:.3f}s")
    
    def train(self, train_dataset, valid_dataset=None, num_workers=64, resumable_with_seed=666):
        """
        Train the model using GRPO
        
        Args:
            train_dataset: Training dataset
            valid_dataset: Validation dataset (optional)
            num_workers: Number of workers for data loading
        """

        # Create training dataloader using the appropriate batching strategy
        if self.batch_size_type == "sample":
            self.train_dataloader = DataLoader(
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
            self.valid_dataloader = DataLoader(
                valid_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                batch_size=self.batch_size,
                shuffle=False,
            )

            self.train_iterator = iter(self.train_dataloader)
            self.valid_iterator = iter(self.valid_dataloader)
    
        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False

            sampler = SequentialSampler(train_dataset)
            batch_sampler = DynamicBatchSampler(
                sampler, self.batch_size, max_samples=self.max_samples, random_seed=resumable_with_seed, drop_last=False
            )
            self.train_dataloader = DataLoader(
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
            self.valid_dataloader = DataLoader(
                valid_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,  
                persistent_workers=True,
                batch_sampler=batch_sampler,
            )
            
            self.train_dataloader, self.valid_dataloader = self.accelerator.prepare(self.train_dataloader, self.valid_dataloader)

            self.train_iterator = iter(self.train_dataloader)
            self.valid_iterator = iter(self.valid_dataloader)

        else:
            raise ValueError(f"batch_size_type must be either 'sample' or 'frame', but received {self.batch_size_type}")
        

        # Setup schedulers
        warmup_steps = self.num_warmup_updates * self.accelerator.num_processes
        total_steps = self.all_steps
        decay_steps = total_steps - warmup_steps
        
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_steps)
        
        self.scheduler = SequentialLR(
            self.optimizer, 
            schedulers=[warmup_scheduler, decay_scheduler], 
            milestones=[warmup_steps]
        )
        
        self.scheduler = self.accelerator.prepare(self.scheduler)
        
        # Load checkpoint if available
        start_step = self.load_checkpoint()
        self.global_step = start_step
        
        # Generate initial batches
        self.generate_mode()
        
        # Training loop
        progress = range(1, self.all_steps + 1)
        
        # Skip steps that are already done
        progress = [step for step in progress if step > start_step]
        if self.is_main:
            progress = tqdm(progress, desc="Training", unit="step")
        
        for step in progress:
            # Get batch from queue or generate more
            batch = self.get_batch()
            while batch is None:
                self.generate_mode()
                batch = self.get_batch()
            
            # GRPO update
            with self.accelerator.accumulate(self.model):
                loss = self.GRPO_step(batch)
                # for param in self.model.parameters():
                #     custom_loss = loss + 0 * param.sum()  
                self.accelerator.backward(loss)
                
                if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                    total_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                else:
                    total_norm = torch.norm(
                        torch.stack([
                            torch.norm(p.grad.detach(), 2)
                            for p in self.model.parameters()
                            if p.grad is not None
                        ]),
                        2
                    )
                
                self.accelerator.log({
                    "grad_norm": total_norm.item()
                }, step=self.global_step)

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            self.global_step += 1
            
            # Log metrics
            if self.is_main:
                self.accelerator.log({
                    "loss": loss.item(),
                    "lr": self.scheduler.get_last_lr()[0],
                    # "avg_reward": batch["rewards"].mean().item(),
                    # "max_reward": batch["rewards"].max().item(),
                    # "min_reward": batch["rewards"].min().item(),
                }, step=self.global_step)
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{self.scheduler.get_last_lr()[0]:.8f}"
                )
            
            # Save checkpoint
            if self.global_step % self.save_per_updates == 0:
                self.save_checkpoint(self.global_step)
                
                # Optional validation logic could be added here
        
        # Save final checkpoint
        self.save_checkpoint(self.global_step, last=True)
        self.accelerator.end_training()
