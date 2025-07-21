from __future__ import annotations

import os
import gc
from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from unimodel import UniModel
from f5_tts.model import CFM
from f5_tts.model.utils import exists, default
from f5_tts.model.dataset import DynamicBatchSampler, collate_fn


# trainer

import math

class RunningStats:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squared differences from the current mean

    def update(self, x):
        """Update the running statistics with a new value x."""
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self):
        """Return the sample variance. Returns NaN if fewer than two samples."""
        return self.M2 / (self.count - 1) if self.count > 1 else float('nan')

    @property
    def std(self):
        """Return the sample standard deviation."""
        return math.sqrt(self.variance)



class Trainer:
    def __init__(
        self,
        model: UniModel,
        epochs,
        learning_rate,
        num_warmup_updates=20000,
        save_per_updates=1000,
        checkpoint_path=None,
        batch_size=32,
        batch_size_type: str = "sample",
        max_samples=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        noise_scheduler: str | None = None,
        duration_predictor: torch.nn.Module | None = None,
        wandb_project="test_e2-tts",
        wandb_run_name="test_run",
        wandb_resume_id: str = None,
        last_per_steps=None,
        log_step=1000,
        accelerate_kwargs: dict = dict(),
        bnb_optimizer: bool = False,
        scale: float = 1.0,
        
        # training parameters for DMDSpeech
        num_student_step: int = 1,
        gen_update_ratio: int = 5,
        lambda_discriminator_loss: float = 1.0,
        lambda_generator_loss: float = 1.0,
        lambda_ctc_loss: float = 1.0,
        lambda_sim_loss: float = 1.0,

        num_GAN: int = 5000,
        num_D: int = 500,
        num_ctc: int = 5000,
        num_sim: int = 10000,
        num_simu: int = 1000,
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        logger = "wandb" if wandb.api.api_key else None
        print(f"Using logger: {logger}")

        self.accelerator = Accelerator(
            log_with=logger,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs,
        )

        if logger == "wandb":
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
                    "noise_scheduler": noise_scheduler,
                },
            )

        self.model = model

        self.scale = scale

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

        self.noise_scheduler = noise_scheduler

        self.duration_predictor = duration_predictor
        
        self.log_step = log_step

        self.gen_update_ratio = gen_update_ratio # number of generator updates per guidance (fake score function and discriminator) update
        self.lambda_discriminator_loss = lambda_discriminator_loss # weight for discriminator loss (L_adv)
        self.lambda_generator_loss = lambda_generator_loss # weight for generator loss (L_adv)
        self.lambda_ctc_loss = lambda_ctc_loss # weight for ctc loss
        self.lambda_sim_loss = lambda_sim_loss # weight for similarity loss
        
        # create distillation schedule for student model
        self.student_steps = (
                torch.linspace(0.0, 1.0, num_student_step + 1)[:-1])
        
        self.GAN = model.guidance_model.gen_cls_loss # whether to use GAN training
        self.num_GAN = num_GAN # number of steps before adversarial training
        self.num_D = num_D # number of steps to train the discriminator before adversarial training 
        self.num_ctc = num_ctc # number of steps before CTC training
        self.num_sim = num_sim # number of steps before similarity training
        self.num_simu = num_simu # number of steps before using simulated data

        # Assuming `self.model.fake_unet.parameters()` and `self.model.guidance_model.parameters()` are accessible
        if bnb_optimizer:
            import bitsandbytes as bnb
            self.optimizer_generator = bnb.optim.AdamW8bit(self.model.feedforward_model.parameters(), lr=learning_rate)
            self.optimizer_guidance = bnb.optim.AdamW8bit(self.model.guidance_model.parameters(), lr=learning_rate)
        else:
            self.optimizer_generator = AdamW(self.model.feedforward_model.parameters(), lr=learning_rate, eps=1e-7)
            self.optimizer_guidance = AdamW(self.model.guidance_model.parameters(), lr=learning_rate, eps=1e-7)

        self.model, self.optimizer_generator, self.optimizer_guidance = self.accelerator.prepare(self.model, self.optimizer_generator, self.optimizer_guidance)

        self.generator_norm = RunningStats()
        self.guidance_norm = RunningStats()

    
    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_checkpoint(self, step, last=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_generator_state_dict=self.accelerator.unwrap_model(self.optimizer_generator).state_dict(),
                optimizer_guidance_state_dict=self.accelerator.unwrap_model(self.optimizer_guidance).state_dict(),
                scheduler_generator_state_dict=self.scheduler_generator.state_dict(),
                scheduler_guidance_state_dict=self.scheduler_guidance.state_dict(),
                step=step,
            )

            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if last:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_last.pt")
                print(f"Saved last checkpoint at step {step}")
            else:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_{step}.pt")

    def load_checkpoint(self):
        if (
            not exists(self.checkpoint_path)
            or not os.path.exists(self.checkpoint_path)
            or not os.listdir(self.checkpoint_path)
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
        # checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", map_location=self.accelerator.device)  # rather use accelerator.load_state ಥ_ಥ
        checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", weights_only=True, map_location="cpu")

        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"], strict=False)
        # self.accelerator.unwrap_model(self.optimizer_generator).load_state_dict(checkpoint["optimizer_generator_state_dict"])
        # self.accelerator.unwrap_model(self.optimizer_guidance).load_state_dict(checkpoint["optimizer_guidance_state_dict"])
        # if self.scheduler_guidance:
        #     self.scheduler_guidance.load_state_dict(checkpoint["scheduler_guidance_state_dict"])
        # if self.scheduler_generator:
        #     self.scheduler_generator.load_state_dict(checkpoint["scheduler_generator_state_dict"])
        step = checkpoint["step"]

        del checkpoint
        gc.collect()
        return step
    

    def train(self, train_dataset: Dataset, num_workers=64, resumable_with_seed: int = None, vocoder: nn.Module = None):
        if exists(resumable_with_seed):
            generator = torch.Generator()
            generator.manual_seed(resumable_with_seed)
        else:
            generator = None

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
        else:
            raise ValueError(f"batch_size_type must be either 'sample' or 'frame', but received {self.batch_size_type}")

        #  accelerator.prepare() dispatches batches to devices;
        #  which means the length of dataloader calculated before, should consider the number of devices
        warmup_steps = (
            self.num_warmup_updates * self.accelerator.num_processes
        )
        
        # consider a fixed warmup steps while using accelerate multi-gpu ddp
        # otherwise by default with split_batches=False, warmup steps change with num_processes
        total_steps = len(train_dataloader) * self.epochs / self.grad_accumulation_steps
        decay_steps = total_steps - warmup_steps
        
        warmup_scheduler_generator = LinearLR(self.optimizer_generator, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps // (self.gen_update_ratio * self.grad_accumulation_steps))
        decay_scheduler_generator = LinearLR(self.optimizer_generator, start_factor=1.0, end_factor=1e-8, total_iters=decay_steps // (self.gen_update_ratio * self.grad_accumulation_steps))
        self.scheduler_generator = SequentialLR(self.optimizer_generator, schedulers=[warmup_scheduler_generator, decay_scheduler_generator], milestones=[warmup_steps // (self.gen_update_ratio * self.grad_accumulation_steps)])

        warmup_scheduler_guidance = LinearLR(self.optimizer_guidance, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
        decay_scheduler_guidance = LinearLR(self.optimizer_guidance, start_factor=1.0, end_factor=1e-8, total_iters=decay_steps)
        self.scheduler_guidance = SequentialLR(self.optimizer_guidance, schedulers=[warmup_scheduler_guidance, decay_scheduler_guidance], milestones=[warmup_steps])

        train_dataloader, self.scheduler_generator, self.scheduler_guidance = self.accelerator.prepare(
            train_dataloader, self.scheduler_generator, self.scheduler_guidance
        )  # actual steps = 1 gpu steps / gpus
        start_step = self.load_checkpoint()
        global_step = start_step

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
                update_generator = global_step % self.gen_update_ratio == 0
                        
                with self.accelerator.accumulate(self.model):
                    metrics = {}
                    text_inputs = batch["text"]
                    mel_spec = batch["mel"].permute(0, 2, 1)
                    mel_lengths = batch["mel_lengths"]
                    
                    mel_spec = mel_spec / self.scale
                    
                    guidance_loss_dict, guidance_log_dict = self.model(inp=mel_spec, 
                                                                text=text_inputs, 
                                                                lens=mel_lengths, 
                                                                student_steps=self.student_steps,
                                                                update_generator=False,
                                                                use_simulated=global_step >= self.num_simu,
                                                                )

                    # if self.GAN and update_generator:
                    #     # only add discriminator loss if GAN is enabled and generator is being updated
                    #     guidance_cls_loss = guidance_loss_dict["guidance_cls_loss"] * (self.lambda_discriminator_loss if global_step >= self.num_GAN and update_generator else 0)
                    #     metrics['loss/discriminator_loss'] = guidance_loss_dict["guidance_cls_loss"]
                    #     self.accelerator.backward(guidance_cls_loss, retain_graph=True)
                        
                    #     if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                    #         metrics['grad_norm_guidance'] = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    guidance_loss = 0
                    guidance_loss += guidance_loss_dict["loss_fake_mean"]
                    metrics['loss/fake_score'] = guidance_loss_dict["loss_fake_mean"]
                    metrics["loss/guidance_loss"] = guidance_loss

                    if self.GAN and update_generator:
                        # only add discriminator loss if GAN is enabled and generator is being updated
                        guidance_cls_loss = guidance_loss_dict["guidance_cls_loss"] * (self.lambda_discriminator_loss if global_step >= self.num_GAN and update_generator else 0)
                        metrics['loss/discriminator_loss'] = guidance_loss_dict["guidance_cls_loss"]

                        guidance_loss += guidance_cls_loss
                    
                    self.accelerator.backward(guidance_loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        metrics['grad_norm_guidance'] = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                        # if self.guidance_norm.count < 100:
                        #     self.guidance_norm.update(metrics['grad_norm_guidance'])

                        # if metrics['grad_norm_guidance'] > self.guidance_norm.mean + 5 * self.guidance_norm.std:
                        #     self.optimizer_generator.zero_grad()
                        #     self.optimizer_guidance.zero_grad()
                        #     print("Gradient explosion detected. Skipping batch.")
                        # elif self.guidance_norm.count >= 100:
                        #     self.guidance_norm.update(metrics['grad_norm_guidance'])


                    self.optimizer_guidance.step()
                    self.scheduler_guidance.step()
                    self.optimizer_guidance.zero_grad()
                    self.optimizer_generator.zero_grad()  # zero out the generator's gradient as well
                    
                    if update_generator:
                        generator_loss_dict, generator_log_dict = self.model(inp=mel_spec, 
                                                                        text=text_inputs, 
                                                                        lens=mel_lengths, 
                                                                        student_steps=self.student_steps,
                                                                        update_generator=True,
                                                                        use_simulated=global_step >= self.num_ctc,
                                                                        )
                        # if self.GAN:
                        #     gen_cls_loss = generator_loss_dict["gen_cls_loss"] * (self.lambda_generator_loss if global_step >= (self.num_GAN + self.num_D) and update_generator else 0)
                        #     metrics["loss/gen_cls_loss"] = generator_loss_dict["gen_cls_loss"]

                        #     self.accelerator.backward(gen_cls_loss, retain_graph=True)

                        #     if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        #         metrics['grad_norm_generator'] = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                        generator_loss = 0
                        generator_loss += generator_loss_dict["loss_dm"]
                        if "loss_mse" in generator_loss_dict:
                            generator_loss += generator_loss_dict["loss_mse"] 
                        generator_loss += generator_loss_dict["loss_ctc"] * (self.lambda_ctc_loss if global_step >= self.num_ctc else 0)
                        generator_loss += generator_loss_dict["loss_sim"] * (self.lambda_sim_loss if global_step >= self.num_sim else 0)
                        generator_loss += generator_loss_dict["loss_kl"] * (self.lambda_ctc_loss if global_step >= self.num_ctc else 0)
                        if self.GAN:
                            gen_cls_loss = generator_loss_dict["gen_cls_loss"] * (self.lambda_generator_loss if global_step >= (self.num_GAN + self.num_D) and update_generator else 0)
                            metrics["loss/gen_cls_loss"] = generator_loss_dict["gen_cls_loss"]
                            generator_loss += gen_cls_loss

                        metrics['loss/dm_loss'] = generator_loss_dict["loss_dm"]
                        metrics['loss/ctc_loss'] = generator_loss_dict["loss_ctc"]

                        metrics['loss/similarity_loss'] = generator_loss_dict["loss_sim"]
                        metrics['loss/generator_loss'] = generator_loss
                        
                        if "loss_mse" in generator_loss_dict and generator_loss_dict["loss_mse"] != 0:
                            metrics['loss/mse_loss'] = generator_loss_dict["loss_mse"]
                        if "loss_kl" in generator_loss_dict and generator_loss_dict["loss_kl"] != 0:
                            metrics['loss/kl_loss'] = generator_loss_dict["loss_kl"]

                        self.accelerator.backward(generator_loss)

                        if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                            metrics['grad_norm_generator'] = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                            # self.generator_norm.update(metrics['grad_norm_generator'])
                            
                            # if metrics['grad_norm_generator'] > self.generator_norm.mean + 15 * self.generator_norm.std:
                            #     self.optimizer_generator.zero_grad()
                            #     self.optimizer_guidance.zero_grad()
                            #     update_generator = False
                            #     print("Gradient explosion detected. Skipping batch.")

                        if update_generator:
                            self.optimizer_generator.step()
                            self.scheduler_generator.step()
                            self.optimizer_generator.zero_grad()
                            self.optimizer_guidance.zero_grad()  # zero out the guidance's gradient as well


                global_step += 1

                if self.accelerator.is_local_main_process:
                    self.accelerator.log({**metrics,
                                          "lr_generator": self.scheduler_generator.get_last_lr()[0],
                                          "lr_guidance": self.scheduler_guidance.get_last_lr()[0],
                                          }
                                         , step=global_step)
                
                if global_step % self.log_step == 0 and self.accelerator.is_local_main_process and vocoder is not None:
                    # log the first batch of the epoch
                    with torch.no_grad():
                        generator_input = generator_log_dict['generator_input'][0].unsqueeze(0).permute(0, 2, 1) * self.scale
                        generator_input = vocoder.decode(generator_input.float().cpu())
                        generator_input = wandb.Audio(
                            generator_input.float().numpy().squeeze(),
                            sample_rate=24000,
                            caption="time: " + str(generator_log_dict['time'][0].float().cpu().numpy())
                        )

                        generator_output = generator_log_dict['generator_output'][0].unsqueeze(0).permute(0, 2, 1) * self.scale
                        generator_output = vocoder.decode(generator_output.float().cpu())
                        generator_output = wandb.Audio(
                            generator_output.float().numpy().squeeze(),
                            sample_rate=24000,
                            caption="time: " + str(generator_log_dict['time'][0].float().cpu().numpy())
                        )
                        
                        generator_cond = generator_log_dict['generator_cond'][0].unsqueeze(0).permute(0, 2, 1) * self.scale
                        generator_cond = vocoder.decode(generator_cond.float().cpu())
                        generator_cond = wandb.Audio(
                            generator_cond.float().numpy().squeeze(),
                            sample_rate=24000,
                            caption="time: " + str(generator_log_dict['time'][0].float().cpu().numpy())
                        )
                        
                        ground_truth = generator_log_dict['ground_truth'][0].unsqueeze(0).permute(0, 2, 1) * self.scale
                        ground_truth = vocoder.decode(ground_truth.float().cpu())
                        ground_truth = wandb.Audio(
                            ground_truth.float().numpy().squeeze(),
                            sample_rate=24000,
                            caption="time: " + str(generator_log_dict['time'][0].float().cpu().numpy())
                        )
                        
                        dmtrain_noisy_inp = generator_log_dict['dmtrain_noisy_inp'][0].unsqueeze(0).permute(0, 2, 1) * self.scale
                        dmtrain_noisy_inp = vocoder.decode(dmtrain_noisy_inp.float().cpu())
                        dmtrain_noisy_inp = wandb.Audio(
                            dmtrain_noisy_inp.float().numpy().squeeze(),
                            sample_rate=24000,
                            caption="dmtrain_time: " + str(generator_log_dict['dmtrain_time'][0].float().cpu().numpy())
                        )
                        
                        dmtrain_pred_real_image = generator_log_dict['dmtrain_pred_real_image'][0].unsqueeze(0).permute(0, 2, 1) * self.scale
                        dmtrain_pred_real_image = vocoder.decode(dmtrain_pred_real_image.float().cpu())
                        dmtrain_pred_real_image = wandb.Audio(
                            dmtrain_pred_real_image.float().numpy().squeeze(),
                            sample_rate=24000,
                            caption="dmtrain_time: " + str(generator_log_dict['dmtrain_time'][0].float().cpu().numpy())
                        )
                        
                        dmtrain_pred_fake_image = generator_log_dict['dmtrain_pred_fake_image'][0].unsqueeze(0).permute(0, 2, 1) * self.scale
                        dmtrain_pred_fake_image = vocoder.decode(dmtrain_pred_fake_image.float().cpu())
                        dmtrain_pred_fake_image = wandb.Audio(
                            dmtrain_pred_fake_image.float().numpy().squeeze(),
                            sample_rate=24000,
                            caption="dmtrain_time: " + str(generator_log_dict['dmtrain_time'][0].float().cpu().numpy())
                        )
                        
                                                
                        self.accelerator.log({"noisy_input": generator_input, 
                                              "output": generator_output,
                                                "cond": generator_cond,
                                                "ground_truth": ground_truth,
                                                "dmtrain_noisy_inp": dmtrain_noisy_inp,
                                                "dmtrain_pred_real_image": dmtrain_pred_real_image,
                                                "dmtrain_pred_fake_image": dmtrain_pred_fake_image,
                                                
                                             }, step=global_step)

                progress_bar.set_postfix(step=str(global_step), metrics=metrics)

                if global_step % (self.save_per_updates * self.grad_accumulation_steps) == 0:
                    self.save_checkpoint(global_step)

                if global_step % self.last_per_steps == 0:
                    self.save_checkpoint(global_step, last=True)

        self.save_checkpoint(global_step, last=True)

        self.accelerator.end_training()
        
        
