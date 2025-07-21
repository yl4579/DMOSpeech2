import os
import torch
import torchaudio
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint
from safetensors.torch import load_file
import IPython.display as ipd

# Import F5-TTS modules
from f5_tts.model import CFM, UNetT, DiT
from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import (
    default, exists, list_str_to_idx, list_str_to_tensor,
    lens_to_mask, mask_from_frac_lengths, get_tokenizer
)
from f5_tts.infer.utils_infer import (
    load_vocoder, preprocess_ref_audio_text, chunk_text,
    convert_char_to_pinyin, transcribe, target_rms,
    target_sample_rate, hop_length, speed
)

# Import custom modules
from unimodel import UniModel
from duration_predictor import SpeechLengthPredictor


class DMOInference:
    """F5-TTS Inference wrapper class for easy text-to-speech generation."""
    
    def __init__(
        self,
        student_checkpoint_path="",
        duration_predictor_path="",
        device="cuda",
        model_type="F5TTS_Base",  # "F5TTS_Base" or "E2TTS_Base"
        tokenizer="pinyin",
        dataset_name="Emilia_ZH_EN",
    ):
        """
        Initialize F5-TTS inference model.
        
        Args:
            student_checkpoint_path: Path to student model checkpoint
            duration_predictor_path: Path to duration predictor checkpoint
            device: Device to run inference on
            model_type: Model architecture type
            tokenizer: Tokenizer type ("pinyin", "char", or "custom")
            dataset_name: Dataset name for tokenizer
            cuda_device_id: CUDA device ID to use
        """
        
        self.device = device
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        
        # Model parameters
        self.target_sample_rate = 24000
        self.n_mel_channels = 100
        self.hop_length = 256
        self.real_guidance_scale = 2
        self.fake_guidance_scale = 0
        self.gen_cls_loss = False
        self.num_student_step = 4
        
        # Initialize components
        self._setup_tokenizer()
        self._setup_models(student_checkpoint_path)
        self._setup_mel_spec()
        self._setup_vocoder()
        self._setup_duration_predictor(duration_predictor_path)
        
    def _setup_tokenizer(self):
        """Setup tokenizer and vocabulary."""
        if self.tokenizer == "custom":
            tokenizer_path = self.tokenizer_path
        else:
            tokenizer_path = self.dataset_name
            
        self.vocab_char_map, self.vocab_size = get_tokenizer(tokenizer_path, self.tokenizer)
        
    def _setup_models(self, student_checkpoint_path):
        """Initialize teacher and student models."""
        # Model configuration
        if self.model_type == "F5TTS_Base":
            model_cls = DiT
            model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
        elif self.model_type == "E2TTS_Base":
            model_cls = UNetT
            model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        # Initialize UniModel (student)
        self.model = UniModel(
            model_cls(**model_cfg, text_num_embeds=self.vocab_size, mel_dim=self.n_mel_channels, 
                      second_time=self.num_student_step > 1),
            checkpoint_path="",
            vocab_char_map=self.vocab_char_map,
            frac_lengths_mask=(0.5, 0.9),
            real_guidance_scale=self.real_guidance_scale,
            fake_guidance_scale=self.fake_guidance_scale,
            gen_cls_loss=self.gen_cls_loss,
            sway_coeff=0,
        )
        
        # Load student checkpoint
        checkpoint = torch.load(student_checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # Setup generator and teacher
        self.generator = self.model.feedforward_model.to(self.device)
        self.teacher = self.model.guidance_model.real_unet.to(self.device)
        
        self.scale = checkpoint['scale']
        
    def _setup_mel_spec(self):
        """Initialize mel spectrogram module."""
        mel_spec_kwargs = dict(
            target_sample_rate=self.target_sample_rate,
            n_mel_channels=self.n_mel_channels,
            hop_length=self.hop_length,
        )
        self.mel_spec = MelSpec(**mel_spec_kwargs)
        
    def _setup_vocoder(self):
        """Initialize vocoder."""
        self.vocos = load_vocoder(is_local=False, local_path="")
        self.vocos = self.vocos.to(self.device)
        
    def _setup_duration_predictor(self, checkpoint_path):
        """Initialize duration predictor."""
        self.wav2mel = MelSpec(
            target_sample_rate=24000,
            n_mel_channels=100,
            hop_length=256,
            win_length=1024,
            n_fft=1024,
            mel_spec_type='vocos'
        ).to(self.device)
        
        self.SLP = SpeechLengthPredictor(
            vocab_size=2545,
            n_mel=100,
            hidden_dim=512,
            n_text_layer=4,
            n_cross_layer=4,
            n_head=8,
            output_dim=301
        ).to(self.device)
        
        self.SLP.eval()
        self.SLP.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['model_state_dict'])
        
    def predict_duration(self, pmt_wav_path, tar_text, pmt_text, dp_softmax_range=0.7, temperature=0):
        """
        Predict duration for target text based on prompt audio.
        
        Args:
            pmt_wav_path: Path to prompt audio
            tar_text: Target text to generate
            pmt_text: Prompt text
            dp_softmax_range: softmax annliation range from rate-based duration
            temperature: temperature for softmax sampling (if 0, will use argmax)
        Returns:
            Estimated duration in frames
        """
        pmt_wav, sr = torchaudio.load(pmt_wav_path)
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            pmt_wav = resampler(pmt_wav)
        if pmt_wav.size(0) > 1:
            pmt_wav = pmt_wav[0].unsqueeze(0)
        pmt_wav = pmt_wav.to(self.device)
        
        pmt_mel = self.wav2mel(pmt_wav).permute(0, 2, 1)
        tar_tokens = self._convert_to_pinyin(list(tar_text))
        pmt_tokens = self._convert_to_pinyin(list(pmt_text))
        
        # Calculate duration
        ref_text_len = len(pmt_tokens)
        gen_text_len = len(tar_tokens)
        ref_audio_len = pmt_mel.size(1)
        duration = int(ref_audio_len / ref_text_len * gen_text_len / speed)
        duration = duration // 10
        
        min_duration = max(int(duration * dp_softmax_range), 0)
        max_duration = min(int(duration * (1 + dp_softmax_range)), 301)
            
        all_tokens = pmt_tokens + [' '] + tar_tokens
        
        text_ids = list_str_to_idx([all_tokens], self.vocab_char_map).to(self.device)
        text_ids = text_ids.masked_fill(text_ids == -1, self.vocab_size)
        
        with torch.no_grad():
            predictions = self.SLP(text_ids=text_ids, mel=pmt_mel)
        predictions = predictions[:, -1, :]
        predictions[:, :min_duration] = float('-inf')
        predictions[:, max_duration:] = float('-inf')
        
        if temperature == 0:
            est_label = predictions.argmax(-1)[..., -1].item() * 10
        else:
            probs = torch.softmax(predictions / temperature, dim=-1)
            sampled_idx = torch.multinomial(probs.squeeze(0), num_samples=1)  # Remove the -1 index
            est_label = sampled_idx.item() * 10
      
        return est_label
        
    def _convert_to_pinyin(self, char_list):
        """Convert character list to pinyin."""
        result = []
        for x in convert_char_to_pinyin(char_list):
            result = result + x
        while result[0] == ' ' and len(result) > 1:
            result = result[1:]
        return result
        
    def generate(
        self,
        gen_text,
        audio_path,
        prompt_text=None,
        teacher_steps=16,
        teacher_stopping_time=0.07,
        student_start_step=1,
        duration=None,
        dp_softmax_range=0.7,
        temperature=0,
        eta=1.0,
        cfg_strength=2.0,
        sway_coefficient=-1.0,
        verbose=False
    ):
        """
        Generate speech from text using teacher-student distillation.
        
        Args:
            gen_text: Text to generate
            audio_path: Path to prompt audio
            prompt_text: Prompt text (if None, will use ASR)
            teacher_steps: Number of teacher guidance steps
            teacher_stopping_time: When to stop teacher sampling
            student_start_step: When to start student sampling
            duration: Total duration (if None, will predict)
            dp_softmax_range: Duration predictor softmax range allowed around rate based duration
            temperature: Temperature for duration predictor sampling (0 means use argmax)
            eta: Stochasticity control (0=DDIM, 1=DDPM)
            cfg_strength: Classifier-free guidance strength
            sway_coefficient: Sway sampling coefficient
            verbose: Output sampling steps
            
        Returns:
            Generated audio waveform
        """
        if prompt_text is None:
            prompt_text = transcribe(audio_path)
            
        # Predict duration if not provided
        if duration is None:
            duration = self.predict_duration(audio_path, gen_text, prompt_text, dp_softmax_range, temperature)
            
        # Preprocess audio and text
        ref_audio, ref_text = preprocess_ref_audio_text(audio_path, prompt_text)
        audio, sr = torchaudio.load(ref_audio)
        
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
            
        # Normalize audio
        rms = torch.sqrt(torch.mean(torch.square(audio)))
        if rms < target_rms:
            audio = audio * target_rms / rms
            
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            audio = resampler(audio)
            
        audio = audio.to(self.device)
        
        # Prepare text
        text_list = [ref_text + gen_text]
        final_text_list = convert_char_to_pinyin(text_list)
        
        # Calculate durations
        ref_audio_len = audio.shape[-1] // self.hop_length
        if duration is None:
            ref_text_len = len(ref_text.encode("utf-8"))
            gen_text_len = len(gen_text.encode("utf-8"))
            duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)
        else:
            duration = ref_audio_len + duration
        
        if verbose:
            print('audio:', audio.shape)
            print('text:', final_text_list)
            print('duration:', duration)
            print('eta (stochasticity):', eta)  # Print eta value for debugging

        # Run inference
        with torch.inference_mode():
            cond, text, step_cond, cond_mask, max_duration, duration_tensor = self._prepare_inputs(
                audio, final_text_list, duration
            )
            
            # Teacher-student sampling
            if teacher_steps > 0 and student_start_step > 0:
                if verbose:
                    print('Start teacher sampling with hybrid DDIM/DDPM (eta={})....'.format(eta))
                x1 = self._teacher_sampling(
                    step_cond, text, cond_mask, max_duration, duration_tensor,  # Use duration_tensor
                    teacher_steps, teacher_stopping_time, eta, cfg_strength, verbose, sway_coefficient
                )
            else:
                x1 = step_cond
            
            if verbose:
                print('Start student sampling...')
            # Student sampling
            x1 = self._student_sampling(x1, cond, text, student_start_step, verbose, sway_coefficient)
            
            # Decode to audio
            mel = x1.permute(0, 2, 1) * self.scale
            generated_wave = self.vocos.decode(mel[..., cond_mask.sum():])
            
        return generated_wave.cpu().numpy().squeeze()
        
    def generate_teacher_only(
        self,
        gen_text,
        audio_path,
        prompt_text=None,
        teacher_steps=32,
        duration=None,
        eta=1.0,
        cfg_strength=2.0,
        sway_coefficient=-1.0
    ):
        """
        Generate speech using teacher model only (no student distillation).
        
        Args:
            gen_text: Text to generate
            audio_path: Path to prompt audio
            prompt_text: Prompt text (if None, will use ASR)
            teacher_steps: Number of sampling steps
            duration: Total duration (if None, will predict)
            eta: Stochasticity control (0=DDIM, 1=DDPM)
            cfg_strength: Classifier-free guidance strength
            sway_coefficient: Sway sampling coefficient
            
        Returns:
            Generated audio waveform
        """
        if prompt_text is None:
            prompt_text = transcribe(audio_path)
            
        # Predict duration if not provided
        if duration is None:
            duration = self.predict_duration(audio_path, gen_text, prompt_text)
            
        # Preprocess audio and text
        ref_audio, ref_text = preprocess_ref_audio_text(audio_path, prompt_text)
        audio, sr = torchaudio.load(ref_audio)
        
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
            
        # Normalize audio
        rms = torch.sqrt(torch.mean(torch.square(audio)))
        if rms < target_rms:
            audio = audio * target_rms / rms
            
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            audio = resampler(audio)
            
        audio = audio.to(self.device)
        
        # Prepare text
        text_list = [ref_text + gen_text]
        final_text_list = convert_char_to_pinyin(text_list)
        
        # Calculate durations
        ref_audio_len = audio.shape[-1] // self.hop_length
        if duration is None:
            ref_text_len = len(ref_text.encode("utf-8"))
            gen_text_len = len(gen_text.encode("utf-8"))
            duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)
        else:
            duration = ref_audio_len + duration
            
        # Run inference
        with torch.inference_mode():
            cond, text, step_cond, cond_mask, max_duration = self._prepare_inputs(
                audio, final_text_list, duration
            )
            
            # Teacher-only sampling
            x1 = self._teacher_sampling(
                step_cond, text, cond_mask, max_duration, duration,
                teacher_steps, 1.0, eta, cfg_strength, sway_coefficient  # stopping_time=1.0 for full sampling
            )
            
            # Decode to audio
            mel = x1.permute(0, 2, 1) * self.scale
            generated_wave = self.vocos.decode(mel[..., cond_mask.sum():])
            
        return generated_wave
        
    def _prepare_inputs(self, audio, text_list, duration):
        """Prepare inputs for generation."""
        lens = None
        max_duration_limit = 4096
        
        cond = audio
        text = text_list
        
        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == 100
            
        cond = cond / self.scale
        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)
            
        # Process text
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch
            
        if exists(text):
            text_lens = (text != -1).sum(dim=-1)
            lens = torch.maximum(text_lens, lens)
            
        # Process duration
        cond_mask = lens_to_mask(lens)
        
        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)
            
        duration = torch.maximum(lens + 1, duration)
        duration = duration.clamp(max=max_duration_limit)
        max_duration = duration.amax()
        
        # Pad conditioning
        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))
        
        return cond, text, step_cond, cond_mask, max_duration, duration
        
    def _teacher_sampling(self, step_cond, text, cond_mask, max_duration, duration,
                         teacher_steps, teacher_stopping_time, eta, cfg_strength, verbose, sway_sampling_coef = -1):
        """Perform teacher model sampling."""
        device = step_cond.device
        
        # Pre-generate noise sequence for stochastic sampling
        noise_seq = None
        if eta > 0:
            noise_seq = [torch.randn(1, max_duration, 100, device=device) 
                        for _ in range(teacher_steps)]
            
        def fn(t, x):
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    if verbose:
                        print(f'current t: {t}')
                    step_frac = 1.0 - t.item()
                    step_idx = min(int(step_frac * len(noise_seq)), len(noise_seq) - 1) if noise_seq else 0
                    
                    # Predict flow
                    pred = self.teacher(
                        x=x, cond=step_cond, text=text, time=t, mask=None,
                        drop_audio_cond=False, drop_text=False
                    )
                    
                    if cfg_strength > 1e-5:
                        null_pred = self.teacher(
                            x=x, cond=step_cond, text=text, time=t, mask=None,
                            drop_audio_cond=True, drop_text=True
                        )
                        pred = pred + (pred - null_pred) * cfg_strength
                        
                    # Add stochasticity if eta > 0
                    if eta > 0 and noise_seq is not None:
                        alpha_t = 1.0 - t.item()
                        sigma_t = t.item()
                        noise_scale = torch.sqrt(torch.tensor(
                            (sigma_t**2) / (alpha_t**2 + sigma_t**2) * eta,
                            device=device
                        ))
                        return pred + noise_scale * noise_seq[step_idx]
                    else:
                        return pred
                        
        # Initialize noise
        y0 = []
        for dur in duration:
            y0.append(torch.randn(dur, 100, device=device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)
        
        # Setup time steps
        t = torch.linspace(0, 1, teacher_steps + 1, device=device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
        t = t[:(t > teacher_stopping_time).float().argmax() + 2]
        t = t[:-1]
        
        # Solve ODE
        trajectory = odeint(fn, y0, t, method="euler")
        
        if teacher_stopping_time < 1.0:
            # If early stopping, compute final step
            pred = fn(t[-1], trajectory[-1])
            test_out = trajectory[-1] + (1 - t[-1]) * pred
            return test_out
        else:
            return trajectory[-1]
            
    def _student_sampling(self, x1, cond, text, student_start_step, verbose, sway_coeff = -1):
        """Perform student model sampling."""
        steps = torch.Tensor([0, 0.25, 0.5, 0.75])
        steps = steps + sway_coeff * (torch.cos(torch.pi / 2 * steps) - 1 + steps)
        steps = steps[student_start_step:]
        
        for step in steps:
            time = torch.Tensor([step]).to(x1.device)
            
            x0 = torch.randn_like(x1)
            t = time.unsqueeze(-1).unsqueeze(-1)
            phi = (1 - t) * x0 + t * x1
            
            if verbose:
                print(f'current step: {step}')
            with torch.no_grad():
                pred = self.generator(
                    x=phi,
                    cond=cond,
                    text=text,
                    time=time,
                    drop_audio_cond=False,
                    drop_text=False
                )
                
                # Predicted mel spectrogram
                output = phi + (1 - t) * pred
                
            x1 = output
            
        return x1
