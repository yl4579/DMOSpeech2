import re
import json
import random
from importlib.resources import files

import torch
import torch.nn.functional as F
import torchaudio
from datasets import Dataset as Dataset_
from datasets import load_from_disk
from torch import nn
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import default




def get_speaker_id(path):
    parts = path.split('/')
    speaker_id = parts[-3]
    return speaker_id


class CustomDataset(Dataset):
    def __init__(
        self,
        custom_dataset: Dataset,
        durations=None,
        target_sample_rate=24_000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
        validation=False,
        validation_num=5000,
        data_augmentation=False,
        return_wavform=False,
        remove_starting_space=True,
        need_prompt_speech=False,
        prompt_repository: dict=None,
    ):
        self.data = custom_dataset
        self.durations = durations
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel

        if not preprocessed_mel:
            self.mel_spectrogram = default(
                mel_spec_module,
                MelSpec(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    target_sample_rate=target_sample_rate,
                    mel_spec_type=mel_spec_type,
                ),
            )
        
        self.validation = validation
        self.validation_num = validation_num

        if (not validation) and data_augmentation:
            print('Using data augmentation.')
            self.augment = Compose([
                AddBackgroundNoise(
                sounds_path="/data5/ESC-50-master",
                min_snr_db=3.0,
                max_snr_db=30.0,
                noise_transform=PolarityInversion(),
                p=0.5
                ),
                AddGaussianNoise(
                    min_amplitude=0.001,
                    max_amplitude=0.015,
                    p=0.5
                ),  
                PitchShift(
                    min_semitones=-12.0,
                    max_semitones=12.0,
                    p=0.8
                ),
                ApplyImpulseResponse(ir_path="/data5/Audio", p=1.0),
                Aliasing(min_sample_rate=4000, max_sample_rate=30000, p=0.3),
                BandPassFilter(min_center_freq=100.0, max_center_freq=6000, p=0.2),
                SevenBandParametricEQ(p=0.2),
                TanhDistortion(
                    min_distortion=0.01,
                    max_distortion=0.7,
                    p=0.2
                ),
            ])
        else:
            print('No data augmentation.')
            self.augment = None

        self.return_wavform = return_wavform
        self.remove_starting_space = remove_starting_space

        if need_prompt_speech:
            if prompt_repository == None:
                self.prompt_repository = {}
                for row in tqdm(self.data):
                    audio_path = row["audio_path"]
                    text = row["text"]
                    duration = row["duration"]
                    spk_id = get_speaker_id(audio_path)
                    assert spk_id != None and spk_id != 'mp3'
                    if spk_id not in self.prompt_repository:
                        self.prompt_repository[spk_id] = [row]
                    else:
                        self.prompt_repository[spk_id].append(row)
            else:
                self.prompt_repository = prompt_repository

            print(f'Grouped samples into {len(self.prompt_repository.keys())} speakers.')           
            self.need_prompt_speech = True

        else:
            self.need_prompt_speech = False


    def get_frame_len(self, index):
        if self.validation:
            index += len(self.data) - self.validation_num

        if (
            self.durations is not None
        ):  # Please make sure the separately provided durations are correct, otherwise 99.99% OOM
            return self.durations[index] * self.target_sample_rate / self.hop_length
        return self.data[index]["duration"] * self.target_sample_rate / self.hop_length

    def __len__(self):
        if not self.validation:
            return len(self.data) - self.validation_num
        return self.validation_num

    def __getitem__(self, index, return_row=True, return_path=False):
        if self.validation:
            index += len(self.data) - self.validation_num

        out = {}

        while True:
            row = self.data[index]
            audio_path = row["audio_path"]
            text = row["text"]
            duration = row["duration"]

            if not isinstance(text, list):
                text = list(text)

            # filter by given length
            if (0.3 <= duration <= 30) and (0 < len(text) < 2048):
                break  # valid

            index = (index + 1) % len(self.data)

        if self.remove_starting_space:
            while len(text) > 1 and text[0] == ' ':
                text = text[1:]
    
        if self.preprocessed_mel:
            mel_spec = torch.tensor(row["mel_spec"])
        else:
            audio, source_sample_rate = torchaudio.load(audio_path)

            # make sure mono input
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            # resample if necessary
            if source_sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
                audio = resampler(audio)

            if not self.validation:
                if self.augment != None:
                    audio = self.augment(audio.squeeze().numpy(), sample_rate=self.target_sample_rate)
                    audio = torch.from_numpy(audio).float().unsqueeze(0)

            # to mel spectrogram
            mel_spec = self.mel_spectrogram(audio)
            mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        out['mel_spec'] = mel_spec
        out['text'] = text
        out['duration'] = duration
        out['target_text'] = self.data[(index + len(self.data) // 2) % len(self.data)]["text"]

        if self.return_wavform:
            out['wav'] = audio

        if return_path:
            out['path'] = audio_path

        if return_row:
            out['row'] = row

        # Sample a prompt speech of the same speaker
        # From prompt_repository
        if self.need_prompt_speech:
            spk = get_speaker_id(audio_path)
            spk_repository = self.prompt_repository[spk]
            _count = 100
            while True:
                pmt_row = random.choice(spk_repository)
                pmt_audio_path = pmt_row['audio_path']
                pmt_text = pmt_row['text']
                pmt_duration = pmt_row['duration']

                if not isinstance(pmt_text, list):
                    pmt_text = list(pmt_text)

                # filter by given length
                if 0.3 <= pmt_duration <= 30 and (0 < len(pmt_text) < 2048):
                    if pmt_text != text:
                        break
                    _count =  _count - 1
                    if _count <= 0:
                        break

            if self.remove_starting_space:
                while len(pmt_text) > 1 and pmt_text[0] == ' ':
                    pmt_text = pmt_text[1:]
        
            if self.preprocessed_mel:
                pmt_mel_spec = torch.tensor(pmt_row["mel_spec"])
            else:
                pmt_audio, source_sample_rate = torchaudio.load(pmt_audio_path)

                # make sure mono input
                if pmt_audio.shape[0] > 1:
                    pmt_audio = torch.mean(pmt_audio, dim=0, keepdim=True)

                # resample if necessary
                if source_sample_rate != self.target_sample_rate:
                    resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
                    pmt_audio = resampler(pmt_audio)

                if not self.validation:
                    if self.augment != None:
                        pmt_audio = self.augment(pmt_audio.squeeze().numpy(), sample_rate=self.target_sample_rate)
                        pmt_audio = torch.from_numpy(pmt_audio).float().unsqueeze(0)

                # to mel spectrogram
                pmt_mel_spec = self.mel_spectrogram(pmt_audio)
                pmt_mel_spec = pmt_mel_spec.squeeze(0)  # '1 d t -> d t'

            out['pmt_mel_spec'] = pmt_mel_spec
            out['pmt_text'] = pmt_text
            out['pmt_duration'] = pmt_duration

            if self.return_wavform:
                out['pmt_wav'] = pmt_audio

            if return_path:
                out['pmt_path'] = pmt_audio_path

            if return_row:
                out['pmt_row'] = pmt_row

        return out


# Dynamic Batch Sampler
class DynamicBatchSampler(Sampler[list[int]]):
    """Extension of Sampler that will do the following:
    1.  Change the batch size (essentially number of sequences)
        in a batch to ensure that the total number of frames are less
        than a certain threshold.
    2.  Make sure the padding efficiency in the batch is high.
    """

    def __init__(
        self, sampler: Sampler[int], frames_threshold: int, max_samples=0, random_seed=None, drop_last: bool = False
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples

        indices, batches = [], []
        data_source = self.sampler.data_source

        # for idx in tqdm(
        #     self.sampler, desc="Sorting with sampler... if slow, check whether dataset is provided with duration"
        # ):
        for idx in self.sampler:
            indices.append((idx, data_source.get_frame_len(idx)))
        indices.sort(key=lambda elem: elem[1])

        batch = []
        batch_frames = 0
        # for idx, frame_len in tqdm(
        #     indices, desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu"
        # ):
        for idx, frame_len in indices:
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    batch = []
                    batch_frames = 0

        if not drop_last and len(batch) > 0:
            batches.append(batch)

        del indices

        # if want to have different batches between epochs, may just set a seed and log it in ckpt
        # cuz during multi-gpu training, although the batch on per gpu not change between epochs, the formed general minibatch is different
        # e.g. for epoch n, use (random_seed + n)
        random.seed(random_seed)
        random.shuffle(batches)

        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


# Load dataset

def load_dataset(
    dataset_name: str,
    tokenizer: str = "pinyin",
    dataset_type: str = "CustomDataset",
    audio_type: str = "raw",
    mel_spec_module: nn.Module | None = None,
    mel_spec_kwargs: dict = dict(),
    split: str = "train",
    data_augmentation: bool = False,
    return_wavform: bool = False,
    remove_starting_space: bool = True,
    need_prompt_speech: bool = False,
    prompt_repository: dict = None
) -> CustomDataset:
    """
    dataset_type    - "CustomDataset" if you want to use tokenizer name and default data path to load for train_dataset
                    - "CustomDatasetPath" if you just want to pass the full path to a preprocessed dataset without relying on tokenizer
    """

    print("Loading dataset ...")

    if dataset_type == "CustomDataset":
        rel_data_path = str(f'/home/yl4579/F5-TTS-diff/F5-TTS-DMD-flow-ds/data/{dataset_name}_{tokenizer}')
        if 'LibriTTS_100_360_500_char_pinyin' in rel_data_path:
            rel_data_path = rel_data_path.replace('LibriTTS_100_360_500_char_pinyin', 'LibriTTS_100_360_500_char')
        if audio_type == "raw":
            try:
                train_dataset = load_from_disk(f"{rel_data_path}/raw")
            except:  # noqa: E722
                train_dataset = Dataset_.from_file(f"{rel_data_path}/raw.arrow")
            preprocessed_mel = False
        elif audio_type == "mel":
            train_dataset = Dataset_.from_file(f"{rel_data_path}/mel.arrow")
            preprocessed_mel = True
        with open(f"{rel_data_path}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(
            train_dataset,
            durations=durations,
            preprocessed_mel=preprocessed_mel,
            mel_spec_module=mel_spec_module,
            **mel_spec_kwargs,
            validation=split == "val",
            data_augmentation=data_augmentation,
            return_wavform=return_wavform,
            remove_starting_space=remove_starting_space,
            need_prompt_speech=need_prompt_speech,
            prompt_repository=prompt_repository
        )

    elif dataset_type == "CustomDatasetPath":
        try:
            train_dataset = load_from_disk(f"{dataset_name}/raw")
        except:  # noqa: E722
            train_dataset = Dataset_.from_file(f"{dataset_name}/raw.arrow")

        with open(f"{dataset_name}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(
            train_dataset, durations=durations, preprocessed_mel=preprocessed_mel, **mel_spec_kwargs
        )

    return train_dataset


# collation
def collate_fn(batch):
    # Extract mel_specs and their lengths
    mel_specs = [item["mel_spec"].squeeze(0) for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()
    
    # Pad mel_specs
    padded_mel_specs = []
    for spec in mel_specs:  # TODO. maybe records mask for attention here
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_mel_specs.append(padded_spec)
    mel_specs = torch.stack(padded_mel_specs)

    text = [item['text'] for item in batch]
    target_text = [item['target_text'] for item in batch]

    text_lengths = torch.LongTensor([len(item) for item in text])

    out = dict(
        mel=mel_specs,
        mel_lengths=mel_lengths,
        text=text,
        text_lengths=text_lengths,
        target_text=target_text,
    )

    if 'pmt_mel_spec' in batch[0]:
        pmt_mel_specs = [item["pmt_mel_spec"].squeeze(0) for item in batch]
        pmt_mel_lengths = torch.LongTensor([spec.shape[-1] for spec in pmt_mel_specs])
        max_pmt_mel_length = pmt_mel_lengths.amax()
        
        # Pad mel_specs
        padded_pmt_mel_specs = []
        for spec in pmt_mel_specs: 
            padding = (0, max_pmt_mel_length - spec.size(-1))
            padded_spec = F.pad(spec, padding, value=0)
            padded_pmt_mel_specs.append(padded_spec)
        pmt_mel_specs = torch.stack(padded_pmt_mel_specs)

        out['pmt_mel_specs'] = pmt_mel_specs

    if 'pmt_text' in batch[0]:
        pmt_text = [item['pmt_text'] for item in batch]
        pmt_text_lengths = torch.LongTensor([len(item) for item in pmt_text])

        out['pmt_text'] = pmt_text
        out['pmt_text_lengths'] = pmt_text_lengths

    return out
