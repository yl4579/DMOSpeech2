# part of the code is borrowed from https://github.com/lawlict/ECAPA-TDNN

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as trans
from pathlib import Path
from torchaudio.models import Conformer

from f5_tts.model.utils import (
    default,
    exists,
    list_str_to_idx,
    list_str_to_tensor,
    lens_to_mask,
    mask_from_frac_lengths,
)

class ResBlock(nn.Module):
    def __init__(self, hidden_dim, n_conv=3, dropout_p=0.2):
        super().__init__()
        self._n_groups = 8
        self.blocks = nn.ModuleList([
            self._get_conv(hidden_dim, dilation=3**i, dropout_p=dropout_p)
            for i in range(n_conv)])


    def forward(self, x):
        for block in self.blocks:
            res = x
            x = block(x)
            x += res
        return x

    def _get_conv(self, hidden_dim, dilation, dropout_p=0.2):
        layers = [
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation),
            nn.ReLU(),
            nn.GroupNorm(num_groups=self._n_groups, num_channels=hidden_dim),
            nn.Dropout(p=dropout_p),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        ]
        return nn.Sequential(*layers)

class ConformerDiscirminator(nn.Module):
    def __init__(self, input_dim, channels=512, num_layers=3, num_heads=8, depthwise_conv_kernel_size=15, use_group_norm=True):
        super().__init__()
        
        self.input_layer = nn.Conv1d(input_dim, channels, kernel_size=3, padding=1)

        self.resblock1 = nn.Sequential(
                ResBlock(channels),
                nn.GroupNorm(num_groups=1, num_channels=channels)
            )
        
        self.resblock2 = nn.Sequential(
                ResBlock(channels),
                nn.GroupNorm(num_groups=1, num_channels=channels)
            )

        self.conformer1 = Conformer(**{"input_dim": channels, 
                "num_heads": num_heads, 
                "ffn_dim": channels * 2, 
                "num_layers": 1, 
                "depthwise_conv_kernel_size": depthwise_conv_kernel_size // 2,
                "use_group_norm": use_group_norm})

        self.conformer2 = Conformer(**{"input_dim": channels, 
                "num_heads": num_heads, 
                "ffn_dim": channels * 2, 
                "num_layers": num_layers - 1, 
                "depthwise_conv_kernel_size": depthwise_conv_kernel_size,
                "use_group_norm": use_group_norm})
        
        self.linear = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x):
        # x = torch.stack(x, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
        x = torch.cat(x, dim=-1)
        x = x.transpose(1, 2)

        x = self.input_layer(x)

        x = self.resblock1(x)
        x = nn.functional.avg_pool1d(x, 2)
        x = self.resblock2(x)
        x = nn.functional.avg_pool1d(x, 2)
        
        # Transpose to (B, T, C) for the conformer.
        x = x.transpose(1, 2)
        batch_size, time_steps, _ = x.shape
        # Create a dummy lengths tensor (all sequences are assumed to be full length).
        lengths = torch.full((batch_size,), time_steps, device=x.device, dtype=torch.int64)
        # The built-in Conformer returns (output, output_lengths); we discard lengths.

        x, _ = self.conformer1(x, lengths)
        x, _ = self.conformer2(x, lengths)
        # Transpose back to (B, C, T).
        x = x.transpose(1, 2)

        # out = self.bn(self.pooling(out))
        out = self.linear(x).squeeze(1)

        return out

if __name__ == "__main__":
    from f5_tts.model.utils import get_tokenizer
    from f5_tts.model import DiT

    bsz = 2
    
    tokenizer = "pinyin"  # 'pinyin', 'char', or 'custom'
    tokenizer_path = None  # if tokenizer = 'custom', define the path to the tokenizer you want to use (should be vocab.txt)
    dataset_name = "Emilia_ZH_EN"
    if tokenizer == "custom":
        tokenizer_path = tokenizer_path
    else:
        tokenizer_path = dataset_name
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)
    
    
    fake_unet = DiT(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4, text_num_embeds=vocab_size, mel_dim=80)

    fake_unet = fake_unet.cuda()

    text = ["hello world"] * bsz
    lens = torch.randint(1, 1000, (bsz,)).cuda()
    inp = torch.randn(bsz, lens.max(), 80).cuda()
    
    batch, seq_len, dtype, device = *inp.shape[:2], inp.dtype, inp.device

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

    # Sample a time
    time = torch.rand((batch,), dtype=dtype, device=device)

    x1 = inp
    x0 = torch.randn_like(x1)
    t = time.unsqueeze(-1).unsqueeze(-1)
    
    phi = (1 - t) * x0 + t * x1
    flow = x1 - x0
    cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

    layers = fake_unet(
        x=phi, 
        cond=cond,
        text=text, 
        time=time, 
        drop_audio_cond=False,
        drop_text=False,
        classify_mode=True
    )

    # layers = torch.stack(layers, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
    # print(layers.shape)

    from ctcmodel import ConformerCTC
    ctcmodel = ConformerCTC(vocab_size=vocab_size, mel_dim=80, num_heads=8, d_hid=512, nlayers=6).cuda()
    real_out, layer = ctcmodel(inp)
    layer = layer[-3:] # only use the last 3 layers
    layer = [F.interpolate(l, mode='nearest', scale_factor=4).transpose(-1, -2) for l in layer]
    if layer[0].size(1) < layers[0].size(1):
        layer = [F.pad(l, (0, 0, 0, layers[0].size(1) - l.size(1))) for l in layer]
    
    layers = layer + layers

    model = ConformerDiscirminator(input_dim=23 * 1024 + 3 * 512, 
                            channels=512
                            )
    

    model = model.cuda()
    print(model)
    out = model(layers)
    print(out.shape)
