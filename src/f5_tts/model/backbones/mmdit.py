"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from x_transformers.x_transformers import RotaryEmbedding

from f5_tts.model.modules import (
    TimestepEmbedding,
    ConvPositionEmbedding,
    MMDiTBlock,
    DiTBlock,
    AdaLayerNormZero_Final,
    precompute_freqs_cis,
    get_pos_embed_indices,
)

from f5_tts.model.utils import (
    default,
    exists,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)
# text embedding


class TextEmbedding(nn.Module):
    def __init__(self, out_dim, text_num_embeds):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, out_dim)  # will use 0 as filler token

        self.precompute_max_pos = 1024
        self.register_buffer("freqs_cis", precompute_freqs_cis(out_dim, self.precompute_max_pos), persistent=False)

    def forward(self, text: int["b nt"], drop_text=False) -> int["b nt d"]:  # noqa: F722
        text = text + 1
        if drop_text:
            text = torch.zeros_like(text)
        text = self.text_embed(text)

        # sinus pos emb
        batch_start = torch.zeros((text.shape[0],), dtype=torch.long)
        batch_text_len = text.shape[1]
        pos_idx = get_pos_embed_indices(batch_start, batch_text_len, max_pos=self.precompute_max_pos)
        text_pos_embed = self.freqs_cis[pos_idx]

        text = text + text_pos_embed

        return text


# noised input & masked cond audio embedding


class AudioEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(2 * in_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(out_dim)

    def forward(self, x: float["b n d"], cond: float["b n d"], drop_audio_cond=False):  # noqa: F722
        if drop_audio_cond:
            cond = torch.zeros_like(cond)
        x = torch.cat((x, cond), dim=-1)
        x = self.linear(x)
        x = self.conv_pos_embed(x) + x
        return x


# Transformer backbone using MM-DiT blocks


class MMDiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        text_depth=4,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        text_num_embeds=256,
        mel_dim=100,
        checkpoint_activations=False,
        text_encoder=True,

    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if text_encoder:
            self.text_encoder = TextEncoder(text_num_embeds=text_num_embeds, 
                                        text_dim=dim,
                                        depth=text_depth,
                                        heads=heads,
                                        dim_head=dim_head,
                                        ff_mult=ff_mult,
                                        dropout=dropout)
        else:
            self.text_encoder = None
            self.text_embed = TextEmbedding(dim, text_num_embeds)
        
        self.audio_embed = AudioEmbedding(mel_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [
                MMDiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=dropout,
                    ff_mult=ff_mult,
                    context_pre_only=i == depth - 1,
                )
                for i in range(depth)
            ]
        )
        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)
        
        self.checkpoint_activations = checkpoint_activations


    def forward(
        self,
        x: float["b n d"],  # nosied input audio  # noqa: F722
        cond: float["b n d"],  # masked cond audio  # noqa: F722
        text: int["b nt"],  # text  # noqa: F722
        time: float["b"] | float[""],  # time step  # noqa: F821 F722
        drop_audio_cond,  # cfg for cond audio
        drop_text,  # cfg for text
        mask: bool["b n"] | None = None,  # noqa: F722
        text_mask: bool["b nt"] | None = None,  # noqa: F722
    ):
        batch = x.shape[0]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning (time), c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(time)
        if self.text_encoder is not None:
            c = self.text_encoder(text, t, mask=text_mask, drop_text=drop_text)
        else:
            c = self.text_embed(text, drop_text=drop_text)
        
        x = self.audio_embed(x, cond, drop_audio_cond=drop_audio_cond)

        seq_len = x.shape[1]
        text_len = text.shape[1]
        rope_audio = self.rotary_embed.forward_from_seq_len(seq_len)
        rope_text = self.rotary_embed.forward_from_seq_len(text_len)
        
        # if mask is not None:
        #     rope_audio = self.rotary_embed.forward_from_seq_len(seq_len + 1)
            
        #     dummy_token = torch.zeros((x.shape[0], 1, x.shape[-1]), device=x.device, dtype=x.dtype)
        #     x = torch.cat([x, dummy_token], dim=1)  # shape is now [b, nw+1, d]
            
        #     # pad the mask so that new dummy token is always masked out
        #     # mask: [b, nw] -> [b, nw+1]
        #     false_col = torch.zeros((x.shape[0], 1), dtype=torch.bool, device=x.device)
        #     mask = torch.cat([mask, false_col], dim=1)
                
        # if text_mask is not None:
        #     rope_text = self.rotary_embed.forward_from_seq_len(text_len + 1)

        #     dummy_token = torch.zeros((c.shape[0], 1, c.shape[-1]), device=c.device, dtype=c.dtype)
        #     c = torch.cat([c, dummy_token], dim=1)  # shape is now [b, nt+1, d]
        
        #     # pad the text mask so that new dummy token is always masked out
        #     # text_mask: [b, nt] -> [b, nt+1]
        #     false_col = torch.zeros((c.shape[0], 1), dtype=torch.bool, device=c.device)
        #     text_mask = torch.cat([text_mask, false_col], dim=1)
                        
        for block in self.transformer_blocks:
            c, x = block(x, c, t, mask=mask, src_mask=text_mask, rope=rope_audio, c_rope=rope_text)

        x = self.norm_out(x, t)
        output = self.proj_out(x)
        

        return output

class TextEncoder(nn.Module):
    def __init__(
        self,
        text_num_embeds: int,
        text_dim: int = 512,
        depth: int = 4,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        """
        A simple text encoder: an embedding layer + multiple DiTBlocks or any other
        transformer blocks for text-only self-attention.
        """
        super().__init__()
        # Embeddings
        self.text_embed = TextEmbedding(text_dim, text_num_embeds)
        self.rotary_embed = RotaryEmbedding(dim_head)
        
        # Example stack of DiTBlocks or any custom blocks
        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=text_dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        text: int["b nt"],  # noqa: F821
        time: float["b"] | float[""],  # time step  # noqa: F821 F722
        mask: bool["b nt"] | None = None,  # noqa: F821 F722
        drop_text: bool = False
    ):
        """
        Encode text into hidden states of shape [b, nt, d].
        """
        batch, seq_len, device = text.shape[0], text.shape[1], text.device

        if drop_text:
            text = torch.zeros_like(text)

        # Basic embedding
        hidden_states = self.text_embed(text, seq_len)  # [b, nt, d]
        
        # lens and mask
        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        # Pass through self-attention blocks
        for block in self.transformer_blocks:
            # Here, you likely want standard self-attn, so no cross-attn
            hidden_states = block(
                x=hidden_states,
                t=time,       # no time embedding for the text encoder by default
                mask=mask,    # or pass a text mask if needed
                rope=rope     # pass a rope if you want rotary embeddings for text
            )
        return hidden_states

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
    
    text = ["hello world"] * bsz
    text_lens = torch.ones((bsz, ), dtype=torch.long) * len("hello world")
    text_lens[-1] = 5
    device = "cuda"
    batch = bsz
    time_embed = TimestepEmbedding(512).to(device)
    
    
    # handle text as string
    if isinstance(text, list):
        if exists(vocab_char_map):
            text = list_str_to_idx(text, vocab_char_map).to(device)
        else:
            text = list_str_to_tensor(text).to(device)
        assert text.shape[0] == batch  
        
    time = torch.rand((batch,), device=device)
    text_mask = lens_to_mask(text_lens).to(device)

    # # test text encoder
    # text_encoder = TextEncoder(
    #     text_num_embeds=vocab_size,
    #     text_dim=512,
    #     depth=4,
    #     heads=8,
    #     dim_head=64,
    #     ff_mult=4,
    #     dropout=0.1
    # ).to('cuda')
    # hidden_states = text_encoder(text, time_embed(time), mask)
    # print(hidden_states.shape)  # [bsz, seq_len, text_dim]
    
    # test MMDiT
    mel_dim = 80
    model = MMDiT(
        dim=512,
        text_depth=4,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        text_num_embeds=vocab_size,
        mel_dim=mel_dim
    ).to(device)
    
    x = torch.rand((batch, 100, mel_dim), device=device)
    cond = torch.rand((batch, 100, mel_dim), device=device)
    lens = torch.ones((batch,), dtype=torch.long) * 100
    mask = lens_to_mask(lens).to(device)
    
    output = model(x, cond, text, time, drop_audio_cond=False, drop_text=False, mask=mask, text_mask=text_mask)
    
    print(output.shape)  # [bsz, seq_len, mel_dim]