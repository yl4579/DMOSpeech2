from torch import nn
import torch 
import copy

from pathlib import Path
from torchaudio.models import Conformer


from f5_tts.model.utils import default
from f5_tts.model.utils import exists
from f5_tts.model.utils import list_str_to_idx
from f5_tts.model.utils import list_str_to_tensor
from f5_tts.model.utils import lens_to_mask
from f5_tts.model.utils import mask_from_frac_lengths


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


class ConformerCTC(nn.Module):
    def __init__(self,
                 vocab_size,
                 mel_dim=100, 
                 num_heads=8, 
                 d_hid=512, 
                 nlayers=6):
        super().__init__()
        
        self.mel_proj = nn.Conv1d(mel_dim, d_hid, kernel_size=3, padding=1)
        
        self.d_hid = d_hid
        
        self.resblock1 = nn.Sequential(
                ResBlock(d_hid),
                nn.GroupNorm(num_groups=1, num_channels=d_hid)
            )
        
        self.resblock2 = nn.Sequential(
                ResBlock(d_hid),
                nn.GroupNorm(num_groups=1, num_channels=d_hid)
            )
        

        self.conf_pre = torch.nn.ModuleList(
            [Conformer(
             input_dim=d_hid,
             num_heads=num_heads,
             ffn_dim=d_hid * 2,
             num_layers=1,
             depthwise_conv_kernel_size=15,
             use_group_norm=True,)
                for _ in range(nlayers // 2)
            ]
        )
        
        self.conf_after = torch.nn.ModuleList(
            [Conformer(
             input_dim=d_hid,
             num_heads=num_heads,
             ffn_dim=d_hid * 2,
             num_layers=1,
             depthwise_conv_kernel_size=7,
             use_group_norm=True,)
                for _ in range(nlayers // 2)
            ]
        )

        self.out = nn.Linear(d_hid, 1 + vocab_size) # 1 for blank

        self.ctc_loss = nn.CTCLoss(blank=vocab_size, zero_infinity=True).cuda()

                
    def forward(self, latent, text=None, text_lens=None):
        layers = []

        x = self.mel_proj(latent.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(1, 2)
        layers.append(nn.functional.avg_pool1d(x, 4))
        # x = x.transpose(1, 2)

        x = self.resblock1(x)
        x = nn.functional.avg_pool1d(x, 2)
        layers.append(nn.functional.avg_pool1d(x, 2))
        x = self.resblock2(x)
        x = nn.functional.avg_pool1d(x, 2)
        layers.append(x)

        x = x.transpose(1, 2)

        batch_size, time_steps, _ = x.shape
        # Create a dummy lengths tensor (all sequences are assumed to be full length).
        input_lengths = torch.full((batch_size,), time_steps, device=x.device, dtype=torch.int64)

        for layer in (self.conf_pre):
            x, _ = layer(x, input_lengths)
            layers.append(x.transpose(1, 2))

        for layer in (self.conf_after):
            x, _ = layer(x, input_lengths)
            layers.append(x.transpose(1, 2))

        x = self.out(x)

        if text_lens is not None and text is not None:
            loss = self.ctc_loss(x.log_softmax(dim=2).transpose(0, 1), text, input_lengths, text_lens)
            return x, layers, loss
        else:
            return x, layers


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
    
    model = ConformerCTC(vocab_size, mel_dim=80, num_heads=8, d_hid=512, nlayers=6).cuda()
    
    text = ["hello world"] * bsz
    lens = torch.randint(1, 1000, (bsz,)).cuda()
    inp = torch.randn(bsz, lens.max(), 80).cuda()
    
    batch, seq_len, dtype, device = *inp.shape[:2], inp.dtype, inp.device
    
    # handle text as string
    text_lens = torch.tensor([len(t) for t in text], device=device)
    if isinstance(text, list):
        if exists(vocab_char_map):
            text = list_str_to_idx(text, vocab_char_map).to(device)
        else:
            text = list_str_to_tensor(text).to(device)
        assert text.shape[0] == batch

    # lens and mask
    if not exists(lens):
        lens = torch.full((batch,), seq_len, device=device)

    out, layers, loss = model(inp, text_lens)

    print(out.shape)
    print(out)
    print(len(layers))
    print(torch.stack(layers, axis=1).shape)
    print(loss)

    probs = out.softmax(dim=2)  # Convert logits to probabilities

    # Greedy decoding
    best_path = torch.argmax(probs, dim=2)

    decoded_sequences = []
    blank_idx = vocab_size

    char_vocab_map = list(vocab_char_map.keys())


    for batch in best_path:
        decoded_sequence = []
        previous_token = None

        for token in batch:
            if token != previous_token:  # Collapse repeated tokens
                if token != blank_idx:  # Ignore blank tokens
                    decoded_sequence.append(token.item())
            previous_token = token

        decoded_sequences.append(decoded_sequence)

    # Convert token indices to characters
    decoded_texts = [''.join([char_vocab_map[token] for token in sequence]) for sequence in decoded_sequences]
    gt_texts = []
    for i in range(text_lens.size(0)):
        gt_texts.append(''.join([char_vocab_map[token] for token in text[i, :text_lens[i]]]))
    
    print(decoded_texts)
    print(gt_texts)