import torch
import torch.nn as nn

# from tts_encode import tts_encode

def calculate_remaining_lengths(mel_lengths):
    B = mel_lengths.shape[0]
    max_L = mel_lengths.max().item()  # Get the maximum length in the batch

    # Create a range tensor: shape (max_L,), [0, 1, 2, ..., max_L-1]
    range_tensor = torch.arange(max_L, device=mel_lengths.device).expand(B, max_L)

    # Compute targets using broadcasting: (L-1) - range_tensor
    remain_lengths = (mel_lengths[:, None] - 1 - range_tensor).clamp(min=0)

    return remain_lengths


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # Shape: (1, max_len, hidden_dim)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x


class SpeechLengthPredictor(nn.Module):

    def __init__(self, 
        vocab_size=2545, n_mel=100, hidden_dim=256, 
        n_text_layer=4, n_cross_layer=4, n_head=8,
        output_dim=1,
    ):
        super().__init__()
        
        # Text Encoder: Embedding + Transformer Layers
        self.text_embedder = nn.Embedding(vocab_size+1, hidden_dim, padding_idx=vocab_size)
        self.text_pe = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_head, dim_feedforward=hidden_dim*2, batch_first=True
        )
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_text_layer)
        
        # Mel Spectrogram Embedder
        self.mel_embedder = nn.Linear(n_mel, hidden_dim)
        self.mel_pe = PositionalEncoding(hidden_dim)

        # Transformer Decoder Layers with Cross-Attention in Every Layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=n_head, dim_feedforward=hidden_dim*2, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_cross_layer)
        
        # Final Classification Layer
        self.predictor = nn.Linear(hidden_dim, output_dim)

    def forward(self, text_ids, mel):
        # Encode text
        text_embedded = self.text_pe(self.text_embedder(text_ids))
        text_features = self.text_encoder(text_embedded) # (B, L_text, D)
        
        # Encode Mel spectrogram
        mel_features = self.mel_pe(self.mel_embedder(mel))  # (B, L_mel, D)
        
        # Causal Masking for Decoder
        seq_len = mel_features.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(mel.device)
        # causal_mask = torch.triu(
        #     torch.full((seq_len, seq_len), float('-inf'), device=mel.device), diagonal=1
        # )

        # Transformer Decoder with Cross-Attention in Each Layer
        decoder_out = self.decoder(mel_features, text_features, tgt_mask=causal_mask)
        
        # Length Prediction
        length_logits = self.predictor(decoder_out).squeeze(-1)
        return length_logits
