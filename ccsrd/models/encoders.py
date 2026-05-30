# encoders.py

import torch
import torch.nn as nn


class ContentEncoder(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True),
            num_layers=6  # paper: base Transformer = 6 layers
        )

    def forward(self, x):
        # x: (B, T, D)
        return self.encoder(x)


class NonContentEncoder(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True),
            num_layers=6
        )

    def forward(self, x):
        return self.encoder(x)


class TranslationDecoder(nn.Module):
    def __init__(self, d_model=512, vocab_size=5000):
        super().__init__()
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=8, batch_first=True),
            num_layers=6  # base Transformer = 6 layers
        )
        # project decoder output to vocab
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_emb, memory):
        # tgt_emb: (B, T_tgt, d_model)  -- already embedded by caller
        # memory:  (B, T,     d_model)  -- Z_c from content encoder
        out = self.decoder(tgt_emb, memory)       # (B, T_tgt, d_model)
        return self.output_projection(out)         # (B, T_tgt, vocab_size)