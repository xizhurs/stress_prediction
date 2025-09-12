import torch.nn as nn
import torch
import numpy as np


class LSTMClassifier(nn.Module):
    def __init__(
        self, feat_dim, hidden=128, num_layers=2, num_classes=4, dropout=0.2, bidir=True
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            feat_dim,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidir,
            dropout=dropout,
        )
        out_dim = hidden * (2 if bidir else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim), nn.Dropout(dropout), nn.Linear(out_dim, num_classes)
        )

    def forward(self, x):  # x: [B, N, F]
        h, _ = self.lstm(x)  # h: [B, N, H*dir]
        h_last = h[:, -1, :]  # take last time step
        return self.head(h_last)  # logits [B, C]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=36):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, T, D]

    def forward(self, x):  # x: [B,T,D]
        return x + self.pe[:, : x.size(1), :]


class TransEncClassifier(nn.Module):
    def __init__(
        self,
        feat_dim,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.2,
        num_classes=4,
    ):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.pos = PositionalEncoding(d_model, max_len=60)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, num_classes)
        )

    def forward(self, x):  # [B,T,F]
        z = self.input_proj(x)
        z = self.pos(z)
        z = self.enc(z)  # [B,T,D]
        z_last = z[:, -1, :]
        return self.head(z_last)
