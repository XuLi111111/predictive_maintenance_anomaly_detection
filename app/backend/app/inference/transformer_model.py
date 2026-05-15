"""TransformerFusionLite — Xu Li's model architecture (T7).

The class definition is copied verbatim from
`src/deep_learning/SKAB_TransformerFusionLite_TrainingSearch_ByXuLi.py`
so that `torch.load(model_transformer.pt)` can resolve it at inference
time. Any change to the trained-model layout MUST be mirrored here, or
loading will silently produce garbage.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TransformerFusionLiteModel(nn.Module):
    """Lightweight Transformer branch + statistical features.

    Architecture (Xu Li, 2026 capstone):
      - Linear input projection (input_dim → d_model)
      - 1+ TransformerEncoderLayer with GELU activation
      - Attention-pooled deep features (d_model)
      - Hand-crafted statistical features (input_dim * 10)
      - Concatenate and MLP head → single logit
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 32,
        nhead: int = 4,
        ff_dim: int = 64,
        dropout: float = 0.1,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn = nn.Linear(d_model, 1)

        stat_dim = input_dim * 10
        self.stat_proj = nn.Sequential(
            nn.Linear(stat_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(d_model + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def extract_stat_features(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = x.mean(dim=1)
        x_std = x.std(dim=1)
        x_max = x.max(dim=1).values
        x_min = x.min(dim=1).values
        x_last = x[:, -1, :]
        x_first = x[:, 0, :]
        x_last_first = x_last - x_first
        x_diff = x[:, 1:, :] - x[:, :-1, :]
        x_diff_mean = x_diff.mean(dim=1)
        x_diff_std = x_diff.std(dim=1)
        x_absmax = x.abs().max(dim=1).values

        return torch.cat(
            [
                x_mean, x_std, x_max, x_min,
                x_last, x_first, x_last_first,
                x_diff_mean, x_diff_std, x_absmax,
            ],
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_deep = self.input_proj(x)
        x_deep = self.transformer(x_deep)
        weights = torch.softmax(self.attn(x_deep), dim=1)
        x_deep = (x_deep * weights).sum(dim=1)

        x_stat = self.extract_stat_features(x)
        x_stat = self.stat_proj(x_stat)

        feat = torch.cat([x_deep, x_stat], dim=1)
        out = self.fc(feat).squeeze()
        # Return (B,) even when batch size is 1 (squeeze would give scalar).
        if out.dim() == 0:
            out = out.unsqueeze(0)
        return out


__all__ = ["TransformerFusionLiteModel"]
