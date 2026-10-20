from __future__ import annotations

import torch
import torch.nn as nn


class TabularMLP(nn.Module):
    def __init__(
        self, input_dim: int, hidden_channels: int = 128, dropout: float = 0.2
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def manual_weighted_bce(logits: torch.Tensor, targets: torch.Tensor, pos_weight: float):
    max_val = torch.clamp(-logits, min=0)
    loss = (1 - targets) * logits + (1 + (pos_weight - 1) * targets) * (
        max_val + torch.log(torch.exp(-max_val) + torch.exp(-logits - max_val))
    )
    return loss.mean()


def manual_unweighted_bce(logits: torch.Tensor, targets: torch.Tensor):
    max_val = torch.clamp(-logits, min=0)
    loss = (
        (1 - targets) * logits
        + max_val
        + torch.log(torch.exp(-max_val) + torch.exp(-logits - max_val))
    )
    return loss.mean()
