from __future__ import annotations

import torch
import torch.nn as nn


class TabularEmbeddingMLP(nn.Module):
    """Two hidden-layer supervised tabular encoder with a 128-d latent output."""

    def __init__(self, input_dim: int, hidden_channels: int = 128, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_channels, 1)

    def forward(self, features: torch.Tensor, return_embeddings: bool = False) -> torch.Tensor:
        embeddings = self.encoder(features)
        if return_embeddings:
            return embeddings
        return self.classifier(embeddings).squeeze(-1)


def weighted_bce(logits: torch.Tensor, targets: torch.Tensor, pos_weight: float) -> torch.Tensor:
    weight = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    return nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=weight)


def unweighted_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(logits, targets)
