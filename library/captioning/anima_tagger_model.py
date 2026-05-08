"""Anima tagger head — multi-label tags + 3-class rating, off frozen PE-Core.

Architecture:

::

    feature [d_in]
        ↓ LayerNorm + Linear(d_in, d_hidden) + GELU + Dropout
    trunk_h [d_hidden]
        ├─→ Linear(d_hidden, n_tags)      → tag_logits
        └─→ Linear(d_hidden, n_ratings)   → rating_logits

The trunk is shared between heads so the rating signal nudges the same
representation that's predicting tags. ``n_tags``/``n_ratings``/``d_in``
all come from ``vocab.json`` + the cached PE feature shape.

Inference receives both heads in one forward; training computes both
losses and combines with ``λ_rating`` (defaults below).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class AnimaTaggerConfig:
    d_in: int
    n_tags: int
    n_ratings: int = 3
    d_hidden: int = 1024
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return {
            "d_in": self.d_in,
            "n_tags": self.n_tags,
            "n_ratings": self.n_ratings,
            "d_hidden": self.d_hidden,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AnimaTaggerConfig":
        return cls(
            d_in=int(d["d_in"]),
            n_tags=int(d["n_tags"]),
            n_ratings=int(d.get("n_ratings", 3)),
            d_hidden=int(d.get("d_hidden", 1024)),
            dropout=float(d.get("dropout", 0.1)),
        )


class AnimaTaggerHead(nn.Module):
    def __init__(self, cfg: AnimaTaggerConfig):
        super().__init__()
        self.cfg = cfg
        self.trunk = nn.Sequential(
            nn.LayerNorm(cfg.d_in),
            nn.Linear(cfg.d_in, cfg.d_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.tag_head = nn.Linear(cfg.d_hidden, cfg.n_tags)
        self.rating_head = nn.Linear(cfg.d_hidden, cfg.n_ratings)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(feat)
        return self.tag_head(h), self.rating_head(h)
