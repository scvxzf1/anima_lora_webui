"""Memory-saving projection helpers for the LoKR plugin."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def lokr_project(x, w1, w2, factor, in_dim, out_dim):
    """Project through the LyCORIS-compatible Kronecker LoKR weight."""

    del factor, in_dim, out_dim
    weight = torch.kron(w1.float(), w2.float())
    return F.linear(x.float(), weight)
