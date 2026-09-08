"""Pérdida de generación (§9.1).

Cross-entropy sobre los tokens de respuesta únicamente. El ``ignore_index``
por defecto es -100 para descartar padding.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def generation_loss(logits: torch.Tensor, labels: torch.Tensor,
                    ignore_index: int = -100) -> torch.Tensor:
    """logits: [B, L, vocab]; labels: [B, L] (long no padding no -100)."""
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )


def truncated_generation_loss(logits: torch.Tensor, labels: torch.Tensor,
                              prefix_len: int, ignore_index: int = -100) -> torch.Tensor:
    """Versión usada por el modelo: solo penaliza los últimos ``labels.size(1)``
    tokens (la respuesta), desplazados tras el prefijo de ``prefix_len`` latents.
    """
    L = labels.size(1)
    resp = logits[:, prefix_len - 1: prefix_len - 1 + L, :]
    return generation_loss(resp, labels, ignore_index)