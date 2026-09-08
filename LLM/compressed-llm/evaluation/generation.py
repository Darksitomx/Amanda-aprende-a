"""Evaluación de calidad de generación (§22).

Métricas: loss/perplexity, comparación contra respuesta elegida, y otras.
En V0 se registran métricas básicas; la preferencia A/B y la evaluación humana
se cubren en `preference.py` y en fases posteriores.
"""
from __future__ import annotations

import math

import torch


@torch.no_grad()
def perplexity_of_output(model, context_ids, context_mask,
                         output_ids, labels) -> float:
    """Devuelve la perplejidad sobre los tokens de respuesta."""
    out = model.forward(context_ids, context_mask, output_ids=output_ids,
                        labels=labels)
    return math.exp(min(out["loss"].item(), 30.0))


@torch.no_grad()
def average_token_logprob(model, context_ids, context_mask,
                          output_ids) -> float:
    """Log-prob media (por token) de la respuesta — útil para preference (§9.2)."""
    out = model.forward(context_ids, context_mask, output_ids=output_ids,
                        labels=None)
    logits = out["logits"]
    # nos quedamos con la región de respuesta
    prefix_len = (model.cfg.compressor.latent_count
                  if model.use_compression else context_ids.size(1))
    L = output_ids.size(1)
    resp = logits[:, prefix_len - 1: prefix_len - 1 + L, :]
    logp = torch.log_softmax(resp, dim=-1)
    return logp.mean().item()