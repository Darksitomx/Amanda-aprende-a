"""Utilidades compartidas: máscaras causales, parametrización de seed,
contadores de parámetros y estimaciones de FLOPs.
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch

from .config import Config


def set_seed(seed: int) -> None:
    """Fija la semilla para random, numpy y torch (más cuda deterministic)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def causal_mask(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Máscara causal triangular inferior: ``1 = allow``, ``0 = mask``.

    Returns
    -------
        ``Tensor[seq_len, seq_len]`` booleano o float (1.0/0.0).
    """
    return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))


def mask_allow_to_attention(mask_allow: torch.Tensor) -> torch.Tensor:
    """Convierte máscara de "permitido" (True=allow) a máscara de atención
    conjunta usada por ``F.scaled_dot_product_attention`` o manual.

    - ``True``  -> 0.0 (no penalizado, se permite atender)
    - ``False`` -> -inf (bloqueado)
    """
    return torch.where(mask_allow, torch.zeros((), device=mask_allow.device),
                       torch.full(mask_allow.shape, float("-inf"), device=mask_allow.device))


def count_parameters(model: torch.nn.Module) -> int:
    """Número total de parámetros entrenables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_non_embedding(model: torch.nn.Module, embedding: torch.nn.Module) -> int:
    """Parámetros entrenables excluyendo un determinado módulo (p. ej. embedding)."""
    total = count_parameters(model)
    emb = count_parameters(embedding)
    return total - emb


def estimate_attention_flops(
    seq_ctx: int, seq_lat: int, heads: int, head_dim: int, layers: int
) -> float:
    """Estimación aproximada de FLOPs (multiplicar-acumular * 2) del módulo de
    atención para una secuencia ``seq_ctx`` atendida por ``seq_lat`` queries.

    Se usa únicamente a efectos de comparación relativa (ver §33): no representa
    el coste total del sistema, solo el término cuadrático cruzado de atención.
    """
    qkv_cost = 2.0 * 3.0 * heads * head_dim * head_dim * seq_lat  # proyecciones de queries
    # cross-attention: queries (seq_lat) atendiendo a keys/values (seq_ctx)
    attn_scores = 2.0 * seq_lat * seq_ctx * heads * head_dim
    attn_apply = 2.0 * seq_lat * seq_ctx * heads * head_dim
    out_proj = 2.0 * heads * head_dim * head_dim * seq_lat
    return (qkv_cost + attn_scores + attn_apply + out_proj) * layers


def metrics_from_config(cfg: Config, model: torch.nn.Module) -> dict:
    """Agrega algunas métricas derivadas de la configuración (parámetros, FLOPs)."""
    return {
        "parameter_count": count_parameters(model),
        "latent_count": cfg.compressor.latent_count,
        "compression_ratio": cfg.training.max_input_tokens / max(1, cfg.compressor.latent_count),
    }