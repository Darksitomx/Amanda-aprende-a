"""Compressor: encoder ligero + bottleneck de latents continuos.

Compone ``LightEncoder`` (local) y ``LatentBottleneck`` (cross-attention) para
producir ``K`` latents ``[B, K, D]`` desde ``N`` tokens ``[B, N, D]``.
"""
from __future__ import annotations

import torch.nn as nn

from compressed_llm.config import CompressorConfig
from compressor.bottleneck import LatentBottleneck
from compressor.encoder import LightEncoder


class Compressor(nn.Module):
    """Módulo completo de compresión contextual.

    Parameters
    ----------
    cfg:
        Configuración del compressor.
    in_dim:
        Dimensión del embedding de entrada (vocab embedding).
    """

    def __init__(self, cfg: CompressorConfig, in_dim: int):
        super().__init__()
        self.cfg = cfg
        self.encoder = LightEncoder(
            in_dim=in_dim,
            out_dim=cfg.hidden_size,
            num_layers=cfg.encoder_layers,
            num_heads=cfg.num_heads,
            window=cfg.local_window,
            intermediate_size=cfg.hidden_size * 4,
            dropout=cfg.dropout,
        )
        self.bottleneck = LatentBottleneck(
            latent_count=cfg.latent_count,
            dim=cfg.latent_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.cross_attention_layers,
            dropout=cfg.dropout,
        )

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """hidden: [B, N, D]; mask: [B, N]. Returns latents [B, K, D]."""
        encoded = self.encoder(hidden, mask)
        latents = self.bottleneck(encoded, mask)
        return latents