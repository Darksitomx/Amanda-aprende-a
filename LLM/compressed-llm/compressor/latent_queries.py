"""Learned latent queries (§5, §6).

Cada query es un vector continuo aprendido ``[latent_dim]``, y el conjunto de
``K`` queries (``[K, latent_dim]``) se usa en la cross-attention del
bottleneck para *leer* información del contexto encoded y producir ``K``
latents continuos.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LearnedLatentQueries(nn.Module):
    """Módulo de queries de latent.

    ``K`` vectores ``[K, latent_dim]`` inicializados aleatoriamente. A los
    efectos del ``state_dict`` se exponen como el parámetro ``query_weights``.
    """

    def __init__(self, latent_count: int, latent_dim: int):
        super().__init__()
        self.latent_count = latent_count
        self.latent_dim = latent_dim
        # Inicialización normal de varianza moderada para evitar colapso inicial (§29)
        self.query_weights = nn.Parameter(torch.empty(latent_count, latent_dim))
        nn.init.normal_(self.query_weights, mean=0.0, std=0.02)

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Devuelve ``[batch_size, K, latent_dim]`` (mismo query replicado por batch)."""
        q = self.query_weights.unsqueeze(0).expand(batch_size, -1, -1)
        return q.to(device)