"""Bottleneck de latents continuos (§5, §6, §9.3).

Toma la secuencia encoded ``[B, N, D]`` y usa ``K`` learned latent queries en
cross-attention para producir ``[B, K, D]``. Con ``K << N``, el coste de esta
etapa se aproxima a O(N*K) en lugar del O(N^2) de un self-attention global.

Se incluye un ``norm`` final y un ``mask`` opcional sobre los latents. La
salida es un *prefijo continuo* que el decoder consumirá.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressed_llm.utils import mask_allow_to_attention
from compressor.latent_queries import LearnedLatentQueries


class CrossAttentionBlock(nn.Module):
    """Bloque de cross-attention: latent queries (Q) sobre contexto encoded (K/V)."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim debe ser divisible por num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, queries: torch.Tensor, ctx: torch.Tensor,
                ctx_mask: torch.Tensor) -> torch.Tensor:
        # queries: [B, K, D]; ctx: [B, N, D]; ctx_mask: [B, N] (1 = keep)
        B, K, D = queries.shape
        H = self.num_heads
        hd = self.head_dim

        q = self.norm_q(queries)
        k = self.norm_kv(ctx)
        v = ctx

        q = self.q(q).view(B, K, H, hd).transpose(1, 2)          # [B, H, K, hd]
        k = self.k(k).view(B, k.size(1), H, hd).transpose(1, 2)  # [B, H, N, hd]
        v = self.v(v).view(B, v.size(1), H, hd).transpose(1, 2)  # [B, H, N, hd]

        scores = (q @ k.transpose(-1, -2)) / (hd ** 0.5)          # [B, H, K, N]
        # enmascarar padding del contexto
        mask = ctx_mask.unsqueeze(1).unsqueeze(1)                 # [B, 1, 1, N]
        attn_mask = mask_allow_to_attention(mask.expand(B, 1, K, ctx.size(1)))
        scores = scores + attn_mask
        attn = torch.softmax(scores, dim=-1)
        attn = F.dropout(attn, self.dropout, training=self.training)

        out = attn @ v                                            # [B, H, K, hd]
        out = out.transpose(1, 2).contiguous().view(B, K, D)
        out = self.out(out)
        return self.norm_out(queries + out)                       # residual + pre-norm


class LatentBottleneck(nn.Module):
    """Compone learned queries + uno o más bloques de cross-attention."""

    def __init__(self, latent_count: int, dim: int, num_heads: int,
                 num_layers: int = 2, dropout: float = 0.0, epsilon: float = 1e-5):
        super().__init__()
        self.queries = LearnedLatentQueries(latent_count, dim)
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.latent_norm = nn.LayerNorm(dim, eps=epsilon)

    def forward(self, ctx: torch.Tensor, ctx_mask: torch.Tensor) -> torch.Tensor:
        """ctx: [B, N, D]; ctx_mask: [B, N]. Returns [B, K, D]."""
        B = ctx.size(0)
        q = self.queries.forward(B, ctx.device)
        x = q
        for block in self.blocks:
            x = block(x, ctx, ctx_mask)
        x = self.latent_norm(x)
        return x