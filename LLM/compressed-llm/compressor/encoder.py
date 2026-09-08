"""Encoder ligero y local (§5, §6).

Objetivo: producir ``[B, N, D]`` desde embeddings de tokens sin pagar un
self-attention global O(N^2). En su lugar se usa atención local por ventana
(blockwise) y una pequeña red feed-forward, lo que mantiene la etapa previa
al bottleneck barata y comparable en coste al “N*K” de la cross-attention.

Se permite atención global opcional para experimentos en los que queramos
comparar con un encoder sin localidad.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressed_llm.utils import mask_allow_to_attention


class BlockwiseWindowAttention(nn.Module):
    """Self-attention por bloques contiguos de longitud ``window``.

    Cada bloque se añade en la secuencia como dimensión de lote, de modo que
    la atención opera localmente sin acceso a posiciones de otros bloques.
    """

    def __init__(self, dim: int, num_heads: int, window: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim debe ser divisible por num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window = window
        self.dropout = dropout
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        B, N, D = x.shape
        window = self.window
        pad_len = (window - N % window) % window
        if pad_len:
            x = F.pad(x, (0, 0, 0, pad_len))  # pad en la dimensión de secuencia
        B_, N_, D_ = x.shape
        num_blocks = N_ // window

        # -> [B, num_blocks, window, D] -> [B*num_blocks, window, D]
        xb = x.view(B_, num_blocks, window, D_).reshape(B_ * num_blocks, window, D_)

        qkv = self.qkv(xb)  # [B*b, window, 3*D]
        q, k, v = qkv.chunk(3, dim=-1)
        H = self.num_heads
        hd = self.head_dim
        q = q.view(B_ * num_blocks, window, H, hd).transpose(1, 2)  # [., H, window, hd]
        k = k.view(B_ * num_blocks, window, H, hd).transpose(1, 2)
        v = v.view(B_ * num_blocks, window, H, hd).transpose(1, 2)

        scores = (q @ k.transpose(-1, -2)) / (hd ** 0.5)  # causal-less block
        # Bloque causal: máscara triangular dentro de cada bloque
        causal = torch.tril(
            torch.ones(window, window, device=x.device, dtype=torch.bool)
        )
        attn_mask = mask_allow_to_attention(causal)
        scores = scores + attn_mask
        attn = torch.softmax(scores, dim=-1)
        attn = F.dropout(attn, self.dropout, training=self.training)

        out = attn @ v  # [., H, window, hd]
        out = out.transpose(1, 2).contiguous().view(B_ * num_blocks, window, D_)
        out = self.out(out).reshape(B_, num_blocks, window, D_).view(B_, N_, D_)

        if pad_len:
            out = out[:, :N, :]
        return out


class EncoderBlock(nn.Module):
    """Bloque de encoder: Atención local + FFN con pre-norm."""

    def __init__(self, dim: int, num_heads: int, window: int,
                 intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = BlockwiseWindowAttention(dim, num_heads, window, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, dim),
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + F.dropout(self.attn(self.norm1(x)), self.dropout, training=self.training)
        x = x + F.dropout(self.ffn(self.norm2(x)), self.dropout, training=self.training)
        return x


class LightEncoder(nn.Module):
    """Encoder ligero: input ``[B, N, D_in]`` -> output ``[B, N, D_out]``.

    Compone una proyección de entrada, varias ``EncoderBlock`` locales y una
    proyección de salida que puede redimensionar ``D_in`` a ``D_out``
    (útil si el embedding usa una dim distinta del latent/diff).
    """

    def __init__(self, in_dim: int, out_dim: int, num_layers: int,
                 num_heads: int, window: int, intermediate_size: int,
                 dropout: float = 0.0):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.blocks = nn.ModuleList([
            EncoderBlock(out_dim, num_heads, window, intermediate_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D_in]; mask: [B, N] booleano (1 = real/keep)
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        # Enmascarar posiciones de padding tras el encoder
        x = x * mask.unsqueeze(-1).to(x.dtype)
        return x