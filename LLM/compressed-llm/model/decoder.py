"""Decoder LLM autoregresivo causal (§6, §9.1).

El decoder consume un prefijo de *embeddings* ya ensamblado (típicamente
``[latents continuos ; embeddings de respuesta]``) y es causal sobre la
secuencia completa. La pérdida de generación se aplica únicamente donde el
``label_mask`` lo indique (las posiciones de respuesta, no el prefijo).

Esto permite usar el mismo decoder en dos modos:

- **Compressed**: el prefijo son ``K`` latents continuos.
- **Baseline** (§17): el prefijo son los embeddings de los tokens de contexto.

La salida retorna logits sobre el vocabulario completo.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressed_llm.utils import causal_mask, mask_allow_to_attention


class CausalMultiHeadAttention(nn.Module):
    """Self-attention causal multi-cabeza."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim debe ser divisible por num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """x: [B, S, D]; key_padding_mask: [B, S] (True = keep)."""
        B, S, D = x.shape
        H = self.num_heads
        hd = self.head_dim

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, S, H, hd).transpose(1, 2)
        k = k.view(B, S, H, hd).transpose(1, 2)
        v = v.view(B, S, H, hd).transpose(1, 2)

        scores = (q @ k.transpose(-1, -2)) / (hd ** 0.5)  # [B, H, S, S]

        # máscara causal
        causal = causal_mask(S, x.device)  # [S, S] True=allow
        attn_mask = mask_allow_to_attention(causal).unsqueeze(0).unsqueeze(0)
        scores = scores + attn_mask

        # padding de keys
        if key_padding_mask is not None:
            allow = key_padding_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, S]
            scores = scores + mask_allow_to_attention(allow.expand(B, 1, S, S))

        attn = torch.softmax(scores, dim=-1)
        attn = F.dropout(attn, self.dropout, training=self.training)
        out = attn @ v  # [B, H, S, hd]
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out(out)


class DecoderBlock(nn.Module):
    """Bloque del decoder: pre-norm + causal attention + FFN."""

    def __init__(self, dim: int, num_heads: int, intermediate_size: int,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalMultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, dim),
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + F.dropout(self.attn(self.norm1(x), key_padding_mask),
                          self.dropout, training=self.training)
        x = x + F.dropout(self.ffn(self.norm2(x)),
                          self.dropout, training=self.training)
        return x


class Decoder(nn.Module):
    """Decoder autoregresivo.

    Parameters
    ----------
    hidden_size:
        Dimensión de los embeddings de entrada (y de los latents).
    num_layers:
        Número de ``DecoderBlock``.
    num_heads:
        Cabezas de atención.
    intermediate_size:
        Dimensión oculta del MLP.
    vocab_size:
        Tamaño del vocabulario de salida.
    dropout:
        Tasa de dropout.
    decoder_input_dim:
        Dimensión real del prefijo recibido (los latents). Útil si el
        compressor usa ``latent_dim`` distinto de ``hidden_size``.
    """

    def __init__(self, hidden_size: int, num_layers: int, num_heads: int,
                 intermediate_size: int, vocab_size: int,
                 dropout: float = 0.0,
                 decoder_input_dim: int = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        in_dim = decoder_input_dim or hidden_size
        self.input_proj = nn.Linear(in_dim, hidden_size) if in_dim != hidden_size else nn.Identity()
        self.blocks = nn.ModuleList([
            DecoderBlock(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def tie_input_embeddings(self, weight: nn.Parameter) -> None:
        """Enlaza ``lm_head`` con el embedding de entrada (weight tying)."""
        if (weight.shape == self.lm_head.weight.shape
                and isinstance(self.input_proj, nn.Identity)):
            self.lm_head.weight = weight

    def forward(self, hidden: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """hidden: [B, S, decoder_input_dim]. Returns logits [B, S, vocab_size]."""
        x = self.input_proj(hidden)
        for block in self.blocks:
            x = block(x, key_padding_mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits