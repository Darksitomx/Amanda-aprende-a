"""Decoder LLM autoregresivo causal."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressed_llm.utils import causal_mask, mask_allow_to_attention


class CausalMultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim debe ser divisible por num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        B, S, D = x.shape
        H, hd = self.num_heads, self.head_dim
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, S, H, hd).transpose(1, 2)
        k = k.view(B, S, H, hd).transpose(1, 2)
        v = v.view(B, S, H, hd).transpose(1, 2)

        scores = (q @ k.transpose(-1, -2)) / (hd ** 0.5)
        causal = causal_mask(S, x.device)
        scores = scores + mask_allow_to_attention(causal).unsqueeze(0).unsqueeze(0)

        if key_padding_mask is not None:
            allow = key_padding_mask.unsqueeze(1).unsqueeze(1)
            scores = scores + mask_allow_to_attention(allow.expand(B, 1, S, S))

        attn = torch.softmax(scores, dim=-1)
        attn = F.dropout(attn, self.dropout, training=self.training)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out(out)


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalMultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, intermediate_size), nn.GELU(), nn.Linear(intermediate_size, dim)
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + F.dropout(self.attn(self.norm1(x), key_padding_mask), self.dropout, training=self.training)
        x = x + F.dropout(self.ffn(self.norm2(x)), self.dropout, training=self.training)
        return x


class Decoder(nn.Module):
    """Decoder causal con posiciones aprendidas."""

    def __init__(self, hidden_size: int, num_layers: int, num_heads: int,
                 intermediate_size: int, vocab_size: int,
                 dropout: float = 0.0, decoder_input_dim: int = None,
                 max_position_embeddings: int = 4096):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        in_dim = decoder_input_dim or hidden_size
        self.input_proj = nn.Linear(in_dim, hidden_size) if in_dim != hidden_size else nn.Identity()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.blocks = nn.ModuleList([
            DecoderBlock(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def tie_input_embeddings(self, weight: nn.Parameter) -> None:
        if weight.shape == self.lm_head.weight.shape and isinstance(self.input_proj, nn.Identity):
            self.lm_head.weight = weight

    def forward(self, hidden: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        B, S, _ = hidden.shape
        if S > self.max_position_embeddings:
            raise ValueError(
                f"Secuencia de decoder ({S}) supera max_position_embeddings ({self.max_position_embeddings})."
            )
        x = self.input_proj(hidden)
        positions = torch.arange(S, device=hidden.device).unsqueeze(0)
        x = x + self.position_embeddings(positions)
        for block in self.blocks:
            x = block(x, key_padding_mask)
        return self.lm_head(self.norm(x))
