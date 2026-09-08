"""Modelo CompressedLLM completo."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressed_llm.config import Config
from compressed_llm.utils import count_parameters
from compressor import Compressor
from model.decoder import Decoder


class CompressedLLM(nn.Module):
    """LLM desde cero con prefijo continuo comprimido o baseline directo."""

    def __init__(self, cfg: Config, use_compression: bool = True):
        super().__init__()
        self.cfg = cfg
        self.use_compression = use_compression
        m, c = cfg.model, cfg.compressor
        self.tok_emb = nn.Embedding(m.vocab_size, m.hidden_size, padding_idx=m.pad_token_id)
        self.compressor = Compressor(c, in_dim=m.hidden_size) if use_compression else None
        self.decoder = Decoder(
            hidden_size=m.hidden_size, num_layers=m.num_layers, num_heads=m.num_heads,
            intermediate_size=m.intermediate_size, vocab_size=m.vocab_size,
            dropout=m.dropout, decoder_input_dim=(c.latent_dim if use_compression else None),
            max_position_embeddings=m.max_position_embeddings,
        )
        self.decoder.tie_input_embeddings(self.tok_emb.weight)

    def compress_context(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.compressor is None:
            raise RuntimeError("compress_context requiere use_compression=True")
        return self.compressor(self.tok_emb(input_ids), attention_mask)

    def embed_output(self, output_ids: torch.Tensor) -> torch.Tensor:
        return self.tok_emb(output_ids)

    def forward(self, context_ids: torch.Tensor, context_mask: torch.Tensor,
                output_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> dict:
        B = context_ids.size(0)
        if self.use_compression:
            prefix = self.compress_context(context_ids, context_mask)
            prefix_mask = torch.ones(B, prefix.size(1), dtype=torch.bool, device=context_ids.device)
        else:
            prefix = self.tok_emb(context_ids)
            prefix_mask = context_mask

        prefix_len = prefix.size(1)
        if output_ids is None:
            seq, seq_mask = prefix, prefix_mask
        else:
            if labels is not None and output_ids.size(1) != labels.size(1):
                raise ValueError("output_ids y labels deben tener la misma longitud.")
            seq = torch.cat([prefix, self.tok_emb(output_ids)], dim=1)
            out_mask = torch.ones(B, output_ids.size(1), dtype=torch.bool, device=context_ids.device)
            seq_mask = torch.cat([prefix_mask, out_mask], dim=1)

        logits = self.decoder(seq, seq_mask)
        result = {"logits": logits}
        if labels is not None:
            # Causal LM:
            # output_ids = [BOS, y0, y1, ...]
            # labels     = [y0,  y1, y2, ... EOS]
            # El logit en prefix_len + i ve el token de output_ids[i] y
            # predice labels[i]. La posición prefix_len - 1 pertenece al
            # último latent/context token y no debe usarse para la respuesta.
            L = labels.size(1)
            resp_logits = logits[:, prefix_len:prefix_len + L, :]
            if resp_logits.size(1) != L:
                raise RuntimeError(f"Alineación inválida: {resp_logits.shape} vs {labels.shape}")
            result["loss"] = F.cross_entropy(
                resp_logits.reshape(-1, resp_logits.size(-1)), labels.reshape(-1), ignore_index=-100
            )
        return result

    def get_num_parameters(self) -> int:
        return count_parameters(self)

    def save_pretrained_like(self, path: str, optimizer=None, step: int = 0):
        import os
        os.makedirs(path, exist_ok=True)
        payload = {"model_state": self.state_dict(), "step": step,
                   "config": self.cfg.to_dict(), "use_compression": self.use_compression}
        if optimizer is not None:
            payload["optimizer_state"] = optimizer.state_dict()
        torch.save(payload, os.path.join(path, "checkpoint.pt"))
        self.cfg.save(os.path.join(path, "config.yaml"))

    @staticmethod
    def load_pretrained_like(path: str, cfg: Config, use_compression: bool = True):
        import os
        model = CompressedLLM(cfg, use_compression=use_compression)
        payload = torch.load(os.path.join(path, "checkpoint.pt"), map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"])
        return model, payload.get("optimizer_state"), payload.get("step", 0)
