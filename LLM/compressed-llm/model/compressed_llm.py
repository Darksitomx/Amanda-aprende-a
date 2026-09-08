"""Modelo CompressedLLM completo (§4, §6).

Orquesta:
    tokens de contexto -> embeddings
        -> [compressor -> K latents continuos]  (modo comprimido)
        -> [embeddings de respuesta]            (modo baseline directo, §17)
        -> decoder causal -> logits

La pérdida de generación se aplica solo sobre las posiciones de *respuesta*
(los tokens objetivo), nunca sobre el prefijo de contexto/latents (§9.1).
"""
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
    """Modelo completo.

    Parametros
    ----------
    cfg:
        Configuración (modelo + compressor).
    use_compression:
        Si ``False`` (baseline, §17) el decoder recibe directamente los
        embeddings del contexto tokenizado en lugar de los latents.
    """

    def __init__(self, cfg: Config, use_compression: bool = True):
        super().__init__()
        self.cfg = cfg
        self.use_compression = use_compression

        m = cfg.model
        c = cfg.compressor

        self.tok_emb = nn.Embedding(m.vocab_size, m.hidden_size,
                                    padding_idx=m.pad_token_id)

        self.compressor = Compressor(c, in_dim=m.hidden_size) if use_compression else None

        self.decoder = Decoder(
            hidden_size=m.hidden_size,
            num_layers=m.num_layers,
            num_heads=m.num_heads,
            intermediate_size=m.intermediate_size,
            vocab_size=m.vocab_size,
            dropout=m.dropout,
            decoder_input_dim=(c.latent_dim if use_compression else None),
        )

        # weight tying entre lm_head y embedding de entrada de tokens
        self.decoder.tie_input_embeddings(self.tok_emb.weight)

    # ------------------------------------------------------------------ #
    # API de alto nivel                                                   #
    # ------------------------------------------------------------------ #
    def compress_context(self, input_ids: torch.Tensor,
                         attention_mask: torch.Tensor) -> torch.Tensor:
        """Devuelve los ``K`` latents ``[B, K, latent_dim]``."""
        hidden = self.tok_emb(input_ids)
        return self.compressor.forward(hidden, attention_mask)

    def embed_output(self, output_ids: torch.Tensor) -> torch.Tensor:
        """Devuelve los embeddings de los tokens de respuesta ``[B, L, D]``."""
        return self.tok_emb(output_ids)

    # ------------------------------------------------------------------ #
    # Forward de entrenamiento                                            #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        output_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """Ensambla el prefijo (latents o contexto) + respuesta y pasa por el decoder.

        Parameters
        ----------
        context_ids:
            ``[B, N]`` tokens del contexto (ya tokenizado/padded a N).
        context_mask:
            ``[B, N]`` booleano ``True`` para posiciones reales.
        output_ids:
            ``[B, L]`` tokens de la respuesta objetivo (incluye BOS si se desea).
        labels:
            ``[B, L]`` — opcional. Si se da, se computa la pérdida.

        Returns
        -------
            dict con ``logits`` y opcionalmente ``loss``.
        """
        B = context_ids.size(0)

        if self.use_compression:
            prefix = self.compress_context(context_ids, context_mask)  # [B, K, latent_dim]
            K = prefix.size(1)
            prefix_mask = torch.ones(B, K, dtype=torch.bool, device=context_ids.device)
        else:
            prefix = self.tok_emb(context_ids)  # [B, N, hidden]
            prefix_mask = context_mask

        if output_ids is not None:
            out_hidden = self.tok_emb(output_ids)  # [B, L, hidden]
            seq = torch.cat([prefix, out_hidden], dim=1)
            out_mask = torch.ones(B, out_hidden.size(1), dtype=torch.bool,
                                  device=context_ids.device)
            seq_mask = torch.cat([prefix_mask, out_mask], dim=1)
        else:
            seq = prefix
            seq_mask = prefix_mask

        logits = self.decoder(seq, seq_mask)  # [B, S, vocab]

        result = {"logits": logits}
        if labels is not None:
            L = labels.size(1)
            # pérdida solo sobre los tokens de respuesta (shift de 1 sobre respuesta)
            resp_logits = logits[:, prefix.size(1) - 1: prefix.size(1) - 1 + L, :]
            loss = F.cross_entropy(
                resp_logits.reshape(-1, resp_logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        return result

    def get_num_parameters(self) -> int:
        return count_parameters(self)

    def save_pretrained_like(self, path: str, optimizer=None, step: int = 0):
        """Guarda modelo (+ optimizer opcional) en formato checkpoint (§35)."""
        import os
        os.makedirs(path, exist_ok=True)
        payload = {
            "model_state": self.state_dict(),
            "step": step,
            "config": self.cfg.to_dict(),
            "use_compression": self.use_compression,
        }
        if optimizer is not None:
            payload["optimizer_state"] = optimizer.state_dict()
        torch.save(payload, os.path.join(path, "checkpoint.pt"))
        self.cfg.save(os.path.join(path, "config.json"))

    @staticmethod
    def load_pretrained_like(path: str, cfg: Config, use_compression: bool = True):
        import os
        model = CompressedLLM(cfg, use_compression=use_compression)
        payload = torch.load(os.path.join(path, "checkpoint.pt"),
                             map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"])
        return model, payload.get("optimizer_state"), payload.get("step", 0)