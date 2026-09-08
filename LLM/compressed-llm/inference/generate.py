"""Generación autoregresiva (§6, §22, §42).

Genera tokens uno a uno condicionando en el prefijo continuo (o contexto
baseline). En V0 el prefijo se re-ensambla completo en cada paso (inferencia
ingenua sin cache de KV), suficiente para la prueba de concepto.
"""
from __future__ import annotations

from typing import List, Optional

import torch

from compressed_llm.config import Config
from model.compressed_llm import CompressedLLM


@torch.no_grad()
def generate(
    model: CompressedLLM,
    context_ids: torch.Tensor,
    context_mask: torch.Tensor,
    max_new_tokens: int = 128,
    bos_token_id: int = 1,
    eos_token_id: int = 2,
    temperature: float = 1.0,
    device: torch.device = None,
) -> torch.Tensor:
    """Genera ``max_new_tokens`` tokens. Devuelve ids ``[B, gen_len]``.

    Parameters
    ----------
    model:
        Modelo CompressedLLM (en modo compressed o baseline).
    context_ids:
        ``[B, N]`` tokens del contexto.
    context_mask:
        ``[B, N]`` máscara válida.
    """
    if device is None:
        device = next(model.parameters()).device

    B = context_ids.size(0)
    generated = torch.full((B, 1), bos_token_id, dtype=torch.long,
                           device=context_ids.device)
    finished = torch.zeros(B, dtype=torch.bool, device=context_ids.device)

    for _ in range(max_new_tokens):
        out = model.forward(context_ids, context_mask, output_ids=generated,
                            labels=None)
        logits = out["logits"]
        # posición previa al último token emitido = última de la respuesta
        last = logits[:, -1, :] / temperature
        probs = torch.softmax(last, dim=-1)
        next_tok = torch.multinomial(probs, 1)  # [B, 1]

        generated = torch.cat([generated, next_tok], dim=1)
        finished = finished | (next_tok.squeeze(-1) == eos_token_id)
        if finished.all():
            break

    # quitar el BOS inicial
    return generated[:, 1:]


@torch.no_grad()
def generate_texts(model: CompressedLLM,
                   context_ids: torch.Tensor,
                   context_mask: torch.Tensor,
                   tokenizer,
                   max_new_tokens: int = 128,
                   bos_token_id: int = 1,
                   eos_token_id: int = 2,
                   temperature: float = 1.0) -> List[str]:
    ids = generate(model, context_ids, context_mask, max_new_tokens,
                   bos_token_id, eos_token_id, temperature)
    return [tokenizer.decode(row.tolist()) for row in ids]