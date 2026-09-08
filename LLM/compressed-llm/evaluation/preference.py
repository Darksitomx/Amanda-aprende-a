"""Evaluación de preferencia (§9.2, §22).

Compara la pérdida/preferencia entre respondedes elegidas y rechazadas. En V0
se expone la utilidad de puntuación; el entrenamiento con preference loss se
habilita cuando se decida (configuración ``loss.preference_weight>0``).
"""
from __future__ import annotations

import torch


@torch.no_grad()
def compare_chosen_rejected(model, context_ids, context_mask,
                            chosen_ids, rejected_ids) -> dict:
    """Puntúa chosen y rejected con log-prob media y devuelve la preferencia."""
    from evaluation.generation import average_token_logprob

    score_chosen = average_token_logprob(model, context_ids, context_mask,
                                         chosen_ids)
    score_rejected = average_token_logprob(model, context_ids, context_mask,
                                           rejected_ids)
    return {
        "chosen_score": score_chosen,
        "rejected_score": score_rejected,
        "prefers_chosen": score_chosen > score_rejected,
    }