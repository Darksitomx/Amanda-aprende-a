"""Pérdida de preferencia (§9.2).

Pérdida de preferencia relativa tipo Bradley-Terry / logistic:

    L_pref = -log(sigmoid(score(chosen) - score(rejected)))

Requiere puntuaciones por lado (chosen y rejected). La métrica en V0 puede
ser el log-verosimilitud de un token [EOS] o el logit medio; se deja la
genérica para experimentos posteriores.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def preference_loss(chosen_score: torch.Tensor,
                    rejected_score: torch.Tensor) -> torch.Tensor:
    """Scores por lote → scalar.

    Parameters
    ----------
    chosen_score, rejected_score:
        ``[B]`` — puntuaciones escalares para la respuesta elegida y rechazada
        (p. ej. log-prob media del texto).
    """
    diff = chosen_score - rejected_score
    return -F.logsigmoid(diff).mean()


def logprob_mean_score(logits: torch.Tensor, labels: torch.Tensor,
                       ignore_index: int = -100) -> torch.Tensor:
    """Puntuación por ejemplo = log-prob media de los tokens de respuesta.

    logits: [B, L, vocab]; labels: [B, L].
    Returns [B].
    """
    log_probs = F.log_softmax(logits, dim=-1)
    gathered = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
    valid = labels != ignore_index
    count = valid.sum(dim=-1).clamp(min=1).float()
    sums = (gathered * valid).sum(dim=-1)
    return sums / count