"""Pérdida de información (§9.3) — diseño experimental.

En V0 se usa ``information_weight = 0.0`` (inactiva), tal como recomienda la
especificación: primero demostrar que el bottleneck aprende con
``L_generation`` antes de añadir pérdidas auxiliares.

Este módulo define la interfaz y un stub de loss contrastive que puede
activarse experimentalmente en fases posteriores.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def contrastive_latent_loss(latents: torch.Tensor,
                            anchor_indices: torch.Tensor,
                            positive_indices: torch.Tensor,
                            negative_indices: torch.Tensor,
                            tau: float = 0.07) -> torch.Tensor:
    """Loss contrastiva entre latents (borrador experimental).

    En fases posteriores puede usarse para agrupar latents de contextos
    similares (p. ej. mismo ``question_id``) y separarlos de contextos
    distintos, forzando información útil por latent (§9.3, §24).

    Nota: API de ejemplo, no está conectada  al entrenamiento en V0.
    """
    raise NotImplementedError(
        "Contrastive latent loss aún no implementada. Proceder primero con "
        "L_generation, según §9.3."
    )


def zero_information_loss() -> torch.Tensor:
    """Stub no operativo: devuelve 0. Mantiene la interfaz para el trainer."""
    return torch.tensor(0.0)