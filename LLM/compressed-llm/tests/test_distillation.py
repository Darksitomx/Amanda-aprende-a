"""Tests del camino de destilación (teacher_mode) añadidos por la etapa V0.

Cubren:
1. teacher_mode activo produce full_context_loss y distillation_loss;
2. la loss combinada es la suma ponderada (pesos > 0);
3. sin pesos (0.0) no se añade ninguna auxiliar;
4. la loss del estudiante con labels desplazadas es ~ln(V), nunca ~0
   (protege contra el copy-cheat de labels=output_ids / fuga causal);
5. el KL respeta el enmascarado de padding (-100) sin NaN.

Usa una config mínima para mantener los tests rápidos.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from compressed_llm.config import Config, ModelConfig, CompressorConfig  # noqa: E402
from model.compressed_llm import CompressedLLM  # noqa: E402


def make_cfg(latent_count: int = 8, hidden: int = 32, vocab: int = 64) -> Config:
    cfg = Config()
    cfg.model = ModelConfig(vocab_size=vocab, hidden_size=hidden,
                            num_layers=2, num_heads=4, intermediate_size=64,
                            max_position_embeddings=128)
    cfg.compressor = CompressorConfig(
        encoder_layers=1, hidden_size=hidden, num_heads=4,
        latent_count=latent_count, latent_dim=hidden,
        cross_attention_layers=1, local_window=8,
    )
    return cfg


@pytest.fixture
def model():
    torch.manual_seed(0)
    cfg = make_cfg(latent_count=8)
    m = CompressedLLM(cfg, use_compression=True)
    m.eval()
    return m


def autoregressive_batch(model, B=2, N=24, L=12):
    """Batch con labels desplazadas (predecir el siguiente token), como SFTDataset."""
    ctx = torch.randint(0, model.cfg.model.vocab_size, (B, N))
    cmask = torch.ones(B, N, dtype=torch.bool)
    out = torch.randint(0, model.cfg.model.vocab_size, (B, L))
    labels = torch.full_like(out, -100)
    labels[:, :-1] = out[:, 1:]  # labels[i] = out[i+1]
    return ctx, cmask, out, labels


def test_teacher_mode_adds_loss_terms(model):
    ctx, cmask, out, labels = autoregressive_batch(model)
    res = model(ctx, cmask, output_ids=out, labels=labels, teacher_mode=True,
                distillation_weight=0.3, full_context_weight=0.1,
                distillation_temperature=2.0)
    assert "full_context_loss" in res
    assert "distillation_loss" in res
    assert torch.isfinite(res["loss"])
    assert torch.isfinite(res["full_context_loss"])
    assert torch.isfinite(res["distillation_loss"])
    assert res["distillation_loss"].item() >= 0.0
    # La combinada = CE + 0.1*CE_teacher + 0.3*KL > solo CE del estudiante.
    assert res["loss"].item() > res["student_ce"].item()


def test_no_auxiliary_losses_with_zero_weights(model):
    ctx, cmask, out, labels = autoregressive_batch(model)
    res = model(ctx, cmask, output_ids=out, labels=labels, teacher_mode=True,
                distillation_weight=0.0, full_context_weight=0.0)
    assert "full_context_loss" not in res
    assert "distillation_loss" not in res
    # Sin auxiliares, loss == student_ce.
    assert abs(res["loss"].item() - res["student_ce"].item()) < 1e-6


def test_teacher_off_produces_plain_student(model):
    ctx, cmask, out, labels = autoregressive_batch(model)
    res = model(ctx, cmask, output_ids=out, labels=labels)
    assert "full_context_loss" not in res
    assert "distillation_loss" not in res
    assert res["loss"].item() == res["student_ce"].item()


def test_student_loss_is_autoregressive_not_copy(model):
    """labels desplazadas: la CE de un modelo aleatorio debe estar clara de 0.

    Protege contra el copy-cheat de usar labels == output_ids (CE≈0 porque el
    modelo "copia" el token que ve en su propia posición vía el tying).
    """
    ctx, cmask, out, labels = autoregressive_batch(model, L=16)
    res = model(ctx, cmask, output_ids=out, labels=labels)
    loss = res["loss"].item()
    assert loss > 2.0, f"CE={loss} sospechosamente baja (posible copy-cheat)"


def test_distillation_ignores_padding_labels(model):
    ctx, cmask, out, labels = autoregressive_batch(model, L=16)
    labels[:, -6:] = -100  # última columna de respuestas = padding
    res = model(ctx, cmask, output_ids=out, labels=labels, teacher_mode=True,
                distillation_weight=0.3, full_context_weight=0.1,
                distillation_temperature=2.0)
    assert torch.isfinite(res["loss"])
    assert torch.isfinite(res["distillation_loss"])
    assert torch.isfinite(res["full_context_loss"])