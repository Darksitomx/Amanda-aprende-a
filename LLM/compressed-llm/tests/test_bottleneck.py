"""Tests obligatorios antes de entrenar (§38).

Cubren:
1. input [B, N] llega al encoder;
2. encoder -> [B, N, D];
3. bottleneck -> [B, K, D];
4. decoder acepta [B, K, D];
5. output con shape correcta;
6. gradientes llegan a encoder y a los latent queries;
7. cambiar K cambia las dimensiones;
8. batch padding/máscaras funcionan;
9. máscara causal del decoder funciona;
10. checkpoint save/load conserva pesos.

Se usa una config mínima para mantener el test rápido.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

# Añadir el raíz del proyecto al path para importar los paquetes
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


def test_1_input_reaches_encoder(model):
    B, N = 2, 24
    ids = torch.randint(0, model.cfg.model.vocab_size, (B, N))
    mask = torch.ones(B, N, dtype=torch.bool)
    hidden = model.tok_emb(ids)
    enc = model.compressor.encoder(hidden, mask)
    assert enc.shape == (B, N, model.cfg.model.hidden_size)


def test_2_encoder_shape(model):
    B, N = 2, 24
    ids = torch.randint(0, model.cfg.model.vocab_size, (B, N))
    mask = torch.ones(B, N, dtype=torch.bool)
    enc = model.compressor.encoder(model.tok_emb(ids), mask)
    assert enc.ndim == 3
    assert enc.shape[0] == B and enc.shape[1] == N


def test_3_bottleneck_shape(model):
    B, N = 2, 24
    ids = torch.randint(0, model.cfg.model.vocab_size, (B, N))
    mask = torch.ones(B, N, dtype=torch.bool)
    hidden = model.tok_emb(ids)
    lat = model.compressor.bottleneck(model.compressor.encoder(hidden, mask), mask)
    assert lat.shape == (B, model.cfg.compressor.latent_count,
                         model.cfg.compressor.latent_dim)


def test_4_decoder_accepts_latents(model):
    B, N = 2, 24
    ids = torch.randint(0, model.cfg.model.vocab_size, (B, N))
    mask = torch.ones(B, N, dtype=torch.bool)
    lat = model.compress_context(ids, mask)
    logits = model.decoder(lat, torch.ones(B, lat.size(1), dtype=torch.bool))
    assert logits.shape[1] == lat.size(1)


def test_5_output_shape(model):
    B, N, L = 2, 24, 16
    ctx = torch.randint(0, model.cfg.model.vocab_size, (B, N))
    cmask = torch.ones(B, N, dtype=torch.bool)
    out = torch.randint(0, model.cfg.model.vocab_size, (B, L))
    labels = torch.randint(0, model.cfg.model.vocab_size, (B, L))
    res = model.forward(ctx, cmask, output_ids=out, labels=labels)
    prefix_len = model.cfg.compressor.latent_count
    assert res["logits"].shape == (B, prefix_len + L, model.cfg.model.vocab_size)
    assert res["loss"] is not None and res["loss"].ndim == 0


def test_6_gradients_reach_encoder_and_queries(model):
    model.train()
    B, N, L = 2, 24, 16
    ctx = torch.randint(0, model.cfg.model.vocab_size, (B, N))
    cmask = torch.ones(B, N, dtype=torch.bool)
    out = torch.randint(0, model.cfg.model.vocab_size, (B, L))
    labels = out  # para simplificar el grad de CE
    model.zero_grad()
    res = model.forward(ctx, cmask, output_ids=out, labels=labels)
    res["loss"].backward()

    enc_param = next(model.compressor.encoder.parameters())
    q_param = model.compressor.bottleneck.queries.query_weights
    assert enc_param.grad is not None
    assert q_param.grad is not None
    assert torch.abs(q_param.grad).sum() > 0


def test_7_K_changes_dimensions():
    for K in [4, 8, 16]:
        cfg = make_cfg(latent_count=K)
        m = CompressedLLM(cfg, use_compression=True)
        B, N = 1, 24
        ids = torch.randint(0, cfg.model.vocab_size, (B, N))
        mask = torch.ones(B, N, dtype=torch.bool)
        lat = m.compress_context(ids, mask)
        assert lat.shape == (B, K, cfg.compressor.latent_dim)


def test_8_padding_masks_work(model):
    B, N, L = 2, 24, 16
    ctx = torch.randint(0, model.cfg.model.vocab_size, (B, N))
    cmask = torch.ones(B, N, dtype=torch.bool)
    cmask[1, :8] = False  # mitad del contexto 2 es padding
    out = torch.randint(0, model.cfg.model.vocab_size, (B, L))
    labels = out
    res = model.forward(ctx, cmask, output_ids=out, labels=labels)
    assert torch.isfinite(res["loss"])


def test_9_causal_mask_works(model):
    S = 16
    attn = model.decoder.blocks[0].attn
    x = torch.randn(1, S, attn.dim)
    out = attn(x)
    assert out.shape == x.shape
    # el último token solo puede atender a sí mismo -> prueba unitaria adicional:
    # comprobamos que la máscara es triangular
    from compressed_llm.utils import causal_mask
    cm = causal_mask(S)
    assert (cm == torch.tril(torch.ones(S, S, dtype=torch.bool))).all()


def test_10_checkpoint_save_load(tmp_path):
    cfg = make_cfg(latent_count=8)
    m1 = CompressedLLM(cfg, use_compression=True)
    m2 = CompressedLLM(cfg, use_compression=True)
    state1 = {k: v.clone() for k, v in m1.state_dict().items()}

    m1.save_pretrained_like(str(tmp_path), step=5)
    loaded, _opt, step = CompressedLLM.load_pretrained_like(str(tmp_path), cfg,
                                                            use_compression=True)
    assert step == 5
    for k in state1:
        assert torch.equal(state1[k], loaded.state_dict()[k])