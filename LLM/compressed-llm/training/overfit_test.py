"""Overfit test (§40).

Entrena el modelo sobre un conjunto diminuto (por defecto 32 ejemplos) y
verifica que la pérdida baja de forma sustancial. Si NO sobreajusta, no seguir
al entrenamiento grande (posibles bugs de máscaras, labels, latents
desconectados, gradientes nulos, loss mal construida o datos mal preparados).

Uso:
    python -m training.overfit_test --config configs/v0.yaml
    (usa datos sintéticos si no se proveen rutas reales)
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

from compressed_llm.config import Config


def build_synthetic_records(n: int, vocab: int = 256) -> list:
    """Genera registros sintéticos con contexto y respuesta para el overfit.

    Los tokens usan la forma ``t{idx}`` que existe en el vocabulario instalado
    por ``_install_raw`` (ids continuos), de modo que cada ejemplo es
    distinguible y el modelo puede memorizar la correspondencia contexto→respuesta.
    """
    max_tok = max(10, min(vocab - 11, 1000))  # rango de tokens 't' disponibles
    recs = []
    for i in range(n):
        length = 24
        ctx = " ".join(f"t{ (i * length + j) % max_tok }" for j in range(length))
        # respuesta distinguishable y consistente por ejemplo
        ans = " ".join(f"t{(i * 7 + k) % max_tok}" for k in range(16))
        recs.append({"context": [{"role": "user", "content": ctx}],
                     "chosen": ans, "question_id": str(i)})
    return recs


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit test (§40)")
    parser.add_argument("--config", default="configs/v0.yaml")
    parser.add_argument("--examples", type=int, default=32)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--train", default=None,
                        help="Opcional: ruta de dataset real preprocesado.")
    parser.add_argument("--raw-tokenizer", action="store_true",
                        help="Usar tokenizer de ids crudos (solo tests).")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    os.makedirs(cfg.training.output_dir, exist_ok=True)

    from tokenizer.tokenizer import BPETokenizer
    tok = BPETokenizer(vocab_size=cfg.model.vocab_size)

    # Usar tokenizer de ids crudos para el overfit rápido (§ tests)
    if args.raw_tokenizer or not hasattr(tok, "_tok") or tok._tok is None:
        _install_raw(tok, cfg.model.vocab_size)

    from training.dataset import SFTDataset, collate_sft
    from training.trainer import Trainer
    from model.compressed_llm import CompressedLLM

    if args.train:
        from datasets import load_from_disk
        raw = load_from_disk(args.train)
        records = list(raw.select(range(min(args.examples, len(raw)))))
    else:
        records = build_synthetic_records(args.examples, cfg.model.vocab_size)

    tr = SFTDataset(records, tok,
                    max_input_tokens=cfg.training.max_input_tokens,
                    max_output_tokens=cfg.training.max_output_tokens)

    collate = lambda b: collate_sft(b, pad_token_id=cfg.model.pad_token_id,
                                    max_input=cfg.training.max_input_tokens,
                                    max_output=cfg.training.max_output_tokens)
    loader = DataLoader(tr, batch_size=cfg.training.batch_size,
                        shuffle=True, collate_fn=collate)

    model = CompressedLLM(cfg, use_compression=True)
    print(f"Parámetros: {model.get_num_parameters():,}")
    trainer = Trainer(model, cfg, loader, None)

    # ejecutamos unos pasos y comprobamos la bajada de loss
    start_loss = None
    for step in range(1, args.steps + 1):
        batch = next(iter(loader))
        batch = {k: v.to(trainer.device) for k, v in batch.items()}
        trainer.optimizer.zero_grad()
        out = model(**batch)
        loss = out["loss"]
        loss.backward()
        trainer.optimizer.step()
        if start_loss is None:
            start_loss = loss.item()
        if step % 50 == 0 or step == args.steps:
            print(f"step {step}: loss={loss.item():.4f}")

    print(f"\nInicio: {start_loss:.4f} -> Final: {loss.item():.4f}")
    if loss.item() < start_loss * 0.3:
        print("OK: el modelo sobreajusta (overfit test superado).")
    else:
        print("WARN: la pérdida no baja lo suficiente. Revisar bugs antes "
              "de entrenar a gran escala (§40).")


def _install_raw(tok, vocab_size: int) -> None:
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    special = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "</s>": 4, "<s>": 5}
    vocab = {**{f"t{i}": i + 10 for i in range(vocab_size - 10)}, **special}
    t = HFTokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    t.pre_tokenizer = Whitespace()
    t.enable_padding(pad_id=0, pad_token="<pad>")
    t.enable_truncation(max_length=1024)
    tok._tok = t
    tok.pad_token_id = 0
    tok.bos_token_id = 1
    tok.eos_token_id = 2


if __name__ == "__main__":
    main()