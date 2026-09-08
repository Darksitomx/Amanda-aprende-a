"""CLI de entrenamiento SFT (§39, §41).

Uso:
    python -m training.train_sft --config configs/v0.yaml \
        --train data/out/train --valid data/out/valid --out results/v0

Flags útiles:
    --baseline      entrena el decoder sin compressor (baseline, §17)
    --overfit N     usa solo N ejemplos (test de overfit, §40)
    --resume PATH   reanuda desde un checkpoint
"""
from __future__ import annotations

import argparse
import os

from datasets import load_from_disk
from torch.utils.data import DataLoader

from compressed_llm.config import Config
from model.compressed_llm import CompressedLLM
from training.dataset import SFTDataset, collate_sft
from training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena CompressedLLM (SFT)")
    parser.add_argument("--config", default="configs/v0.yaml")
    parser.add_argument("--train", default="data/out/train")
    parser.add_argument("--valid", default="data/out/valid")
    parser.add_argument("--out", default="results/v0")
    parser.add_argument("--baseline", action="store_true", help="Baseline sin compressor")
    parser.add_argument("--overfit", type=int, default=None,
                        help="Entrenar solo con las primeras N filas (test §40)")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)

    # Dataset y tokenizer (V0: BPE de entrada (§27))
    from tokenizer.tokenizer import BPETokenizer
    tok = BPETokenizer(vocab_size=cfg.model.vocab_size)
    # En esta fase el tokenizer se asume ya entrenado (ver data/tokenizer). Para
    # arrancar sin tokenizer entrenado se usa un fallback de ids crudos solo en
    # tests; en producción, cargar un tokenizer.json con tok.load(path).

    train_ds = load_from_disk(args.train)
    valid_ds = load_from_disk(args.valid)

    if args.overfit:
        train_ds = train_ds.select(range(min(args.overfit, len(train_ds))))
        valid_ds = train_ds

    records_train = list(train_ds)
    records_valid = list(valid_ds)

    # Si el tokenizer no está entrenado, usamos ids crudos (0,1,2,...) solo para
    # validar shapes; en un entrenamiento real esto debe sustituirse por un BPE.
    if getattr(tok, "_tok", None) is None:
        print("[WARN] Tokenizer sin entrenar: se usarán ids crudos (solo para tests).")
        _install_raw_tokenizer(tok, cfg.model.vocab_size)

    tr_ds = SFTDataset(records_train, tok,
                       max_input_tokens=cfg.training.max_input_tokens,
                       max_output_tokens=cfg.training.max_output_tokens)
    va_ds = SFTDataset(records_valid, tok,
                       max_input_tokens=cfg.training.max_input_tokens,
                       max_output_tokens=cfg.training.max_output_tokens)

    collate = lambda b: collate_sft(b, pad_token_id=cfg.model.pad_token_id,
                                    max_input=cfg.training.max_input_tokens,
                                    max_output=cfg.training.max_output_tokens)
    train_loader = DataLoader(tr_ds, batch_size=cfg.training.batch_size,
                              shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(va_ds, batch_size=cfg.training.batch_size,
                              shuffle=False, collate_fn=collate)

    model = CompressedLLM(cfg, use_compression=not args.baseline)
    print(f"Parámetros del modelo: {model.get_num_parameters():,}")
    print(f"Uso de compresión: {not args.baseline}")

    trainer = Trainer(model, cfg, train_loader, valid_loader,
                      resume_from=args.resume)
    trainer.train(output_dir=args.out)


def _install_raw_tokenizer(tok, vocab_size: int) -> None:
    """Instala un tokenizer "crudo" (char->id) solo para tests de shapes."""
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    special = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    vocab = {**{f"t{i}": i + 10 for i in range(max(0, vocab_size - 10))}, **special}
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