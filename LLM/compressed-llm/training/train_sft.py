"""CLI SFT. Para el flujo completo recomendado: ``python train.py --config v0``."""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader

from compressed_llm.config import Config
from model.compressed_llm import CompressedLLM
from training.dataset import SFTDataset, collate_sft
from training.trainer import Trainer


def load_records(path: str):
    """Carga Arrow si existe o JSONL para entornos Windows compatibles."""
    source = Path(path)
    if source.is_file() or source.suffix.lower() == ".jsonl":
        with source.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    return list(load_from_disk(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena CompressedLLM (SFT)")
    parser.add_argument("--config", default="configs/v0.yaml")
    parser.add_argument("--train", default="data/out/train.jsonl")
    parser.add_argument("--valid", default="data/out/valid.jsonl")
    parser.add_argument("--out", default="results/v0")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--overfit", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    from tokenizer.tokenizer import BPETokenizer
    tok = BPETokenizer(vocab_size=cfg.model.vocab_size)
    tok.load(args.tokenizer)
    cfg.model.vocab_size = tok.vocab_size
    cfg.model.pad_token_id = tok.pad_token_id
    cfg.model.bos_token_id = tok.bos_token_id
    cfg.model.eos_token_id = tok.eos_token_id

    train_ds = load_records(args.train)
    valid_ds = load_records(args.valid)
    if args.overfit:
        train_ds = train_ds[:min(args.overfit, len(train_ds))]
        valid_ds = train_ds

    tr = SFTDataset(train_ds, tok, cfg.training.max_input_tokens, cfg.training.max_output_tokens)
    va = SFTDataset(valid_ds, tok, cfg.training.max_input_tokens, cfg.training.max_output_tokens)
    # `partial` en vez de un lambda local para que sea picklable con workers (spawn en Windows).
    collate = partial(collate_sft, pad_token_id=tok.pad_token_id,
                      max_input=cfg.training.max_input_tokens,
                      max_output=cfg.training.max_output_tokens)
    loader_kwargs = {"num_workers": cfg.training.num_workers,
                     "pin_memory": cfg.training.pin_memory and torch.cuda.is_available()}
    train_loader = DataLoader(tr, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=collate, **loader_kwargs)
    valid_loader = DataLoader(va, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate, **loader_kwargs)

    model = CompressedLLM(cfg, use_compression=not args.baseline)
    print(f"Parámetros: {model.get_num_parameters():,}")
    print(f"Compresión: {not args.baseline}")
    Trainer(model, cfg, train_loader, valid_loader, resume_from=args.resume).train(output_dir=args.out)


if __name__ == "__main__":
    main()
