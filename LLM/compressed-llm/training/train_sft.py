"""CLI SFT. Para el flujo completo recomendado: ``python train.py --config v0``."""
from __future__ import annotations

import argparse

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

    train_ds = load_from_disk(args.train)
    valid_ds = load_from_disk(args.valid)
    if args.overfit:
        train_ds = train_ds.select(range(min(args.overfit, len(train_ds))))
        valid_ds = train_ds

    tr = SFTDataset(list(train_ds), tok, cfg.training.max_input_tokens, cfg.training.max_output_tokens)
    va = SFTDataset(list(valid_ds), tok, cfg.training.max_input_tokens, cfg.training.max_output_tokens)
    collate = lambda b: collate_sft(b, tok.pad_token_id, cfg.training.max_input_tokens, cfg.training.max_output_tokens)
    train_loader = DataLoader(tr, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(va, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate)

    model = CompressedLLM(cfg, use_compression=not args.baseline)
    print(f"Parámetros: {model.get_num_parameters():,}")
    print(f"Compresión: {not args.baseline}")
    Trainer(model, cfg, train_loader, valid_loader, resume_from=args.resume).train(output_dir=args.out)


if __name__ == "__main__":
    main()
