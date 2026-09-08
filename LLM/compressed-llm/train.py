"""Pipeline único de entrenamiento.

Uso recomendado:
    python train.py --config v0

Hace, en orden:
1. descarga una muestra Spanish del dataset Arena mediante streaming;
2. preprocesa chosen/rejected;
3. entrena el BPE real y guarda tokenizer/tokenizer.json;
4. crea train/valid/test en data/out;
5. ejecuta los tests del bottleneck;
6. hace un overfit corto sobre 32 ejemplos reales;
7. entrena V0 y guarda resultados/checkpoints.

No usa el fallback de IDs crudos para entrenamiento real.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from datasets import Dataset, load_dataset

ROOT = Path(__file__).resolve().parent
DATASET_NAME = "lmarena-ai/arena-human-preference-100k"


def log(msg: str) -> None:
    print(f"\n{'=' * 72}\n{msg}\n{'=' * 72}", flush=True)


def stream_spanish(num_rows: int) -> List[Dict]:
    log(f"[1/7] Descargando hasta {num_rows} filas Spanish de {DATASET_NAME} (streaming)")
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)
    ds = ds.filter(lambda ex: ex.get("language") == "Spanish")
    rows: List[Dict] = []
    for row in ds:
        rows.append(row)
        if len(rows) >= num_rows:
            break
    if len(rows) < 32:
        raise RuntimeError(f"Solo se encontraron {len(rows)} filas Spanish; se necesitan al menos 32.")
    print(f"Muestra obtenida: {len(rows)} filas", flush=True)
    return rows


def preprocess_rows(rows: List[Dict]) -> List[Dict]:
    from data.preprocess import preprocess_row
    processed = []
    for row in rows:
        rec = preprocess_row(row)
        if rec is not None and rec.get("context") and rec.get("chosen"):
            processed.append(rec)
    if len(processed) < 32:
        raise RuntimeError(f"Solo quedaron {len(processed)} registros utilizables para SFT.")
    return processed


def save_jsonl(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_corpus(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from training.dataset import serialize_context
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(serialize_context(rec["context"]) + "\n")
            f.write(str(rec["chosen"]) + "\n")
            # El rechazado también aporta vocabulario de entrada/salida, pero no
            # participa como target SFT.
            if rec.get("rejected"):
                f.write(str(rec["rejected"]) + "\n")


def grouped_split(records: List[Dict], seed: int):
    groups: Dict[str, List[Dict]] = {}
    for rec in records:
        groups.setdefault(str(rec["question_id"]), []).append(rec)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    n_train = max(1, int(n * 0.8))
    n_valid = max(1, int(n * 0.1))
    train_keys = keys[:n_train]
    valid_keys = keys[n_train:n_train + n_valid]
    test_keys = keys[n_train + n_valid:]
    if not test_keys:
        test_keys = valid_keys[-1:]
        valid_keys = valid_keys[:-1] or train_keys[-1:]
    materialize = lambda ks: [r for k in ks for r in groups[k]]
    return materialize(train_keys), materialize(valid_keys), materialize(test_keys)


def prepare_data(args, cfg):
    raw_rows = stream_spanish(args.rows)
    raw_path = ROOT / "data" / "raw" / "arena_spanish_sample.jsonl"
    save_jsonl(raw_rows, raw_path)

    log("[2/7] Preprocesando conversaciones A/B")
    records = preprocess_rows(raw_rows)
    print(f"Registros SFT utilizables: {len(records)}", flush=True)
    processed_path = ROOT / "data" / "raw" / "arena_spanish_processed.jsonl"
    save_jsonl(records, processed_path)

    log("[3/7] Entrenando tokenizer BPE real")
    corpus = ROOT / "data" / "raw" / "tokenizer_corpus.txt"
    build_corpus(records, corpus)
    from tokenizer.tokenizer import BPETokenizer
    tok = BPETokenizer(vocab_size=args.tokenizer_vocab)
    tok_path = ROOT / "tokenizer" / "tokenizer.json"
    tok.train([str(corpus)])
    tok.save(str(tok_path))
    print(f"Tokenizer guardado: {tok_path}", flush=True)
    print(f"Vocabulario real: {tok.vocab_size}", flush=True)

    # El modelo debe usar exactamente el vocabulario resultante del BPE.
    cfg.model.vocab_size = tok.vocab_size
    cfg.model.pad_token_id = tok.pad_token_id
    cfg.model.bos_token_id = tok.bos_token_id
    cfg.model.eos_token_id = tok.eos_token_id

    log("[4/7] Creando train/valid/test")
    train_records, valid_records, test_records = grouped_split(records, args.seed)
    out_dir = ROOT / "data" / "out"
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in (("train", train_records), ("valid", valid_records), ("test", test_records)):
        Dataset.from_list(subset).save_to_disk(str(out_dir / name))
        print(f"{name}: {len(subset)}", flush=True)

    return tok, train_records, valid_records, test_records


def run_tests() -> None:
    log("[5/7] Ejecutando tests obligatorios")
    cmd = [sys.executable, "-m", "pytest", "tests/test_bottleneck.py", "-q"]
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit("Los tests fallaron. Se detiene antes del entrenamiento V0.")


def run_overfit(cfg, tok, records: List[Dict], steps: int, device: str | None) -> None:
    log(f"[6/7] Overfit test sobre {min(32, len(records))} ejemplos reales ({steps} pasos)")
    import torch
    from torch.utils.data import DataLoader
    from model.compressed_llm import CompressedLLM
    from training.dataset import SFTDataset, collate_sft
    from training.trainer import Trainer

    small = records[:min(32, len(records))]
    ds = SFTDataset(small, tok, cfg.training.max_input_tokens, cfg.training.max_output_tokens)
    loader = DataLoader(
        ds, batch_size=cfg.training.batch_size, shuffle=True,
        collate_fn=lambda b: collate_sft(
            b, tok.pad_token_id, cfg.training.max_input_tokens, cfg.training.max_output_tokens
        ),
    )
    model = CompressedLLM(cfg, use_compression=True)
    trainer = Trainer(model, cfg, loader, None, device=device)

    model.train()
    iterator = iter(loader)
    losses = []
    for step in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {k: v.to(trainer.device) for k, v in batch.items()}
        trainer.optimizer.zero_grad(set_to_none=True)
        out = model(**batch)
        loss = out["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError("Overfit produjo una loss no finita.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip or 1.0)
        trainer.optimizer.step()
        losses.append(float(loss.item()))
        if (step + 1) % max(1, steps // 5) == 0:
            print(f"overfit step={step + 1}/{steps} loss={losses[-1]:.4f}", flush=True)

    start, end = losses[0], losses[-1]
    print(f"Overfit: {start:.4f} -> {end:.4f}", flush=True)
    if end >= start * 0.5:
        raise RuntimeError(
            "El overfit no redujo la loss al menos 50%. No se inicia V0; "
            "revisa masks, labels, tokenizer o bottleneck."
        )


def train_v0(cfg, tok, train_records, valid_records, out_dir: Path, args) -> None:
    log("[7/7] Entrenamiento V0")
    from torch.utils.data import DataLoader
    from model.compressed_llm import CompressedLLM
    from training.dataset import SFTDataset, collate_sft
    from training.trainer import Trainer

    train_ds = SFTDataset(train_records, tok, cfg.training.max_input_tokens, cfg.training.max_output_tokens)
    valid_ds = SFTDataset(valid_records, tok, cfg.training.max_input_tokens, cfg.training.max_output_tokens)
    collate = lambda b: collate_sft(
        b, tok.pad_token_id, cfg.training.max_input_tokens, cfg.training.max_output_tokens
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate)

    model = CompressedLLM(cfg, use_compression=True)
    print(f"Parámetros: {model.get_num_parameters():,}", flush=True)
    print(f"Input tokens: {cfg.training.max_input_tokens} -> {cfg.compressor.latent_count} latents", flush=True)
    trainer = Trainer(model, cfg, train_loader, valid_loader, device=args.device)
    result = trainer.train(max_steps=args.steps, output_dir=str(out_dir))

    (out_dir / "config_resolved.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "run_summary.json").write_text(
        json.dumps({
            "dataset": DATASET_NAME,
            "rows_requested": args.rows,
            "train": len(train_records),
            "valid": len(valid_records),
            "test": len(getattr(args, "test_records", train_records)),
            "tokenizer_vocab": tok.vocab_size,
            "latent_count": cfg.compressor.latent_count,
            "compression_ratio": cfg.training.max_input_tokens / cfg.compressor.latent_count,
            "final_step": result["final_step"],
        }, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compressed LLM: pipeline único de preparación + tests + entrenamiento")
    parser.add_argument("--config", default="v0", help="Nombre de YAML en configs/ sin .yaml")
    parser.add_argument("--rows", type=int, default=2048, help="Filas Spanish a obtener por streaming")
    parser.add_argument("--tokenizer-vocab", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overfit-steps", type=int, default=150)
    parser.add_argument("--steps", type=int, default=None, help="Steps V0; por defecto usa max_steps del YAML")
    parser.add_argument("--device", default=None, help="cuda, cpu o auto por defecto")
    parser.add_argument("--clean", action="store_true", help="Borrar data/out antes de crear splits")
    args = parser.parse_args()

    if Path(args.config).suffix:
        config_path = Path(args.config)
    else:
        config_path = ROOT / "configs" / f"{args.config}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No existe configuración: {config_path}")

    sys.path.insert(0, str(ROOT))
    from compressed_llm.config import Config
    cfg = Config.from_yaml(str(config_path))
    if args.steps is None:
        args.steps = cfg.training.max_steps

    tok, train_records, valid_records, test_records = prepare_data(args, cfg)
    args.test_records = test_records
    run_tests()
    run_overfit(cfg, tok, train_records, args.overfit_steps, args.device)

    result_dir = ROOT / "results" / args.config
    result_dir.mkdir(parents=True, exist_ok=True)
    train_v0(cfg, tok, train_records, valid_records, result_dir, args)
    log("FINALIZADO")
    print(f"Resultados: {result_dir}")


if __name__ == "__main__":
    main()
