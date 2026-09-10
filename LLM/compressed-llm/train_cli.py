"""CLI visual para entrenar Compressed LLM por etapas.

Diseñado para:
- elegir cuántas filas acumuladas del dataset Spanish usar;
- conservar el tokenizer después de la primera etapa;
- reutilizar el último checkpoint y su optimizer;
- aumentar batch size para aprovechar mejor la GPU;
- continuar el entrenamiento por bloques de steps.

Ejecutar desde LLM/compressed-llm:
    python train_cli.py
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
DATASET_NAME = "lmarena-ai/arena-human-preference-100k"
DEFAULT_CONFIG = "v0"
TOKENIZER_PATH = ROOT / "tokenizer" / "tokenizer.json"
RAW_PATH = ROOT / "data" / "raw" / "arena_spanish_sample.jsonl"
PROCESSED_PATH = ROOT / "data" / "raw" / "arena_spanish_processed.jsonl"
OUT_DIR = ROOT / "data" / "out"
RESULTS_DIR = ROOT / "results" / DEFAULT_CONFIG
STATE_PATH = RESULTS_DIR / "training_state.json"


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def ask_int(prompt: str, default: int, minimum: int = 1) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
            if value < minimum:
                raise ValueError
            return value
        except ValueError:
            print(f"Introduce un entero >= {minimum}.")


def ask_choice(prompt: str, choices: Iterable[str], default: str) -> str:
    choices = list(choices)
    while True:
        raw = input(f"{prompt} ({'/'.join(choices)}) [{default}]: ").strip().lower()
        value = raw or default
        if value in choices:
            return value
        print("Opción no válida.")


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def existing_raw_rows() -> int:
    return count_jsonl(RAW_PATH)


def fetch_more_rows(target_total: int) -> List[Dict]:
    current = existing_raw_rows()
    if target_total <= current:
        print(f"Dataset local ya contiene {current} filas; no se descarga nada.")
        return load_jsonl(RAW_PATH)

    needed = target_total - current
    banner(f"Descargando {needed} filas nuevas (objetivo acumulado: {target_total})")
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)
    ds = ds.filter(lambda ex: ex.get("language") == "Spanish")
    ds = ds.skip(current)

    new_rows: List[Dict] = []
    for row in ds:
        new_rows.append(row)
        print(f"  filas: {current + len(new_rows)}/{target_total}", end="\r", flush=True)
        if len(new_rows) >= needed:
            break
    print()

    if len(new_rows) < needed:
        raise RuntimeError(
            f"El dataset solo permitió obtener {len(new_rows)} filas nuevas; "
            f"objetivo pendiente: {needed}."
        )

    append_jsonl(new_rows, RAW_PATH)
    print(f"Dataset local: {count_jsonl(RAW_PATH)} filas Spanish.")
    return load_jsonl(RAW_PATH)


def preprocess_all(raw_rows: List[Dict]) -> List[Dict]:
    from data.preprocess import preprocess_row

    processed = []
    for row in raw_rows:
        rec = preprocess_row(row)
        if rec is not None and rec.get("context") and rec.get("chosen"):
            processed.append(rec)
    if len(processed) < 32:
        raise RuntimeError(f"Solo quedaron {len(processed)} registros utilizables; se necesitan al menos 32.")
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROCESSED_PATH.open("w", encoding="utf-8") as f:
        for rec in processed:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return processed


def ensure_tokenizer(records: List[Dict], vocab_size: int):
    from tokenizer.tokenizer import BPETokenizer
    from training.dataset import serialize_context

    tok = BPETokenizer(vocab_size=vocab_size)
    if TOKENIZER_PATH.exists():
        tok.load(str(TOKENIZER_PATH))
        print(f"Tokenizer existente reutilizado: vocab={tok.vocab_size}")
        return tok

    banner("Entrenando tokenizer BPE (solo primera etapa)")
    corpus = ROOT / "data" / "raw" / "tokenizer_corpus.txt"
    corpus.parent.mkdir(parents=True, exist_ok=True)
    with corpus.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(serialize_context(rec["context"]) + "\n")
            f.write(str(rec["chosen"]) + "\n")
            if rec.get("rejected"):
                f.write(str(rec["rejected"]) + "\n")
    tok.train([str(corpus)])
    tok.save(str(TOKENIZER_PATH))
    print(f"Tokenizer guardado: {TOKENIZER_PATH}")
    print(f"Vocabulario: {tok.vocab_size}")
    return tok


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


def build_splits(records: List[Dict], seed: int):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_records, valid_records, test_records = grouped_split(records, seed)
    for name, subset in (("train", train_records), ("valid", valid_records), ("test", test_records)):
        Dataset.from_list(subset).save_to_disk(str(OUT_DIR / name))
    print(f"Splits: train={len(train_records)} valid={len(valid_records)} test={len(test_records)}")
    return train_records, valid_records, test_records


def latest_checkpoint() -> Optional[Path]:
    if not RESULTS_DIR.exists():
        return None
    candidates = []
    for path in RESULTS_DIR.glob("step_*.pt"):
        match = re.fullmatch(r"step_(\d+)\.pt", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None))[1]


def checkpoint_step(path: Optional[Path]) -> int:
    if path is None:
        return 0
    match = re.fullmatch(r"step_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else 0


def detect_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def recommended_batch_size() -> int:
    if not torch.cuda.is_available():
        return 2
    try:
        gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        return 4
    if gb >= 10:
        return 16
    if gb >= 6:
        return 8
    if gb >= 4:
        return 4
    return 2


def print_gpu_info() -> None:
    device = detect_device()
    print(f"Dispositivo: {device}")
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        total = p.total_memory / (1024 ** 3)
        print(f"GPU: {p.name} | VRAM total: {total:.2f} GB")
        print(f"CUDA: {torch.version.cuda}")


def make_loaders(cfg, tok, train_records, valid_records, batch_size: int):
    from training.dataset import SFTDataset, collate_sft

    train_ds = SFTDataset(train_records, tok, cfg.training.max_input_tokens, cfg.training.max_output_tokens)
    valid_ds = SFTDataset(valid_records, tok, cfg.training.max_input_tokens, cfg.training.max_output_tokens)
    collate = lambda b: collate_sft(
        b, tok.pad_token_id, cfg.training.max_input_tokens, cfg.training.max_output_tokens
    )
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        pin_memory=pin,
        persistent_workers=False,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        pin_memory=pin,
        persistent_workers=False,
    )
    return train_loader, valid_loader


def train_stage(cfg, tok, train_records, valid_records, checkpoint: Optional[Path],
                target_steps: int, batch_size: int, accum: int, device: str):
    from model.compressed_llm import CompressedLLM
    from training.trainer import Trainer

    cfg.training.batch_size = batch_size
    cfg.training.gradient_accumulation = accum
    train_loader, valid_loader = make_loaders(cfg, tok, train_records, valid_records, batch_size)

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    model = CompressedLLM(cfg, use_compression=True)
    print(f"Parámetros: {model.get_num_parameters():,}")
    print(f"Micro-batch: {batch_size} | accumulation: {accum} | batch efectivo: {batch_size * accum}")
    if checkpoint:
        print(f"Reanudando: {checkpoint}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model,
        cfg,
        train_loader,
        valid_loader,
        resume_from=str(checkpoint) if checkpoint else None,
        device=device,
    )
    start_step = trainer.global_step
    if target_steps <= start_step:
        raise ValueError(f"El checkpoint ya está en step {start_step}; objetivo recibido: {target_steps}.")

    print(f"Entrenando step {start_step} -> {target_steps}")
    result = trainer.train(max_steps=target_steps, output_dir=str(RESULTS_DIR))
    return result


def save_state(rows: int, steps: int, batch_size: int, accum: int, tokenizer_vocab: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps({
        "dataset_rows": rows,
        "training_steps": steps,
        "batch_size": batch_size,
        "gradient_accumulation": accum,
        "effective_batch_size": batch_size * accum,
        "tokenizer_vocab": tokenizer_vocab,
        "checkpoint": str(latest_checkpoint()) if latest_checkpoint() else None,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def new_stage() -> None:
    from compressed_llm.config import Config

    banner("NUEVA ETAPA")
    print_gpu_info()
    target_rows = ask_int("¿Cuántas filas Spanish acumuladas quieres usar?", max(256, existing_raw_rows() or 1024), 32)
    target_steps_increment = ask_int("¿Cuántos steps quieres entrenar en esta etapa?", 1000, 1)
    vocab = ask_int("Vocabulario BPE (solo afecta si todavía no existe tokenizer)", 8192, 256)
    batch_default = recommended_batch_size()
    batch = ask_int("Micro-batch", batch_default, 1)
    effective = ask_int("Batch efectivo objetivo", batch * 4, batch)
    accum = max(1, math.ceil(effective / batch))

    cfg = Config.from_yaml(str(ROOT / "configs" / f"{DEFAULT_CONFIG}.yaml"))
    raw_rows = fetch_more_rows(target_rows)
    records = preprocess_all(raw_rows)
    tok = ensure_tokenizer(records, vocab)

    cfg.model.vocab_size = tok.vocab_size
    cfg.model.pad_token_id = tok.pad_token_id
    cfg.model.bos_token_id = tok.bos_token_id
    cfg.model.eos_token_id = tok.eos_token_id

    train_records, valid_records, _ = build_splits(records, cfg.training.seed)
    previous = latest_checkpoint()
    current_step = checkpoint_step(previous)
    target_step = current_step + target_steps_increment

    banner("ENTRENAMIENTO")
    print(f"Filas acumuladas: {len(raw_rows)}")
    print(f"Registros utilizables: {len(records)}")
    print(f"Checkpoint previo: {previous or 'ninguno'}")
    print(f"Objetivo: step {current_step} -> {target_step}")
    device = detect_device()
    train_stage(cfg, tok, train_records, valid_records, previous, target_step, batch, accum, device)
    save_state(len(raw_rows), target_step, batch, accum, tok.vocab_size)
    banner("ETAPA TERMINADA")
    print(f"Último checkpoint: {latest_checkpoint()}")


def continue_stage() -> None:
    from compressed_llm.config import Config

    banner("CONTINUAR ENTRENAMIENTO")
    print_gpu_info()
    checkpoint = latest_checkpoint()
    if checkpoint is None:
        print("No existe ningún checkpoint en results/v0. Primero ejecuta una etapa nueva.")
        return

    current_step = checkpoint_step(checkpoint)
    target_rows = ask_int("Filas Spanish acumuladas que quieres tener", existing_raw_rows() or 1024, 32)
    additional_steps = ask_int("¿Cuántos steps adicionales?", 1000, 1)
    batch_default = recommended_batch_size()
    batch = ask_int("Micro-batch", batch_default, 1)
    effective = ask_int("Batch efectivo objetivo", batch * 4, batch)
    accum = max(1, math.ceil(effective / batch))

    cfg = Config.from_yaml(str(ROOT / "configs" / f"{DEFAULT_CONFIG}.yaml"))
    raw_rows = fetch_more_rows(target_rows)
    records = preprocess_all(raw_rows)

    from tokenizer.tokenizer import BPETokenizer
    if not TOKENIZER_PATH.exists():
        raise RuntimeError("Falta tokenizer/tokenizer.json; no se debe reentrenar durante un resume.")
    tok = BPETokenizer()
    tok.load(str(TOKENIZER_PATH))

    cfg.model.vocab_size = tok.vocab_size
    cfg.model.pad_token_id = tok.pad_token_id
    cfg.model.bos_token_id = tok.bos_token_id
    cfg.model.eos_token_id = tok.eos_token_id

    train_records, valid_records, _ = build_splits(records, cfg.training.seed)
    target_step = current_step + additional_steps

    banner("RESUME")
    print(f"Checkpoint: {checkpoint}")
    print(f"Dataset acumulado: {len(raw_rows)} filas")
    print(f"Objetivo: step {current_step} -> {target_step}")
    device = detect_device()
    train_stage(cfg, tok, train_records, valid_records, checkpoint, target_step, batch, accum, device)
    save_state(len(raw_rows), target_step, batch, accum, tok.vocab_size)
    banner("CONTINUACIÓN TERMINADA")
    print(f"Último checkpoint: {latest_checkpoint()}")


def inspect() -> None:
    banner("ESTADO DEL EXPERIMENTO")
    print_gpu_info()
    print(f"Filas locales: {existing_raw_rows()}")
    print(f"Tokenizer: {'OK' if TOKENIZER_PATH.exists() else 'NO'}")
    cp = latest_checkpoint()
    print(f"Checkpoint: {cp or 'NO'}")
    print(f"Step actual: {checkpoint_step(cp)}")
    if STATE_PATH.exists():
        try:
            data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            print(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception:
            pass


def main() -> None:
    os.chdir(ROOT)
    while True:
        banner("COMPRESSED LLM — TRAINING CLI")
        print("1) Nueva etapa de entrenamiento")
        print("2) Continuar desde el último checkpoint")
        print("3) Ver estado")
        print("4) Salir")
        choice = ask_choice("Selecciona", ["1", "2", "3", "4"], "1")
        try:
            if choice == "1":
                new_stage()
            elif choice == "2":
                continue_stage()
            elif choice == "3":
                inspect()
            else:
                print("Hasta luego.")
                return
        except KeyboardInterrupt:
            print("\nInterrumpido por el usuario. El checkpoint más reciente queda intacto.")
        except Exception as exc:
            print(f"\nERROR: {type(exc).__name__}: {exc}")
        input("\nPulsa ENTER para volver al menú...")


if __name__ == "__main__":
    main()
