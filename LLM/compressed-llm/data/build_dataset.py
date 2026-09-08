"""Orquesta el pipeline completo de datos (§49).

    Hugging Face → Spanish → validación → chosen/rejected → split agrupado →
    Arrow listo para entrenamiento.

Uso:
    python -m data.build_dataset --out-dir data/out
"""
from __future__ import annotations

import argparse
import os

from datasets import load_dataset

from data.download import download
from data.filter_spanish import filter_spanish
from data.preprocess import preprocess
from data.split import group_split

DATASET_NAME = "lmarena-ai/arena-human-preference-100k"


def run(output_dir: str, cache_dir: str = None, seed: int = 42,
        limit: int = None) -> dict:
    """Ejecuta la cadena completa y guarda los tres splits en ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"[1/4] Descargando {DATASET_NAME} ...")
    raw = download(name=DATASET_NAME, cache_dir=cache_dir)
    if limit:
        raw = raw.select(range(min(limit, len(raw))))
    print(f"      {len(raw)} filas")

    print("[2/4] Filtrando Spanish ...")
    es = filter_spanish(raw)
    print(f"      {len(es)} filas en español")

    print("[3/4] Preprocesando chosen/rejected ...")
    processed = preprocess(es)
    print(f"      {len(processed)} registros válidos")

    print("[4/4] Split agrupado por question_id ...")
    train, valid, test = group_split(processed, seed=seed)

    for name, ds in [("train", train), ("valid", valid), ("test", test)]:
        path = os.path.join(output_dir, name)
        ds.save_to_disk(path)
        print(f"      {name}: {len(ds)} -> {path}")

    return {
        "raw": len(raw),
        "spanish": len(es),
        "processed": len(processed),
        "train": len(train),
        "valid": len(valid),
        "test": len(test),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline completo de datos")
    parser.add_argument("--out-dir", default="data/out",
                        help="Directorio de salida para los splits.")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limitar filas (para pruebas rápidas).")
    args = parser.parse_args()

    stats = run(args.out_dir, cache_dir=args.cache_dir, seed=args.seed,
                limit=args.limit)
    print("\nResumen:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()