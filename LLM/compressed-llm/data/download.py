"""Paso A — Descarga o carga del dataset (dependencia de `datasets`).

Dataset: ``lmarena-ai/arena-human-preference-100k``
Referencia (§1):
    https://huggingface.co/datasets/lmarena-ai/arena-human-preference-100k
"""
from __future__ import annotations

import argparse
import os

from datasets import load_dataset


def download(name: str = "lmarena-ai/arena-human-preference-100k",
             cache_dir: str = None,
             split: str = "train") -> "datasets.Dataset":
    """Carga el dataset. Si ``cache_dir`` se da, se usa como caché local."""
    return load_dataset(name, cache_dir=cache_dir, split=split)


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga 'arena-human-preference-100k'")
    parser.add_argument("--cache-dir", default=None,
                        help="Directorio de caché de HF datasets.")
    parser.add_argument("--num-rows", type=int, default=None,
                        help="Limitar el número de filas (debug).")
    args = parser.parse_args()

    ds = download(cache_dir=args.cache_dir)
    print(f"Descargado: {len(ds)} filas")
    print("Columnas:", ds.column_names)
    if args.num_rows:
        ds = ds.select(range(min(args.num_rows, len(ds))))
        print(f"Seleccionadas {len(ds)} filas para inspección")
        print(ds.to_pandas().head(2).to_string())


if __name__ == "__main__":
    main()