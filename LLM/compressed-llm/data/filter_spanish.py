"""Paso B — Filtro obligatorio: ``language == "Spanish"`` (§1, Paso B)."""
from __future__ import annotations

import argparse

from datasets import Dataset


def filter_spanish(ds: Dataset) -> Dataset:
    """Conserva únicamente ejemplos en español."""
    if "language" not in ds.column_names:
        raise KeyError("El dataset no tiene columna 'language'.")
    return ds.filter(lambda ex: ex.get("language") == "Spanish", num_proc=4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filtra 'Spanish' del dataset.")
    parser.add_argument("--input", required=True,
                        help="Ruta del dataset cacheado o nombre HF.")
    parser.add_argument("--output", required=True,
                        help="Ruta arrow/parquet de salida o referencia.")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    from datasets import load_from_disk, load_dataset

    if os.path.isdir(args.input):
        ds = load_from_disk(args.input)
    else:
        ds = load_dataset(args.input, cache_dir=args.cache_dir, split="train")

    before = len(ds)
    filtered = filter_spanish(ds)
    print(f"Filtrado: {before} -> {len(filtered)} filas en español")
    filtered.save_to_disk(args.output)


if __name__ == "__main__":
    import os
    main()