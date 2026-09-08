"""Paso E — Inspección del dataset (Paso E de la especificación).

Muestra:
- número de filas;
- distribución de winners;
- longitud de contexto y respuestas;
- distribución de turnos;
- ejemplos aleatorios;
- porcentaje de code/refusal.
"""
from __future__ import annotations

import argparse
import random

from datasets import load_from_disk


def inspect(ds, k: int = 3, seed: int = 42) -> None:
    print("=" * 70)
    print(f"Filas: {len(ds)}")
    print("=" * 70)

    pdf = ds.to_pandas()

    if "chosen" in pdf.columns:
        lens = pdf["chosen"].astype(str).str.len()
        print(f"\nLongitud respuestas 'chosen': media={lens.mean():.1f} "
              f"min={lens.min()} max={lens.max()}")

    if "winner" in pdf.columns:
        print("\nDistribución de winners:")
        print(pdf["winner"].value_counts().to_string())

    if "turn" in pdf.columns:
        print("\nDistribución de turnos:")
        print(pdf["turn"].value_counts().sort_index().to_string())

    meta = pdf.get("metadata")
    if meta is not None:
        is_code = [m.get("is_code", False) if isinstance(m, dict) else False for m in meta]
        is_refusal = [m.get("is_refusal", False) if isinstance(m, dict) else False for m in meta]
        if meta.shape[0]:
            print(f"\n% is_code: {100.0 * sum(is_code) / len(meta):.2f}%")
            print(f"% is_refusal: {100.0 * sum(is_refusal) / len(meta):.2f}%")

    print("\nEjemplos aleatorios:")
    rng = random.Random(seed)
    sample_idx = rng.sample(range(len(ds)), min(k, len(ds)))
    for i in sample_idx:
        row = ds[i]
        print("-" * 70)
        print(f"question_id={row.get('question_id')} winner={row.get('winner')}")
        print("context:", str(row.get('context'))[:200])
        chosen = str(row.get('chosen', ''))[:200]
        print("chosen:", chosen)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspecciona el dataset preprocesado")
    parser.add_argument("--input", required=True)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    ds = load_from_disk(args.input)
    inspect(ds, k=args.k)


if __name__ == "__main__":
    main()