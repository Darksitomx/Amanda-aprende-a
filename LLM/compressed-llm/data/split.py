"""Paso D — Split agrupado por ``question_id`` (§12, §13).

- 80% train, 10% validation, 10% test
- El split se hace *agrupado* por ``question_id`` para evitar que partes
  relacionadas de una misma conversación acaben en splits distintos.
"""
from __future__ import annotations

import argparse
import random
from typing import Tuple

from datasets import Dataset, load_from_disk


def group_split(ds: Dataset, seed: int = 42,
                train=0.8, val=0.1, test=0.1) -> Tuple[Dataset, Dataset, Dataset]:
    """Devuelve ``(train, validation, test)`` con grupos por ``question_id``."""
    if "question_id" not in ds.column_names:
        raise KeyError("Falta 'question_id' para el split agrupado.")

    # Índices de las filas agrupadas por question_id (sin duplicar)
    seen = {}
    group_idx = []
    for i, q in enumerate(ds["question_id"]):
        if q not in seen:
            seen[q] = True
            group_idx.append(i)

    rng = random.Random(seed)
    rng.shuffle(group_idx)
    n = len(group_idx)
    n_train = int(n * train)
    n_val = int(n * val)

    def _select(idx_list):
        # Expandir índices de grupo a índices de fila (varias filas por grupo)
        rows = []
        existing = set()
        for gi in idx_list:
            for j in range(len(ds)):
                # O(N^2) en el peor caso; se indica para conjuntos grandes,
                # en V0 el dataset es acotado.
                if j in existing:
                    continue
                if ds[j]["question_id"] == ds[gi]["question_id"]:
                    rows.append(j)
                    existing.add(j)
        return ds.select(rows)

    train_ds = _select(group_idx[:n_train])
    val_ds = _select(group_idx[n_train:n_train + n_val])
    test_ds = _select(group_idx[n_train + n_val:])
    return train_ds, val_ds, test_ds


def main() -> None:
    parser = argparse.ArgumentParser(description="Split agrupado por question_id")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ds = load_from_disk(args.input)
    train, val, test = group_split(ds, seed=args.seed)
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    train.save_to_disk(os.path.join(args.out_dir, "train"))
    val.save_to_disk(os.path.join(args.out_dir, "valid"))
    test.save_to_disk(os.path.join(args.out_dir, "test"))
    print(f"train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()