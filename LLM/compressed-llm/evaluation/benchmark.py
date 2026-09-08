"""Benchmark general (§18, §23).

Agrupa la ejecución de varias evaluaciones y escribe las métricas en
``metrics.csv`` y una carpeta de plots. En V0 define el esqueleto; los plots
de la curva calidad-vs-compresión se generan cuando haya resultados.
"""
from __future__ import annotations

import csv
import os


METRIC_FIELDS = [
    "input_tokens", "latent_tokens", "compression_ratio", "output_tokens",
    "train_loss", "validation_loss", "perplexity", "latency_ms",
    "throughput_tokens_per_second", "VRAM_MB", "estimated_FLOPs",
    "parameter_count", "training_steps",
]


def write_metrics_csv(path: str, rows: list) -> None:
    """``rows`` es una lista de dicts con las claves de ``METRIC_FIELDS``."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in METRIC_FIELDS})


def make_plots(metrics_csv: str, out_dir: str) -> None:
    """Genera los plots de calidad/IRS/latencia vs compresión (§23).

    Stub: se completa cuando haya resultados reales de experimentos.
    """
    raise NotImplementedError(
        "Generación de plots pendiente de tener resultados experimentales (§23)."
    )