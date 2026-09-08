"""Benchmark factual personalizado (§20).

Genera preguntas a partir del contexto original y compara:
    contexto completo -> respuesta
        vs.
    latents únicamente -> respuesta

Medir si la compresión destruyó información recuperable. En V0 se define la
interfaz y un pipeline básico de probes extraídas del texto.
"""
from __future__ import annotations

import re


YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
NAME_RE = re.compile(r"\b[A-Z][a-zñ]+(?:\s[A-Z][a-zñ]+)*\b")


def extract_probes(context_text: str) -> list:
    """Extrae preguntas (probe) del contexto con su respuesta esperada.

    Heurística simple (borrador): fechas, cifras y nombres propios. Función
    deliberadamente limitada; §20 prevé una extensión con LLM later.
    """
    probes = []

    years = YEAR_RE.findall(context_text)
    if years:
        first = years[0]
        probes.append({
            "question": "¿Qué año se menciona?",
            "expected": first,
        })

    for num in set(NUM_RE.findall(context_text)):
        probes.append({
            "question": f"¿Qué cantidad es {num}?",
            "expected": num,
        })

    for name in NAME_RE.findall(context_text)[:3]:
        probes.append({
            "question": f"¿Se mencionó el nombre {name}?",
            "expected": "sí",
        })

    return probes


def factual_benchmark(context_text: str, generated: str) -> dict:
    """Corre el matcher simple sobre las probes y devuelve métrica."""
    from evaluation.information_retention import information_retention_score

    probes = extract_probes(context_text)
    if not probes:
        return {"n_probes": 0, "irs": 0.0, "per_probe": []}

    expected = [p["expected"] for p in probes]
    responses = [generated] * len(probes)
    irs, ok = information_retention_score(responses, expected)
    return {
        "n_probes": len(probes),
        "irs": irs,
        "per_probe": [
            {"question": p["question"], "expected": p["expected"], "ok": o}
            for p, o in zip(probes, ok)
        ],
    }