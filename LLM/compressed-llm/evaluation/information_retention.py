"""Information Retention Score (IRS) (§19).

``IRS = respuestas_ok / preguntas_evaluadas``, donde el *ok* lo decide un
matcher (exacto o por subcadena). En V0 se define un matcher simple y una
colección básica de preguntas derivadas del contexto (borrador). Se guardan
también ejemplos cualitativos de fallos.
"""
from __future__ import annotations

import json
import re
from typing import List, Tuple

# Preguntas sobre hechos: (pregunta, forma de buscar en la respuesta generada)
PROBE_TEMPLATES = [
    ("¿Cuándo fue fundado?", "2018"),
]


def simple_answer_matcher(generated: str, expected: str) -> bool:
    """Matcher básico: busca la respuesta esperada (sin acentos/case) como subcadena."""
    def norm(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[áàäâ]", "a", s)
        s = re.sub(r"[éèëê]", "e", s)
        s = re.sub(r"[íìïî]", "i", s)
        s = re.sub(r"[óòöô]", "o", s)
        s = re.sub(r"[úùüû]", "u", s)
        s = re.sub(r"[ñ]", "n", s)
        return re.sub(r"[^a-z0-9 ]", "", s)
    return norm(expected) in norm(generated)


def information_retention_score(responses: List[str],
                                expected_answers: List[str]) -> Tuple[float, List[bool]]:
    """Calcula IRS y devuelve el acierto booleano por ejemplo."""
    if len(responses) != len(expected_answers) or len(responses) == 0:
        raise ValueError("queries y respuestas deben tener igual longitud > 0")
    ok = [simple_answer_matcher(r, e) for r, e in zip(responses, expected_answers)]
    return sum(ok) / len(ok), ok


def save_failures(samples: List[dict], path: str) -> None:
    """Guarda ejemplos cualitativos de fallos (§19)."""
    fails = [s for s in samples if not s.get("ok")]
    with open(path, "w", encoding="utf-8") as f:
        for s in fails:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")