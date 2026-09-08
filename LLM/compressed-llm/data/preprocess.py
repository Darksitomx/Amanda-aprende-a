"""Paso C — Convierte A/B + winner en registros ``context/chosen/rejected`` (§10, §11).

Reglas (§10):
- ``winner == "model_a"``: chosen = conversation_a, rejected = conversation_b
- ``winner == "model_b"``: chosen = conversation_b, rejected = conversation_a
- ``winner == "tie"``    : sin preferencia binaria; se etiqueta como ``is_tie``
- ``winner == "both_bad"``: se excluye del SFT estándar; se marca ``is_both_bad``

La salida conserva ambas respuestas y metadatos (is_code, is_refusal,
category_tag, turn, question_id, language).
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

from datasets import Dataset


def _extract_conversation_text(item: Dict) -> str:
    """Serializa la conversación a texto plano conservando turnos."""
    return json.dumps(item, ensure_ascii=False)


def _split_messages(item) -> List[Dict]:
    """Devuelve la lista de turnos ``[{"role": ..., "content": ...}]``."""
    if isinstance(item, list):
        return item
    if isinstance(item, dict):
        # algunos formatos vienen como {role: content} o con 'messages'
        if "messages" in item:
            return item["messages"]
        return [item]
    return []


def preprocess_row(row: Dict) -> Dict | None:
    """Convierte una fila brute en el esquema interno. Devuelve ``None`` si debe
    descartarse (p. ej. estructura corrupta)."""
    qid = str(row.get("question_id", ""))
    winner = row.get("winner")

    conv_a = _split_messages(row.get("conversation_a"))
    conv_b = _split_messages(row.get("conversation_b"))
    text_a = _extract_conversation_text(conv_a)
    text_b = _extract_conversation_text(conv_b)

    if not qid or not text_a or not text_b:
        return None  # corrupto / incompleto

    meta = {
        "is_code": bool(row.get("is_code", False)),
        "is_refusal": bool(row.get("is_refusal", False)),
        "category_tag": row.get("category_tag", ""),
        "dedup_tag": row.get("dedup_tag", ""),
    }

    rec = {
        "question_id": qid,
        "language": row.get("language", "Spanish"),
        "turn": row.get("turn", 0),
        "winner": winner,
        "metadata": meta,
        "archived": False,
    }

    # ---- Preferencia binaria + conservar ambos lados (§10) ----
    if winner == "model_a":
        rec["context"] = conv_a
        rec["chosen"] = text_a
        rec["rejected"] = text_b
        rec["is_tie"] = False
        rec["is_both_bad"] = False
    elif winner == "model_b":
        rec["context"] = conv_a
        rec["chosen"] = text_b
        rec["rejected"] = text_a
        rec["is_tie"] = False
        rec["is_both_bad"] = False
    elif winner == "tie":
        rec["context"] = conv_a
        rec["chosen"] = text_a
        rec["rejected"] = None
        rec["is_tie"] = True
        rec["is_both_bad"] = False
    elif winner == "both_bad":
        rec["context"] = conv_a
        rec["chosen"] = None
        rec["rejected"] = None
        rec["is_tie"] = False
        rec["is_both_bad"] = True
    else:
        return None  # winner desconocido

    return rec


def preprocess(ds: Dataset) -> Dataset:
    """Aplica ``preprocess_row`` sobre todas las filas y descarta las corruptas."""
    rows = []
    for r in ds:
        p = preprocess_row(r)
        if p is not None:
            rows.append(p)
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocesa A/B+winner -> chosen/rejected")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from datasets import load_from_disk
    ds = load_from_disk(args.input)
    out = preprocess(ds)
    print(f"Preprocesado: {len(ds)} -> {len(out)} registros válidos")
    print("Distribución de winners:", out.to_pandas()["winner"].value_counts().to_dict())
    out.save_to_disk(args.output)


if __name__ == "__main__":
    main()