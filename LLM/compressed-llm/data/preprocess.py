"""Preprocesado de Arena A/B a registros para SFT y preferencia.

El objetivo de SFT debe ser únicamente la última respuesta del asistente,
condicionada por los mensajes anteriores de la conversación. No se entrena
contra la representación JSON de la conversación completa.
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple

from datasets import Dataset


def _split_messages(item) -> List[Dict]:
    """Devuelve la lista de turnos ``[{"role": ..., "content": ...}]``."""
    if isinstance(item, list):
        return item
    if isinstance(item, dict):
        if "messages" in item and isinstance(item["messages"], list):
            return item["messages"]
        return [item]
    return []


def _message_content(message: Dict) -> str:
    """Normaliza el contenido de un mensaje a texto plano."""
    content = message.get("content", "")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", part.get("content", ""))))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


def _conversation_to_context_and_answer(messages: List[Dict]) -> Optional[Tuple[List[Dict], str]]:
    """Separa una conversación en contexto previo + última respuesta del asistente."""
    if not messages:
        return None

    last_assistant = None
    for i in range(len(messages) - 1, -1, -1):
        role = str(messages[i].get("role", "")).lower()
        if role == "assistant":
            last_assistant = i
            break

    if last_assistant is None:
        return None

    answer = _message_content(messages[last_assistant]).strip()
    if not answer:
        return None

    context = messages[:last_assistant]
    cleaned_context = []
    for message in context:
        role = str(message.get("role", "user"))
        content = _message_content(message)
        cleaned_context.append({"role": role, "content": content})

    return cleaned_context, answer


def preprocess_row(row: Dict) -> Dict | None:
    """Convierte una fila de Arena al esquema interno de entrenamiento."""
    qid = str(row.get("question_id", ""))
    winner = row.get("winner")
    if not qid:
        return None

    conv_a = _split_messages(row.get("conversation_a"))
    conv_b = _split_messages(row.get("conversation_b"))
    a_parts = _conversation_to_context_and_answer(conv_a)
    b_parts = _conversation_to_context_and_answer(conv_b)
    if a_parts is None or b_parts is None:
        return None

    context_a, answer_a = a_parts
    context_b, answer_b = b_parts

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

    if winner == "model_a":
        rec.update({
            "context": context_a,
            "chosen": answer_a,
            "rejected": answer_b,
            "is_tie": False,
            "is_both_bad": False,
        })
    elif winner == "model_b":
        rec.update({
            "context": context_b,
            "chosen": answer_b,
            "rejected": answer_a,
            "is_tie": False,
            "is_both_bad": False,
        })
    elif winner == "tie":
        # Para SFT podemos usar A, pero conservamos la etiqueta de empate.
        rec.update({
            "context": context_a,
            "chosen": answer_a,
            "rejected": None,
            "is_tie": True,
            "is_both_bad": False,
        })
    elif winner == "both_bad":
        rec.update({
            "context": context_a,
            "chosen": None,
            "rejected": None,
            "is_tie": False,
            "is_both_bad": True,
        })
    else:
        return None

    return rec


def preprocess(ds: Dataset) -> Dataset:
    rows = []
    for row in ds:
        processed = preprocess_row(row)
        if processed is not None:
            rows.append(processed)
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocesa Arena A/B a contexto + respuesta")
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
