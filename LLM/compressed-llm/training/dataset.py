"""Construcción de batches de entrenamiento SFT a partir de los registros
``context/chosen`` preprocesados (§8, §9.1).

Para cada ejemplo:
- ``context`` (lista de mensajes) se serializa a texto y se tokeniza.
- ``chosen`` se tokeniza como la respuesta objetivo.

Al alinear, cada posición de la respuesta apunta al siguiente token (shift).
"""
from __future__ import annotations

import json
from typing import Dict, Iterable, List

import torch
from torch.utils.data import Dataset as TorchDataset


def serialize_context(messages: List[Dict]) -> str:
    """Convierte la lista de turnos en texto plano (mantiene roles)."""
    if isinstance(messages, list):
        parts = []
        for m in messages:
            if isinstance(m, dict):
                role = m.get("role", "user")
                content = m.get("content", "")
                parts.append(f"{role}: {content}")
            else:
                parts.append(str(m))
        return "\n".join(parts)
    return str(messages)


class SFTDataset(TorchDataset):
    """Dataset PyTorch sobre registros preprocesados.

    Parameters
    ----------
    records:
        Iterable de dicts con ``context``, ``chosen`` y ``question_id``.
    tokenizer:
        Instancia de ``tokenizer.tokenizer.BPETokenizer``.
    max_input_tokens:
        Longitud máxima del contexto.
    max_output_tokens:
        Longitud máxima de la respuesta.
    """

    def __init__(self, records: Iterable[Dict], tokenizer,
                 max_input_tokens: int = 256, max_output_tokens: int = 256):
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_input = max_input_tokens
        self.max_output = max_output_tokens

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        rec = self.records[idx]
        ctx_text = serialize_context(rec.get("context", []))
        ans_text = rec.get("chosen", "")

        ctx_ids = self.tokenizer.encode(ctx_text)[:self.max_input]
        # respuesta: ponemos BOS para que el decoder aprenda a arrancar
        ans_ids = self.tokenizer.encode(ans_text)
        out_ids = [self.tokenizer.bos_token_id or 1] + ans_ids[:self.max_output - 1]

        pad = self.tokenizer.pad_token_id or 0
        ctx_len = len(ctx_ids)
        labels = out_ids + [-100] * (self.max_output - len(out_ids))

        return {
            "context_ids": torch.tensor(list(ctx_ids), dtype=torch.long),
            "context_mask": torch.ones(ctx_len, dtype=torch.bool),
            "output_ids": torch.tensor(out_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __getitems__(self, indices: List[int]) -> List[Dict]:
        return [self.__getitem__(i) for i in indices]


def collate_sft(batch: List[Dict], pad_token_id: int = 0,
                max_input: int = 256, max_output: int = 256) -> Dict:
    """Apila un batch con padding a la longitud máxima dentro del batch."""
    B = len(batch)

    def _pad_to(seqs, target, value):
        out = []
        masks = []
        for seq in seqs:
            arr = list(seq)
            mask = [1] * len(arr)
            if len(arr) < target:
                arr = arr + [value] * (target - len(arr))
                mask = mask + [0] * (target - len(mask))
            else:
                arr = arr[:target]
                mask = mask[:target]
            out.append(arr)
            masks.append(mask)
        return (torch.tensor(out, dtype=torch.long),
                torch.tensor(masks, dtype=torch.bool))

    ctx_ids = [b["context_ids"] for b in batch]
    ctx_mask = [b["context_mask"] for b in batch]
    out_ids = [b["output_ids"] for b in batch]
    labels = [b["labels"] for b in batch]

    ctx_ids_p, ctx_mask_p = _pad_to(ctx_ids, max_input, pad_token_id)
    out_ids_p, _ = _pad_to(out_ids, max_output, pad_token_id)
    labels_p, _ = _pad_to(labels, max_output, -100)

    return {
        "context_ids": ctx_ids_p,
        "context_mask": ctx_mask_p,
        "output_ids": out_ids_p,
        "labels": labels_p,
    }