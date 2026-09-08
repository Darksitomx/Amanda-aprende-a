"""Dataset y collate para SFT causal."""
from __future__ import annotations

from typing import Dict, Iterable, List

import torch
from torch.utils.data import Dataset as TorchDataset


def serialize_context(messages: List[Dict]) -> str:
    if isinstance(messages, list):
        parts = []
        for message in messages:
            if isinstance(message, dict):
                role = message.get("role", "user")
                content = message.get("content", "")
                if isinstance(content, list):
                    content = "".join(
                        str(part.get("text", part.get("content", "")))
                        if isinstance(part, dict) else str(part)
                        for part in content
                    )
                parts.append(f"{role}: {content}")
            else:
                parts.append(str(message))
        return "\n".join(parts)
    return str(messages)


class SFTDataset(TorchDataset):
    """Convierte registros ``context/chosen`` a entrenamiento causal."""

    def __init__(self, records: Iterable[Dict], tokenizer,
                 max_input_tokens: int = 256, max_output_tokens: int = 256):
        self.records = [r for r in records if r.get("context") and r.get("chosen")]
        self.tokenizer = tokenizer
        self.max_input = max_input_tokens
        self.max_output = max_output_tokens

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        rec = self.records[idx]
        ctx_ids = self.tokenizer.encode(serialize_context(rec["context"]))[:self.max_input]

        # Causal LM correcto:
        # input  = [BOS] + answer[:-1]
        # target = answer + [EOS]
        answer_ids = self.tokenizer.encode(str(rec["chosen"]))
        answer_ids = answer_ids[:max(0, self.max_output - 1)]
        input_ids = [self.tokenizer.bos_token_id] + answer_ids
        labels = answer_ids + [self.tokenizer.eos_token_id]

        return {
            "context_ids": torch.tensor(ctx_ids, dtype=torch.long),
            "context_mask": torch.ones(len(ctx_ids), dtype=torch.bool),
            "output_ids": torch.tensor(input_ids[:self.max_output], dtype=torch.long),
            "labels": torch.tensor(labels[:self.max_output], dtype=torch.long),
        }


def collate_sft(batch: List[Dict], pad_token_id: int = 0,
                max_input: int = 256, max_output: int = 256) -> Dict:
    def pad(seqs, target, value):
        out, masks = [], []
        for seq in seqs:
            arr = seq.tolist() if isinstance(seq, torch.Tensor) else list(seq)
            arr = arr[:target]
            mask = [1] * len(arr)
            if len(arr) < target:
                arr += [value] * (target - len(arr))
                mask += [0] * (target - len(mask))
            out.append(arr)
            masks.append(mask)
        return torch.tensor(out, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)

    context_ids, context_mask = pad([x["context_ids"] for x in batch], max_input, pad_token_id)
    output_ids, _ = pad([x["output_ids"] for x in batch], max_output, pad_token_id)
    labels, _ = pad([x["labels"] for x in batch], max_output, -100)

    return {
        "context_ids": context_ids,
        "context_mask": context_mask,
        "output_ids": output_ids,
        "labels": labels,
    }
