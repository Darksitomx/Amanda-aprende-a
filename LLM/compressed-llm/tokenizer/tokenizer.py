"""Tokenizer BPE/subword para Compressed LLM."""
from __future__ import annotations

import os
from typing import List, Optional

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders, models, normalizers, pre_tokenizers, trainers


class BPETokenizer:
    """Wrapper sobre Hugging Face tokenizers para un BPE reproducible."""

    def __init__(self, vocab_size: int = 32000,
                 special_tokens: Optional[List[str]] = None) -> None:
        self._vocab_size = vocab_size
        self.special_tokens = special_tokens or ["<pad>", "<bos>", "<eos>", "<unk>"]
        self._tok: Optional[HFTokenizer] = None
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

    def _build_pipeline(self) -> HFTokenizer:
        tok = HFTokenizer(models.BPE(unk_token="<unk>"))
        # Conservamos mayúsculas: nombres propios, código y detalles factuales.
        tok.normalizer = normalizers.NFC()
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tok.decoder = decoders.ByteLevel()
        return tok

    def train(self, corpus_files: List[str]) -> None:
        tok = self._build_pipeline()
        trainer = trainers.BpeTrainer(
            vocab_size=self._vocab_size,
            special_tokens=self.special_tokens,
            min_frequency=1,
            show_progress=True,
        )
        tok.train(corpus_files, trainer=trainer)
        self._tok = tok
        self._register_ids()

    def load(self, path: str) -> None:
        self._tok = HFTokenizer.from_file(path)
        self._register_ids()

    def save(self, path: str) -> None:
        if self._tok is None:
            raise RuntimeError("No hay tokenizer construido/cargado.")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._tok.save(path)

    def _register_ids(self) -> None:
        if self._tok is None:
            return
        for name, token in (
            ("pad_token_id", "<pad>"),
            ("bos_token_id", "<bos>"),
            ("eos_token_id", "<eos>"),
            ("unk_token_id", "<unk>"),
        ):
            value = self._tok.token_to_id(token)
            if value is not None:
                setattr(self, name, value)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size() if self._tok is not None else self._vocab_size

    @vocab_size.setter
    def vocab_size(self, value: int) -> None:
        self._vocab_size = value

    def encode(self, text: str, add_special: bool = False) -> List[int]:
        if self._tok is None:
            raise RuntimeError("Tokenizer no cargado. Usa train() o load().")
        ids = self._tok.encode(text).ids
        if add_special:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        if self._tok is None:
            raise RuntimeError("Tokenizer no cargado. Usa train() o load().")
        return self._tok.decode(ids, skip_special_tokens=skip_special)

    @staticmethod
    def pad_sequence(seq: List[int], max_len: int, pad_id: int = 0) -> List[int]:
        return seq[:max_len] + [pad_id] * max(0, max_len - len(seq))
