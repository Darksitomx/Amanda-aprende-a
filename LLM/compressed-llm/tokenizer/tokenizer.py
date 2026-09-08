"""Módulo del tokenizer.

En V0 se usa un tokenizer BPE/subword convencional (§27) para aislar la
hipótesis de compresión del problema de tokenización desde bytes. El pipeline
completo (bytes -> encoder -> latents) se explora en fases posteriores (§27).

El wrapper expone las operaciones mínimas necesarias para el resto del
proyecto (``encode``/``decode``/``to_ids``/``from_ids``) y las constantes de
tokens especiales que el modelo usa (pad/bos/eos), consistentes con
``ModelConfig``.
"""
from __future__ import annotations

import os
from typing import List, Optional

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders, models, normalizers, pre_tokenizers, trainers


class BPETokenizer:
    """Wrapper mínimo sobre un tokenizer BPE de Hugging Face ``tokenizers``.

    Permite tanto construir un BPETrainer desde un corpus como cargar un
    tokenizer persistido desde disco.

    Parameters
    ----------
    vocab_size:
        Tamaño de vocabulario objetivo al entrenar.
    special_tokens:
        Lista de tokens especiales en orden; pad/bos/eos deben estar presentes
        para alinearse con ``ModelConfig``.
    """

    def __init__(self, vocab_size: int = 32000,
                 special_tokens: Optional[List[str]] = None) -> None:
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or [
            "<pad>", "<bos>", "<eos>", "<unk>"
        ]
        self._tok: Optional[HFTokenizer] = None

    # ------------------------------------------------------------------ #
    # Construcción                                                          #
    # ------------------------------------------------------------------ #
    def _build_pipeline(self) -> HFTokenizer:
        tok = HFTokenizer(models.BPE(unk_token="<unk>"))
        tok.normalizer = normalizers.Sequence(
            [normalizers.NFC(), normalizers.Lowercase()]
        )
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tok.decoder = decoders.ByteLevel()
        return tok

    def train(self, corpus_files: List[str]) -> None:
        """Entrena un BPE sobre una lista de rutas a ficheros de texto plano."""
        tok = self._build_pipeline()
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            min_frequency=1,
        )
        tok.train(corpus_files, trainer)
        self._tok = tok
        self._register_ids()

    def load(self, path: str) -> None:
        """Carga un tokenizer persistido (``tokenizer.json``)."""
        self._tok = HFTokenizer.from_file(path)
        self._register_ids()

    def save(self, path: str) -> None:
        if self._tok is None:
            raise RuntimeError("No hay tokenizer construido/cargado para guardar.")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._tok.save(path)

    def _register_ids(self) -> None:
        if self._tok is None:
            return
        for name, tok in zip(
            ("pad_token_id", "bos_token_id", "eos_token_id", "unk_token_id"),
            ("<pad>", "<bos>", "<eos>", "<unk>"),
        ):
            try:
                setattr(self, name, self._tok.token_to_id(tok))
            except Exception:
                setattr(self, name, 0)

    # ------------------------------------------------------------------ #
    # API pública                                                          #
    # ------------------------------------------------------------------ #
    @property
    def vocab_size(self) -> int:
        if self._tok is not None:
            return self._tok.get_vocab_size()
        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, value: int) -> None:
        self._vocab_size = value

    def encode(self, text: str, add_special: bool = False) -> List[int]:
        """Devuelve los ids de un texto."""
        if self._tok is None:
            raise RuntimeError("Tokenizer no cargado. Llama a train() o load().")
        enc = self._tok.encode(text)
        return enc.ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Convierte ids de vuelta a texto."""
        if self._tok is None:
            raise RuntimeError("Tokenizer no cargado. Llama a train() o load().")
        return self._tok.decode(ids, skip_special_tokens=skip_special)

    @staticmethod
    def pad_sequence(seq: List[int], max_len: int, pad_id: int = 0) -> List[int]:
        """Trunca o rellena a ``max_len`` con el id de padding."""
        if len(seq) >= max_len:
            return seq[:max_len]
        return seq + [pad_id] * (max_len - len(seq))