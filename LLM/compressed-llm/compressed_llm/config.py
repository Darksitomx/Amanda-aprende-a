"""Carga y validación de configuración en YAML.

Permite leer ``configs/*.yaml`` y exponer los valores como un objeto
con atributos accesibles, aplicando valores por defecto para evitar
errores silenciosos por claves faltantes.
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml

# --------------------------------------------------------------------------- #
# Dataclasses de configuración                                                    #
# --------------------------------------------------------------------------- #


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    intermediate_size: int = 1024
    max_position_embeddings: int = 4096
    dropout: float = 0.1
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2


@dataclass
class CompressorConfig:
    encoder_layers: int = 2
    hidden_size: int = 256
    num_heads: int = 4
    latent_count: int = 32
    latent_dim: int = 256
    cross_attention_layers: int = 2
    local_window: int = 64
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    seed: int = 42
    max_input_tokens: int = 256
    max_output_tokens: int = 256
    batch_size: int = 2
    gradient_accumulation: int = 16
    learning_rate: float = 3e-4
    warmup_steps: int = 100
    max_steps: int = 10000
    mixed_precision: str = "fp16"
    weight_decay: float = 0.0
    log_steps: int = 10
    save_steps: int = 1000
    eval_steps: int = 500
    output_dir: str = "results"
    grad_clip: Optional[float] = 1.0


@dataclass
class LossConfig:
    preference_weight: float = 0.1
    information_weight: float = 0.0


@dataclass
class Config:
    """Configuración completa con valores por defecto satisfechos."""

    name: str = "base"
    model: ModelConfig = field(default_factory=ModelConfig)
    compressor: CompressorConfig = field(default_factory=CompressorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    # ------------------------------------------------------------------ #
    # (de)serialización                                                      #
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model": dataclasses.asdict(self.model),
            "compressor": dataclasses.asdict(self.compressor),
            "training": dataclasses.asdict(self.training),
            "loss": dataclasses.asdict(self.loss),
        }

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_yaml())

    @staticmethod
    def _coerce(datacls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Convierte valores que PyYAML resuelve como str (p. ej. '3e-4') al
        tipo que el campo espera (int/float), para robustez frente a la
        notación científica YAML."""
        out = {}
        for name, value in kwargs.items():
            ann = datacls.__dataclass_fields__[name].type if name in datacls.__dataclass_fields__ else None
            ann_name = ann if isinstance(ann, str) else (ann.__name__ if ann is not None else "")
            if isinstance(value, str) and ann_name in ("int", "float"):
                try:
                    if value.lstrip("+-").lower().startswith("0x"):
                        value = int(value, 16)
                    elif ann_name == "int":
                        value = int(float(value)) if value.replace(".", "", 1).replace("-", "", 1).isdigit() else value
                    else:  # float
                        value = float(value.replace(",", "."))
                except ValueError:
                    pass
            out[name] = value
        return out

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Config":
        cfg = Config()
        if "name" in data:
            cfg.name = data["name"]
        if "model" in data:
            cfg.model = ModelConfig(**Config._coerce(ModelConfig, {**dataclasses.asdict(cfg.model), **data["model"]}))
        if "compressor" in data:
            cfg.compressor = CompressorConfig(
                **Config._coerce(CompressorConfig, {**dataclasses.asdict(cfg.compressor), **data["compressor"]})
            )
        if "training" in data:
            cfg.training = TrainingConfig(
                **Config._coerce(TrainingConfig, {**dataclasses.asdict(cfg.training), **data["training"]})
            )
        if "loss" in data:
            cfg.loss = LossConfig(
                **Config._coerce(LossConfig, {**dataclasses.asdict(cfg.loss), **data["loss"]})
            )
        return cfg

    @staticmethod
    def from_yaml(path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cfg = Config.from_dict(data)
        cfg.name = os.path.splitext(os.path.basename(path))[0]
        return cfg