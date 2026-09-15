"""Crecimiento de modelo ("model growing") para CompressedLLM.

Permite pasar de una arquitectura pequeña ya entrenada a una más grande
(``hidden_size``, ``num_layers``, ``num_heads`` (si es compatible),
``intermediate_size``, ``latent_count``, ``latent_dim``,
``encoder_layers``, ``cross_attention_layers``, ``max_position_embeddings``)
sin perder lo aprendido, en vez de reiniciar los pesos aleatoriamente.

Estrategia (warm-start genérico, no es Net2Net exacto):

1. Para cada tensor del ``state_dict`` del modelo nuevo que también existe en
   el modelo viejo:
   - Si tiene la misma forma: se copia tal cual.
   - Si es más grande en alguna dimensión: se copia la sub-región que se
     solapa (esquina superior-izquierda) y el resto queda con la
     inicialización aleatoria fresca del modelo nuevo.
2. Para bloques *enteramente nuevos* (p. ej. si ``num_layers`` sube de 4 a 6,
   los bloques 4 y 5 no existen en el checkpoint viejo), se ponen a cero los
   pesos (y bias) de la última proyección de cada sub-módulo residual
   (``attn.out``, ``ffn.2`` en decoder/encoder; ``out`` en cross-attention).
   Esto hace que, al insertarlos, el bloque nuevo actúe inicialmente como una
   función identidad (``x = x + f(norm(x))`` con ``f(...) = 0``), así el
   modelo no "olvida" de golpe lo aprendido al agregar profundidad; luego el
   entrenamiento adicional los va activando gradualmente.

Limitación conocida: cuando una dimensión existente crece (p. ej.
``hidden_size`` 256 -> 384), las filas/columnas nuevas quedan con init
aleatoria fresca, no con una transformación que preserve la función exacta
(a diferencia del Net2Net original de Chen et al.). En la práctica funciona
bien como warm-start porque el resto de la red ya sabe qué hacer con la
sub-región vieja, y el ruido nuevo se amortigua rápido con unos cientos de
steps adicionales.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

# Prefijos de contenedores de bloques residuales que soportan "identity init"
# cuando un bloque es enteramente nuevo. Cada entrada mapea el prefijo del
# contenedor a las claves relativas (dentro del bloque) que hay que poner a
# cero para que el bloque sea un no-op al insertarse.
_BLOCK_CONTAINERS = {
    "compressor.encoder.blocks": ["attn.out.weight", "ffn.2.weight", "ffn.2.bias"],
    "compressor.bottleneck.blocks": ["out.weight"],
    "decoder.blocks": ["attn.out.weight", "ffn.2.weight", "ffn.2.bias"],
}

_BLOCK_INDEX_RE = re.compile(r"^(?P<prefix>.+\.blocks)\.(?P<idx>\d+)\.")


@dataclass
class GrowReport:
    copied_exact: List[str]
    copied_partial: List[str]
    fresh_random: List[str]
    zeroed_new_blocks: List[str]
    new_block_indices: Dict[str, List[int]]

    def summary(self) -> str:
        lines = [
            f"Copiados exactos:      {len(self.copied_exact)}",
            f"Copiados parciales:    {len(self.copied_partial)} "
            f"(dimensión creció, se rellenó el solape)",
            f"Nuevos (init fresca):  {len(self.fresh_random)}",
            f"Bloques nuevos:        "
            + (", ".join(f"{k}={v}" for k, v in self.new_block_indices.items()) or "ninguno"),
        ]
        return "\n".join(lines)


def _overlap_slices(old_shape: torch.Size, new_shape: torch.Size) -> Tuple[slice, ...]:
    return tuple(slice(0, min(o, n)) for o, n in zip(old_shape, new_shape))


def _max_block_index(state_dict: Dict[str, torch.Tensor], prefix: str) -> int:
    max_idx = -1
    needle = prefix + "."
    for key in state_dict:
        if key.startswith(needle):
            match = _BLOCK_INDEX_RE.match(key)
            if match and match.group("prefix") == prefix:
                max_idx = max(max_idx, int(match.group("idx")))
    return max_idx


def transplant_weights(old_state_dict: Dict[str, torch.Tensor],
                       new_model: nn.Module) -> Tuple[Dict[str, torch.Tensor], GrowReport]:
    """Combina ``old_state_dict`` con la inicialización fresca de ``new_model``.

    No modifica ``new_model`` in-place; devuelve un ``state_dict`` listo para
    ``new_model.load_state_dict(...)`` junto con un reporte de qué se copió.
    """
    new_sd = new_model.state_dict()
    copied_exact: List[str] = []
    copied_partial: List[str] = []
    fresh_random: List[str] = []

    for key, new_tensor in new_sd.items():
        old_tensor = old_state_dict.get(key)
        if old_tensor is None:
            fresh_random.append(key)
            continue
        if tuple(old_tensor.shape) == tuple(new_tensor.shape):
            new_sd[key] = old_tensor.clone()
            copied_exact.append(key)
        elif len(old_tensor.shape) == len(new_tensor.shape):
            merged = new_tensor.clone()
            slices = _overlap_slices(old_tensor.shape, new_tensor.shape)
            merged[slices] = old_tensor[slices]
            new_sd[key] = merged
            copied_partial.append(key)
        else:
            # Rango de dimensiones distinto (no debería pasar en este modelo);
            # se conserva la init fresca por seguridad.
            fresh_random.append(key)

    # Identidad para bloques enteramente nuevos (agregados por num_layers++).
    zeroed_new_blocks: List[str] = []
    new_block_indices: Dict[str, List[int]] = {}
    for prefix, zero_suffixes in _BLOCK_CONTAINERS.items():
        old_max = _max_block_index(old_state_dict, prefix)
        new_max = _max_block_index(new_sd, prefix)
        new_indices = [i for i in range(old_max + 1, new_max + 1)]
        if new_indices:
            new_block_indices[prefix] = new_indices
        for idx in new_indices:
            for suffix in zero_suffixes:
                full_key = f"{prefix}.{idx}.{suffix}"
                if full_key in new_sd:
                    new_sd[full_key] = torch.zeros_like(new_sd[full_key])
                    zeroed_new_blocks.append(full_key)

    report = GrowReport(
        copied_exact=copied_exact,
        copied_partial=copied_partial,
        fresh_random=fresh_random,
        zeroed_new_blocks=zeroed_new_blocks,
        new_block_indices=new_block_indices,
    )
    return new_sd, report


def grow_model(old_model: nn.Module, new_model: nn.Module) -> GrowReport:
    """Transplanta pesos de ``old_model`` a ``new_model`` in-place.

    Ambos deben ser instancias de ``CompressedLLM`` (o cualquier módulo con
    ``state_dict``/``load_state_dict`` compatibles a nivel de nombres de
    claves). Devuelve el reporte de qué se copió/reinicializó.
    """
    new_sd, report = transplant_weights(old_model.state_dict(), new_model)
    new_model.load_state_dict(new_sd)
    return report
