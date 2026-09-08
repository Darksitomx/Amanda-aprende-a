# Compressed LLM

LLM desde cero cuyo contexto de entrada se transforma por un **bottleneck de
representaciones continuas (continuous latents)** antes del decoder LLM
autoregresivo. Dataset: `lmarena-ai/arena-human-preference-100k` (español).

Especificación completa en [`PROJECT_SPEC.md`](PROJECT_SPEC.md) y
`../LLM/COMPRESSED_LLM_PROJECT.md`.

## Hipótesis

> Una conversación puede representarse con muchos menos vectores continuos que
> tokens convencionales, conservando la información necesaria para generar
> respuestas de alta calidad.

## Arquitectura (§4)

```
contexto -> tokenizer -> encoder ligero -> K continuous latents -> decoder LLM -> respuesta
```

## Requisitos

- Python 3.10+
- PyTorch 2.x (CUDA opcional para aceleración)
- `datasets`, `tokenizers`, `pyyaml`, `tqdm`, `numpy`

Instalar con: `pip install -r requirements.txt`
El entorno actual ya incluye torch 2.10+cu130 (RTX 4050, CUDA disponible).

## Estructura (§36)

```
compressed-llm/
  configs/       # configuraciones YAML (base, v0, baseline, 4x..64x)
  compressed_llm/  # paquete raíz reutilizable (config, utils)
  data/          # pipeline de datos (download, filter, preprocess, split, inspect)
  tokenizer/     # wrapper BPE convencional
  compressor/    # encoder ligero + bottleneck cross-attention
  model/         # decoder LLM causal + CompressedLLM
  losses/        # generation, preference, information
  training/      # trainer SFT, CLI, overfit test
  evaluation/    # latent_usage, IRS, factual, generation, preference, benchmark
  inference/     # generación autoregresiva
  tests/         # tests obligatorios de shapes/gradients (§38)
  experiments/   # salidas por fase (v0..v3)
  results/       # métricas, configs, checkpoints, plots
```

## Importación

Todos los comandos se ejecutan **desde `compressed-llm/`**. Los imports son
absolutos por paquete (`compressed_llm`, `model`, `compressor`, `tokenizer`,
`training`, `data`, `evaluation`, `inference`, `losses`).

## Prueba rápida (shapes + gradientes)

```bash
cd compressed-llm
python -m pytest tests/test_bottleneck.py -q
```

## Overfit test (§40)

```bash
cd compressed-llm
python -m training.overfit_test --config configs/v0.yaml --examples 32 --steps 300
```

## Pipeline de datos (§49)

```bash
cd compressed-llm
python -m data.build_dataset --out-dir data/out
```

## Entrenamiento SFT

```bash
cd compressed-llm
python -m training.train_sft --config configs/v0.yaml \
    --train data/out/train --valid data/out/valid --out results/v0
```

## Estado

Esqueleto completo implementado (configs, data, tokenizer, compressor, model,
losses, training, evaluation, inference, tests). Pendiente: tokenizer real,
build de datos, entrenamiento V0, escalera de compresión y evaluación (§ más
abajo en `PROJECT_SPEC.md`).