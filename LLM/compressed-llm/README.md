# Compressed LLM

LLM desde cero cuyo contexto de entrada se transforma por un **bottleneck de representaciones continuas (continuous latents)** antes del decoder LLM autoregresivo. Dataset: `lmarena-ai/arena-human-preference-100k` (español).

Especificación completa en [`PROJECT_SPEC.md`](PROJECT_SPEC.md) y `../COMPRESSED_LLM_PROJECT.md`.

## Hipótesis

> Una conversación puede representarse con muchos menos vectores continuos que tokens convencionales, conservando la información necesaria para generar respuestas de alta calidad.

## Arquitectura

```text
contexto -> BPE -> N tokens -> encoder local -> K latents continuos -> decoder causal -> respuesta
```

## Requisitos

- Python 3.10+
- PyTorch 2.x (CUDA opcional)
- `datasets`, `tokenizers`, `pyyaml`, `tqdm`, `pytest`, `numpy`

Instalar:

```bash
pip install -r requirements.txt
```

## Entrenamiento en un solo comando

Desde `LLM/compressed-llm/`:

```bash
python train.py --config v0
```

El pipeline hace automáticamente:

```text
Arena streaming
  -> Spanish
  -> preprocess chosen/rejected
  -> BPE real
  -> train/valid/test
  -> pytest
  -> overfit 32 ejemplos
  -> V0 256 -> 64 latents
```

Por defecto procesa hasta 2048 filas Spanish y entrena un BPE de hasta 8192 tokens. El dataset completo NO se materializa durante este flujo.

Opciones útiles:

```bash
python train.py --config v0 --rows 512 --overfit-steps 100 --steps 500
python train.py --config v0 --device cuda
python train.py --config v0 --clean
```

El tokenizer real queda en `tokenizer/tokenizer.json`, los splits en `data/out/` y los resultados en `results/v0/`.

## Estado

Implementados:

- tokenizer BPE real
- pipeline Arena Spanish por streaming
- chosen/rejected
- split agrupado por `question_id`
- compressor local + latent bottleneck
- decoder causal con posiciones
- SFT end-to-end
- checkpoints y reanudación
- tests de shapes/gradientes
- overfit gate antes del entrenamiento grande
- baseline y configuraciones de compresión

Pendiente como fases experimentales:

- entrenamiento V0 real y sus resultados
- escalera 4x..128x
- benchmark IRS/factual completo
- plots de calidad, retención e inferencia
- contextos 1024+ y compresión jerárquica
