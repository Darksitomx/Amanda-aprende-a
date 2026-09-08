# Compressed LLM — Especificación

Este directorio contiene la implementación del proyecto **Compressed LLM**,
definido en `../LLM/COMPRESSED_LLM_PROJECT.md`.

## Hipótesis central

> Una conversación puede representarse con muchos menos vectores continuos que
> tokens convencionales, conservando la información necesaria para generar
> respuestas de alta calidad.

Arquitectura: tokens -> encoder ligero -> **K continuous latents** -> decoder LLM autoregresivo.

## Estado actual (esqueleto)

- [x] Estructura de proyecto (§36)
- [x] Configuraciones base / V0 / baseline / escalera de compresión (§14, §43)
- [x] Config loader con defaults (`compressed_llm/config.py`)
- [x] Tokenizer BPE convencional (wrapper) (§27)
- [x] Compressor: encoder ligero + bottleneck cross-attention (§5, §6)
- [x] Decoder LLM causal + CompressedLLM (§6)
- [x] Losses: generation, preference (Bradley-Terry), information (stub) (§9)
- [x] Pipeline de datos completo (download/filter/preprocess/split/inspect) (§49)
- [x] Trainer SFT + CLI + overfit test (§34, §40)
- [x] Evaluación: latent usage, IRS, factual, generation, preference (§18-22)
- [x] Tests obligatorios de shapes/gradients (§38)

## Pendiente (fases siguientes)

- [ ] Entrenar un tokenizer BPE sobre corpus real de entrenamiento
- [ ] Build final del pipeline de datos con `data/build_dataset.py`
- [ ] Entrenamiento V0 (256 -> 64 latents) + baseline (§41)
- [ ] Escalera de compresión 4x..128x (§14)
- [ ] Information Retention Score / benchmark factual completo (§19, §20)
- [ ] Plots calidad-vs-compresión (§23)

## Convención de imports

Los módulos se importan relativos desde `compressed-llm/`. El paquete raíz es
`compressed_llm`. Para ejecutar tests o scripts, correr desde dentro de
`compressed-llm/`.

## Hardware objetivo

GPU ~6-8 GB VRAM (§34). El entorno actual es una RTX 4050 Laptop (6 GB).

## Métricas a registrar por experimento (§18)

`input_tokens, latent_tokens, compression_ratio, output_tokens, train_loss,
validation_loss, perplexity, latency_ms, throughput_tokens_per_second,
VRAM_MB, estimated_FLOPs, parameter_count, training_steps`