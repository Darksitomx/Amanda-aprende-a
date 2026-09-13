# Compressed LLM

Implementación del experimento **Compressed LLM**: `tokens -> encoder ligero -> K continuous latents -> decoder LLM autoregresivo`.

## Entrenamiento por etapas

Para entrenar de forma incremental usa:

```bash
python train_cli.py
```

El menú permite:

1. **Nueva etapa**: eliges cuántas filas Spanish acumuladas usar y cuántos steps adicionales entrenar.
2. **Continuar**: aumenta el dataset y reanuda desde el último `step_*.pt`, conservando pesos y estado del optimizer.
3. **Estado**: muestra dataset, tokenizer, checkpoint, step y GPU detectada.

El tokenizer BPE se entrena únicamente en la primera etapa y después se reutiliza. Esto es importante porque cambiar el vocabulario cambiaría el tamaño de la matriz de embeddings y no sería compatible con el checkpoint existente.

Durante los primeros steps, `v0` y `fast_v1` usan una destilación temporal: el decoder también consulta el contexto completo como referencia mientras aprende a responder desde los latents. Después de `distillation_steps`, se desactiva esa ruta auxiliar y el entrenamiento continúa únicamente con la entrada comprimida.

## GPU

El CLI detecta CUDA y propone un micro-batch más alto según la VRAM. En CUDA activa FP16/AMP, transferencias `non_blocking` con `pin_memory` y TF32 cuando corresponde. La acumulación de gradiente mantiene un batch efectivo grande sin exigir toda la memoria de una sola vez.

Para la RTX 4050 de 6 GB del proyecto, el valor recomendado inicial es normalmente `micro-batch=8`; si aparece un `CUDA out of memory`, baja a 4 o 2.

## Entrenamiento V0 (un solo paso)
Por defecto procesa hasta 2048 filas Spanish y entrena un BPE de hasta 8192 tokens. El dataset completo NO se materializa durante este flujo.

Opciones útiles:

```bash
python train.py --config v0 --rows 512 --overfit-steps 100 --steps 500
python train.py --config v0 --device cuda
python train.py --config v0 --clean
```

El tokenizer real queda en `tokenizer/tokenizer.json`, los splits JSONL en `data/out/` y los resultados en `results/v0/`.

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
