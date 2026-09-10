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

## GPU

El CLI detecta CUDA y propone un micro-batch más alto según la VRAM. En CUDA activa FP16/AMP, transferencias `non_blocking` con `pin_memory` y TF32 cuando corresponde. La acumulación de gradiente mantiene un batch efectivo grande sin exigir toda la memoria de una sola vez.

Para la RTX 4050 de 6 GB del proyecto, el valor recomendado inicial es normalmente `micro-batch=8`; si aparece un `CUDA out of memory`, baja a 4 o 2.
