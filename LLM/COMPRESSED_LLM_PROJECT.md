# Compressed LLM — Especificación y Plan de Construcción

## 0. Propósito

Construir desde cero un modelo de lenguaje conversacional cuyo contexto de entrada sea transformado por un **bottleneck de representaciones continuas (continuous latents)** antes de llegar al decoder LLM.

La hipótesis del proyecto es:

> Una conversación puede representarse con muchos menos vectores continuos que tokens convencionales, conservando la información necesaria para generar respuestas de alta calidad.

El objetivo no es simplemente resumir texto ni reconstruirlo. El objetivo es maximizar la **información útil por latent** y reducir la longitud contextual que procesa el decoder.

---

## 1. Dataset principal

Dataset:

`lmarena-ai/arena-human-preference-100k`

Referencia:

`https://huggingface.co/datasets/lmarena-ai/arena-human-preference-100k/viewer/default/train?f%5Blanguage%5D%5Bvalue%5D=%27Spanish%27&p=1`

### Filtro obligatorio

Usar inicialmente exclusivamente:

```python
example["language"] == "Spanish"
```

### Campos relevantes

- `question_id`
- `model_a`
- `model_b`
- `winner`
- `conversation_a`
- `conversation_b`
- `turn`
- `language`
- `is_code`
- `is_refusal`
- `category_tag`
- `dedup_tag`

### Interpretación

`conversation_a` y `conversation_b` contienen respuestas alternativas al mismo contexto. `winner` contiene la preferencia humana.

Valores de `winner` pueden incluir:

- `model_a`
- `model_b`
- `tie`
- `both_bad`

No tratar `tie` ni `both_bad` como preferencia binaria normal.

### Restricciones del dataset

El dataset es principalmente un dataset de conversaciones/preferencias, no un corpus completo de preentrenamiento general. Sirve para validar la arquitectura de compresión y aprendizaje conversacional, no para afirmar que el modelo adquiere conocimiento general comparable con un LLM grande.

Revisar siempre las condiciones de licencia del dataset y, especialmente, los términos aplicables a las salidas de los modelos proveedores.

---

# 2. Objetivo científico

Responder experimentalmente:

> ¿Cuántos tokens convencionales pueden sustituirse por cuántos latents continuos antes de perder información útil para responder?

Y medir:

1. calidad de respuesta;
2. retención de información;
3. capacidad factual;
4. preferencia humana;
5. coste computacional;
6. latencia;
7. memoria;
8. densidad de información por latent;
9. escalabilidad a contextos largos.

---

# 3. Concepto fundamental: latents continuos

Los latents **NO son palabras ni tokens discretos**.

Ejemplo conceptual:

```text
Texto:
"Chilla's Art fue fundado alrededor de 2018 por Yasuka Taira..."

                    ↓

z1 = vector continuo
z2 = vector continuo
z3 = vector continuo
...
z32 = vector continuo
```

Cada latent puede codificar información distribuida y mezclada. No asumir correspondencia 1:1 entre latent y concepto humano.

Los latents deben poder contener relaciones semánticas, entidades, hechos, contexto y señales necesarias para producir la respuesta.

---

# 4. Arquitectura objetivo

```text
                    CONVERSACIÓN
                          │
                          ▼
                   Tokenizador BPE
                          │
                          ▼
                       N tokens
                          │
                          ▼
                ┌──────────────────┐
                │ Encoder ligero   │
                │ local / barato   │
                └────────┬─────────┘
                         │
                         ▼
                K continuous latents
                         │
                         ▼
                ┌──────────────────┐
                │  Decoder LLM     │
                │ autoregresivo    │
                └────────┬─────────┘
                         │
                         ▼
                     respuesta
```

El decoder NO debe ejecutar self-attention sobre todos los tokens originales del contexto.

---

# 5. Restricción de eficiencia

El compressor no debe ser tan grande que anule la ganancia.

Evitar:

```text
N tokens
   ↓
Transformer grande O(N²)
   ↓
K latents
```

Preferir un encoder ligero y/o local, seguido de cross-attention de latent queries.

Una posibilidad conceptual:

```text
input embeddings
      ↓
local encoder / conv / local attention
      ↓
encoded sequence
      ↓
K learned latent queries
      ↓
cross-attention
      ↓
K latents
```

Con `K << N`, la etapa de bottleneck puede aproximarse a un coste proporcional a `N*K` en la cross-attention, en lugar de requerir self-attention global sobre todos los tokens.

---

# 6. Arquitectura V0 recomendada

Objetivo: prueba de concepto, no modelo final.

### Input

- máximo inicial: 256 tokens;
- después: 512 y 1024;
- tokenizer subword existente y estable.

### Encoder

- embedding: 256–384 dim;
- 2 capas ligeras;
- local attention, convolución o equivalente eficiente;
- evitar self-attention global caro.

### Bottleneck

- 8 / 16 / 32 / 64 learned latent queries según experimento;
- latent dimension igual o cercana al hidden size del decoder;
- 1–2 bloques de cross-attention en V0.

### Decoder

- aproximadamente 5–30M parámetros según experimento;
- 4–6 capas;
- hidden 256–384;
- 4–8 heads;
- causal autoregressive generation.

El decoder debe aceptar los latents como contexto/prefix continuo y luego generar tokens de respuesta.

---

# 7. Flujo de entrenamiento

La primera versión debe entrenarse end-to-end.

```text
contexto
  ↓
tokenizer
  ↓
encoder
  ↓
latent bottleneck
  ↓
continuous latents
  ↓
decoder LLM
  ↓
respuesta
  ↓
loss
  ↓
backpropagation
```

El compressor y el decoder deben aprender conjuntamente.

---

# 8. Objetivo principal: respuesta, no reconstrucción literal

No utilizar como objetivo principal:

```text
contexto → latents → reconstruir exactamente el contexto
```

Esto puede gastar capacidad recordando detalles superficiales como puntuación, orden exacto o redacción.

Objetivo principal:

```text
contexto → latents → generar una respuesta correcta/útil
```

Una reconstruction loss puede existir como experimento auxiliar, pero no debe dominar el entrenamiento.

---

# 9. Pérdidas

Base conceptual:

```text
L_total = L_generation + λ1 * L_preference + λ2 * L_information
```

## 9.1 Generation loss

Causal language modeling loss sobre la respuesta objetivo.

Para SFT:

```text
contexto → compressor → latents → respuesta elegida
```

Aplicar cross-entropy únicamente a los tokens objetivo de la respuesta, no al contexto, salvo que un experimento específico lo requiera.

## 9.2 Preference loss

Cuando existe winner válido:

```text
chosen > rejected
```

Puede usarse una pérdida tipo Bradley-Terry / logistic preference:

```text
L_pref = -log(sigmoid(score(chosen) - score(rejected)))
```

La implementación exacta puede cambiar, pero debe conservar el principio de preferencia relativa.

## 9.3 Information loss

Debe diseñarse experimentalmente. Posibles objetivos:

- recuperación de hechos;
- contraste entre ejemplos;
- pregunta-respuesta sobre contexto;
- clasificación de atributos presentes en el contexto;
- contrastive learning.

No asumir desde el principio una loss compleja. Primero demostrar que el bottleneck aprende con `L_generation`.

---

# 10. Uso de A/B del dataset

Conservar ambas respuestas.

```text
                    mismo contexto
                         │
                 ┌───────┴───────┐
                 ▼               ▼
          respuesta A       respuesta B
                 │               │
                 └───────┬───────┘
                         ▼
                       winner
```

### Caso `model_a`

- chosen = A
- rejected = B

### Caso `model_b`

- chosen = B
- rejected = A

### Caso `tie`

No usar como preferencia binaria.

Puede conservarse como ejemplo de SFT especial o excluirse en la primera versión.

### Caso `both_bad`

Excluir del entrenamiento SFT estándar y de preference loss binaria de la V0.

---

# 11. Formato interno recomendado

Después de preprocessing, producir algo como:

```json
{
  "question_id": "...",
  "language": "Spanish",
  "context": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "chosen": "...",
  "rejected": "...",
  "winner": "model_a",
  "turn": 1,
  "metadata": {
    "is_code": false,
    "is_refusal": false,
    "category_tag": "..."
  }
}
```

El esquema final puede simplificarse según implementación.

---

# 12. Limpieza del dataset

Pipeline recomendado:

```text
Arena 100k
   ↓
language == Spanish
   ↓
validar estructura
   ↓
quitar ejemplos corruptos
   ↓
quitar duplicados usando metadata disponible
   ↓
separar casos tie/both_bad
   ↓
crear chosen/rejected
   ↓
split por conversación/question_id
```

No crear train/test de forma que partes relacionadas de una misma conversación terminen en splits diferentes.

---

# 13. Split

Primera configuración:

- 80% train
- 10% validation
- 10% test

El split debe hacerse de forma agrupada por `question_id` o identificador equivalente para reducir data leakage.

---

# 14. Experimento principal de compresión

Mantener arquitectura y procedimiento lo más constantes posible y cambiar principalmente el número de latents.

### Baseline

```text
N tokens → decoder
```

### 4×

```text
1024 → 256 latents
```

### 8×

```text
1024 → 128
```

### 16×

```text
1024 → 64
```

### 32×

```text
1024 → 32
```

### 64×

```text
1024 → 16
```

### 128×

```text
1024 → 8
```

No asumir que todas las configuraciones serán viables. Ejecutarlas como una escalera experimental.

---

# 15. V0 experimental pequeña

Primera prueba recomendada:

```text
Input:       256 tokens
Latents:      64
Decoder:    ~10M params
```

Después:

```text
256 → 32
256 → 16
256 → 8
```

Objetivo de V0:

> comprobar que el bottleneck continuo puede conservar suficiente información para que el decoder aprenda a responder.

No intentar todavía demostrar superioridad computacional.

---

# 16. Escalera de experimentos

## V0

```text
256 → 64/32/16/8
```

## V1

```text
512 → 128/64/32/16
```

## V2

```text
1024 → 128/64/32/16
```

## V3

```text
2048 → 128/64/32
```

## V4

```text
4096 → 256/128/64
```

Solo avanzar cuando la fase previa sea estable.

---

# 17. Baseline obligatorio

Construir un baseline sin compresión.

```text
contexto → tokenizer → decoder → respuesta
```

y compararlo contra:

```text
contexto → tokenizer → compressor → latents → decoder → respuesta
```

Controlar, en la medida de lo posible:

- dataset;
- optimizer;
- training steps;
- presupuesto de parámetros;
- hidden size;
- tamaño de respuesta objetivo;
- semillas;
- evaluación.

La comparación no debe basarse únicamente en "mi modelo es más pequeño".

---

# 18. Métricas obligatorias

Registrar por experimento:

```text
input_tokens
latent_tokens
compression_ratio
output_tokens
train_loss
validation_loss
perplexity (cuando sea apropiada)
latency_ms
throughput_tokens_per_second
VRAM_MB
estimated_FLOPs
parameter_count
training_steps
```

---

# 19. Information Retention Score (IRS)

Crear un benchmark específico de retención de información.

Idea:

```text
IRS = preguntas respondidas correctamente / preguntas evaluadas
```

Ejemplos de preguntas:

- nombres;
- fechas;
- cantidades;
- entidades;
- relaciones;
- hechos explícitos;
- detalles pequeños.

No depender solamente de una métrica automática. Guardar también ejemplos cualitativos de fallos.

---

# 20. Benchmark factual personalizado

Generar un conjunto de preguntas a partir del contexto original.

Ejemplo:

```text
Contexto:
"Chilla's Art fue fundado alrededor de 2018..."

Pregunta:
"¿Cuándo fue fundado?"

Respuesta esperada:
"Alrededor de 2018."
```

Comparar:

```text
contexto completo → respuesta
```

vs.

```text
latents únicamente → respuesta
```

Esto mide si la compresión destruyó información recuperable.

---

# 21. Preguntas adversariales

Crear preguntas que exijan detalles fáciles de perder:

- ¿qué número se mencionó?
- ¿qué nombre concreto apareció?
- ¿qué opción ocupaba la segunda posición?
- ¿se mencionó X?
- ¿qué ocurrió antes de Y?
- ¿qué palabra/entidad fue usada en relación con Z?

El objetivo es detectar compresión que mantiene la idea general pero destruye detalles.

---

# 22. Calidad de generación

Evaluar respuestas con:

1. loss/perplexity;
2. evaluación automática;
3. comparación contra respuesta elegida;
4. preferencia A/B cuando sea posible;
5. evaluación humana en una muestra pequeña.

Nunca usar una sola métrica como prueba definitiva.

---

# 23. Curva principal del proyecto

Crear gráficas de:

```text
Quality vs Compression Ratio
IRS vs Compression Ratio
Latency vs Compression Ratio
VRAM vs Compression Ratio
FLOPs vs Compression Ratio
Information Density vs Latent Count
```

La gráfica más importante:

```text
               QUALITY
                  │
100% ─────────────┤
                  │
                  │
                  │
                  └──────────────────────
                    4x 8x 16x 32x 64x
                         compression
```

Buscar el punto donde la calidad empieza a caer de forma abrupta.

---

# 24. Information density

Registrar una métrica aproximada de densidad:

```text
information_density = retained_information / number_of_latents
```

No asumir que el score es una unidad física de información. Es una métrica experimental para comparar configuraciones.

---

# 25. Contexto largo

El beneficio de la compresión debería crecer con la longitud del contexto.

Después de V0/V1, probar:

```text
1024 tokens
2048 tokens
4096 tokens
8192 tokens
```

Con bottlenecks mucho menores que N.

---

# 26. Compresión jerárquica futura

No implementar en V0 salvo que sea necesario.

Posible arquitectura futura:

```text
4096 tokens
     ↓
chunks
     ↓
local latents
     ↓
global compressor
     ↓
global latents
     ↓
LLM
```

Ejemplo:

```text
4096 tokens
 ↓
256 local latents
 ↓
32 global latents
 ↓
decoder
```

Esto puede ayudar a escalar a contextos muy largos.

---

# 27. Consideración crítica sobre el tokenizador

En la primera versión **sí usar un tokenizer BPE/subword convencional para la entrada**.

Esto no contradice la investigación.

Queremos aislar primero la hipótesis:

> "¿Funciona el bottleneck continuo?"

Después se puede investigar:

```text
bytes/chars → encoder ligero → continuous latents → LLM
```

Pero empezar directamente desde bytes mezcla dos problemas distintos:

1. representación de entrada;
2. compresión contextual.

Primero demostrar el segundo.

---

# 28. Qué significa "desde cero"

El decoder LLM debe entrenarse desde cero para este experimento, no limitarse a insertar un compressor delante de un LLM ya preentrenado.

Sin embargo, el tokenizer de entrada puede ser un tokenizer convencional en la V0 para evitar confundir el experimento con un proyecto de tokenización desde bytes.

El objetivo de la investigación es que el **decoder aprenda a operar sobre representaciones continuas comprimidas desde el principio del entrenamiento**.

---

# 29. Evitar colapso del bottleneck

Posibles problemas:

- todos los latents se vuelven similares;
- el decoder ignora parte de los latents;
- algunos latents nunca contienen información útil;
- el compressor aprende una representación demasiado pobre;
- el decoder memoriza patrones superficiales.

Instrumentar:

- norma media de cada latent;
- varianza por dimensión;
- similitud coseno entre latents;
- utilización de cada latent;
- gradientes del bottleneck;
- attention weights del latent cross-attention;
- sensibilidad de la respuesta al enmascaramiento de latents.

---

# 30. Prueba de utilización de latents

Para saber si todos los latents sirven, ocultar uno o varios durante inferencia:

```text
z1 z2 z3 ... z32
```

convertir, por ejemplo:

```text
z17 = 0
```

y medir cambio de calidad.

Esto permite detectar latents muertos o redundantes.

---

# 31. Otra prueba crucial: información perturbada

Tomar un contexto y comparar:

```text
contexto original → latents
```

contra:

```text
contexto ligeramente alterado → latents
```

Las respuestas deberían cambiar cuando cambia información relevante.

Ejemplo:

```text
2018 → 2019
```

Si la respuesta sigue afirmando 2018, la representación puede estar ignorando un detalle.

---

# 32. Presupuesto computacional

La eficiencia debe analizarse por separado en:

### Entrenamiento

El compressor todavía debe leer el contexto completo.

### Inferencia

El gran decoder opera sobre K latents y no N tokens de contexto.

Por tanto, el proyecto busca principalmente:

- reducir coste de procesamiento contextual del decoder;
- reducir memoria contextual;
- permitir contextos más largos;
- mejorar throughput/latencia cuando N es muy grande.

No afirmar que la compresión reduce el coste total por un factor igual al ratio de compresión.

---

# 33. Regla de interpretación computacional

Nunca decir:

> "1024 → 32 significa 32× menos FLOPs totales."

Decir:

> "1024 → 32 reduce aproximadamente 32× la longitud contextual procesada por la etapa que consume los latents, aunque el ahorro total depende de arquitectura, atención, MLP, encoder, proyecciones y generación."

En términos de pares de atención, pasar de N a K cambia un término cuadrático de `N²` a `K²` cuando corresponde a self-attention sobre esas secuencias, pero no debe interpretarse como ahorro total del sistema.

---

# 34. Entrenamiento en hardware modesto

V0 debe ser compatible con una GPU de aproximadamente 6–8 GB de VRAM.

Recomendaciones:

- mixed precision;
- batch pequeño;
- gradient accumulation;
- sequence length moderado;
- checkpointing si hace falta;
- evitar modelos innecesariamente grandes;
- logging frecuente pero no excesivo.

La primera meta es una prueba reproducible, no máxima escala.

---

# 35. Reproducibilidad

Cada experimento debe guardar:

```text
config.json
seed
model_size
latent_count
max_input_tokens
learning_rate
batch_size
gradient_accumulation
training_steps
dataset_revision
code/git commit si existe
metrics
checkpoint path
```

No mezclar resultados de configuraciones distintas sin registrar sus parámetros.

---

# 36. Estructura de proyecto sugerida

```text
compressed-llm/
│
├── README.md
├── PROJECT_SPEC.md
├── requirements.txt
├── configs/
│   ├── v0.yaml
│   ├── 4x.yaml
│   ├── 8x.yaml
│   ├── 16x.yaml
│   ├── 32x.yaml
│   └── 64x.yaml
│
├── data/
│   ├── download.py
│   ├── filter_spanish.py
│   ├── preprocess.py
│   ├── split.py
│   └── inspect.py
│
├── tokenizer/
│   └── tokenizer.py
│
├── compressor/
│   ├── encoder.py
│   ├── latent_queries.py
│   └── bottleneck.py
│
├── model/
│   ├── decoder.py
│   └── compressed_llm.py
│
├── losses/
│   ├── generation.py
│   ├── preference.py
│   └── information.py
│
├── training/
│   ├── train_sft.py
│   ├── train_preference.py
│   └── trainer.py
│
├── evaluation/
│   ├── generation.py
│   ├── factual.py
│   ├── preference.py
│   ├── information_retention.py
│   ├── latent_usage.py
│   └── benchmark.py
│
├── inference/
│   └── generate.py
│
├── experiments/
│   ├── v0/
│   ├── v1/
│   ├── v2/
│   └── v3/
│
└── results/
    ├── metrics.csv
    ├── configs/
    ├── checkpoints/
    └── plots/
```

---

# 37. Requisitos técnicos mínimos del código

El código debe:

- usar PyTorch;
- permitir CPU/GPU automáticamente;
- permitir cambiar `N` y `K` desde configuración;
- guardar checkpoints;
- reanudar entrenamiento;
- registrar métricas;
- poder ejecutar evaluación separadamente;
- evitar hardcodear rutas locales;
- validar shapes de tensores;
- incluir tests básicos para el bottleneck;
- documentar cada módulo.

Preferir dependencias maduras y pequeñas.

---

# 38. Tests obligatorios antes de entrenar

Crear tests para comprobar:

1. input `[B, N]` llega correctamente al encoder;
2. encoder produce `[B, N, D]`;
3. bottleneck produce `[B, K, D]`;
4. decoder acepta `[B, K, D]`;
5. output tiene shape correcta;
6. gradients llegan al encoder y latents;
7. cambiar K cambia correctamente las dimensiones;
8. batch padding/attention masks funcionan;
9. causal mask del decoder funciona;
10. checkpoint save/load conserva pesos.

---

# 39. Primer entregable de implementación

NO empezar entrenando inmediatamente.

Primero construir:

### Paso A

`download.py`

Descarga o carga el dataset de Hugging Face.

### Paso B

`filter_spanish.py`

Aplica `language == "Spanish"`.

### Paso C

`preprocess.py`

Convierte A/B + winner en registros `context/chosen/rejected`.

### Paso D

`split.py`

Hace split agrupado por `question_id`.

### Paso E

`inspect.py`

Muestra:

- número de filas;
- distribución de winners;
- longitud de contexto;
- longitud de respuestas;
- distribución de turnos;
- ejemplos aleatorios;
- porcentaje de code/refusal.

### Paso F

Construir un batch sintético y verificar shapes.

### Paso G

Construir el modelo V0.

### Paso H

Hacer overfit deliberado sobre un dataset diminuto.

---

# 40. Overfit test

Antes del entrenamiento completo, usar un conjunto muy pequeño, por ejemplo 32 ejemplos.

Esperar que el modelo pueda sobreajustar fuertemente.

Si NO puede sobreajustar, no continuar al entrenamiento grande.

Posibles causas:

- bug en máscaras;
- bug en labels;
- latents desconectados;
- gradients nulos;
- loss mal construida;
- datos mal preparados.

---

# 41. Primer experimento científico

Una vez que V0 pase los tests:

```text
Dataset: Spanish
Input max: 256
Decoder: ~10M
Latents: 64
```

Entrenar baseline y compressed bajo configuración comparable.

Después probar:

```text
64 latents
32 latents
16 latents
8 latents
```

Mantener fijo el resto tanto como sea razonable.

---

# 42. Qué debe producir cada experimento

Cada experimento debe terminar con:

```text
config.json
checkpoint
metrics.json
metrics.csv
sample_outputs.jsonl
plots/
README.md o report.md
```

`sample_outputs.jsonl` debe incluir ejemplos con:

- contexto;
- respuesta esperada;
- respuesta del baseline;
- respuesta comprimida;
- número de tokens;
- número de latents;
- score de evaluación si existe.

---

# 43. Ejemplo de configuración

```yaml
seed: 42

model:
  vocab_size: 32000
  hidden_size: 256
  num_layers: 4
  num_heads: 4
  intermediate_size: 1024

compressor:
  encoder_layers: 2
  latent_count: 32
  latent_dim: 256
  cross_attention_layers: 2
  local_window: 64

training:
  max_input_tokens: 256
  max_output_tokens: 256
  batch_size: 2
  gradient_accumulation: 16
  learning_rate: 3e-4
  warmup_steps: 100
  max_steps: 10000
  mixed_precision: fp16

loss:
  preference_weight: 0.1
  information_weight: 0.0
```

Los valores son iniciales y deben poder cambiarse fácilmente.

---

# 44. Resultado esperado de la investigación

El éxito del proyecto NO es simplemente producir un chatbot.

Éxito significa poder cuantificar una relación como:

```text
compression ratio
        vs
quality
        vs
information retention
        vs
compute
```

Idealmente descubrir algo como:

```text
1024 tokens → 32 latents

calidad ≈ baseline
IRS ≈ baseline
coste contextual << baseline
```

Pero **no asumir que ocurrirá**. El experimento debe determinarlo.

---

# 45. Criterio para declarar que la hipótesis funciona

Una conclusión positiva necesita evidencia de que:

1. el decoder depende realmente de los latents;
2. los latents retienen información factual;
3. el rendimiento no cae catastróficamente con compresión significativa;
4. el encoder no consume más computación de la que se ahorra de manera que anule el beneficio;
5. los resultados se reproducen en distintas semillas o al menos en más de una ejecución cuando sea factible;
6. el baseline está correctamente implementado.

---

# 46. Posibles extensiones futuras

No implementar todavía salvo necesidad experimental:

- latent count adaptativo;
- routing de latents;
- MoE compressor;
- quantización de latents;
- discrete latents / VQ;
- byte-level input;
- hierarchical compression;
- memory latents persistentes;
- retrieval sobre latents;
- compresión recurrente de conversaciones;
- distillation de un LLM teacher;
- entrenamiento multiidioma;
- contextos de 16k/32k/64k tokens.

---

# 47. Filosofía del proyecto

No optimizar prematuramente.

La secuencia correcta es:

```text
          ¿funciona?
              ↓
         ¿retiene datos?
              ↓
        ¿cuánto comprime?
              ↓
       ¿cuánto cuesta?
              ↓
        ¿escala a N grande?
              ↓
      ¿es mejor que baseline?
```

Cada afirmación debe venir acompañada de una métrica o experimento.

---

# 48. Regla final para futuras IAs que continúen el proyecto

Antes de modificar la arquitectura, revisar:

1. cuál es la hipótesis que se está probando;
2. qué baseline existe;
3. qué métrica puede cambiar;
4. qué coste computacional añade el cambio;
5. si el cambio destruye la comparabilidad con experimentos anteriores.

No cambiar simultáneamente tokenizer + compressor + decoder + loss + dataset y luego llamar al resultado una mejora.

Cuando sea posible, cambiar **una variable experimental importante a la vez**.

---

# 49. Primera tarea que debe ejecutar la IA constructora

Implementar el pipeline de datos para el dataset Arena:

```text
Hugging Face
   ↓
Spanish only
   ↓
validación
   ↓
chosen/rejected
   ↓
split agrupado
   ↓
JSONL / Arrow listo para entrenamiento
```

Después implementar el modelo V0 y tests de shapes/gradients.

**No comenzar por entrenar un modelo grande.**

---

# 50. Resumen de una línea

> Construir un LLM desde cero que aprenda a responder a partir de un número pequeño de continuous latent vectors obtenidos mediante un compressor ligero, usando Arena Human Preference 100K en español para medir cuánta información útil puede comprimirse sin destruir la calidad de respuesta.
