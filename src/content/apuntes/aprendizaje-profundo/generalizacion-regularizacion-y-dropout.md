---
titulo: "Generalización, regularización y dropout"
asignatura: aprendizaje-profundo
fecha: 2026-10-30
orden: 6
tema: "Sesión 6"
tags: [generalización, validación, regularización, early-stopping, dropout, aprendizaje-profundo]
resumen: "Interpretación de curvas de entrenamiento y validación, elección de checkpoints, regularización, parada temprana y dropout para controlar la generalización y el comportamiento en evaluación."
---

## Página 1 — Generalization, regularization and dropout

**Semana 2, sesión 6**

**Horario:** viernes, 16:00–17:30.

**Pregunta central:** ¿Cómo entrenamos un modelo que siga siendo útil con nuevos datos?

**Objetivos de la sesión:**

- interpretar curvas de entrenamiento y validación;
- seleccionar y restaurar un checkpoint;
- explicar e implementar dropout invertido;
- separar el comportamiento de entrenamiento y evaluación.

**Modelo de referencia:** 2→4→3 con ReLU y tres logits de salida.

**Convención de la presentación:** la diapositiva principal abarca páginas 1–26; los apéndices de referencia van de la 27 a la 34.

## Página 2 — Pérdida de entrenamiento y riesgo esperado

Para un clasificador, la pérdida por ejemplo es:

$$
\ell(\theta; x, y) = -\log p_\theta(y \mid x).
$$

Lo que el entrenamiento puede medir es:

$$
\hat{R}_{\mathrm{train}}(\theta) = \frac{1}{N_{\mathrm{tr}}} \sum_{i\in \mathcal{D}_{\mathrm{tr}}} \ell(\theta; x_i, y_i).
$$

Es una muestra finita; los propios ejemplos también determinan los parámetros ajustados.

Lo que queremos es que funcione en:

$$
R(\theta) = \mathbb{E}_{(x,y)\sim P}[\ell(\theta; x, y)].
$$

La optimización mejora el objetivo de entrenamiento. La evaluación pregunta cómo se comporta el modelo más allá de la muestra de entrenamiento.

Una pérdida de entrenamiento baja es evidencia de ajuste a los datos observados.

## Página 3 — La brecha de generalización

Para un modelo ajustado $\hat{\theta}$, la brecha poblacional es:

$$
R(\hat{\theta}) - \hat{R}_{\mathrm{train}}(\hat{\theta}).
$$

Inspeccionamos una comparación basada en una muestra:

$$
\hat{R}_{\mathrm{val}}(\hat{\theta}) - \hat{R}_{\mathrm{train}}(\hat{\theta}).
$$

**Patrón observado** | **Pregunta a investigar**

- Ambas pérdidas siguen altas: ¿es apropiado el modelo? ¿La optimización ha tenido éxito?
- Entrenamiento mejora y validación empeora: ¿estamos ajustando detalles específicos de la muestra?
- La pérdida de validación es menor: ¿son comparables las condiciones de evaluación? ¿La muestra es más fácil?

Las curvas de aprendizaje apoyan un diagnóstico; una sola brecha no demuestra nada.

## Página 4 — Curvas de aprendizaje de un experimento pequeño

Modelo 2→4→3 con ReLU; 24 puntos de entrenamiento y 600 de validación; Adam con $\eta = 0.01$.

La figura muestra pérdida cruzada media (CE) frente a periodo de entrenamiento.

- El mínimo se alcanza en la época 83.
- En la época 83: CE de entrenamiento $= 0.341$ y CE de validación $= 0.645$.
- En la época 600: CE de entrenamiento $= 0.056$ y CE de validación $= 1.937$.

Se trata de un experimento original en NumPy con clases Gaussianas superpuestas, nueva inicialización, sin dropout ni penalización por peso. Ambas curvas usan el mismo checkpoint determinista. La ejecución completa se muestra retrospectivamente; no se puntúa un conjunto de prueba.

## Página 5 — Pérdidas de entrenamiento y validación comparables

Antes de interpretar una brecha, conviene comprobar qué mide cada curva.

| Condición | Posible diferencia |
|---|---|
| Registro de entrenamiento | Puede incluir varios estados a lo largo de una época |
| Parámetros | Un checkpoint compartido para comparación |
| Modo de red | Aleatoriedad del dropout |
| Pipeline de entrada | Aumento aleatorio |
| Cantidad reportada | Pérdida de datos + penalización |

Se debe reevaluar el conjunto de entrenamiento con las mismas condiciones que la validación.

La validación por debajo del entrenamiento puede ser válida. Hay que comprobar las definiciones.

## Página 6 — Entrenamiento, validación y prueba

| Partición | Papel principal | Qué puede influir |
|---|---|---|
| Entrenamiento | Ajustar parámetros con gradientes | Pesos, sesgos, preprocesado ajustado |
| Validación | Comparar candidatas | Arquitectura, tasa de aprendizaje, dropout, época de parada |
| Prueba | Estimar rendimiento final | Informe final tras fijar elecciones |

Puntos clave:

- Las etiquetas de validación influyen en el modelo seleccionado, aunque no haya retropropagación.
- Usar el resultado de prueba para elegir otro candidato hace que ese conjunto forme parte de la selección.
- Elegir la partición y la métrica adecuadas según el despliegue previsto.

## Página 7 — El preprocesado forma parte de la partición

Ejemplo: estandarizar la característica $j$ usando solo los datos de entrenamiento:

$$
\mu_j = \frac{1}{N_{\mathrm{tr}}}\sum_{i\in \mathcal{D}_{\mathrm{tr}}} x_{ij},
\quad
\sigma_j^2 = \frac{1}{N_{\mathrm{tr}}}\sum_{i\in \mathcal{D}_{\mathrm{tr}}}(x_{ij}-\mu_j)^2.
$$

Para cada partición, aplicar la misma transformación ajustada:

$$
 x'_{ij} = \frac{x_{ij} - \mu_j}{\max(\sigma_j,\varepsilon)}, \quad \varepsilon > 0.
$$

**Regla:** se ajustan $\mu, \sigma$ con los datos de entrenamiento y se transforma entrenamiento/validación/prueba con la misma regla.

Esto se aplica también a imputaciones aprendidas, selección de características y ajuste de vocabulario.

## Página 8 — Fugas de información y unidad de independencia

Pregunta clave: ¿qué será nuevo cuando se despliegue el modelo?

| Objetivo de despliegue | División engañosa | Principio de partición adecuado |
|---|---|---|
| Nuevas personas o pacientes | Registros de una persona en ambos conjuntos | Mantener a cada persona en una partición |
| Nuevos vídeos | Frames adyacentes repartidos aleatoriamente | Separar vídeos o grabaciones originales |
| Nuevos documentos | Duplicados cercanos entre particiones | Agrupar duplicados y fuentes relacionadas |
| Observaciones futuras | Entrenamiento usa información posterior | Respetar tiempo y disponibilidad de características |

Dividir filas no basta si las filas comparten información.

## Página 9 — La validación guía la selección del modelo

Se sintoniza arquitectura, tasa de aprendizaje, weight decay, tasa de dropout y duración.

1. Fijar la tarea de predicción, particiones y métrica de selección.
2. Ajustar cada candidato con los datos de entrenamiento; compararlos con validación.
3. Elegir el modelo y el checkpoint usando ese criterio.
4. Evaluar el procedimiento seleccionado sobre el conjunto de prueba no usado.

**Dos límites:**

- muchas pruebas adaptativas pueden sobreajustar la validación;
- una sola semilla mezcla el efecto del método con la aleatoriedad del entrenamiento.

Tratar $p_{\mathrm{drop}} = 0$ como baseline útil; elegir tasas empíricamente.

## Página 10 — Regularización tras AdamW

| Mecanismo | Qué cambia |
|---|---|
| Penalización por peso | El objetivo: $J(\theta) = \hat{R}_{\mathrm{train}}(\theta) + \frac{\lambda}{2}\|w\|_2^2$ |
| AdamW | El update: contracción desacoplada de pesos seleccionados |
| Early stopping | Cantidad de entrenamiento y checkpoint seleccionado |
| Dropout | El cálculo de entrenamiento mediante máscaras aleatorias |

Aquí $w$ contiene 20 pesos de matriz; los 7 sesgos quedan excluidos por decisión explícita del modelado.

Cada mecanismo cambia el aprendizaje; su beneficio es una pregunta empírica.

## Página 11 — La pérdida que optimizamos y la métrica que reportamos

Una penalización explícita por peso da un objetivo de entrenamiento:

$$
J(\theta)=\frac{1}{N_{\mathrm{tr}}}\sum_{i\in\mathcal{D}_{\mathrm{tr}}}\ell(\theta;x_i,y_i) + \frac{\lambda}{2}\|w\|_2^2.
$$

Para una medida comparable de ajuste predictivo, reportamos:

$$
CE_{S}(\theta) = \frac{1}{|S|}\sum_{i\in S} -\log p_\theta(y_i\mid x_i), \quad S \in \{\mathcal{D}_{\mathrm{tr}}, \mathcal{D}_{\mathrm{val}}\}.
$$

- Usar el mismo modelo fijo, modo de evaluación y preprocesado determinista.
- Informar la penalización por separado si ayuda a diagnosticar el entrenamiento.
- En Keras, `val_loss` puede incluir pérdidas de regularización de capas.

## Página 12 — Early stopping y el mejor checkpoint

Se monitoriza la CE de validación después de cada época; cuanto menor mejor.

1. Si la métrica mejora, guardar una copia de los pesos.
2. Contar épocas consecutivas sin mejora.
3. Detener cuando el contador llega a `patience`.
4. Restaurar el mejor checkpoint aceptado para evaluar.

**Nuestra convención:**

$$
\text{min\_delta}=0: \text{una mejora significa } CE_{\mathrm{actual}} < CE_{\mathrm{best}}.
$$

En la curva observada, con `patience = 30`, se detiene en la época 113 y se restaura la época 83. La curva completa de 600 épocas se conserva para diagnóstico.

## Página 13 — Tu turno: ¿qué checkpoint sobrevive?

Secuencia construida en clase — CE de validación.

| Época | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CE | 0.900 | 0.700 | 0.620 | 0.610 | 0.615 | 0.618 | 0.617 |
| contador de espera | ? | ? | ? | ? | ? | ? | ? |

Con `mode="min"`, `patience = 3`, `min_delta = 0`, monitorizando desde la época 1.

1. ¿Qué épocas sustituyen al checkpoint guardado?
2. ¿Después de qué época termina el entrenamiento?
3. ¿Qué modelo debería evaluarse después?

## Página 14 — Solución del checkpoint y restauración

| Época | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CE | 0.900 | 0.700 | 0.620 | 0.610 | 0.615 | 0.618 | 0.617 |
| contador de espera | 0 | 0 | 0 | 0 | 1 | 2 | 3 |
| Guardar pesos | Sí | Sí | Sí | Sí | No | No | No |

- Se para después de la época 7.
- Se restauran los pesos guardados en la época 4.

Para reanudar el entrenamiento, puede ser necesario restaurar también el optimizador, el generador aleatorio y el orden de los datos.

Una referencia a pesos mutables no es un checkpoint guardado.

## Página 15 — Dropout como máscara aleatoria

La red ahora usa una operación estocástica después de la ReLU oculta.

$$
 h \in \mathbb{R}^4, \qquad \tilde{h} \in \mathbb{R}^4, \qquad s \in \mathbb{R}^3, \qquad \ell \in \mathbb{R}.
$$

Con probabilidad $p$, se elimina una activación; $q = 1-p$ es la probabilidad de mantenerla. Se muestrea una máscara $m_j \sim \mathrm{Bernoulli}(q)$.

- La máscara se muestrea durante el entrenamiento; las coordenadas son independientes aquí.
- Las activaciones descartadas se vuelven cero en ese paso. Los pesos siguen siendo parámetros.
- Dropout no añade parámetros entrenables; este modelo aún tiene 27.

## Página 16 — Dropout invertido

Durante el entrenamiento:

$$
\tilde{h} = m \odot \frac{h}{q}, \qquad 0 < q \le 1.
$$

Las activaciones mantenidas se multiplican por $1/q$.

Durante la evaluación:

$$
\tilde{h} = h.
$$

La capa dropout es la identidad; no se aplica escala adicional.

Si $h_j = 2$ y $q=0.5$, la salida de entrenamiento es:

$$
\tilde{h}_j =
\begin{cases}
0,& \text{con probabilidad } 0.5,\\
4,& \text{con probabilidad } 0.5.
\end{cases}
$$

En las APIs, `p = 0.2` o `rate = 0.2` significa $q = 0.8$.

## Página 17 — ¿Por qué dividir por la probabilidad de mantener?

Fijando la activación actual $h_j$, promediamos solo sobre la máscara.

$$
\mathbb{E}[\tilde{h}_j \mid h_j] = h_j \cdot q \cdot \mathbb{E}[m_j] = h_j q \cdot q = h_j.
$$

Sin esa escala:

$$
\mathbb{E}[m_j h_j \mid h_j] = q h_j.
$$

**Comprobación numérica:** si $h_j = 2$ y $q=0.5$:

$$
\mathbb{E}[\tilde{h}_j] = 0.5\cdot 0 + 0.5\cdot 4 = 2.
$$

- La igualdad se cumple en esperanza, no para cada máscara.
- Un vector muestreado no tiene por qué tener la misma suma ni la misma norma que el original.
- Operaciones no lineales posteriores requieren su propio análisis.

## Página 18 — La varianza introducida por dropout

Con $h_j$ fijado y $\mathrm{Var}(m_j) = q(1-q) = q p$:

$$
\mathrm{Var}(\tilde{h}_j\mid h_j) = h_j^2 q^2 \cdot q p = p q h_j^2.
$$

| $h_j$ | $p$ | $q$ | Media | Varianza |
|---:|---:|---:|---:|---:|
| 2 | 0 | 1 | 2 | 0 |
| 2 | 0.2 | 0.8 | 2 | 1 |
| 2 | 0.5 | 0.5 | 2 | 4 |

A $h_j$ fijo, una tasa de descarte mayor inyecta más varianza en la activación.

Medias iguales pueden convivir con cálculos estocásticos muy distintos.

## Página 19 — La retropropagación usa la misma máscara

Una vez muestreada la máscara hacia delante, se trata como fija al calcular el gradiente hacia atrás.

$$
\tilde{h} = m \odot \frac{h}{q}
\quad\Longrightarrow\quad
\frac{\partial \ell}{\partial h} = \frac{m}{q} \odot \frac{\partial \ell}{\partial \tilde{h}}.
$$

- Si $m_j = 0$, la gradiente local de esa activación es cero en ese paso.
- Si $m_j = 1$, la gradiente de entrada se multiplica por $1/q$.
- Usar una nueva máscara en backward diferenciaría un cálculo distinto.

La regla de la cadena sigue la de la sesión 3.

## Página 20 — Paso enmascarado a través de nuestro clasificador

**Parámetros fijos originales**

$$
W_1 =
\begin{pmatrix}
1 & 0 \\
0 & 1 \\
1 & -1 \\
-1 & 1
\end{pmatrix},
\quad
b_1 =
\begin{pmatrix}
0 \\
0 \\
0 \\
-2
\end{pmatrix}
$$

$$
W_2 =
\begin{pmatrix}
1 & 0 & 1 & 0 \\
0 & 1 & 0 & 1 \\
-1 & -1 & 1 & 1
\end{pmatrix},
\quad
b_2 =
\begin{pmatrix}
0 \\
1 \\
0
\end{pmatrix}.
$$

Entrada:

$$
 x = (2,1)^\top, \quad y = 0.
$$

**Paso hacia delante con $q=0.5$**

$$
 z = W_1 x + b_1 = (2,1,1,-3)^\top,
$$

$$
 h = \mathrm{ReLU}(z) = (2,1,1,0)^\top,
$$

$$
 m = (1,0,1,1)^\top,
$$

$$
\tilde{h} = (4,0,2,0)^\top,
$$

$$
 s = W_2\tilde{h} + b_2 = (6,1,-2)^\top.
$$

La probabilidad de clase 0 es aproximadamente:

$$
 p_0 \approx 0.992976,
$$

y la pérdida es:

$$
\ell \approx 0.007049.
$$

Sin dropout se tendría $s = (3,2,-2)^\top$ y $\ell \approx 0.318175$.

## Página 21 — Media de activaciones y media de predicciones

Para este vector oculto fijo y una cabeza afín de salida,

$$
\mathbb{E}[\tilde{h}] = h,
\quad
\mathbb{E}[s] = W_2 h + b_2,
\quad
\mathbb{E}[\mathrm{softmax}(s)] \ne \mathrm{softmax}(\mathbb{E}[s]).
$$

La figura compara las probabilidades deterministas con la media sobre máscaras para $q=0.8$ y $q=0.5$. Con etiqueta $y=0$, la media exacta de las probabilidades. Con $h_4 = 0$, solo 8 resultados distintos tienen probabilidad real.

### Cálculo medio

| Modo | CE promedio |
|---|---:|
| Determinista | 0.318175 |
| Máscaras, $q=0.8$ | 0.491935 |
| Máscaras, $q=0.5$ | 0.904415 |

Se enumera exactamente el conjunto de 16 máscaras. Esto es una comparación aritmética, no un estudio de rendimiento de modelo entrenado.

## Página 22 — PyTorch: dropout y evaluación

```python
import torch
from torch import nn

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4, 3),
)
criterion = nn.CrossEntropyLoss()

model.train()
loss = criterion(model(xb), yb)
# Luego: backward + optimizer.step()

model.eval()
with torch.no_grad():
    logits = model(xv)
    val_ce = criterion(logits, yv)
```

- `eval()`: cambia el comportamiento de las capas.
- `no_grad()`: desactiva el registro de gradiente.
- Estas dos cosas controlan cosas distintas.

`xb`: $(B,2)$ flotantes; `yb`: $(B,)$ enteros largos.

## Página 23 — TensorFlow/Keras: el argumento training

```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.Input(shape=(2,)),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(3),
])

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

with tf.GradientTape() as tape:
    s = model(xb, training=True)
    loss = loss_fn(yb, s)
    grads = tape.gradient(loss, model.trainable_variables)

s = model(xv, training=False)
val_ce = loss_fn(yv, s)
```

`training=False` controla dropout; `trainable=False` no lo desactiva.

## Página 24 — Tu turno: una nueva máscara y su gradiente

Mantener la cabeza de salida original y

$$
 h = (2,1,1,0)^\top,
\quad q = 0.5,
\quad m = (0,1,1,0)^\top,
$$

con

$$
W_2 =
\begin{pmatrix}
1 & 0 & 1 & 0 \\
0 & 1 & 0 & 1 \\
-1 & -1 & 1 & 1
\end{pmatrix},
\quad
b_2 =
\begin{pmatrix}
0 \\
1 \\
0
\end{pmatrix}.
$$

1. Calcular $\tilde{h}$ y los logits $s = W_2\tilde{h} + b_2$.
2. Para un gradiente local prescrito $\partial\ell/\partial\tilde{h} = (1,-2,3,4)^\top$, hallar $\partial\ell/\partial h$.
3. ¿Cuántos parámetros entrenables tiene el modelo ahora?

Se mantiene fija la máscara hacia delante al aplicar la regla de la cadena.

## Página 25 — Solución: activaciones, logits y backward

$$
\tilde{h} = 2(0,1,1,0)^\top \odot (2,1,1,0)^\top = (0,2,2,0)^\top,
$$

$$
 s = W_2\tilde{h} + b_2 = (2,3,0)^\top.
$$

Y la gradiente hacia atrás es:

$$
\frac{\partial\ell}{\partial h}
= 2(0,1,1,0)^\top \odot (1,-2,3,4)^\top
= (0,-4,6,0)^\top.
$$

Número de parámetros entrenables:

$$
4\cdot 2 + 4 + 3\cdot 4 + 3 = 27.
$$

La máscara aleatoria no es entrenable.

## Página 26 — Exit ticket y próxima sesión

1. El CE de entrenamiento cae; el CE de validación sube. ¿Qué inspeccionaría y conservaría?
2. ¿Por qué es un problema un escalador ajustado sobre todos los datos para la evaluación?
3. Con probabilidad de descarte 0.2, ¿cuál es la escala sobre una activación mantenida durante el entrenamiento? ¿Qué ocurre durante la evaluación?
4. ¿Preservar la media de las activaciones preserva la media de las probabilidades?
5. ¿`no_grad()` o `trainable=False` desactivan dropout?

**Siguiente sesión:** convolución unidimensional.

## Página 27 — Apéndice A. Momentos condicionales del dropout

Fijar $h$, tomar $m_j$ independientes con distribución Bernoulli$(q)$ y definir

$$
\tilde{h}_j = \frac{m_j h_j}{q}.
$$

Entonces:

$$
\mathbb{E}[\tilde{h}_j \mid h] = h_j,
$$

$$
\mathbb{E}[\tilde{h}_j^2 \mid h] = h_j^2 q,
$$

$$
\mathrm{Var}(\tilde{h}_j \mid h) = p q h_j^2,
$$

$$
\mathrm{Cov}(\tilde{h}_i,\tilde{h}_j\mid h)=0 \quad (i\ne j).
$$

Por tanto,

$$
\mathrm{Cov}(\tilde{h}\mid h) = p q \mathrm{diag}(h_1^2,\ldots,h_d^2).
$$

## Página 28 — Apéndice B. Las ocho máscaras efectivas

Con $q=0.5$, $h=(2,1,1,0)^\top$, $y=0$. Cada fila tiene probabilidad $1/8$.

| Máscara $(m_1,m_2,m_3)$ | Logits $s$ | $p_0$ | $-\log p_0$ |
|---|---|---:|---:|
| 000 | $(0,1,0)$ | 0.211942 | 1.551445 |
| 001 | $(2,1,2)$ | 0.422319 | 0.861995 |
| 010 | $(0,3,-2)$ | 0.047123 | 3.054985 |
| 011 | $(2,3,0)$ | 0.259496 | 1.349012 |
| 100 | $(4,1,-4)$ | 0.952270 | 0.048907 |
| 101 | $(6,1,-2)$ | 0.992976 | 0.007049 |
| 110 | $(4,3,-6)$ | 0.731034 | 0.313295 |
| 111 | $(6,3,-4)$ | 0.952533 | 0.048631 |

Se verifican media y pérdida media:

$$
\mathbb{E}[p_0] = 0.571212,
\quad
\mathbb{E}[-\log p_0] = 0.904415.
$$

## Página 29 — Apéndice C. Identidad lineal con pérdida cuadrática

Para un predictor lineal escalar, con $h$, $w$, $y$ fijos, máscaras independientes y pérdida cuadrática media:

$$
\frac{1}{2}\mathbb{E}[(y-w^\top \tilde{h})^2\mid h]
= \frac{1}{2}(y-\mathbb{E}[w^\top \tilde{h}])^2 + \frac{1}{2}\mathrm{Var}(w^\top \tilde{h}).
$$

Esto equivale a:

$$
\frac{1}{2}(y-w^\top h)^2 + \frac{p}{2q}\sum_j w_j^2 h_j^2.
$$

El término extra es una penalización cuadrática dependiente de los datos sobre los pesos de salida.

## Página 30 — Apéndice D. Objetivo de entrenamiento estocástico

Con dropout, el entrenamiento apunta a un objetivo de pérdida esperada enmascarada:

$$
J_{\mathrm{drop}}(\theta)
= \frac{1}{N_{\mathrm{tr}}}\sum_{i\in\mathcal{D}_{\mathrm{tr}}}\mathbb{E}_m[\ell(f_\theta(x_i;m), y_i)].
$$

Para un espacio finito de máscaras cuya probabilidad no depende de $\theta$, en puntos diferenciables:

$$
\nabla_\theta \mathbb{E}[\ell] = \mathbb{E}[\nabla_\theta \ell].
$$

Con un minibatch uniforme, el gradiente enmascarado medio es no sesgado para $\nabla J_{\mathrm{drop}}$.

## Página 31 — Apéndice E. Early stopping en Keras

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalCrossentropy(
        name="ce", from_logits=True)],
)

stop = keras.callbacks.EarlyStopping(
    monitor="val_ce",
    mode="min",
    min_delta=0.0,
    patience=3,
    restore_best_weights=True,
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[stop],
)
```

- Restaura pesos del modelo, no el historial del optimizador.
- `validation_split` toma las últimas muestras antes de mezclar.

## Página 32 — Apéndice F. Modos, máscaras y checkpoints

| Operación | Significado |
|---|---|
| `model.eval()` | Cambia el comportamiento del dropout; el autograd puede seguir registrando operaciones |
| `torch.no_grad()` | Detiene el registro del gradiente; no activa dropout |
| `training=False` en Keras | Ejecuta dropout como identidad |
| `trainable=False` | Congela pesos; no desactiva dropout |

Guardar un checkpoint de PyTorch con pesos mejores:

```python
import copy
best_weights = copy.deepcopy(model.state_dict())
# Después:
model.load_state_dict(best_weights)
model.eval()
```

Una máscara broadcast con forma $(1,H)$ comparte decisiones entre ejemplos y cambia el modelo estocástico.

## Página 33 — Apéndice G. Evaluación reproducible

| Registro | Por qué importa |
|---|---|
| División y agrupación | Define qué ejemplos son independientes |
| Preprocesado ajustado | Debe viajar con el checkpoint |
| Métrica y reducción | La CE media es una suma sobre ejemplos dividida por el total |
| Checkpoint y modo | Fija valores de parámetros y comportamiento de capa |
| Semillas y software | Permite reproducibilidad dentro del entorno computacional |

Para batch de tamaños $B_b$, se calcula:

$$
CE = \frac{\sum_b B_b \cdot CE_b}{\sum_b B_b}.
$$

Un promedio sin peso de medias de batch es incorrecto cuando los tamaños varían.

## Página 34 — Apéndice H. Referencias y lecturas

### Libros

- Christopher M. Bishop y Hugh Bishop, *Deep Learning: Foundations and Concepts*, 2024, capítulo 9: *Regularization*.
- Ian Goodfellow, Yoshua Bengio y Aaron Courville, *Deep Learning*, 2016, secciones 5.2–5.3, 7.1, 7.8 y 7.12.

### Artículo original

- N. Srivastava et al., *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*, JMLR, 2014.

### APIs y material auxiliar

- PyTorch: `Dropout`, `CrossEntropyLoss`, `eval()`, `no_grad()`, checkpoints.
- Keras: `Dropout`, `EarlyStopping`, `SparseCategoricalCrossentropy`, `training=False`.

Fuentes primarias consultadas el 17 de octubre de 2026. Exposición docente original, ejemplos numéricos y figuras.
