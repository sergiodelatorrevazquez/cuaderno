---
titulo: "De las puntuaciones a las probabilidades y pérdidas: softmax, temperatura y entropía cruzada"
asignatura: aprendizaje-profundo
fecha: 2026-10-22
orden: 2
tema: "Semana 1 - Sesión 2"
tags: [softmax, temperatura, entropia-cruzada, log-verosimilitud, pytorch, tensorflow]
resumen: "Desarrollo página por página de la conversión de logits en probabilidades, la softmax y su estabilidad numérica, la verosimilitud, la entropía cruzada, la temperatura y su implementación en PyTorch y Keras."
---

## Página 1 — De las puntuaciones a las probabilidades y las pérdidas

**Fundamentos matemáticos y computacionales del aprendizaje profundo.**

**Título:** De las puntuaciones a las probabilidades y las pérdidas.  
**Subtítulo:** Softmax, temperatura y entropía cruzada.  
**Sesión:** Semana 1, sesión 2.  
**Horario indicado:** jueves, 19:20–20:20.  
**Docente:** Francisco R. Villatoro.

**Objetivo de la sesión:** conectar la salida de una red neuronal con un modelo de probabilidad y con la cantidad que se minimiza durante el entrenamiento.

## Página 2 — El flujo de predicción

El diagrama de la página muestra el flujo de izquierda a derecha:

1. **Entrada:** $x\in\mathbb{R}^{2}$.
2. **Red neuronal:** calcula las puntuaciones $s=f_\theta(x)$.
3. **Softmax:** transforma las puntuaciones en probabilidades $p$.
4. **Pérdida:** se calcula como $\ell=-\log p_y$, usando la etiqueta observada $y\in\{0,1,2\}$.

La etiqueta observada alimenta la etapa de pérdida; no es una etapa de la red ni de la softmax.

**Ruta temporal de la sesión:**

| Minutos | Contenido |
|---|---|
| 0–10 | Etiqueta, puntuación y probabilidad condicional |
| 10–25 | Softmax, ejes y estabilidad numérica |
| 25–40 | Verosimilitud y entropía cruzada con $T=1$ |
| 40–50 | Temperatura manteniendo fijos los logits |
| 50–60 | Comprobación numérica y contrato de las bibliotecas |

## Página 3 — Un modelo condicional para la clase

Para una entrada dada $x$, la red define la probabilidad condicional de cada clase candidata:

$$
p_\theta(y=k\mid x),\qquad k\in\{0,1,2\}.
$$

**Notación:**

- $x$: la entrada que se ha observado.
- $y$: la clase que se quiere predecir.
- $\theta$: los parámetros actuales del modelo.
- $k$: una clase candidata.

Una distribución válida debe cumplir:

$$
p_\theta(y=k\mid x)\geq 0,\qquad
\sum_{k=0}^{2}p_\theta(y=k\mid x)=1.
$$

La suma se realiza sobre las clases para una misma entrada. Estas son las probabilidades del modelo; la fiabilidad de esas probabilidades debe evaluarse.

**Referencias de la diapositiva:** Bishop y Bishop (2024), capítulo 6; Goodfellow, Bengio y Courville (2016), capítulos 3 y 6.

## Página 4 — Una etiqueta, tres puntuaciones y tres probabilidades

Continúa el ejemplo calculado a mano en la sesión 1:

$$
s=(1,3,-3)^\top,\qquad y=1,\qquad T=1.
$$

| Cantidad | Clase 0 | Clase 1 | Clase 2 |
|---|---:|---:|---:|
| Indicador de la clase observada | 0 | 1 | 0 |
| Logit $s_k$ | 1 | 3 | $-3$ |
| Probabilidad $p_k$ | 0.118943 | 0.878878 | 0.002179 |

**Comprobación:** los logits suman accidentalmente 1 en este ejemplo. Eso no los convierte en probabilidades: las probabilidades son las de la última fila.

Se conserva $s$ para denotar los logits de salida, como en la sesión 1. Algunas referencias usan $z$ para los logits; aquí $z$ denota una preactivación oculta.

Las puntuaciones proceden del cálculo ilustrativo hecho a mano, no de los pesos ajustados en el experimento.

## Página 5 — Softmax a temperatura uno

La función softmax convierte cualquier vector finito de puntuaciones reales en probabilidades de clase:

$$
p_k=\operatorname{softmax}(s)_k
=\frac{\exp(s_k)}{\sum_{j=0}^{2}\exp(s_j)}.
$$

**Exponenciación:** $\exp(s_k)>0$, por lo que todas las clases reciben masa positiva.

**Normalización:**

$$
\sum_{k=0}^{2}p_k
=\frac{\sum_k\exp(s_k)}{\sum_j\exp(s_j)}
=1.
$$

El denominador acopla las clases: compiten entre sí por la masa total de probabilidad.

**Referencia de la diapositiva:** Goodfellow, Bengio y Courville (2016), capítulo 6, unidades de salida softmax.

## Página 6 — Ejemplo de cálculo

Para las puntuaciones del ejemplo:

$$
s=(1,3,-3)^\top
\quad\Longrightarrow\quad
\exp(s)\approx
(2.718282,\ 20.085537,\ 0.049787)^\top.
$$

| Cantidad | Clase 0 | Clase 1 | Clase 2 |
|---|---:|---:|---:|
| Masa sin normalizar | 2.718282 | 20.085537 | 0.049787 |
| Probabilidad al dividir por 22.853606 | 0.118943 | 0.878878 | 0.002179 |

La clase predicha es:

$$
\hat{y}=\arg\max_k p_k=1.
$$

La probabilidad asignada a la etiqueta observada es $p_y=p_1\approx0.878878$. La etiqueta es un índice; $p_y$ selecciona la componente de la distribución correspondiente a ese índice.

## Página 7 — Normalizar a lo largo del eje de clases

Un lote almacena los ejemplos en filas. Para tres clases, $S\in\mathbb{R}^{B\times3}$. El ejemplo de dos filas muestra:

$$
S=
\begin{bmatrix}
1&3&-3\\
0&0&0
\end{bmatrix}
\quad\longrightarrow\quad
P=
\begin{bmatrix}
0.118943&0.878878&0.002179\\
1/3&1/3&1/3
\end{bmatrix}.
$$

En cada fila se aplica softmax a las clases y cada fila suma 1: hay una distribución por ejemplo.

```python
P = torch.softmax(S, dim=1)  # S tiene forma (B, 3)
P = tf.nn.softmax(S, axis=-1)  # mismo eje de clases
```

Con esta forma, `dim=1` y `dim=-1` seleccionan el mismo eje. Normalizar sobre el lote mezclaría ejemplos distintos.

**Referencias de la diapositiva:** documentación oficial de softmax de PyTorch y `tf.nn.softmax` de TensorFlow.

## Página 8 — Sumar una constante no cambia ninguna probabilidad

Sea $c\in\mathbb{R}$ una constante igual para todas las clases. Por la definición de softmax:

$$
\operatorname{softmax}(s+c\mathbf{1})_k
=\frac{\exp(s_k+c)}{\sum_j\exp(s_j+c)}
=\frac{\exp(c)\exp(s_k)}{\exp(c)\sum_j\exp(s_j)}
=\operatorname{softmax}(s)_k.
$$

En aritmética exacta, los vectores $(1,3,-3)$ y $(1001,1003,997)$ producen las mismas probabilidades. Solo importan las puntuaciones relativas: un desplazamiento común se cancela.

## Página 9 — Restar el máximo antes de exponenciar

Se elige $m=\max_j s_j$ y se usa la invariancia a desplazamientos comunes:

$$
p_k=\frac{\exp(s_k-m)}{\sum_j\exp(s_j-m)}.
$$

Para el ejemplo:

$$
(1,3,-3)-(3,3,3)=(-2,0,-6),
$$

$$
\exp(s-m\mathbf{1})=(e^{-2},1,e^{-6})^\top.
$$

**Por qué ayuda:** cada puntuación desplazada es como máximo cero, la exponencial mayor es 1 y, con tres clases, la suma de las exponenciales queda entre 1 y 3. El desplazamiento común evita el desbordamiento de las exponenciales.

Las probabilidades muy pequeñas todavía pueden sufrir subdesbordamiento. Para calcular pérdidas, es preferible calcular directamente los logaritmos de las probabilidades.

**Referencia de la diapositiva:** Goodfellow, Bengio y Courville (2016), capítulo 4, cálculo numérico.

## Página 10 — ¿Qué hace verosímiles las etiquetas observadas?

Para los datos observados $\mathcal{D}=\{(x_i,y_i)\}_{i=1}^{N}$, se define la verosimilitud:

$$
L(\theta;\mathcal{D})
=\prod_{i=1}^{N}p_\theta(y_i\mid x_i).
$$

**Supuesto del modelo:** dadas sus entradas y $\theta$, las etiquetas son condicionalmente independientes y cada predicción utiliza su propia entrada. Los pares de ejemplos independientes e idénticamente distribuidos constituyen el marco introductorio estándar.

- **Probabilidad:** se fijan $x$ y $\theta$ y se comparan las posibles etiquetas $y$.
- **Verosimilitud:** se fijan los datos observados y se comparan los valores posibles de los parámetros $\theta$.

El entrenamiento debe hacer verosímiles las etiquetas observadas.

**Referencias de la diapositiva:** Bishop y Bishop (2024), capítulo 6; Goodfellow, Bengio y Courville (2016), capítulo 5.

## Página 11 — De la verosimilitud a la pérdida de entrenamiento

El logaritmo es una función creciente y transforma productos en sumas:

$$
\log L(\theta;\mathcal{D})
=\sum_{i=1}^{N}\log p_\theta(y_i\mid x_i).
$$

Maximizar esta expresión equivale a minimizar la log-verosimilitud negativa media:

$$
\mathcal{L}(\theta)
=\frac{1}{N}\sum_{i=1}^{N}
\underbrace{-\log p_\theta(y_i\mid x_i)}_{\ell_i(\theta)}.
$$

Para un solo ejemplo, $\ell=-\log p_y$. En el ejemplo de esta sesión, $\ell\approx0.129109$.

El uso de logaritmos naturales hace que la pérdida se mida en *nats*. Se usa $T=1$, sin ponderación de clases ni suavizado de etiquetas (*label smoothing*).

**Referencias de la diapositiva:** Goodfellow, Bengio y Courville (2016), capítulos 5 y 6.

## Página 12 — Por qué es entropía cruzada

La etiqueta observada se codifica como un objetivo *one-hot*:

$$
q_k=\mathbf{1}[k=y],\qquad
y=1\quad\Longrightarrow\quad q=(0,1,0)^\top.
$$

La entropía cruzada entre el objetivo $q$ y la predicción $p$ es:

$$
H(q,p)=-\sum_{k=0}^{2}q_k\log p_k=-\log p_y=\ell.
$$

- **Entropía cruzada $H(q,p)$:** utiliza el objetivo y evalúa cuánta probabilidad le asigna el modelo.
- **Entropía de la predicción $H(p)$:** $-\sum_k p_k\log p_k$ mide la dispersión de la distribución.

Una etiqueta entera y su codificación *one-hot* representan el mismo objetivo.

**Referencia de la diapositiva:** Goodfellow, Bengio y Courville (2016), capítulo 3, teoría de la información.

## Página 13 — ¿Qué recompensa la pérdida?

La gráfica representa la entropía cruzada, en *nats*, frente a la probabilidad $p_y$ asignada a la clase verdadera. El eje horizontal abarca probabilidades de 0 a 1 y el vertical valores de pérdida desde 0 hasta algo más de 3. La curva decrece cuando aumenta $p_y$, según la relación $-\log p_y$.

La gráfica marca, entre otros, estos valores:

- Para $p_y=0.9$, $-\log 0.9\approx0.105$.
- Para $p_y=0.5$, $\log 2\approx0.693$.
- Para $p_y=0.1$, $-\log 0.1=\log 10\approx2.303$.
- Con distribución uniforme entre tres clases, $p_y=1/3$ y $\ell=\log3\approx1.099$.

La pérdida recompensa asignar probabilidad a la etiqueta observada. La misma clase predicha puede tener pérdidas muy diferentes. La exactitud comprueba cuál clase gana; la entropía cruzada utiliza la probabilidad asignada al objetivo.

## Página 14 — Calcular la pérdida directamente a partir de los logits

Al sustituir softmax en la pérdida, con $T=1$:

$$
\ell=-\log\frac{e^{s_y}}{\sum_j e^{s_j}}
=\underbrace{\log\sum_j e^{s_j}}_{\operatorname{LSE}(s)}-s_y.
$$

Con $m=\max_j s_j$, la forma estable es:

$$
\ell=(m-s_y)+\log\sum_j e^{s_j-m}.
$$

Para $s=(1,3,-3)$ e $y=1$:

$$
\ell=(3-3)+\log(e^{-2}+1+e^{-6})\approx0.129109.
$$

Se recomienda usar una pérdida que combine *log-softmax* con la selección del objetivo, en lugar de calcular una probabilidad potencialmente subdesbordada y después su logaritmo.

**Referencias de la diapositiva:** `CrossEntropyLoss` de PyTorch; pérdidas categóricas de Keras con `from_logits=True`.

## Página 15 — La temperatura reescala logits fijos

Para logits fijos $s=(1,3,-3)$, la distribución a temperatura $T$ se define como:

$$
p_k(T)=\frac{\exp(s_k/T)}{\sum_j\exp(s_j/T)},\qquad T>0.
$$

Los tres gráficos de barras muestran las probabilidades para las clases 0, 1 y 2:

| Temperatura | Clase 0 | Clase 1 | Clase 2 |
|---|---:|---:|---:|
| $T=0.5$ | 0.018 | 0.982 | $6.03\times10^{-6}$ |
| $T=1$ | 0.119 | 0.879 | 0.00218 |
| $T=2$ | 0.259 | 0.705 | 0.0351 |

Con logits fijos, una temperatura menor concentra la distribución; una temperatura mayor la hace más dispersa.

## Página 16 — Una temperatura positiva conserva el orden

La razón entre las probabilidades de dos clases es:

$$
\frac{p_i(T)}{p_j(T)}
=\exp\left(\frac{s_i-s_j}{T}\right).
$$

Como $T>0$ y la función exponencial es creciente:

$$
s_i>s_j\quad\Longleftrightarrow\quad p_i(T)>p_j(T).
$$

Las puntuaciones iguales siguen empatadas. Por tanto:

$$
\arg\max_k p_k(T)=\arg\max_k s_k.
$$

La clase ganadora del ejemplo es la clase 1 para $T=0.5$, $T=1$ y $T=2$. Estas afirmaciones se refieren a logits finitos y aritmética exacta.

## Página 17 — Elegir una clase o muestrear una

**Elegir la clase más probable:**

$$
\hat{y}=\arg\max_k p_k(T).
$$

Es una decisión determinista con una regla fija para resolver empates. En el ejemplo siempre se devuelve la clase 1.

**Muestrear de la distribución:**

$$
\tilde{y}\sim\operatorname{Categorical}(p(T)).
$$

Muestreos independientes repetidos tienen como proporciones esperadas las probabilidades $p(T)$. Cambiar $T$ cambia esas proporciones.

**La confianza no equivale a corrección.** Si la clase ganadora fuese incorrecta, reducir $T$ haría esa predicción errónea más confiada.

La **calibración** pregunta si las probabilidades declaradas coinciden con las frecuencias observadas. Que una distribución esté normalizada no garantiza que esté calibrada.

## Página 18 — Ejercicio: seguir una predicción

Trabajar en parejas durante 90 segundos con:

$$
s=(1,3,-3)^\top,\qquad y=1,\qquad T=1.
$$

1. Restar el logit mayor. ¿Cuáles son los tres valores desplazados?
2. Usar $e^{-2}\approx0.135335$ y $e^{-6}\approx0.002479$ para calcular $p_y$.
3. Calcular $\ell=-\log p_y$. ¿Es menor que $\log3$?
4. Para un lote de $B$ ejemplos, indicar las formas de los logits, las etiquetas enteras, las probabilidades y la pérdida media.

Antes de calcular, indicar la clase objetivo y el eje de clases.

**Solución numérica:** apéndice A.  
**Respuestas desarrolladas:** apéndice F.

## Página 19 — PyTorch: pasar los logits a la pérdida

```python
import torch
from torch import nn

S = torch.tensor([[1., 3., -3.]])  # (B, C) = (1, 3)
y = torch.tensor([1], dtype=torch.long)  # (B,) = (1,)
loss = nn.CrossEntropyLoss()(S, y)  # escalar; media
P = torch.softmax(S, dim=1)  # para inspección
# loss ~ 0.129109; P ~ [[0.118943, 0.878878, 0.002179]]
```

**Contrato de la función:**

| Elemento | Significado |
|---|---|
| Salida del modelo | Logits sin normalizar; no aplicar softmax antes de esta pérdida |
| Objetivo de este ejemplo | Índices enteros de clase `0, ..., C−1`; tipo `long` |
| Orden de argumentos | `loss_fn(prediction, target)` |

PyTorch 2.14 también admite objetivos de probabilidad; véase el apéndice C.

**Referencia de la diapositiva:** documentación de PyTorch 2.14 para [`CrossEntropyLoss`](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html).

## Página 20 — Keras y TensorFlow: declarar el contrato

```python
import tensorflow as tf
from tensorflow import keras

S = tf.constant([[1., 3., -3.]], dtype=tf.float32)
y = tf.constant([1], dtype=tf.int32)
loss_fn = keras.losses.SparseCategoricalCrossentropy(
    from_logits=True)
loss = loss_fn(y, S)  # media ~ 0.129109
P = tf.nn.softmax(S, axis=-1)  # para inspección
```

**Contrato de la función:**

- **Dispersa (`Sparse`):** las etiquetas son índices enteros.
- **`from_logits=True`:** las entradas son puntuaciones anteriores a softmax.
- **Orden de argumentos:** `loss_fn(target, prediction)`.
- El eje de clases es el último.
- Keras independiente utiliza `import keras`; se aplica el mismo contrato para esta pérdida.

**Referencias de la diapositiva:** pérdidas probabilísticas de [Keras](https://keras.io/api/losses/probabilistic_losses/) y [`SparseCategoricalCrossentropy`](https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy) de TensorFlow.

## Página 21 — Preguntas de salida (*exit ticket*)

Responder cada pregunta en una frase:

1. Si se suma 1000 a todos los logits, ¿qué ocurre con las probabilidades y la pérdida?
2. Si se cambia $T$ de 1 a 2, ¿cambia la clase predicha?
3. En este ejemplo, ¿PyTorch `CrossEntropyLoss` debe recibir `softmax(S)`?
4. ¿Qué pérdida de Keras y qué opción corresponden a etiquetas enteras y logits sin normalizar?

El flujo de la sesión se resume así:

$$
s\longrightarrow p(T)\longrightarrow\ell=-\log p_y.
$$

**Próxima sesión:** ¿cómo cambia esta pérdida al modificar un parámetro?  
**Respuestas:** apéndice F.

## Página 22 — Apéndice A. Temperatura y pérdida

Se mantienen fijos $s=(1,3,-3)$ y la etiqueta observada $y=1$. Se usan logaritmos naturales.

| $T$ | $p_0(T)$ | $p_1(T)$ | $p_2(T)$ | $-\log p_1(T)$ |
|---:|---:|---:|---:|---:|
| 0.5 | 0.017986 | 0.982008 | 0.000006 | 0.018156 |
| 1 | 0.118943 | 0.878878 | 0.002179 | 0.129109 |
| 2 | 0.259496 | 0.705385 | 0.035119 | 0.349012 |

Con $T=1$, las puntuaciones desplazadas son $(-2,0,-6)$:

$$
Z'=e^{-2}+1+e^{-6}\approx1.137814,\qquad
p_y=\frac{1}{Z'},\qquad
\ell=\log Z'\approx0.129109.
$$

Las probabilidades se redondean por separado; por ello, las sumas de las filas pueden diferir ligeramente de 1.

El cálculo es original; los materiales incluyen una referencia reproducible en NumPy. La diapositiva indica volver al ejercicio.

## Página 23 — Apéndice B. Un lote: una pérdida por ejemplo

Se consideran:

$$
S=
\begin{bmatrix}
1&3&-3\\
0&0&0
\end{bmatrix},\qquad
y=
\begin{bmatrix}
1\\2
\end{bmatrix}.
$$

Las pérdidas por ejemplo son:

$$
(\ell_1,\ell_2)\approx(0.129109,\ 1.098612).
$$

La pérdida media del lote es:

$$
\mathcal{L}_{\mathrm{batch}}
=\frac{\ell_1+\ell_2}{2}\approx0.613861.
$$

Se pueden inspeccionar las pérdidas individuales antes de promediarlas:

```python
# PyTorch: primero los logits
per_example = nn.CrossEntropyLoss(reduction="none")(S, y)

# Keras: primero los objetivos
per_example = keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction=None)(y, S)
```

Aquí $S$ e $y$ representan tensores de la biblioteca correspondiente. Sin ponderaciones ni máscaras, las reducciones predeterminadas dan la media mostrada.

**Referencias de la diapositiva:** documentación oficial de las pérdidas de PyTorch y Keras; ejemplo numérico original.

## Página 24 — Apéndice C. Representaciones del objetivo

Los logits sin normalizar $S$ tienen forma `(B, C)`.

| Objetivo | Forma | PyTorch | Keras / `tf.keras` |
|---|---|---|---|
| Índices de clase | `(B,)` | `CrossEntropyLoss`; `long` | `SparseCategoricalCrossentropy` |
| *One-hot* | `(B, C)` | `CrossEntropyLoss`; `float` | `CategoricalCrossentropy` |
| Objetivos suaves (*soft targets*) | `(B, C)` | `CrossEntropyLoss`; `float` | `CategoricalCrossentropy` |

En esta configuración, todas las pérdidas de Keras indicadas deben usar `from_logits=True`.

Ejemplo de codificación *one-hot* de la clase 1 en Keras/TensorFlow:

```python
Q = tf.constant([[0., 1., 0.]], dtype=tf.float32)
loss = keras.losses.CategoricalCrossentropy(
    from_logits=True)(Q, S)  # S = [[1., 3., -3.]]
```

Los objetivos de probabilidad válidos cumplen $q_k\geq0$ y $\sum_k q_k=1$. Un objetivo *one-hot* recupera $-\log p_y$; un objetivo suave utiliza:

$$
-\sum_k q_k\log p_k.
$$

**Referencias de la diapositiva:** API `CrossEntropyLoss` de PyTorch y API de pérdidas probabilísticas de Keras. En estos ejemplos no se usan ponderaciones de clase ni suavizado de etiquetas.

## Página 25 — Apéndice D. Referencia estable en NumPy

```python
import numpy as np

def probabilities_and_losses(S, y, T=1.0):
    S = np.asarray(S, dtype=np.float64)
    a = (S - S.max(axis=1, keepdims=True)) / T
    logP = a - np.log(np.exp(a).sum(axis=1, keepdims=True))
    losses = -logP[np.arange(S.shape[0]), y]
    return np.exp(logP), losses

S = np.array([[1., 3., -3.], [0., 0., 0.]])
y = np.array([1, 2])
P, losses = probabilities_and_losses(S, y)
```

Para $S=(1000,-1000,0)$, $y=1$ y $T=1$, la pérdida es aproximadamente 2000. La probabilidad seleccionada puede sufrir subdesbordamiento y convertirse en 0, mientras que su log-probabilidad continúa siendo finita.

**Entradas:** logits finitos de forma `(B, C)`, etiquetas enteras de forma `(B,)` y $T>0$.

## Página 26 — Apéndice E. Límites de la temperatura y confianza

Para logits finitos y $M=\{k:s_k=\max_j s_j\}$, el límite cuando la temperatura tiende a cero por valores positivos es:

$$
\lim_{T\to0^+}p_k(T)=
\begin{cases}
1/|M|,& k\in M,\\
0,& k\notin M.
\end{cases}
$$

Cuando la temperatura tiende a infinito, la distribución converge a la uniforme entre tres clases:

$$
\lim_{T\to\infty}p_k(T)=\frac{1}{3}.
$$

**Si la clase ganadora es errónea**, se consideran las mismas puntuaciones, pero hipotéticamente $y=0$:

| Temperatura $T$ | 0.5 | 1 | 2 |
|---|---:|---:|---:|
| $-\log p_0(T)$ | 4.018156 | 2.129109 | 1.349012 |
| Clase predicha | 1 | 1 | 1 |

En este caso, reducir $T$ hace que la predicción incorrecta sea más confiada. El escalado de temperatura para calibración debe ajustarse y evaluarse con datos reservados apropiados.

## Página 27 — Apéndice F. Soluciones del ejercicio y de las preguntas de salida

### Ejercicio numérico

1. Puntuaciones desplazadas: $(-2,0,-6)$.
2. $p_y=p_1\approx0.878878$.
3. $\ell\approx0.129109<\log3$.
4. Las formas para un lote de $B$ ejemplos son:
   - Logits: `(B, 3)`.
   - Etiquetas enteras: `(B,)`.
   - Probabilidades: `(B, 3)`.
   - Pérdida media: escalar.

### *Exit ticket*

1. Un desplazamiento común no cambia ni las probabilidades ni la pérdida en aritmética exacta.
2. Una temperatura positiva conserva la clase ganadora: clase 1.
3. Se pasan los logits sin normalizar a la entropía cruzada de PyTorch (CE).
4. En Keras se utiliza `SparseCategoricalCrossentropy` con `from_logits=True`.

La diapositiva indica volver al ejercicio y volver a las preguntas de salida.

## Página 28 — Apéndice G. Lecturas y referencias de API

### Conceptos y derivaciones matemáticas

- C. M. Bishop y H. Bishop (2024). *Deep Learning: Foundations and Concepts*. Capítulo 6: salidas de clasificación y objetivos de redes neuronales. [bishopbook.com](https://www.bishopbook.com/)
- I. Goodfellow, Y. Bengio y A. Courville (2016). *Deep Learning*. Capítulo 3: probabilidad e información; capítulo 4: estabilidad numérica; capítulo 5: máxima verosimilitud; capítulo 6: salidas softmax. [deeplearningbook.org](https://www.deeplearningbook.org/)

### Contratos de las bibliotecas

- PyTorch 2.14: [`CrossEntropyLoss`](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html) y [softmax](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.softmax.html).
- Keras: [pérdidas probabilísticas](https://keras.io/api/losses/probabilistic_losses/).
- TensorFlow: [`SparseCategoricalCrossentropy`](https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy) y [`tf.nn.softmax`](https://www.tensorflow.org/api_docs/python/tf/nn/softmax).

La documentación se comprobó el 17 de octubre de 2026. Los diagramas, ejemplos y textos docentes son originales. Los resultados numéricos se comprobaron con NumPy y los fragmentos de los frameworks se contrastaron con las API enlazadas.
