---
titulo: "Fundamentos matemáticos y computacionales del aprendizaje profundo: introducción a las redes neuronales"
asignatura: aprendizaje-profundo
fecha: 2026-10-22
tema: "Semana 1 - Sesión 1"
tags: [fundamentos, tensores, capas-afines, relu, perceptrón-multicapa, parámetros]
resumen: "Introducción a los modelos de aprendizaje profundo como composición de funciones. Tensores, formas, parámetros, capas afines, activaciones ReLU y un primer ejemplo con una red de dos capas."
borrador: false
---

## Objetivos de la sesión

Al finalizar esta sesión, deberías ser capaz de:

- Leer una red neuronal pequeña como una composición de funciones.
- Seguir la forma (*shape*) de cada array en una red.
- Identificar y contar los parámetros entrenables.
- Explicar qué cambia durante el entrenamiento.

**Requisitos previos:** multiplicación de matrices, derivadas elementales y probabilidad condicional.

## Estructura del curso

El curso se divide en cuatro bloques complementarios:

| Bloque | Pregunta principal | Profesor/a |
| --- | --- | --- |
| Fundamentos | ¿Cómo representamos un modelo, cómo computamos con él y cómo lo entrenamos? | Francisco R. Villatoro |
| Imágenes y vídeo | ¿Cómo resuelven las arquitecturas convolucionales las tareas visuales? | Enrique |
| Secuencias neuronales | ¿Cómo representan las secuencias los *tokens*, la recurrencia y los *transformers*? | Leo |
| Modelos de lenguaje | ¿Cómo apoyan los modelos n-grama, los *embeddings* y los modelos de lenguaje a NLP y RAG? | Ezequiel |

El lenguaje común de todo el curso son: datos, tensores, parámetros, predicciones y aprendizaje.

## Un modelo como una función entrenable

Un modelo de aprendizaje profundo se puede entender como una función parametrizada $f_\theta(\cdot)$ que transforma una entrada $x$ en una predicción.

Durante el **entrenamiento**, lo que cambia son los **parámetros** $\theta = \{W^{(l)}, b^{(l)}\}$ del modelo, con el objetivo de que las predicciones se ajusten mejor a los datos.

## Preguntas diagnósticas

### 1) Formas

Si $x \in \mathbb{R}^2$ y $z = Wx + b \in \mathbb{R}^4$, ¿qué formas tienen $W$ y $b$?

- $W \in \mathbb{R}^{4\times 2}$
- $b \in \mathbb{R}^{4}$

### 2) Derivada a través de dos operaciones

Sea $h = 2x + 1$ y $u = h^2$. ¿Cuál es $\frac{du}{dx}$ en $x=1$?

$$
h(1)=3,\quad \frac{du}{dh}=2h=6,\quad \frac{dh}{dx}=2,\quad \frac{du}{dx}=\frac{du}{dh}\cdot\frac{dh}{dx}=12
$$

### 3) Probabilidad condicional

Entre 20 mensajes que contienen la palabra *offer*, 15 son *spam*. ¿Qué probabilidad empírica de *spam* se estima?

$$
P(\text{spam}\mid\text{offer}) \approx \frac{15}{20} = 0.75
$$

## Tensores, vectores y lotes (*batches*)

En aprendizaje profundo trabajamos con **arrays multidimensionales** (tensores). Es fundamental ser coherentes con el **orden de ejes**.

### Vectores como arrays

Un vector columna matemático se corresponde con un array de una dimensión en código. Por ejemplo, con NumPy:

```python
import numpy as np
x = np.array([1.0, 2.0])  # shape (2,)
```

La vectorización permite que los núcleos optimizados de arrays operen eficientemente sobre **lotes** completos.

### Batches (*lotes*) de ejemplos

Es habitual almacenar los ejemplos en **filas**. Si tenemos $B$ ejemplos con $d_{in}$ características, escribimos:

$$
X \in \mathbb{R}^{B\times d_{in}}
$$

Cada fila de $X$ corresponde a un ejemplo distinto.

## Operaciones con arrays y *broadcasting*

Sea $X \in \mathbb{R}^{B\times d}$, $A \in \mathbb{R}^{B\times d}$ y $a \in \mathbb{R}^{d}$.

| Expresión NumPy | Operación | Forma de salida |
| --- | --- | --- |
| `X @ a` | Suma de productos a lo largo del eje de características | `(B,)` |
| `X * A` | Producto elemento a elemento | `(B, d)` |
| `X + a` | Suma del mismo vector de características a todas las filas (*broadcasting*) | `(B, d)` |
| `X.mean(axis=0)` | Media sobre ejemplos | `(d,)` |
| `X.mean(axis=1)` | Media sobre características | `(B,)` |

El *broadcasting* permite reutilizar valores cuando las formas son compatibles.

## La capa afín (*Linear/Dense*)

Una **capa afín** (también llamada *fully connected* o *densa*) combina las entradas con pesos y sesgo:

$$
z = Wx + b,\quad W \in \mathbb{R}^{d_{out}\times d_{in}},\ x \in \mathbb{R}^{d_{in}},\ b \in \mathbb{R}^{d_{out}},\ z \in \mathbb{R}^{d_{out}}
$$

Componentes por coordenada:

$$
z_j = \sum_{k=1}^{d_{in}} W_{jk}\,x_k + b_j
$$

- Cada fila de $W$ define una coordenada de salida.
- El sesgo $b$ desplaza la salida independientemente de la entrada.
- Con sesgo, una capa afín tiene $d_{out}d_{in} + d_{out}$ parámetros entrenables.
- En PyTorch se denomina `Linear`, en Keras `Dense`.

### Capa oculta: afín + activación

Para una unidad oculta:

$$
z_j = \sum_{k} W_{jk} x_k + b_j,\quad h_j = \sigma(z_j)
$$

Donde $z_j$ es la **pre-activación** y $h_j$ es la **activación**. Varias unidades comparten la misma entrada y se calculan en paralelo.

## La función de activación ReLU

La función *Rectified Linear Unit* (ReLU) se define como:

$$
\operatorname{ReLU}(z) = \max(0, z)
$$

Aplicada elemento a elemento:

$$
h = \operatorname{ReLU}(z)
$$

- Sin parámetros entrenables.
- Cambia de pendiente en $z=0$.
- Es la activación no lineal más habitual en las primeras capas.

## Composición de capas: ¿por qué necesitamos no linealidad?

Si ambas capas son afines:

$$
h = W^{(1)} x + b^{(1)},\quad s = W^{(2)} h + b^{(2)}
$$

Sustituyendo:

$$
s = W^{(2)}W^{(1)}\,x + \bigl(W^{(2)}b^{(1)} + b^{(2)}\bigr) = W_{eff}\,x + b_{eff}
$$

El resultado sigue siendo **afín**. Al insertar una **activación no lineal** entre capas (como ReLU), la composición deja de ser afín y el modelo puede representar funciones mucho más complejas.

## Formas con lotes (*batch*) y convenciones de pesos

Con ejemplos en filas ($X \in \mathbb{R}^{B\times d_{in}}$):

$$
Z = X W^T + \mathbf{1} b^T \in \mathbb{R}^{B\times d_{out}}
$$

El término $b$ se propaga (*broadcast*) a todas las filas.

| Framework | Forma de pesos almacenados | Cálculo |
| --- | --- | --- |
| PyTorch (`nn.Linear`) | $(out, in)$ | `X @ weight.T + bias` |
| Keras (`Dense`) | $(in, out)$ | `X @ kernel + bias` |

Ambas convenciones representan la misma transformación afín.

## Ejemplo: clasificador con una capa oculta

Consideremos una red con 2 características de entrada, 4 unidades ocultas con ReLU y 3 salidas (*logits*):

$$
h = \operatorname{ReLU}(W^{(1)} x + b^{(1)}),\quad s = W^{(2)} h + b^{(2)}
$$

- Las salidas $s \in \mathbb{R}^3$ se llaman **logits** (puntuaciones sin normalizar).
- La predicción se obtiene con $\hat{y} = \arg\max_k s_k$.
- En la sesión 2 convertiremos los *logits* en probabilidades.

## Ejercicio práctico

Para la red 2-4-3:

1. Dar las formas de $W^{(1)}, b^{(1)}, W^{(2)}, b^{(2)}$.
2. Contar el total de parámetros entrenables (incluidos sesgos).
3. Dar las formas de $X, H, S$ para un lote de 16 ejemplos.
4. ¿Añade parámetros entrenables la operación ReLU?

**Solución:** $W^{(1)}\in\mathbb{R}^{4\times 2}, b^{(1)}\in\mathbb{R}^{4}, W^{(2)}\in\mathbb{R}^{3\times 4}, b^{(2)}\in\mathbb{R}^{3}$, total **27** parámetros. Para $B=16$: $X\in\mathbb{R}^{16\times 2}, H\in\mathbb{R}^{16\times 4}, S\in\mathbb{R}^{16\times 3}$. **ReLU no añade parámetros entrenables**.

## Implementación en PyTorch

```python
import torch
from torch import nn

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 3),
)

X = torch.randn(16, 2)
S = model(X)        # (16, 3) - logits
y_hat = S.argmax(dim=1)  # (16,)
```

## Entrenamiento: ¿qué cambia?

- **Antes del entrenamiento:** las regiones de decisión son caóticas o poco útiles; la precisión es cercana al azar.
- **Después del entrenamiento:** las regiones de decisión se ajustan a los datos; la precisión sube drásticamente tanto en entrenamiento como en *test* (*held-out*).
- **Lo que se ajusta:** únicamente los **pesos y sesgos**. Las activaciones ocultas cambian como consecuencia.
- **Lo que guía el entrenamiento:** la **función de pérdida** (*cross-entropy* media). Las métricas en *test* no se usan para actualizar los parámetros.

## Preguntas de repaso (*exit ticket*)

1. ¿Qué cantidades pueden cambiar cuando solo cambia la entrada: pesos, sesgos o activaciones ocultas?
2. ¿Qué cantidades ajusta el entrenamiento?
3. ¿Por qué este modelo incluye una ReLU entre sus capas afines?
4. ¿Son ya probabilidades las tres salidas del modelo?

**Respuestas:**
1. Solo las **activaciones ocultas**.
2. Los **pesos y sesgos**.
3. Porque componer solo capas afines sigue siendo afín; la no linealidad permite representar funciones no afines.
4. **No**. Son **logits** (puntuaciones sin normalizar). Hay que normalizarlos (por ejemplo, con *softmax*) para obtener probabilidades.

## Apéndices

### A1. Formas y 27 parámetros entrenables

| Cantidad | Significado | Forma | Parámetros |
| --- | --- | --- | --- |
| $W^{(1)}$ | Pesos capa oculta | $(4,2)$ | 8 |
| $b^{(1)}$ | Sesgo capa oculta | $(4,)$ | 4 |
| $W^{(2)}$ | Pesos capa salida | $(3,4)$ | 12 |
| $b^{(2)}$ | Sesgo capa salida | $(3,)$ | 3 |

Total: $(4\times 2 + 4) + (3\times 4 + 3) = 12 + 15 = 27$. Para $B=16$: $X\in\mathbb{R}^{16\times 2}, H\in\mathbb{R}^{16\times 4}, S\in\mathbb{R}^{16\times 3}$.

### A2. Pase hacia adelante numérico (capa oculta)

$$
x=\begin{bmatrix}1\\2\end{bmatrix},\quad
W^{(1)}=\begin{bmatrix}
1&0\\
0&1\\
1&-1\\
-1&1
\end{bmatrix},\quad
b^{(1)}=\begin{bmatrix}
0\\0\\0\\-2
\end{bmatrix},\quad
z=W^{(1)}x+b^{(1)}=\begin{bmatrix}1\\2\\-1\\-1\end{bmatrix},\quad
h=\operatorname{ReLU}(z)=\begin{bmatrix}1\\2\\0\\0\end{bmatrix}
$$

### A3. Pase hacia adelante numérico (capa de salida)

$$
h=\begin{bmatrix}1\\2\\0\\0\end{bmatrix},\quad
W^{(2)}=\begin{bmatrix}
1&0&1&0\\
0&1&0&1\\
-1&-1&1&1
\end{bmatrix},\quad
b^{(2)}=\begin{bmatrix}0\\1\\0\end{bmatrix},\quad
s=W^{(2)}h+b^{(2)}=\begin{bmatrix}1\\3\\-3\end{bmatrix},\quad
\hat{y}=\arg\max_k s_k = 1
$$

### A6. Respuestas al *exit ticket*

1. Las **activaciones ocultas** dependen de la entrada. Los pesos y los sesgos son compartidos en un paso fijo.
2. El entrenamiento ajusta los **pesos y los sesgos**.
3. Componer solo capas afines produce una función afín. La función **ReLU** introduce no linealidad, lo que permite representar funciones no afines.
4. **No**. Son *logits*. Se necesita normalizarlos (por ejemplo, con *softmax*) para obtener probabilidades.

## Lecturas recomendadas

- Christopher M. Bishop & Hugh Bishop (2024). *Deep Learning: Foundations and Concepts*. Springer. Capítulos 1 y 6.
- Ian Goodfellow, Yoshua Bengio & Aaron Courville (2016). *Deep Learning*. MIT Press. Capítulos 1, 2, 5 y 6.
- [PyTorch nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- [Keras Dense](https://keras.io/api/layers/core_layers/dense/)
- [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)