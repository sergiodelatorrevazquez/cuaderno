---
titulo: "Fundamentos matemáticos y computacionales del aprendizaje profundo: introducción a las redes neuronales"
asignatura: aprendizaje-profundo
fecha: 2026-10-22
orden: 1
tema: "Semana 1 - Sesión 1"
tags: [fundamentos, tensores, capas-afines, ReLU, perceptrón-multicapa, parámetros, entrenamiento, PyTorch, Keras]
resumen: "Introducción a las redes neuronales como funciones entrenables: tareas de aprendizaje, parámetros y activaciones, tensores y lotes, operaciones vectorizadas, capas afines, ReLU, clasificación y entrenamiento."
---

## Página 1 — Fundamentos matemáticos y computacionales del aprendizaje profundo

**Título:** Fundamentos matemáticos y computacionales del aprendizaje profundo.
**Subtítulo:** Introducción: una función entrenable.
**Curso y sesión:** Aprendizaje profundo, semana 1, sesión 1.
**Duración indicada:** 90 minutos.
**Horario impreso:** jueves, 19:20–20:20.
**Docente:** Francisco R. Villatoro.
**Contexto:** curso de máster en Aprendizaje Profundo. Las diapositivas están en inglés; la discusión en el aula es principalmente en español.

## Página 2 — Un curso, cuatro bloques complementarios

El curso comprende cuatro bloques, cada uno con una pregunta principal y un docente:

| Bloque | Pregunta principal | Docente |
|---|---|---|
| Fundamentos | ¿Cómo representamos un modelo, computamos con él y lo entrenamos? | Francisco |
| Imágenes y vídeo | ¿Cómo resuelven las arquitecturas convolucionales las tareas visuales? | Enrique |
| Secuencias neuronales | ¿Cómo representan las secuencias los *tokens*, la recurrencia y los *transformers*? | Leo |
| Modelos de lenguaje | ¿Cómo contribuyen los n-gramas, los *embeddings* y los modelos de lenguaje a NLP y RAG? | Ezequiel |

El lenguaje compartido por los bloques comprende **datos, tensores, parámetros, predicciones y aprendizaje**. La estructura del curso fue proporcionada por el equipo docente.

## Página 3 — Objetivos y recorrido de la sesión

Al finalizar la sesión, se espera poder:

- Leer una red pequeña como una composición de funciones.
- Seguir la forma de cada array.
- Identificar y contar los parámetros entrenables.
- Explicar qué cambia durante el entrenamiento.

| Minutos | Tema |
|---|---|
| 0–10 | Curso y sesión |
| 10–20 | Preguntas diagnósticas |
| 20–35 | Modelos y aprendizaje |
| 35–55 | Tensores y lotes |
| 55–75 | Capas afines y ReLU |
| 75–90 | Un modelo pequeño completo |

**Conocimientos previos que se repasan:** multiplicación de matrices, derivadas elementales y probabilidad condicional.

## Página 4 — Tres preguntas diagnósticas

### 1. Formas

Si $x\in\mathbb{R}^{2}$ y $z=Wx+b\in\mathbb{R}^{4}$, ¿qué formas tienen $W$ y $b$?

### 2. Derivada a través de dos operaciones

Sean $h=2x+1$ y $u=h^2$. ¿Cuál es $\frac{du}{dx}$ en $x=1$?

### 3. Probabilidad condicional

De 20 mensajes que contienen la palabra *offer*, 15 son *spam*. ¿Qué probabilidad empírica estima $P(\text{spam}\mid\text{offer})$?

**Instrucción de trabajo:** resolver individualmente durante dos minutos y luego comparar las respuestas con otra persona.

## Página 5 — Respuestas diagnósticas

### 1. Formas

$$
W\in\mathbb{R}^{4\times2},\qquad b\in\mathbb{R}^{4}.
$$

Cada fila de $W$ produce una coordenada de salida.

### 2. Regla de la cadena

$$
\frac{du}{dx}
=\frac{du}{dh}\frac{dh}{dx}
=2h\cdot2.
$$

Como $h(1)=3$:

$$
\left.\frac{du}{dx}\right|_{x=1}=12.
$$

### 3. Probabilidad condicional

$$
\widehat{P}(\text{spam}\mid\text{offer})=\frac{15}{20}=0.75.
$$

Condicionar selecciona el grupo de referencia. La condicional inversa utiliza otro grupo de referencia.

## Página 6 — Las tareas de aprendizaje definen entradas y objetivos

| Tarea | Entrada | Objetivo o predicción |
|---|---|---|
| Regresión | Mediciones de sensores | Un valor numérico, como el consumo de energía |
| Clasificación | Una imagen o un vector de características | Una clase, como uno de tres tipos de objetos |
| Predicción del siguiente símbolo | Los símbolos precedentes | Una distribución sobre el siguiente símbolo |

Un conjunto de datos supervisado contiene pares de entrada y objetivo:

$$
\mathcal{D}=\{(x_i,y_i)\}_{i=1}^{N}.
$$

En la predicción autosupervisada, los propios datos pueden proporcionar el objetivo; por ejemplo, el siguiente símbolo de una secuencia.

**Referencias:** Bishop y Bishop, capítulos 1 y 6; Goodfellow y colaboradores, capítulos 1 y 5.

## Página 7 — El aprendizaje profundo construye representaciones intermedias

Una red puede expresarse como una secuencia de transformaciones:

$$
h^{(1)}=f_1(x;\theta_1),\qquad
h^{(2)}=f_2(h^{(1)};\theta_2),\qquad
s=f_3(h^{(2)};\theta_3).
$$

Una **representación** es una descripción de la entrada que una operación posterior puede utilizar. El entrenamiento ajusta los parámetros que producen esas descripciones intermedias.

- **Profundidad:** varias transformaciones sucesivas.
- **Anchura:** número de coordenadas o unidades de una representación oculta.

El primer modelo utiliza una capa oculta para que cada operación siga siendo visible.

**Referencias:** Bishop y Bishop, capítulo 6; Goodfellow y colaboradores, capítulos 1 y 6.

## Página 8 — Parámetros, hiperparámetros y activaciones

El modelo se representa como:

$$
s=f_\theta(x).
$$

| Cantidad | Ejemplo | Función |
|---|---|---|
| Parámetro entrenable | Un peso o un sesgo | El algoritmo de entrenamiento ajusta su valor |
| Hiperparámetro | Anchura de la capa oculta o tasa de aprendizaje | Elección del modelo o del procedimiento de entrenamiento |
| Activación | Un vector oculto $h(x)$ | Cantidad que el modelo calcula para la entrada actual |

Los mismos parámetros entrenados sirven para muchas entradas; sus activaciones pueden diferir.

**Referencia:** Goodfellow y colaboradores, capítulos 5 y 6.

## Página 9 — Entrenamiento e inferencia

### Entrenamiento

Se comparan las predicciones con los objetivos observados, se mide una pérdida y se ajustan los parámetros. Una pérdida media se expresa como:

$$
\mathcal{L}(\theta)=\frac{1}{N}\sum_{i=1}^{N}
\ell\bigl(f_\theta(x_i),y_i\bigr).
$$

### Inferencia

Se evalúa una entrada nueva con los parámetros ya ajustados, que se mantienen fijos.

### Los datos cumplen funciones distintas

- Los datos de **entrenamiento** ajustan los parámetros.
- Los datos de **validación** orientan las decisiones del modelo.
- Los datos de **prueba** evalúan la elección final.

La división debe reflejar el uso previsto, incluida la estructura temporal o por grupos cuando sea pertinente.

**Referencia:** Goodfellow y colaboradores, capítulo 5. La regularización y la evaluación se retoman en la semana 2.

## Página 10 — Los tensores almacenan números a lo largo de ejes con significado

En las bibliotecas numéricas, un **tensor** es un array multidimensional. Sus ejes adquieren significado a partir del modelo y de los datos.

| Objeto | Forma de ejemplo | Significado |
|---|---|---|
| Escalar | `()` | Un valor de pérdida |
| Vector | `(2,)` | Dos características de un ejemplo |
| Matriz | `(32, 2)` | 32 ejemplos con dos características cada uno |
| Array de cuatro ejes | `(32, 3, 64, 64)` | 32 imágenes RGB en disposición canales primero |

El número de ejes y el rango de una matriz en álgebra lineal son conceptos diferentes.

**Referencias:** Goodfellow y colaboradores, capítulo 2; convenciones de tensores y convolución de PyTorch.

## Página 11 — Un ejemplo y un lote de ejemplos

### Un ejemplo: vector columna

$$
x_1=\begin{bmatrix}1\\2\end{bmatrix}\in\mathbb{R}^{2}.
$$

Sus dos coordenadas son características. En Python, un array con forma `(2,)` tiene un eje.

### Un lote: un ejemplo por fila

$$
X=
\begin{bmatrix}
x_1^\top\\
x_2^\top\\
x_3^\top
\end{bmatrix}
=
\begin{bmatrix}
1&2\\
0&-1\\
2&0
\end{bmatrix}
\in\mathbb{R}^{3\times2}.
$$

La matriz contiene 3 ejemplos por 2 características. En los ejemplos tabulares, $X$ tiene forma $(B,d)$: la fila $b$ identifica un ejemplo y la columna $j$ una característica:

$$
X_{bj}=(x_b)_j.
$$

Esta notación se utiliza en todo el bloque.

## Página 12 — Los mismos números pueden tener distribuciones de ejes diferentes

Para un lote de imágenes RGB pueden usarse dos convenciones:

- **Canales primero:** $(B,C,H,W)$.
- **Canales al final:** $(B,H,W,C)$.

| Convención | Forma concreta | Dónde aparece |
|---|---|---|
| Canales primero | `(32, 3, 64, 64)` | `Conv2d` de PyTorch |
| Canales al final | `(32, 64, 64, 3)` | `Conv2D` de Keras con `channels_last` |

Cambiar el orden de los ejes requiere una permutación. Hay que nombrar los ejes antes de elegir una operación `reshape` o una transposición.

**Referencias:** documentación oficial de `Conv2d` de PyTorch y `Conv2D` de Keras; enlaces en las referencias.

## Página 13 — La vectorización aplica una regla a un lote

Para cada ejemplo se quiere calcular:

$$
r_b=X_{b1}+2X_{b2}.
$$

### Un bucle sobre los ejemplos

```python
import numpy as np

r = np.zeros(X.shape[0])
for b in range(X.shape[0]):
    r[b] = X[b, 0] + 2 * X[b, 1]
```

### Una operación sobre arrays

```python
a = np.array([1.0, 2.0])
r = X @ a
```

Para los datos de la página 11:

$$
\underbrace{
\begin{bmatrix}
1&2\\
0&-1\\
2&0
\end{bmatrix}}_{(3,2)}
\underbrace{
\begin{bmatrix}1\\2\end{bmatrix}}_{(2,)}
=
\underbrace{
\begin{bmatrix}5\\-2\\2\end{bmatrix}}_{(3,)}.
$$

Los vectores columna matemáticos corresponden en este código a arrays de un eje. La vectorización permite que núcleos de arrays optimizados procesen el lote.

**Nota de la diapositiva:** ejemplo numérico original y notación NumPy.

## Página 14 — Las operaciones con arrays tienen significados distintos

Sean $X,A\in\mathbb{R}^{B\times d}$ y $a\in\mathbb{R}^{d}$.

| Expresión NumPy | Operación | Forma de salida |
|---|---|---|
| `X @ a` | Suma de productos a lo largo del eje de características | `(B,)` |
| `X * A` | Multiplicación de elementos correspondientes | `(B, d)` |
| `X + a` | Suma del mismo vector de características a cada fila | `(B, d)` |
| `X.mean(axis=0)` | Media sobre los ejemplos | `(d,)` |
| `X.mean(axis=1)` | Media sobre las características | `(B,)` |

El *broadcasting* reutiliza valores a lo largo de ejes compatibles.

**Referencia:** convenciones de broadcasting y operaciones de arrays de NumPy.

## Página 15 — Una capa afín combina las coordenadas de entrada

La transformación de una capa se expresa como:

$$
z=Wx+b,\qquad
W\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}},
\quad
x\in\mathbb{R}^{d_{\mathrm{in}}},
\quad
b\in\mathbb{R}^{d_{\mathrm{out}}},
\quad
z\in\mathbb{R}^{d_{\mathrm{out}}}.
$$

La coordenada de salida $j$ es:

$$
z_j=\sum_{k=1}^{d_{\mathrm{in}}}W_{jk}x_k+b_j.
$$

- Cada fila de $W$ define una coordenada de salida.
- El sesgo $b$ desplaza la salida independientemente de la entrada.
- Con sesgo, una capa totalmente conectada tiene $d_{\mathrm{out}}d_{\mathrm{in}}+d_{\mathrm{out}}$ parámetros.
- En el software se llama `Linear` en PyTorch y `Dense` en Keras.

Con sesgo, la transformación matemática es **afín**.

**Referencias:** Bishop y Bishop, capítulo 6; documentación oficial de `Linear` de PyTorch y `Dense` de Keras.

## Página 16 — Una unidad oculta: transformación afín y activación

El esquema representa dos entradas, $x_1$ y $x_2$, conectadas a una unidad. Los pesos entrenables $W_{j1}$ y $W_{j2}$ y el sesgo $b_j$ producen la preactivación:

$$
z_j=W_{j1}x_1+W_{j2}x_2+b_j.
$$

La función de activación $\phi$ transforma esa preactivación en la activación calculada:

$$
h_j=\phi(z_j).
$$

La ilustración distingue los **pesos y sesgo entrenables** de la **activación calculada**. Una capa evalúa varias unidades de este tipo usando el mismo vector de entrada. $z_j$ es la **preactivación** y $h_j$ la **activación**.

**Referencia:** Bishop y Bishop, capítulo 6; esquema original.

## Página 17 — ReLU introduce un cambio de pendiente

La unidad lineal rectificada (**ReLU**, *Rectified Linear Unit*) se define como:

$$
\operatorname{ReLU}(z)=\max(0,z).
$$

El gráfico tiene $z$ en el eje horizontal, con marcas en $-2,-1,1,2$, y $\operatorname{ReLU}(z)$ en el vertical, con marcas en 1 y 2. Para $z<0$ el resultado es cero; para $z>0$ sigue la recta de pendiente 1. La pendiente cambia en cero.

Aplicada a cada coordenada:

$$
\begin{bmatrix}
1\\2\\-1\\-1
\end{bmatrix}
\longmapsto
\begin{bmatrix}
1\\2\\0\\0
\end{bmatrix}.
$$

ReLU no tiene parámetros entrenables. La derivada y las convenciones de diferenciación automática se retoman en la sesión 3.

**Referencia:** Goodfellow y colaboradores, capítulo 6.

## Página 18 — Dos capas afines se componen en una sola transformación afín

Supongamos que ambas etapas son afines:

$$
h=W^{(1)}x+b^{(1)},\qquad
s=W^{(2)}h+b^{(2)}.
$$

Al sustituir:

$$
s=
\underbrace{W^{(2)}W^{(1)}}_{W_{\mathrm{eff}}}x
+
\underbrace{W^{(2)}b^{(1)}+b^{(2)}}_{b_{\mathrm{eff}}}.
$$

La función resultante sigue siendo afín en $x$. Intercalar una activación no lineal permite una composición generalmente no afín.

**Referencias:** derivación original; véanse Bishop y Bishop, y Goodfellow y colaboradores, capítulo 6.

## Página 19 — Los lotes en filas explican las formas almacenadas de los pesos

Con ejemplos en filas y $X\in\mathbb{R}^{B\times d_{\mathrm{in}}}$:

$$
Z=XW^\top+\mathbf{1}_B b^\top
\in\mathbb{R}^{B\times d_{\mathrm{out}}}.
$$

El segundo término añade el mismo vector de sesgo a todas las filas.

| Biblioteca | Forma almacenada de los pesos | Cálculo sobre arrays |
|---|---|---|
| PyTorch | `(out, in)` | `X @ weight.T + bias` |
| Keras | `(in, out)` | `X @ kernel + bias` |

Ambas convenciones representan la misma transformación afín cuando sus pesos se corresponden por transposición.

**Referencias:** documentación oficial de `Linear` de PyTorch y `Dense` de Keras.

## Página 20 — Clasificador pequeño con una capa oculta

El esquema muestra dos características de entrada, cuatro activaciones ocultas y tres puntuaciones de salida. Las dos transformaciones afines tienen parámetros entrenables $W^{(1)},b^{(1)}$ y $W^{(2)},b^{(2)}$; entre ellas se aplica ReLU:

$$
h=\operatorname{ReLU}\bigl(W^{(1)}x+b^{(1)}\bigr),
\qquad
s=W^{(2)}h+b^{(2)}.
$$

Hay dos transformaciones afines entrenables y una operación ReLU. Las puntuaciones de salida se llaman **logits**. La sesión 2 las convertirá en probabilidades.

El esquema original muestra las conexiones entre las entradas $x_1,x_2$, las cuatro unidades ocultas $h_1,\ldots,h_4$ y las tres salidas $s_0,s_1,s_2$. Las ecuaciones incluyen los sesgos.

## Página 21 — Tarea concreta: tres clases de puntos

La entrada es un punto de dos coordenadas:

$$
x=(x_1,x_2)^\top.
$$

El objetivo es una etiqueta de clase $y\in\{0,1,2\}$. La red calcula tres puntuaciones:

$$
s=f_\theta(x),\qquad
\hat{y}=\arg\max_{k\in\{0,1,2\}}s_k.
$$

El gráfico del conjunto de prueba presenta 150 puntos sintéticos en un plano cuyos ejes $x_1$ y $x_2$ abarcan aproximadamente de $-1.8$ a $1.8$. Los colores y marcadores indican las etiquetas observadas: la clase 0 aparece como puntos azules agrupados cerca del origen; la clase 1, como cuadrados naranjas en una corona intermedia; y la clase 2, como triángulos verdes en una corona exterior.

**Referencia de la figura:** ejemplo sintético original; el conjunto de datos y el protocolo se describen en el apéndice A7.

## Página 22 — Ejercicio: formas y parámetros entrenables

Para la red con dos entradas, cuatro unidades ocultas y tres salidas:

$$
h=\operatorname{ReLU}\bigl(W^{(1)}x+b^{(1)}\bigr),
\qquad
s=W^{(2)}h+b^{(2)}.
$$

1. Indicar las formas de $W^{(1)},b^{(1)},W^{(2)},b^{(2)}$.
2. Contar todos los parámetros escalares entrenables, incluidos los sesgos.
3. Indicar las formas de $X,H,S$ para un lote de 16 ejemplos.
4. ¿Añade ReLU parámetros entrenables?

**Solución desarrollada:** apéndice A1.

## Página 23 — El mismo modelo en código

```python
import torch
from torch import nn

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 3),
)

X = torch.randn(16, 2)
S = model(X)  # (16, 3)
y_hat = S.argmax(dim=1)  # (16,)
```

### Seguimiento de las formas

$$
X:(16,2),\qquad
Z,H:(16,4),\qquad
S:(16,3).
$$

Los sesgos se propagan a todas las filas. $S$ contiene logits. Calcular `model(X)` evalúa el modelo actual.

La arquitectura equivalente en Keras se encuentra en el apéndice A4.

**Referencias:** API oficiales de `Linear` y `Sequential` de PyTorch; ilustración de la arquitectura.

## Página 24 — El entrenamiento cambia las regiones de decisión

La figura compara dos paneles, **antes del entrenamiento** y **después del entrenamiento**, sobre los mismos ejes $x_1,x_2$, aproximadamente de $-1.8$ a $1.8$. El fondo de cada panel indica la clase predicha; los marcadores superpuestos son las mismas 150 etiquetas observadas del conjunto reservado.

- **Antes:** el fondo se divide en regiones amplias, angulares y aproximadamente triangulares que no reproducen la organización radial de los puntos.
- **Después:** las regiones se ordenan alrededor del origen en zonas aproximadamente concéntricas: una región central para la clase 0, una corona intermedia para la clase 1 y una zona exterior para la clase 2.

La leyenda identifica la clase 0 con azul y círculos, la clase 1 con naranja y cuadrados y la clase 2 con verde y triángulos.

El modelo ajusta sus **27 parámetros** utilizando 300 ejemplos de entrenamiento distintos.

**Referencia de la figura:** cálculo original con una inicialización y un protocolo de entrenamiento fijos; véase el apéndice A7.

## Página 25 — Pérdida, exactitud y datos utilizados para ajustar el modelo

### Curva de pérdida

El gráfico de la izquierda muestra la entropía cruzada media, en *nats*, frente a las actualizaciones de entrenamiento de 0 a 5.000. La curva comienza por encima de 1, desciende con rapidez en las primeras actualizaciones y después se aproxima gradualmente a cero. La pérdida guía el entrenamiento; aquí se usa entropía cruzada media, que se introduce en la sesión 2.

### Exactitud antes y después

| Conjunto | Antes | Después |
|---|---:|---:|
| Entrenamiento | 35.0 % | 98.3 % |
| Reservado para prueba (*held out*) | 35.3 % | 97.3 % |

La exactitud cuenta las etiquetas predichas correctamente. Se realizaron 5.000 actualizaciones de parámetros con un protocolo fijo.

El ejemplo es un experimento sintético original. La curva usa solo los datos de entrenamiento; los datos reservados no guían las decisiones del modelo.

## Página 26 — Preguntas de salida

Responder:

1. Si solo cambia la entrada, ¿qué cantidades pueden cambiar: los pesos, los sesgos o las activaciones ocultas?
2. ¿Qué cantidades ajusta el entrenamiento?
3. ¿Por qué este modelo incluye una ReLU entre sus capas afines?
4. ¿Son ya probabilidades sus tres puntuaciones de salida?

### Próxima sesión

Probabilidades condicionales, softmax, temperatura y entropía cruzada.

Las respuestas desarrolladas están en el apéndice A6. A continuación aparecen referencias y cálculos adicionales.

## Página 27 — Apéndice A1. Formas y 27 parámetros entrenables

| Cantidad | Significado | Forma del array | Parámetros |
|---|---|---|---:|
| $W^{(1)}$ | Pesos ocultos | `(4, 2)` | 8 |
| $b^{(1)}$ | Sesgo oculto | `(4,)` | 4 |
| $h$ | Activación oculta | `(4,)` | 0 |
| $W^{(2)}$ | Pesos de salida | `(3, 4)` | 12 |
| $b^{(2)}$ | Sesgo de salida | `(3,)` | 3 |

El recuento es:

$$
(4\cdot2+4)+(3\cdot4+3)=12+15=27.
$$

Para $B=16$: $X:(16,2)$, $H:(16,4)$ y $S:(16,3)$. Los pesos con valor cero siguen siendo parámetros entrenables. La propagación del sesgo y ReLU no añaden parámetros.

La diapositiva indica volver al ejercicio.

## Página 28 — Apéndice A2. Pase hacia delante numérico: capa oculta

Los pesos enteros pequeños sirven para ilustrar el cálculo y son distintos de los pesos entrenados en el experimento:

$$
x=\begin{bmatrix}1\\2\end{bmatrix},\qquad
W^{(1)}=
\begin{bmatrix}
1&0\\
0&1\\
1&-1\\
-1&1
\end{bmatrix},\qquad
b^{(1)}=
\begin{bmatrix}
0\\0\\0\\-2
\end{bmatrix}.
$$

La preactivación y la activación resultan:

$$
z=W^{(1)}x+b^{(1)}
=
\begin{bmatrix}
1\\2\\-1\\-1
\end{bmatrix},
\qquad
h=\operatorname{ReLU}(z)
=
\begin{bmatrix}
1\\2\\0\\0
\end{bmatrix}.
$$

El vector oculto es una representación calculada para esta entrada concreta.

## Página 29 — Apéndice A3. Pase hacia delante numérico: capa de salida

Se continúa con el vector oculto del apéndice A2:

$$
h=\begin{bmatrix}1\\2\\0\\0\end{bmatrix},\qquad
W^{(2)}=
\begin{bmatrix}
1&0&1&0\\
0&1&0&1\\
-1&-1&1&1
\end{bmatrix},\qquad
b^{(2)}=
\begin{bmatrix}0\\1\\0\end{bmatrix}.
$$

La salida es:

$$
s=W^{(2)}h+b^{(2)}
=
\begin{bmatrix}1\\3\\-3\end{bmatrix},
\qquad
\hat{y}=\arg\max_{k\in\{0,1,2\}}s_k=1.
$$

La puntuación mayor corresponde a la clase 1, utilizando etiquetas de clase que empiezan en cero. Las puntuaciones pueden ser negativas y no tienen por qué sumar uno.

## Página 30 — Apéndice A4. Arquitecturas equivalentes en PyTorch y Keras

### PyTorch

```python
from torch import nn

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 3),
)
n = sum(p.numel() for p in model.parameters())
# n = 27
```

### Keras con TensorFlow

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(3),
])
n = model.count_params()
# n = 27
```

Ambos modelos tienen un sesgo en cada capa densa y producen tres logits. Para obtener predicciones coincidentes, hay que asignar explícitamente los pesos y sesgos correspondientes.

**Referencias:** documentación oficial de `Linear` de PyTorch y `Dense` de Keras.

## Página 31 — Apéndice A5. Vectores columna, formas y orden de ejes

### Un vector matemático puede tener varias representaciones de array

```python
import numpy as np

x = np.array([1.0, 2.0])  # forma (2,)
x_col = x[:, None]        # forma (2, 1)
x_row = x[None, :]        # forma (1, 2)
```

### Las entradas de lotes siguen una convención de ejes declarada

```python
# X guarda B ejemplos como filas, con d características.
Z = X @ W.T + b  # (B, d_out)

# Disposición de imagen: (B, C, H, W) -> (B, H, W, C)
images_last = images_first.transpose(0, 2, 3, 1)
```

Un `reshape` cambia el agrupamiento de las entradas. Una permutación de ejes cambia qué eje ocupa cada posición.

**Nota:** convenciones de arrays de NumPy. El ejemplo de imágenes supone un array NumPy.

## Página 32 — Apéndice A6. Respuestas desarrolladas al *exit ticket*

1. Las activaciones ocultas dependen de la entrada actual. Los pesos y los sesgos se comparten entre ejemplos en un paso fijo del entrenamiento.
2. En este modelo, el entrenamiento ajusta los pesos y los sesgos. Las activaciones ocultas cambian como consecuencia.
3. La ReLU intermedia permite que la red represente funciones que, en general, no son afines. Componer transformaciones afines solamente da como resultado una transformación afín.
4. Los valores de salida son logits. Una normalización adecuada, que se introduce en la sesión 2, los convierte en probabilidades de clase.

La diapositiva indica volver al *exit ticket*.

## Página 33 — Apéndice A7. Experimento sintético y reproducibilidad

### Datos

Hay 100 puntos de entrenamiento y 50 puntos reservados para prueba por clase. Se toma un ángulo uniforme y se reescala su dirección para que tenga norma $L_1$ igual a uno. Después se multiplica por un radio elegido uniformemente de los siguientes intervalos, según la clase:

| Clase | Intervalo del radio |
|---|---|
| 0 | $[0.04,0.45]$ |
| 1 | $[0.75,1.00]$ |
| 2 | $[1.30,1.65]$ |

### Modelo y entrenamiento

- Dos entradas, cuatro unidades ocultas ReLU, tres logits de salida y sesgos en ambas capas afines.
- Implementación NumPy con 27 parámetros en precisión `float64`.
- Adam de lote completo (*full-batch*), 5.000 actualizaciones.
- Tasa de aprendizaje: 0.015.
- $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$.
- Semilla de los datos: `20261017`.
- Semilla de inicialización: `17`.

### Evaluación

Se usa un protocolo fijo. Solo los datos de entrenamiento determinan las actualizaciones y la curva de pérdida. Las métricas del conjunto reservado se calculan al inicializar y después de la última actualización.

Es un ejemplo original. Los materiales que lo acompañan incluyen los datos, los parámetros finales y el guion completo de reproducción.

## Página 34 — Apéndice A8. Lecturas y referencias

### Libros de texto

- Christopher M. Bishop y Hugh Bishop (2024). *Deep Learning: Foundations and Concepts*. Springer, capítulos 1 y 6. [bishopbook.com](https://www.bishopbook.com/)
- Ian Goodfellow, Yoshua Bengio y Aaron Courville (2016). *Deep Learning*. MIT Press, capítulos 1, 2, 5 y 6. [deeplearningbook.org](https://www.deeplearningbook.org/)

### Referencias oficiales de implementación

- [`Linear` de PyTorch](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html) y [`Conv2d` de PyTorch](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html).
- [`Dense` de Keras](https://keras.io/api/layers/core_layers/dense/) y [`Conv2D` de Keras](https://keras.io/api/layers/convolution_layers/convolution2d/).
- [*Broadcasting* de NumPy](https://numpy.org/doc/stable/user/basics.broadcasting.html) y [multiplicación matricial de NumPy](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html).

### Repaso sugerido

Volver a resolver las preguntas diagnósticas y el ejemplo de los 27 parámetros. Seguir el pase numérico hacia delante de los apéndices A2 y A3.

Todos los diagramas y figuras numéricas de esta presentación son originales. La documentación se comprobó en octubre de 2026.
