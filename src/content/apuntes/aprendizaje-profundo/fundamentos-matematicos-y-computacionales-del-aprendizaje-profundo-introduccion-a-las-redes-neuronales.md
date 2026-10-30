---
titulo: "Fundamentos matemáticos y computacionales del aprendizaje profundo: introducción a las redes neuronales"
asignatura: aprendizaje-profundo
fecha: 2026-10-22
orden: 1
tema: "Semana 1 - Sesión 1"
tags: [fundamentos, tensores, capas-afines, relu, perceptron-multicapa, parametros, entrenamiento, pytorch, keras]
resumen: "Fundamentos de las redes neuronales como funciones entrenables: tareas de aprendizaje, tensores y lotes, operaciones vectorizadas, capas afines, ReLU, formas, parámetros, entrenamiento y un experimento reproducible de clasificación."
---

# Fundamentos matemáticos y computacionales del aprendizaje profundo

## Contexto y objetivos de la sesión

Esta sesión introductoria de 90 minutos presenta una red neuronal como una **función entrenable** y construye un clasificador pequeño para hacer visibles sus operaciones. Las diapositivas están en inglés; la discusión de clase se plantea principalmente en español.

Al finalizar, se debería poder:

- Leer una red pequeña como una composición de funciones.
- Seguir las dimensiones (*shapes*) de los arrays en cada operación.
- Identificar y contar sus parámetros entrenables.
- Explicar qué cantidades cambian durante el entrenamiento y cuáles al cambiar la entrada.

**Secuencia de la sesión:** curso y contexto (0-10 min), preguntas diagnósticas (10-20), modelos y aprendizaje (20-35), tensores y lotes (35-55), capas afines y ReLU (55-75), y construcción de un modelo completo (75-90).

**Conocimientos previos que se repasan:** multiplicación de matrices, derivadas elementales y probabilidad condicional.

## Estructura del curso

El curso se organiza en cuatro bloques complementarios:

| Bloque | Pregunta principal | Docente |
|---|---|---|
| Fundamentos | ¿Cómo representamos un modelo, computamos con él y lo entrenamos? | Francisco R. Villatoro |
| Imágenes y vídeo | ¿Cómo resuelven las arquitecturas convolucionales las tareas visuales? | Enrique |
| Secuencias neuronales | ¿Cómo representan las secuencias los *tokens*, la recurrencia y los *transformers*? | Leo |
| Modelos de lenguaje | ¿Cómo contribuyen los n-gramas, los *embeddings* y los modelos de lenguaje a NLP y RAG? | Ezequiel |

El lenguaje común entre los bloques son los **datos, tensores, parámetros, predicciones y aprendizaje**.

## Tareas de aprendizaje

La tarea determina qué se usa como entrada y qué produce o predice el modelo:

| Tarea | Entrada | Objetivo o predicción |
|---|---|---|
| Regresión | Mediciones de sensores | Un valor numérico, como el consumo de energía |
| Clasificación | Una imagen o un vector de características | Una clase, por ejemplo, uno de tres tipos de objetos |
| Predicción del siguiente símbolo | Los símbolos anteriores | Una distribución sobre el siguiente símbolo |

En aprendizaje supervisado, un conjunto de datos está formado por pares de entrada y objetivo:

$$
\mathcal{D} = \{(x_i,y_i)\}_{i=1}^{N}.
$$

En la predicción autosupervisada, los propios datos pueden proporcionar el objetivo. Por ejemplo, a partir de una secuencia se puede entrenar un modelo para predecir el siguiente símbolo.

## Un modelo como función entrenable

Un modelo se representa como una función parametrizada:

$$
s = f_\theta(x).
$$

La entrada $x$ se transforma en una salida o predicción. En una red profunda, el cálculo se descompone en transformaciones sucesivas y representaciones intermedias:

$$
h^{(1)} = f_1(x;\theta_1),\qquad
h^{(2)} = f_2(h^{(1)};\theta_2),\qquad
s = f_3(h^{(2)};\theta_3).
$$

Una **representación** es una descripción de la entrada que una operación posterior puede utilizar. El entrenamiento ajusta los parámetros que producen esas representaciones intermedias.

- **Profundidad**: número de transformaciones sucesivas.
- **Anchura**: número de coordenadas o unidades de una representación oculta.

En este primer modelo se utiliza una sola capa oculta para poder seguir cada operación.

### Parámetros, hiperparámetros y activaciones

Estas cantidades tienen papeles distintos:

| Cantidad | Ejemplo | Función |
|---|---|---|
| Parámetro entrenable | Un peso o un sesgo | El algoritmo de entrenamiento ajusta su valor |
| Hiperparámetro | Anchura de la capa oculta o tasa de aprendizaje | Elección del modelo o del procedimiento de entrenamiento |
| Activación | Un vector oculto $h(x)$ | Resultado calculado por el modelo para una entrada concreta |

Los mismos parámetros entrenados se reutilizan para muchas entradas; las activaciones pueden cambiar de una entrada a otra.

### Entrenamiento, inferencia y particiones de datos

Durante el **entrenamiento**, se comparan las predicciones con los objetivos observados, se calcula una pérdida y se ajustan los parámetros. Una pérdida empírica media puede escribirse como:

$$
\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^{N}\ell\bigl(f_\theta(x_i),y_i\bigr).
$$

Durante la **inferencia**, el modelo evalúa una nueva entrada con los parámetros ya ajustados, que se mantienen fijos.

Los datos suelen separarse según su función:

- **Entrenamiento**: se utiliza para ajustar los parámetros.
- **Validación**: se utiliza para orientar elecciones del modelo o del procedimiento.
- **Prueba** (*test*): se utiliza para evaluar la elección final.

La partición debe reflejar el uso previsto. Por ejemplo, si los datos tienen estructura temporal o por grupos, esa estructura debe tenerse en cuenta para evitar una evaluación engañosa.

## Preguntas diagnósticas

### 1. Formas de una transformación lineal

Si $x\in\mathbb{R}^{2}$ y $z=Wx+b\in\mathbb{R}^{4}$, ¿qué formas tienen $W$ y $b$?

$$
W\in\mathbb{R}^{4\times2},\qquad b\in\mathbb{R}^{4}.
$$

Cada fila de $W$ produce una coordenada de salida.

### 2. Derivada a través de dos operaciones

Sean $h=2x+1$ y $u=h^2$. Para calcular $\frac{du}{dx}$ en $x=1$, se aplica la regla de la cadena:

$$
h(1)=3,\qquad
\frac{du}{dh}=2h=6,\qquad
\frac{dh}{dx}=2,\qquad
\left.\frac{du}{dx}\right|_{x=1}
= \frac{du}{dh}\frac{dh}{dx}=12.
$$

### 3. Probabilidad condicional

De 20 mensajes que contienen la palabra *offer*, 15 son *spam*. La probabilidad empírica estimada es:

$$
\widehat{P}(\text{spam}\mid\text{offer}) = \frac{15}{20}=0.75.
$$

La condición selecciona el grupo de referencia: mensajes que contienen *offer*. La probabilidad inversa, $P(\text{offer}\mid\text{spam})$, usaría otro grupo de referencia y no tiene por qué ser igual.

## Tensores, vectores y lotes

En las bibliotecas numéricas, un **tensor** es un array multidimensional. Sus ejes adquieren significado a partir de los datos y del modelo.

| Objeto | Forma de ejemplo | Significado |
|---|---|---|
| Escalar | `()` | Un valor, por ejemplo, una pérdida |
| Vector | `(2,)` | Dos características de un ejemplo |
| Matriz | `(32, 2)` | 32 ejemplos con dos características cada uno |
| Array de cuatro ejes | `(32, 3, 64, 64)` | 32 imágenes RGB en disposición canales primero |

El **número de ejes** de un tensor y el **rango lineal-algebraico** de una matriz son conceptos diferentes.

### Un ejemplo y su representación en código

Matemáticamente, un ejemplo con dos características puede escribirse como un vector columna:

$$
x_1 = \begin{bmatrix}1\\2\end{bmatrix}\in\mathbb{R}^{2}.
$$

En NumPy, un array de una dimensión tiene forma `(2,)`; no es explícitamente un vector fila ni columna:

```python
import numpy as np

x = np.array([1.0, 2.0])  # shape (2,)
x_col = x[:, None]        # shape (2, 1)
x_row = x[None, :]        # shape (1, 2)
```

### Lotes: una fila por ejemplo

Para procesar varios ejemplos juntos, es habitual almacenarlos en las filas. Por ejemplo:

$$
X =
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

Hay tres ejemplos y dos características. En general, si $B$ es el tamaño del lote (*batch*) y $d$ el número de características, $X$ tiene forma $(B,d)$. La fila $b$ identifica el ejemplo y la columna $j$ la característica:

$$
X_{bj} = (x_b)_j.
$$

### Disposición de ejes en imágenes

Las mismas imágenes pueden almacenarse con distintos órdenes de ejes:

- **Canales primero**: $(B,C,H,W)$; por ejemplo, `(32, 3, 64, 64)`.
- **Canales al final**: $(B,H,W,C)$; por ejemplo, `(32, 64, 64, 3)`.

PyTorch `Conv2d` utiliza habitualmente canales primero; Keras `Conv2D` admite canales al final. Para cambiar el orden hay que permutar los ejes. Conviene nombrar los ejes antes de hacer un `reshape` o una transposición: cambiar el agrupamiento de elementos no es lo mismo que cambiar la posición de los ejes.

```python
# De (B, C, H, W) a (B, H, W, C), con un array NumPy.
images_last = images_first.transpose(0, 2, 3, 1)
```

## Vectorización y operaciones con arrays

La **vectorización** aplica una misma regla a un lote sin escribir explícitamente un bucle en Python para cada ejemplo. Para el lote anterior, se puede calcular $r_b=X_{b1}+2X_{b2}$ con un bucle:

```python
r = np.zeros(X.shape[0])
for b in range(X.shape[0]):
    r[b] = X[b, 0] + 2 * X[b, 1]
```

O con una operación matricial:

```python
a = np.array([1.0, 2.0])
r = X @ a
```

En el ejemplo, el cálculo es:

$$
\begin{bmatrix}
1&2\\
0&-1\\
2&0
\end{bmatrix}
\begin{bmatrix}1\\2\end{bmatrix}
=
\begin{bmatrix}5\\-2\\2\end{bmatrix}.
$$

Los vectores columna de las expresiones matemáticas corresponden a arrays de un eje en este código. La vectorización permite que los núcleos optimizados de arrays procesen el lote.

Para $X,A\in\mathbb{R}^{B\times d}$ y $a\in\mathbb{R}^{d}$:

| Expresión NumPy | Operación | Forma de salida |
|---|---|---|
| `X @ a` | Suma de productos a lo largo del eje de características | `(B,)` |
| `X * A` | Multiplicación elemento a elemento | `(B, d)` |
| `X + a` | Suma del mismo vector de características a cada fila | `(B, d)` |
| `X.mean(axis=0)` | Media sobre los ejemplos | `(d,)` |
| `X.mean(axis=1)` | Media sobre las características | `(B,)` |

El *broadcasting* reutiliza valores en ejes compatibles. Por ejemplo, `X + a` aplica el vector $a$ a cada fila de $X$.

## Capas afines

Una capa totalmente conectada o **densa** combina las coordenadas de entrada mediante pesos y un sesgo:

$$
z = Wx+b,
\qquad
W\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}},
\quad
x\in\mathbb{R}^{d_{\mathrm{in}}},
\quad
b,z\in\mathbb{R}^{d_{\mathrm{out}}}.
$$

La coordenada $j$ de la salida es:

$$
z_j = \sum_{k=1}^{d_{\mathrm{in}}}W_{jk}x_k+b_j.
$$

- Cada fila de $W$ define una coordenada de salida.
- El sesgo $b$ desplaza cada coordenada de salida independientemente de la entrada.
- Con sesgo, la capa tiene $d_{\mathrm{out}}d_{\mathrm{in}}+d_{\mathrm{out}}$ parámetros entrenables.
- En PyTorch se llama `Linear`; en Keras, `Dense`.

Aunque a menudo se le llama capa “lineal”, al incluir un sesgo la transformación matemática es **afín**.

### Unidad oculta: transformación afín y activación

Una unidad oculta aplica primero una transformación afín y luego una función de activación:

$$
z_j = \sum_k W_{jk}x_k+b_j,\qquad h_j=\phi(z_j).
$$

$z_j$ es la **preactivación** y $h_j$ la **activación**. Varias unidades usan la misma entrada y se calculan en paralelo. En el esquema de una unidad con dos entradas, $z_j=W_{j1}x_1+W_{j2}x_2+b_j$ y la activación transforma ese resultado en $h_j$.

### Función de activación ReLU

La unidad lineal rectificada, **ReLU** (*Rectified Linear Unit*), se define como:

$$
\operatorname{ReLU}(z)=\max(0,z).
$$

Se aplica elemento a elemento. Por ejemplo:

$$
\begin{bmatrix}1\\2\\-1\\-1\end{bmatrix}
\longmapsto
\begin{bmatrix}1\\2\\0\\0\end{bmatrix}.
$$

ReLU no tiene parámetros entrenables y su pendiente cambia en cero. El estudio de su derivada y de la diferenciación automática se retoma más adelante en el curso.

## Por qué insertar una no linealidad

Si dos etapas son afines:

$$
h=W^{(1)}x+b^{(1)},\qquad
s=W^{(2)}h+b^{(2)},
$$

al sustituir la primera expresión en la segunda se obtiene:

$$
s =
\underbrace{W^{(2)}W^{(1)}}_{W_{\mathrm{eff}}}x
+
\underbrace{W^{(2)}b^{(1)}+b^{(2)}}_{b_{\mathrm{eff}}}.
$$

La composición sigue siendo afín en $x$. Intercalar una activación no lineal, como ReLU, hace que la composición sea, en general, no afín y permite representar relaciones más complejas.

## Lotes y convenciones para los pesos

Con ejemplos en filas, $X\in\mathbb{R}^{B\times d_{\mathrm{in}}}$, la transformación por lotes es:

$$
Z = XW^\top + \mathbf{1}_B b^\top
\in\mathbb{R}^{B\times d_{\mathrm{out}}}.
$$

El término del sesgo se añade a todas las filas mediante *broadcasting*.

| Framework | Forma almacenada de los pesos | Cálculo para un lote |
|---|---|---|
| PyTorch (`nn.Linear`) | `(out, in)` | `X @ weight.T + bias` |
| Keras (`Dense`) | `(in, out)` | `X @ kernel + bias` |

Las dos convenciones representan la misma transformación afín si los pesos correspondientes se relacionan por transposición.

## Clasificador con una capa oculta

Consideremos una red con dos características de entrada, cuatro unidades ocultas con ReLU y tres salidas:

$$
h = \operatorname{ReLU}\bigl(W^{(1)}x+b^{(1)}\bigr),
\qquad
s = W^{(2)}h+b^{(2)}.
$$

Las tres coordenadas de $s\in\mathbb{R}^{3}$ son **logits**, es decir, puntuaciones sin normalizar. La clase predicha se obtiene mediante:

$$
\hat{y} = \arg\max_{k\in\{0,1,2\}}s_k.
$$

Los logits pueden ser negativos y no tienen por qué sumar uno: todavía no son probabilidades. La sesión siguiente introduce la conversión de puntuaciones a probabilidades.

### Tarea sintética de tres clases

La entrada es un punto $x=(x_1,x_2)^\top$ y el objetivo es una etiqueta $y\in\{0,1,2\}$. En la figura del conjunto de prueba aparecen 150 puntos sintéticos: la clase 0 (azul) se concentra cerca del centro, la clase 1 (naranja) forma una corona intermedia y la clase 2 (verde) una corona exterior. Las coordenadas de los puntos son las entradas; el color y el marcador representan las etiquetas observadas. El modelo produce tres puntuaciones para cada punto y predice la clase de mayor puntuación.

### Ejercicio: formas y parámetros

Para la red de dos entradas, cuatro unidades ocultas y tres salidas:

1. Indicar las formas de $W^{(1)},b^{(1)},W^{(2)},b^{(2)}$.
2. Contar todos los parámetros escalares entrenables, incluidos los sesgos.
3. Indicar las formas de $X,H,S$ para un lote de 16 ejemplos.
4. Determinar si ReLU añade parámetros entrenables.

**Solución:**

$$
W^{(1)}\in\mathbb{R}^{4\times2},\quad
b^{(1)}\in\mathbb{R}^{4},\quad
W^{(2)}\in\mathbb{R}^{3\times4},\quad
b^{(2)}\in\mathbb{R}^{3}.
$$

El total es:

$$
(4\cdot2+4)+(3\cdot4+3)=12+15=27.
$$

Para $B=16$, las formas son $X:(16,2)$, $H:(16,4)$ y $S:(16,3)$. ReLU no añade parámetros entrenables.

## Implementación en PyTorch y Keras

La misma arquitectura en PyTorch:

```python
import torch
from torch import nn

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 3),
)

X = torch.randn(16, 2)
S = model(X)                  # (16, 3), logits
y_hat = S.argmax(dim=1)       # (16,)
```

El seguimiento de las formas muestra el recorrido del lote:

$$
X:(16,2),\qquad Z,H:(16,4),\qquad S:(16,3).
$$

Los sesgos se propagan a todas las filas. La operación `model(X)` evalúa el modelo actual con sus parámetros actuales.

Equivalente en TensorFlow/Keras:

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(3),
])

n = model.count_params()  # 27
```

El modelo de PyTorch contiene dos capas `Linear` y el de Keras dos capas `Dense`; ambas incluyen sesgos y producen tres logits. Para que las predicciones coincidan, hay que asignar explícitamente los pesos y sesgos correspondientes, teniendo en cuenta la transposición entre sus convenciones de almacenamiento.

## Entrenamiento y evaluación del modelo

El modelo tiene 27 parámetros. Antes de entrenar, sus regiones de decisión son poco útiles: en el ejemplo visual aparecen sectores poligonales que no separan bien las tres clases. Después de entrenarlo, las regiones se ordenan aproximadamente en torno a las clases: una zona central para la clase 0, una corona intermedia para la clase 1 y una región exterior para la clase 2. Los marcadores de ambas figuras son las mismas 150 etiquetas observadas de prueba; el fondo indica la clase predicha.

En el experimento se utilizaron 300 ejemplos distintos para entrenar el modelo. La tabla compara la exactitud antes y después de entrenar:

| Partición | Antes | Después |
|---|---:|---:|
| Entrenamiento | 35.0 % | 98.3 % |
| Prueba (datos reservados) | 35.3 % | 97.3 % |

La pérdida usada para guiar el entrenamiento es la **entropía cruzada media**, expresada en nats y presentada en la sesión siguiente. En la curva del experimento cae rápidamente durante las primeras actualizaciones y luego continúa descendiendo más lentamente. Se realizaron 5.000 actualizaciones de parámetros con un protocolo fijo.

La curva de pérdida se calcula con datos de entrenamiento. Los datos de prueba reservados no guían las actualizaciones ni la selección del modelo; sus métricas se calculan al inicio y tras la última actualización.

- Antes de entrenar, la precisión está cerca del azar y las regiones de decisión no siguen adecuadamente las clases.
- Después, las regiones reflejan la estructura de los datos y mejora la exactitud tanto en entrenamiento como en prueba.
- En este modelo se ajustan los pesos y sesgos. Las activaciones ocultas cambian como consecuencia de la entrada y de los parámetros.
- La pérdida guía el ajuste de parámetros; la exactitud de prueba no se utiliza para actualizarlos.

## Preguntas de repaso (*exit ticket*)

1. Si solo cambia la entrada, ¿qué puede cambiar: pesos, sesgos o activaciones ocultas?
2. ¿Qué cantidades ajusta el entrenamiento en este modelo?
3. ¿Por qué se incluye ReLU entre las capas afines?
4. ¿Son ya probabilidades las tres salidas del modelo?

**Respuestas:**

1. Cambian las **activaciones ocultas**, porque dependen de la entrada. Los pesos y los sesgos se comparten entre ejemplos en un paso de entrenamiento fijo.
2. El entrenamiento ajusta los **pesos y los sesgos**; las activaciones cambian como consecuencia.
3. La composición de transformaciones afines sigue siendo afín. ReLU introduce una no linealidad que permite representar funciones generalmente no afines.
4. **No**. Son logits. Es necesario normalizarlos, por ejemplo mediante *softmax*, para obtener probabilidades de clase.

## Apéndices y cálculos complementarios

### A1. Formas y 27 parámetros entrenables

| Cantidad | Significado | Forma del array | Parámetros |
|---|---|---:|---:|
| $W^{(1)}$ | Pesos de la capa oculta | `(4, 2)` | 8 |
| $b^{(1)}$ | Sesgo de la capa oculta | `(4,)` | 4 |
| $h$ | Activación oculta | `(4,)` | 0 |
| $W^{(2)}$ | Pesos de la capa de salida | `(3, 4)` | 12 |
| $b^{(2)}$ | Sesgo de la capa de salida | `(3,)` | 3 |

La cuenta completa es:

$$
(4\cdot2+4)+(3\cdot4+3)=12+15=27.
$$

Para un lote de $B=16$, $X:(16,2)$, $H:(16,4)$ y $S:(16,3)$. Los pesos con valor cero siguen siendo parámetros entrenables. El *broadcasting* del sesgo y la operación ReLU no añaden parámetros.

### A2. Pase hacia delante numérico: capa oculta

Los pesos enteros de este ejemplo permiten comprobar el cálculo a mano; son independientes de los pesos entrenados en el experimento:

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

Primero se calcula la preactivación:

$$
z=W^{(1)}x+b^{(1)}
=
\begin{bmatrix}
1\\2\\-1\\-1
\end{bmatrix}.
$$

Después se aplica ReLU a cada coordenada:

$$
h=\operatorname{ReLU}(z)
=
\begin{bmatrix}
1\\2\\0\\0
\end{bmatrix}.
$$

El vector oculto es una representación calculada para esta entrada concreta.

### A3. Pase hacia delante numérico: capa de salida

Se continúa con $h$ del apéndice A2:

$$
W^{(2)}=
\begin{bmatrix}
1&0&1&0\\
0&1&0&1\\
-1&-1&1&1
\end{bmatrix},\qquad
b^{(2)}=
\begin{bmatrix}
0\\1\\0
\end{bmatrix}.
$$

La salida es:

$$
s=W^{(2)}h+b^{(2)}
=
\begin{bmatrix}
1\\3\\-3
\end{bmatrix},
\qquad
\hat{y}=\arg\max_{k\in\{0,1,2\}}s_k=1.
$$

La puntuación mayor corresponde a la clase 1 usando etiquetas de clase que empiezan en cero. Las puntuaciones pueden ser negativas y no tienen por qué sumar uno.

### A4. Número de parámetros en PyTorch y Keras

PyTorch permite contar los parámetros del modelo así:

```python
n = sum(p.numel() for p in model.parameters())
# n = 27
```

En Keras:

```python
n = model.count_params()
# n = 27
```

Ambos modelos tienen un sesgo en cada capa densa y producen tres logits.

### A5. Vectores columna, formas y orden de ejes

El mismo vector matemático puede representarse con distintas formas de array:

```python
x = np.array([1.0, 2.0])  # (2,)
x_col = x[:, None]        # (2, 1)
x_row = x[None, :]        # (1, 2)
```

Con ejemplos en filas, la operación de una capa se puede escribir:

```python
# X contiene B ejemplos en filas y d características.
Z = X @ W.T + b  # (B, d_out)
```

Una permutación convierte una imagen de canales primero a canales al final:

```python
# (B, C, H, W) -> (B, H, W, C)
images_last = images_first.transpose(0, 2, 3, 1)
```

`reshape` cambia cómo se agrupan los elementos; una permutación cambia qué eje ocupa cada posición.

### A6. Respuestas desarrolladas al *exit ticket*

1. Las activaciones ocultas dependen de la entrada. Los pesos y sesgos se comparten entre ejemplos en un paso fijo.
2. En esta red, el entrenamiento ajusta pesos y sesgos. Las activaciones ocultas cambian como resultado de los parámetros y la entrada.
3. La composición de transformaciones afines por sí sola sigue siendo afín. La función ReLU permite que la red represente funciones generalmente no afines.
4. Las salidas son logits. Una normalización adecuada, que se introduce en la sesión 2, las convierte en probabilidades de clase.

### A7. Experimento sintético y reproducibilidad

**Datos:** se generan 100 ejemplos de entrenamiento y 50 ejemplos reservados para prueba por cada clase. Para cada punto se toma un ángulo uniforme, se escala su dirección para que tenga norma $L_1$ igual a uno y se multiplica por un radio uniforme según la clase:

| Clase | Intervalo del radio |
|---|---|
| 0 | $[0.04,\,0.45]$ |
| 1 | $[0.75,\,1.00]$ |
| 2 | $[1.30,\,1.65]$ |

Esto genera puntos en torno al origen para la clase 0, una corona intermedia para la clase 1 y una corona exterior para la clase 2. Los datos de entrenamiento y prueba son ejemplos distintos.

**Modelo y entrenamiento:**

- Dos entradas, cuatro unidades ocultas con ReLU, tres logits de salida y sesgo en ambas capas afines.
- Implementación NumPy con 27 parámetros y precisión `float64`.
- Optimizador Adam de lote completo (*full-batch*), 5.000 actualizaciones.
- Tasa de aprendizaje $0.015$; $\beta_1=0.9$, $\beta_2=0.999$ y $\epsilon=10^{-8}$.
- Semilla de los datos: `20261017`; semilla de inicialización: `17`.

**Evaluación:** se mantiene un protocolo fijo. Solo los datos de entrenamiento determinan las actualizaciones y la curva de pérdida. Las métricas de los datos reservados se calculan al inicio y tras la última actualización. El ejemplo muestra una inicialización y un protocolo concretos, por lo que las cifras describen ese experimento.

### A8. Lecturas y referencias

**Libros:**

- Christopher M. Bishop y Hugh Bishop (2024). *Deep Learning: Foundations and Concepts*. Springer, capítulos 1 y 6. [bishopbook.com](https://www.bishopbook.com/)
- Ian Goodfellow, Yoshua Bengio y Aaron Courville (2016). *Deep Learning*. MIT Press, capítulos 1, 2, 5 y 6. [deeplearningbook.org](https://www.deeplearningbook.org/)

**Documentación oficial y conceptos relacionados:**

- [PyTorch `Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- [PyTorch `Conv2d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
- [PyTorch `Sequential`](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)
- [Keras `Dense`](https://keras.io/api/layers/core_layers/dense/)
- [Keras `Conv2D`](https://keras.io/api/layers/convolution_layers/convolution2d/)
- [NumPy: broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [NumPy: multiplicación matricial](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)

**Repaso sugerido:** volver a resolver las preguntas diagnósticas, rehacer la cuenta de los 27 parámetros y seguir los cálculos de los apéndices A2 y A3.
