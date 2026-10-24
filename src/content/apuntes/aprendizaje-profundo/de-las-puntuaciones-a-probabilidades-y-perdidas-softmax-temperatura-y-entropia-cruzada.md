---
titulo: "De las puntuaciones a probabilidades y pérdidas: softmax, temperatura y entropía cruzada"
asignatura: aprendizaje-profundo
fecha: 2026-10-22
tema: "Semana 1 - Sesión 2"
tags: [softmax, entropía-cruzada, pérdida, temperatura, clasificación-multiclase, verosimilitud, logits]
resumen: "Conversión de logits a distribuciones de probabilidad con softmax. Verosimilitud, entropía cruzada como función de pérdida, estabilidad numérica y efecto de la temperatura."
borrador: false
---

## Objetivo de la sesión

Conectar la salida de la red ($\text{logits}$) con un modelo probabilístico condicional y con la cantidad que se minimiza durante el entrenamiento.

**Ruta de la sesión:**
- 0-10 min: Etiqueta, puntuación (*score*), probabilidad condicional
- 10-25 min: *Softmax*, ejes y estabilidad numérica
- 25-40 min: Verosimilitud y entropía cruzada con $T=1$
- 40-50 min: Temperatura con los *logits* fijos
- 50-60 min: Comprobación numérica y contrato con las bibliotecas

## El pipeline de predicción

$$
x \in \mathbb{R}^2 \xrightarrow{f_\theta} s = f_\theta(x) \in \mathbb{R}^K \xrightarrow{\text{softmax}} p_\theta(\cdot \mid x) \in \Delta^{K-1} \xrightarrow{-\log p_\theta(y\mid x)} \mathcal{L}
$$

Donde $\Delta^{K-1}$ es el simplex de distribuciones sobre $K$ clases.

## Un modelo condicional para la clase

Dado un ejemplo de entrada $x$, la red define

$$
p_\theta(y=k \mid x), \quad k \in \{0,\ldots,K-1\}.
$$

- $x$: entrada observada
- $y$: clase que queremos predecir
- $\theta$: parámetros actuales del modelo
- $k$: clase candidata

Para que sea una distribución válida:

$$
p_\theta(y=k \mid x) \ge 0,\qquad \sum_{k=0}^{K-1} p_\theta(y=k \mid x) = 1.
$$

Los valores $s_k = (f_\theta(x))_k$ que produce la última capa afín se denominan **logits**. No están normalizados, pueden ser negativos y no tienen por qué sumar 1.

## De logits a probabilidades con softmax

Definimos la función *softmax* (con temperatura $T=1$):

$$
p_\theta(y=k \mid x) = \frac{e^{s_k}}{\sum_{j=0}^{K-1} e^{s_j}}, \qquad s_k \in \mathbb{R}.
$$

Propiedades importantes:

- $p_k > 0$ para todo $k$.
- $\sum_k p_k = 1$.
- Es monótona: si $s_i > s_j$, entonces $p_i > p_j$.
- No cambia el orden relativo de los *logits* (preserva el argumento que maximiza).
- Es una generalización multiclase de la función sigmoide.

### Ejemplo numérico (3 clases)

Supongamos

$$
s = \begin{bmatrix}s_0\\s_1\\s_2\end{bmatrix} = \begin{bmatrix}1.0\\ 3.0\\ -1.0\end{bmatrix}.
$$

Calculamos exponenciales:

$$
e^{1.0}\approx 2.718,\quad e^{3.0}\approx 20.086,\quad e^{-1.0}\approx 0.368,\quad \sum \approx 23.172.
$$

Por tanto:

$$
p = \begin{bmatrix}
2.718/23.172\\
20.086/23.172\\
0.368/23.172
\end{bmatrix}
\approx \begin{bmatrix}
0.117\\
0.866\\
0.016
\end{bmatrix},\qquad \sum p \approx 1.0,\qquad \arg\max p = 1.
$$

## Ejes y lotes

Con un lote de $B$ ejemplos, $S \in \mathbb{R}^{B\times K}$ donde cada fila es $s^{(i)}$. Aplicamos *softmax* **fila a fila** (a lo largo del eje de clases):

$$
P_{i,k} = \frac{e^{S_{i,k}}}{\sum_{j=0}^{K-1} e^{S_{i,j}}}.
$$

En NumPy/PyTorch esto corresponde a `softmax` con `dim = -1` (o `dim=1` según convención).

## Estabilidad numérica

Los valores $e^{s_k}$ pueden desbordar (*overflow*) o subdesbordar (*underflow*) si los *logits* son muy grandes en valor absoluto. Para evitarlo usamos la **identidad de invarianza por constante**:

$$
\frac{e^{s_k}}{\sum_j e^{s_j}} = \frac{e^{s_k - c}}{\sum_j e^{s_j - c}}, \qquad \forall c \in \mathbb{R}.
$$

La elección práctica más habitual es

$$
c = \max_{j} s_j.
$$

Así, el mayor exponente pasa a 0, los demás son $\le 0$ y los valores $e^{s_k - c}$ quedan en $(0,1]$, lo que mejora enormemente la estabilidad numérica.

**Regla práctica:** nunca calcules $e^{s_k}$ directamente para *logits* grandes; resta el máximo por fila.

## Verosimilitud (*likelihood*)

Dado un ejemplo $(x^{(i)}, y^{(i)})$ con etiqueta verdadera $y^{(i)}$, la probabilidad que el modelo asigna a la clase correcta es

$$
p_\theta(y^{(i)} \mid x^{(i)}).
$$

Si asumimos ejemplos **independientes e idénticamente distribuidos (i.i.d.)**, la verosimilitud del conjunto de datos es

$$
\mathcal{L}(\theta) = \prod_{i=1}^{N} p_\theta(y^{(i)} \mid x^{(i)}).
$$

## Entropía cruzada (*cross-entropy*)

Maximizar $\mathcal{L}(\theta)$ es equivalente a **minimizar** el logaritmo negativo:

$$
-\log \mathcal{L}(\theta) = -\sum_{i=1}^{N} \log p_\theta(y^{(i)} \mid x^{(i)}).
$$

Definimos la **pérdida de entropía cruzada** por ejemplo como

$$
\ell_i(\theta) = -\log p_\theta(y^{(i)} \mid x^{(i)}).
$$

Para un lote de $B$ ejemplos usamos la media:

$$
J(\theta) = \frac{1}{B}\sum_{i=1}^{B} -\log p_\theta(y^{(i)} \mid x^{(i)}).
$$

**Interpretación:** penaliza con mayor fuerza asignar probabilidad baja a la clase correcta. Si el modelo asigna probabilidad 1 a la clase correcta, $\ell_i = 0$. Si asigna probabilidad cercana a 0, la pérdida tiende a $+\infty$.

### Ejemplo numérico

Con $p \approx [0.117, 0.866, 0.016]$ y etiqueta verdadera $y=1$:

$$
\ell = -\log(p_1) = -\log(0.866) \approx 0.144
$$

Si en cambio $p_1 \approx 0.3$, $\ell = -\log(0.3) \approx 1.204$ (pérdida mucho mayor).

## Temperatura $T$

Para analizar o controlar la "nitidez" de la distribución, introducimos un parámetro de temperatura $T > 0$:

$$
p_k^{(T)} = \frac{e^{s_k / T}}{\sum_{j=0}^{K-1} e^{s_j / T}}.
$$

- $T \to 0^+$: la distribución se vuelve muy **afilada** (*peaked*). Tiende a un *one-hot*: toda la masa va a la clase con mayor *logit*.
- $T = 1$: *softmax* estándar.
- $T > 1$: la distribución se vuelve más **uniforme** (*suave*). Las diferencias entre clases se atenúan.
- $T \to +\infty$: $p_k^{(T)} \to 1/K$ para todas las clases.

**Importante:** al entrenar con entropía cruzada estándar se usa $T=1$. El parámetro $T$ es útil para análisis, calibración o para ciertos esquemas como *knowledge distillation*.

### Ejemplo con los mismos logits, distinta $T$

Con $s=[1.0, 3.0, -1.0]$:

| $T$ | $p_0$ | $p_1$ | $p_2$ | Comentario |
| --- | --- | --- | --- | --- |
| 0.5 | ~0.013 | ~0.986 | ~0.001 | Muy afilado, casi determinista |
| 1.0 | ~0.117 | ~0.866 | ~0.016 | Estándar |
| 2.0 | ~0.212 | ~0.665 | ~0.123 | Más suave, menos confiado |
| 10.0 | ~0.326 | ~0.352 | ~0.322 | Muy cercano a uniforme |

## ¿Qué cambia y qué no durante el entrenamiento?

- **Con $T$ fijo ($T=1$):** los **parámetros** $\theta$ (y, por tanto, los *logits* $s$) se actualizan para minimizar $J(\theta)=\frac{1}{B}\sum -\log p_\theta(y^{(i)}\mid x^{(i)})$.
- **Softmax + entropía cruzada:** juntas convierten las puntuaciones sin normalizar en una pérdida diferenciable adecuada para clasificación multiclase.
- **El orden de predicción:** con $T=1$, $\arg\max s_k = \arg\max p_k$. Cambiar $T$ altera las probabilidades, pero **manteniendo los *logits* fijos**, el ganador con mayor *logit* sigue siendo el que tiene mayor probabilidad para cualquier $T>0$.
- **Temperatura:** no es un parámetro entrenable en el ajuste estándar; es un hiperparámetro o herramienta de análisis.

## Contrato con las bibliotecas

La gran mayoría de frameworks siguen un convenio muy extendido:

- **PyTorch:** `nn.CrossEntropyLoss` espera **logits** (sin pasar por `softmax`) como entrada y la **etiqueta de clase** (`y` con índices). Internamente aplica `LogSoftmax` + `NLLLoss` de forma numéricamente estable.
- **Keras/TensorFlow:** `SparseCategoricalCrossentropy(from_logits=True)` espera logits. Si `from_logits=False`, espera probabilidades ya normalizadas.
- **NumPy puro:** puedes implementar `softmax` con la resta del máximo y después `-np.log(p[correct])`.

**Truco clave:** no apliques *softmax* dos veces. Si usas una pérdida que ya espera *logits*, pasa los *logits* directamente.

## Ejercicios rápidos

1. **Formas:** Para $B=16, K=3$, ¿qué forma tiene $P = \text{softmax}(S)$? ¿Y qué debe sumar cada fila?
2. **Estabilidad:** ¿Por qué restamos $\max_j s_j$ antes de exponentiar?
3. **Pérdida:** Si $p_{correct}=0.25$, ¿cuál es $\ell = -\log p_{correct}$ (aprox.)?
4. **Temperatura:** Con *logits* fijos, ¿cambia $\arg\max p^{(T)}$ al variar $T>0$?
5. **Bibliotecas:** ¿Por qué `CrossEntropyLoss` de PyTorch recibe *logits* en lugar de probabilidades?

**Soluciones rápidas:**
1. $P \in \mathbb{R}^{16\times 3}$, cada fila suma 1.
2. Para evitar desbordamiento/subdesbordamiento numérico gracias a $e^{s-c}/\sum e^{s-c}$.
3. $-\log(0.25)\approx 1.386$.
4. **No**, para $T>0$ el máximo se mantiene en la misma clase (solo cambia la confianza).
5. Porque es más estable numéricamente y evita aplicar *softmax* explícitamente dos veces.

## Apéndices

### A1. Derivación intuitiva de la identidad de softmax

$$
\frac{e^{s_k - c}}{\sum_j e^{s_j - c}} = \frac{e^{-c}e^{s_k}}{\sum_j e^{-c}e^{s_j}} = \frac{e^{-c}e^{s_k}}{e^{-c}\sum_j e^{s_j}} = \frac{e^{s_k}}{\sum_j e^{s_j}}.
$$

### A2. Relación con la regresión logística (caso binario)

Para $K=2$, con $s_1 - s_0$ o directamente con un único logit $z$, se recupera la sigmoide:

$$
p(y=1\mid x)=\sigma(z)=\frac{1}{1+e^{-z}},\qquad p(y=0\mid x)=1-\sigma(z).
$$

### A3. Ejemplo completo paso a paso

**Logits:**
$$
s = [2.0,\ 1.0,\ 0.1]
$$

**Restamos máximo ($c=2.0$):**
$$
s-c = [0.0,\ -1.0,\ -1.9]
$$

**Exponenciales:**
$$
e^0=1.0,\ e^{-1.0}\approx 0.3679,\ e^{-1.9}\approx 0.1496,\ \sum \approx 1.5175
$$

**Probabilidades:**
$$
p \approx [0.6586,\ 0.2424,\ 0.0990],\qquad \sum=1.0
$$

**Si $y=0$ (correcta):**
$$
\ell = -\log(0.6586) \approx 0.417
$$

## Lecturas recomendadas

- Bishop, C. M. & Bishop, H. (2024). *Deep Learning: Foundations and Concepts*. Springer. (Capítulos sobre clasificación probabilística)
- Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*. MIT Press. [deeplearningbook.org](https://www.deeplearningbook.org/)
- PyTorch Docs: [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- TensorFlow/Keras: [SparseCategoricalCrossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy)
- 3Blue1Brown: [But what is a neural network?](https://www.3blue1brown.com/lessons/neural-networks) y series relacionadas con clasificación
