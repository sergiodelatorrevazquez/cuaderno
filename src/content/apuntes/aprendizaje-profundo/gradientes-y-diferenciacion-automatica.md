---
titulo: "Gradientes y diferenciación automática"
asignatura: aprendizaje-profundo
fecha: 2026-10-23
orden: 3
tema: "Sesión 3"
tags: [gradientes, autograd, backpropagación, softmax, entropía-cruzada, PyTorch, TensorFlow, Keras]
resumen: "Extensión de la sesión anterior sobre la sensibilidad del error frente a los pesos, la regla de la cadena en reversa, la diferenciación automática y la actualización de parámetros con gradientes."
---

## Página 1 — Gradientes y diferenciación automática

**Tema:** Fundamentos matemáticos y computacionales del aprendizaje profundo.

**Pregunta central:** ¿Cómo indica una pérdida a cada peso qué cambio debe hacer?

**Objetivos de la sesión:**

- seguir una pasada hacia atrás,
- comprobar las formas de los gradientes,
- explicar una llamada a `autograd`.

**Idioma de las diapositivas:** inglés. **Enseñanza y discusión:** español.

## Página 2 — La misma red y la misma pérdida

La red de ejemplo sigue siendo:

$$
z = W_1 x + b_1, \quad h = \operatorname{ReLU}(z), \quad s = W_2 h + b_2, \quad p = \operatorname{softmax}(s), \quad \ell = -\log p_y.
$$

**Ejemplo de trabajo:**

$$
x = (1, 2)^\top, \quad y = 1.
$$

Con los valores dados:

$$
s = (1, 3, -3)^\top, \quad p_y \approx 0.878878.
$$

$$
\ell \approx 0.129109 \text{ nats}, \quad T = 1.
$$

**Pregunta del día:**

Los 27 parámetros están en

$$
\theta = \{W_1, b_1, W_2, b_2\}.
$$

¿Cómo de sensible es $\ell$ a cada uno de ellos?

Se mantiene fijo el input y la etiqueta, y se diferencia la pérdida.

La ilustración manual se toma de las sesiones 1–2. Los pesos completos aparecen en el apéndice A.

## Página 3 — Sensibilidad local de la pérdida

La gráfica muestra la pérdida en función de una variación del sesgo de la clase 1, $\beta = b_2[1]$.

- El eje horizontal representa cambio local en $\beta$.
- El eje vertical representa la pérdida de entropía cruzada en nats.
- La curva exacta y la tangente en $\beta = 1$ aparecen superpuestas.

La pérdida evaluada en ese punto es:

$$
\ell = 0.129109.
$$

Se varía únicamente el sesgo de la clase 1:

$$
\beta = b_2[1].
$$

Se obtiene:

$$
\ell'(1) \approx -0.121122.
$$

Un pequeño aumento positivo en $\beta$ reduce localmente la pérdida.

En general:

$$
\ell(\beta + \Delta) \approx \ell(\beta) + \ell'(\beta) \Delta.
$$

La tangente es una aproximación local.

## Página 4 — Derivadas parciales y gradiente

Se reúnen los parámetros en un vector:

$$
\theta \in \mathbb{R}^{27}.
$$

El gradiente se define como:

$$
\nabla_\theta \ell =
\begin{bmatrix}
\frac{\partial \ell}{\partial \theta_1} \
\vdots \\
\frac{\partial \ell}{\partial \theta_{27}}
\end{bmatrix}.
$$

Puntos clave:

- Una derivada parcial varía una coordenada y mantiene fijas las demás.
- Su signo predice el efecto local de aumentar esa coordenada.
- En conjunto, las entradas proporcionan un cambio de primer orden en cualquier dirección:

$$
\ell(\theta + \Delta \theta) \approx \ell(\theta) + (\nabla_\theta \ell)^\top \Delta \theta.
$$

Un gradiente describe la sensibilidad en los valores actuales de los parámetros.

## Página 5 — Formas del gradiente

Para una pérdida escalar, el gradiente respecto a un array tiene la misma forma que ese array.

| Cantidad | Forma | Gradiente |
|---|---:|---:|
| Entrada $x$ | $2$ | $\nabla_x \ell$ tiene forma $2$ |
| Pesos de la primera capa $W_1$ | $4 \times 2$ | $\nabla_{W_1} \ell$ tiene forma $4 \times 2$ |
| Sesgo de la primera capa $b_1$ | $4$ | $\nabla_{b_1} \ell$ tiene forma $4$ |
| Logits de salida $s$ | $3$ | $g_s := \nabla_s \ell$ tiene forma $3$ |

También se cumple:

$$
(\nabla_W \ell)_{ij} = \frac{\partial \ell}{\partial W_{ij}}.
$$

Se escribe $g_v$ para el gradiente de la pérdida con respecto a $v$.

Los vectores se consideran columnas en las ecuaciones. Un gradiente puede existir para una entrada aunque esa entrada no sea entrenable.

## Página 6 — Diferencias finitas

Una derivada puede comprobarse usando dos evaluaciones cercanas de la función:

$$
\frac{\partial \ell}{\partial \theta_i} \approx \frac{\ell(\theta + \varepsilon e_i) - \ell(\theta - \varepsilon e_i)}{2\varepsilon}.
$$

Aquí, $e_i$ selecciona una coordenada y todas las demás se mantienen fijas.

**Utilidad:**

- comparar una derivada propuesta con una estimación numérica independiente.

**Coste y error numérico:**

- dos evaluaciones de la pérdida por coordenada,
- $\varepsilon$ grande: error de aproximación,
- $\varepsilon$ demasiado pequeño: cancelación y redondeo.

Para entrenamiento, reutilizaremos la estructura del cálculo.

## Página 7 — Métodos de diferenciación

| Método | Idea principal | Papel en este curso |
|---|---|---|
| Manual / simbólico | Derivar expresiones aplicando reglas del cálculo. | Entender y derivar reglas locales. |
| Diferencias finitas | Perturbar entradas y comparar valores de la función. | Comprobar numéricamente un ejemplo pequeño. |
| Diferenciación automática | Aplicar reglas locales de derivadas a las operaciones ejecutadas. | Calcular gradientes del modelo de manera eficiente. |

La retropropagación aplica la regla de la cadena en orden inverso a través del grafo de la red.

La diferenciación automática sigue usando aritmética de precisión finita.

**Lectura recomendada:** Bishop & Bishop, Ch. 8; Goodfellow, Bengio & Courville, Ch. 6.

## Página 8 — Un grafo computacional

Ejemplo de calentamiento escalar:

$$
u = wx + b, \quad h = \operatorname{ReLU}(\nu), \quad \ell = \frac{1}{2}(h - t)^2.
$$

El esquema muestra:

- cada nodo: un valor calculado,
- cada flecha: una dependencia,
- **avance**: evaluar valores,
- **retroceso**: propagar sensibilidades de la pérdida.

## Página 9 — Valores hacia delante

Con:

$$
x = 2, \quad w = 0.5, \quad b = 0, \quad t = 0.
$$

| Operación | Valor |
|---|---:|
| $u = wx + b$ | $1$ |
| $h = \operatorname{ReLU}(u)$ | $1$ |
| $\ell = \frac{1}{2}(h-t)^2$ | $0.5$ |

Se conservan los valores que se necesitarán más adelante:

- $h - t$ para el error al cuadrado,
- el signo de $u$ para ReLU,
- $x$ y $w$ para la multiplicación.

**Pregunta de predicción:** si $w$ aumenta ligeramente, ¿aumenta o disminuye esta pérdida?

## Página 10 — La regla de la cadena en sentido inverso

Se empieza con:

$$
\frac{\partial \ell}{\partial \ell} = 1
$$

y se trabaja desde la pérdida escalar hacia atrás.

| Gradiente de pérdida | Gradiente entrante | $\times$ derivada local | Valor |
|---|---:|---:|---:|
| $g_h$ | $1$ | $(h-t)$ | $1$ |
| $g_u$ | $g_h$ | $\operatorname{ReLU}'(u)$ | $1$ |
| $g_w$ | $g_u$ | $x$ | $2$ |
| $g_b$ | $g_u$ | $1$ | $1$ |
| $g_x$ | $g_u$ | $w$ | $0.5$ |

Entonces:

$$
\frac{\partial \ell}{\partial w}
= \frac{\partial \ell}{\partial h}
  \frac{\partial h}{\partial u}
  \frac{\partial u}{\partial w}
= 1 \cdot 1 \cdot 2 = 2.
$$

Se reutiliza el gradiente entrante para calcular cada gradiente previo.

## Página 11 — Variables compartidas y ramas

La función es:

$$
f(a) = a \cdot a + a.
$$

La derivada en $a = 2$ es:

$$
\frac{df}{da} = 2 + 2 + 1 = 5.
$$

Se observa que cada uso de la variable compartida aporta a la misma derivada. En una ruta se multiplica; en diferentes caminos se suma.

## Página 12 — Tu turno: una pasada inversa escalar

Se define:

$$
u = wx + b, \quad h = \operatorname{ReLU}(\nu), \quad \ell = \frac{1}{2}(h-t)^2.
$$

Con:

$$
w = 1, \quad x = -1, \quad b = 2, \quad t = 2.
$$

**Tareas:**

1. Calcular $u$, $h$ y $\ell$.
2. Encontrar $g_h$, luego $g_w$, $g_b$ y $g_x$.
3. Cambiar solo $b$ a $0$. ¿Qué ocurre con la pérdida y los gradientes?

**Discusión:** ¿puede la pérdida ser positiva si todos los gradientes de parámetros son cero?

Se propone 2 minutos de cálculo y 3 de discusión. Las soluciones están en el apéndice B.

## Página 13 — El clasificador revisitado

Las operaciones hacia delante son:

$$
z = W_1 x + b_1 \in \mathbb{R}^4,
\quad h = \operatorname{ReLU}(z) \in \mathbb{R}^4,
\quad s = W_2 h + b_2 \in \mathbb{R}^3,
\quad \ell = \operatorname{LSE}(s) - s_y \in \mathbb{R}.
$$

Con:

$$
x = (1,2)^\top, \quad y = 1,
$$

los valores son:

$$
z = (1,2,-1,-1)^\top,
\quad h = (1,2,0,0)^\top,
\quad s = (1,3,-3)^\top,
\quad \ell \approx 0.129109.
$$

El orden inverso es:

$$
\text{pérdida} \rightarrow s \rightarrow h \rightarrow z.
$$

Se usa:

$$
\operatorname{LSE}(s) = \log \sum_j e^{s_j}.
$$

Con $T = 1$. Los pesos completos aparecen en el apéndice A.

## Página 14 — Entropía cruzada y gradiente de los logits

Para un ejemplo con clase $y$:

$$
q_k = \mathbf{1}\{k = y\}.
$$

La pérdida es:

$$
\ell = -\log p_y = \log \sum_j e^{s_j} - s_y.
$$

Y su gradiente respecto a cada logit es:

$$
\frac{\partial \ell}{\partial s_k} = \frac{e^{s_k}}{\sum_j e^{s_j}} - \mathbf{1}\{k = y\} = p_k - q_k.
$$

Por tanto:

$$
g_s = p - q.
$$

- Para la clase objetivo, $p_y - 1 < 0$.
- Para las demás clases, $p_k > 0$.
- Sus gradientes suman cero.

La propagación inversa del clasificador comienza con **predicción menos objetivo**.

Se trata de una sola muestra, con entropía cruzada sin ponderar y $T = 1$. La derivación con $T>0$ fijo aparece en el apéndice C.

## Página 15 — Paso hacia atrás por una capa afín

Se tiene:

$$
z = Wu + b, \quad W \in \mathbb{R}^{m\times d}, \quad u \in \mathbb{R}^{d}, \quad z,b \in \mathbb{R}^m.
$$

Supuesto que el cálculo posterior ya ha proporcionado $g_z = \nabla_z \ell$.

| Qué necesitamos | Regla | Forma |
|---|---|---|
| Gradiente de peso | $\nabla_W \ell = g_z u^\top$ | $m \times d$ |
| Gradiente de sesgo | $\nabla_b \ell = g_z$ | $m$ |
| Gradiente de entrada | $g_u = W^\top g_z$ | $d$ |

La regla usa un producto exterior para los pesos y la transpuesta para la entrada.

Las mismas tres reglas aplican a ambas capas afines. La derivación entrada a entrada está en el apéndice D.

## Página 16 — Gradientes de la capa de salida

Se tiene:

$$
s = W_2 h + b_2,
$$

con:

$$
h = (1,2,0,0)^\top, \quad y = 1.
$$

El gradiente de la pérdida respecto a los logits es:

$$
g_s = p - q \approx
\begin{bmatrix}
0.118943 \\
-0.121122 \\
0.002179
\end{bmatrix}.
$$

Aplicando las reglas afines:

$$
\nabla_{W_2}\ell = g_s h^\top \in \mathbb{R}^{3\times 4},
\quad
\nabla_{b_2}\ell = g_s,
\quad
g_h = W_2^\top g_s.
$$

**Inspección de un peso:**

- unidad oculta 2 a clase 1,
- en indexación de Python: $W_2[1,1]$.

$$
g_{W_2[1,1]} \approx (-0.121122) \cdot 2 \approx -0.242244.
$$

Los índices de clase son 0, 1, 2; “unidad 2” significa la segunda unidad oculta. Los gradientes completos están en el apéndice E.

## Página 17 — ReLU y la máscara activa

La derivada de ReLU es:

$$
\operatorname{ReLU}'(z_i) =
\begin{cases}
1, & z_i > 0, \\
0, & z_i < 0.
\end{cases}
$$

Entonces:

$$
g_z = g_h \odot \mathbf{1}\{z > 0\}.
$$

El gradiente entrante se transmite o se anula, elemento a elemento.

Para el ejemplo:

| $g_h$ | máscara | $g_z$ |
|---|---:|---:|
| $0.116765$ | $1$ | $0.116765$ |
| $-0.123300$ | $1$ | $-0.123300$ |
| $0.121122$ | $0$ | $0$ |
| $-0.118943$ | $0$ | $0$ |

En $z_i = 0$, ReLU no tiene derivada clásica. La regla estándar de retroceso de ReLU usa $0$ por convención.

Esto es la mecánica de `autograd` de PyTorch y la implementación del gradiente de ReLU en TensorFlow. Ninguna de nuestras preactivaciones está en cero.

## Página 18 — Gradientes de la capa oculta

Se aplica la misma regla afín a:

$$
z = W_1 x + b_1.
$$

Entonces:

$$
\nabla_{W_1}\ell = g_z x^\top \approx
\begin{bmatrix}
0.116765 & 0.233529 \\
-0.123300 & -0.246601 \\
0 & 0 \\
0 & 0
\end{bmatrix},
\quad
\nabla_{b_1}\ell = g_z \approx
\begin{bmatrix}
0.116765 \\
-0.123300 \\
0 \\
0
\end{bmatrix}.
$$

Puntos clave:

- las dos filas cero provienen de ReLUs inactivas para esta entrada,
- también existe un gradiente de entrada: $g_x = W_1^\top g_z \in \mathbb{R}^2$,
- durante el entrenamiento ordinario, la entrada se mantiene fija y se actualizan los parámetros.

Una pasada inversa proporciona todas las 27 derivadas de parámetros.

## Página 19 — Un paso de descenso de gradiente

Se usa:

$$
\theta_{\text{new}} = \theta_{\text{old}} - \eta \nabla_\theta \ell(\theta_{\text{old}}), \quad \eta = 0.1.
$$

Para una coordenada concreta:

$$
W_2[1,1] = 1 - 0.1(-0.242244) \approx 1.024224.
$$

Todos los parámetros usan gradientes calculados con los mismos valores antiguos de parámetros.

**Recalcular la pasada hacia delante:**

- después de actualizar los 27 parámetros,
- $\ell: 0.129109 \rightarrow 0.098396$,
- $p_y: 0.878878 \rightarrow 0.906290$.

Calcular un gradiente y actualizar un parámetro son operaciones separadas.

Este paso reduce la pérdida para este ejemplo. Un tamaño de paso arbitrario no tiene por qué hacerlo.

## Página 20 — Diferenciación automática del grafo

1. **Avance**: ejecutar operaciones tensores y registrar dependencias.
2. **Guardar lo necesario**: conservar valores seleccionados para las reglas de derivación local.
3. **Inverso**: comenzar desde la pérdida escalar y combinar las reglas locales.

**¿Por qué el modo inverso encaja con el entrenamiento?**

- muchos parámetros y una sola pérdida escalar,
- todos los gradientes requeridos se calculan en un único recorrido inverso.

**Lo que necesita el programa:**

- un camino registrado y diferenciable desde parámetros hasta pérdida,
- memoria para los valores necesarios en el retroceso.

PyTorch: mecánica de `autograd`; TensorFlow: diferenciación automática.

## Página 21 — PyTorch: gradientes y una actualización

Ejemplo manual de un modelo 2–4–3:

- `X`: tensor flotante de forma `(1,2)`,
- `y`: tensor entero `torch.long` con valor `[1]`.

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
optimizer.zero_grad(set_to_none=True)

S = model(X)          # logits: (B, 3)
loss = loss_fn(S, y)  # escalar, pérdida media
loss.backward()       # calcula gradientes
print(model[0].weight.grad.shape)  # torch.Size([4, 2])
optimizer.step()      # actualiza parámetros
```

**Nota:** `Linear.weight` tiene forma `(out, in)`. La capa final devuelve logits; no hace falta un `softmax` explícito para esta pérdida.

PyTorch: `backward`; `CrossEntropyLoss`. La inicialización completa aparece en el archivo Python complementario.

## Página 22 — TensorFlow y Keras: un pase hacia delante registrado

Se usa Keras con el backend de TensorFlow; mismo modelo, entradas y etiquetas enteras.

```python
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

with tf.GradientTape() as tape:
    S = model(X, training=True)    # logits: (B, 3)
    loss = loss_fn(y, S)          # objetivo, predicción
    params = model.trainable_variables
    grads = tape.gradient(loss, params)  # lista de gradientes
    optimizer.apply_gradients(zip(grads, params))
```

**Nota:** `Dense.kernel` tiene forma `(in, out)`. El primer kernel y su gradiente tienen forma `(2, 4)`: la transpuesta de $W_1$.

TensorFlow: `GradientTape`; Keras: un bucle de entrenamiento de TensorFlow.

## Página 23 — Gradientes, buffers y actualizaciones

| Operación o mecanismo | Significado |
|---|---|
| Ramas en un mismo grafo | Sumar cada contribución matemática a una derivada. |
| `PyTorch backward()` | Acumular gradientes en el buffer `parameter.grad`. |
| `PyTorch zero_grad()` | Reiniciar esos buffers antes de un nuevo paso de optimización. |
| `TensorFlow tape.gradient()` | Devolver gradientes para las variables pedidas. |
| Actualización del optimizador | Cambiar valores de parámetros usando los gradientes calculados. |

**Pregunta:** ¿`zero_grad()` reinicia los pesos?

PyTorch: `zero_grad`; TensorFlow: API `GradientTape`.

## Página 24 — Comprobación de gradientes en la práctica

Gráfico de error máximo absoluto frente al tamaño de paso finito:

- variación del tamaño de paso $\varepsilon$ en $10^{-12}, 10^{-9}, 10^{-6}, 10^{-3}, 10^{-1}$,
- red de 27 parámetros + 2 entradas,
- $\varepsilon = 10^{-5}$.

Se comparan diferencias centradas con las fórmulas de retroceso. Con $\varepsilon = 10^{-5}$:

$$
\max_i |g_{FD,i} - g_i| \approx 2.03 \times 10^{-11}.
$$

Todas las perturbaciones se mantienen alejadas de los puntos de kinks de ReLU.

La mejor elección de paso depende de la escala y de la precisión.

## Página 25 — Tu turno: un peso, un sesgo y una forma

Se usa el mismo clasificador: clase objetivo 1; activación oculta $h_2 = 2$.

$$
(g_s)_1 \approx -0.121122, \quad W_2[1,1] = 1, \quad \eta = 0.1.
$$

**Preguntas:**

1. ¿Cuál es el gradiente de pérdida para el peso desde la unidad oculta 2 a la clase 1?
2. ¿Cuál es el gradiente para el sesgo de la salida de la clase 1?
3. ¿Cuál es el nuevo valor de ese peso tras un paso?
4. ¿Qué forma tiene el gradiente completo de los pesos de salida en PyTorch? ¿Y en Keras?

Se propone 2 minutos en parejas + 2 minutos de corrección. “Unidad 2” significa la segunda unidad oculta. Las soluciones están en el apéndice B.

## Página 26 — Ticket de salida

1. ¿`backward()` cambia los pesos?
2. Si una ReLU está inactiva para esta entrada, ¿debe estarlo para todas las entradas?
3. Una pérdida por batch es la media de $B$ pérdidas de ejemplo. ¿Qué factor aparece en su gradiente?
4. ¿Un paso finito más pequeño siempre es más preciso?

La cadena completa es:

$$
\text{valores hacia delante} \rightarrow \text{pérdida} \rightarrow \text{gradientes} \rightarrow \text{actualización de parámetros}.
$$

Siguiente tema: mini-batches y optimización para aprendizaje profundo.

Las respuestas están en el apéndice B. Las diapositivas restantes son material de referencia, fuera del horario de 90 minutos.

## Página 27 — Apéndice A. Red manual: pesos y valores

Estos son los pesos elegidos a mano de la ilustración de la sesión 1.

$$
W_1 =
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
1 & -1 \\
-1 & 1
\end{bmatrix},
\quad
b_1 =
\begin{bmatrix}
0 \\
0 \\
0 \\
-2
\end{bmatrix},
\quad
W_2 =
\begin{bmatrix}
1 & 0 & 1 & 0 \\
0 & 1 & 0 & 1 \\
-1 & -1 & 1 & 1
\end{bmatrix},
\quad
b_2 =
\begin{bmatrix}
0 \\
1 \\
0
\end{bmatrix}.
$$

Con:

$$
x =
\begin{bmatrix} 1 \\ 2 \end{bmatrix},
\quad
z =
\begin{bmatrix} 1 \\ 2 \\ -1 \\ -1 \end{bmatrix},
\quad
h =
\begin{bmatrix} 1 \\ 2 \\ 0 \\ 0 \end{bmatrix},
\quad
s =
\begin{bmatrix} 1 \\ 3 \\ -3 \end{bmatrix},
\quad y = 1.
$$

La distribución de probabilidad es:

$$
p \approx (0.118943, 0.878878, 0.002179)^\top,
\quad
\ell \approx 0.129109.
$$

Hay 27 parámetros: $8 + 4 + 12 + 3$. Los kernels de Keras almacenan $W_1^\top$ y $W_2^\top$.

## Página 28 — Apéndice B. Respuestas del ejercicio y del ticket de salida

**Ejercicio escalar:**

$$
w = 1, \quad x = -1, \quad t = 2.
$$

| $b$ | $u$ | $h$ | $\ell$ | $g_h$ | $g_w$ | $g_b$ | $g_x$ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 1 | 0.5 | -1 | 1 | -1 | -1 |
| 0 | -1 | 0 | 2 | -2 | 0 | 0 | 0 |

**Ejercicio del clasificador:**

$$
g_{W_2[1,1]} \approx -0.242244,
\quad
g_{b_2[1]} \approx -0.121122.
$$

Nuevo peso:

$$
1.024224.
$$

Formas del gradiente:

- PyTorch: $(3,4)$,
- Keras: $(4,3)$.

**Ticket de salida:**

1. `backward()` calcula gradientes; el optimizador cambia los pesos.
2. La ReLU puede activarse para otra entrada.
3. $\nabla J = \frac{1}{B}\sum_{n=1}^{B}\nabla \ell_n$: el promedio aparece exactamente una vez.
4. Los pasos muy pequeños pueden amplificar el error de cancelación y redondeo.

## Página 29 — Apéndice C. Derivadas de la entropía cruzada

Sea $q_k \ge 0$ y $\sum_k q_k = 1$. Las etiquetas one-hot son el caso principal.

$$
\ell = -\sum_k q_k \log p_k = -\sum_k q_k s_k + \log\sum_j e^{s_j}.
$$

Entonces:

$$
\frac{\partial \ell}{\partial s_i} = -q_i + \frac{e^{s_i}}{\sum_j e^{s_j}} = p_i - q_i.
$$

Si se usa una temperatura positiva fija dentro de la pérdida:

$$
p_k(T) = \frac{e^{s_k/T}}{\sum_j e^{s_j/T}},
\quad
\ell_T = -\sum_k q_k \log p_k(T),
$$

entonces:

$$
\nabla_s \ell_T = p(T) - q_T.
$$

Se diferencian los logits originales y se mantiene $T$ fijo.

## Página 30 — Apéndice D. Derivadas de una capa afín

Sea:

$$
z_i = \sum_{j=1}^{d} W_{ij}u_j + b_i,
\quad (g_z)_i = \frac{\partial \ell}{\partial z_i}.
$$

Entonces:

$$
\frac{\partial \ell}{\partial W_{ij}} = \sum_{r=1}^{m}\frac{\partial \ell}{\partial z_r}\frac{\partial z_r}{\partial W_{ij}} = (g_z)_i u_j,
$$

por lo que:

$$
\nabla_W \ell = g_z u^\top.
$$

Además:

$$
\frac{\partial \ell}{\partial b_i} = (g_z)_i,
\quad
\nabla_b \ell = g_z,
$$

y:

$$
\frac{\partial \ell}{\partial u_j} = \sum_{i=1}^{m}(g_z)_i W_{ij},
\quad
g_u = W^\top g_z.
$$

Las dimensiones son:

- $(m\times 1)(1\times d) = (m\times d)$,
- $(d \times m)(m\times 1) = (d\times 1)$.

## Página 31 — Apéndice E. Todos los 27 gradientes de parámetros

Un ejemplo; $T=1$; valores redondeados a seis decimales.

$$
\nabla_{W_2}\ell \approx
\begin{bmatrix}
0.118943 & 0.237886 & 0 & 0 \\
-0.121122 & -0.242244 & 0 & 0 \\
0.002179 & 0.004357 & 0 & 0
\end{bmatrix},
\quad
\nabla_{b_2}\ell \approx
\begin{bmatrix}
0.118943 \\
-0.121122 \\
0.002179
\end{bmatrix}.
$$

$$
\nabla_{W_1}\ell \approx
\begin{bmatrix}
0.116765 & 0.233529 \\
-0.123300 & -0.246601 \\
0 & 0 \\
0 & 0
\end{bmatrix},
\quad
\nabla_{b_1}\ell \approx
\begin{bmatrix}
0.116765 \\
-0.123300 \\
0 \\
0
\end{bmatrix}.
$$

Además:

$$
\nabla_x \ell \approx (0.116765, -0.123300)^\top.
$$

Las dos derivadas de entrada no forman parte de los 27 parámetros entrenables.

## Página 32 — Apéndice F. Mini-batches y disposiciones de pesos

Los ejemplos son filas:

$$
X \in \mathbb{R}^{B\times 2}, \quad H \in \mathbb{R}^{B\times 4}, \quad S, P, Q \in \mathbb{R}^{B\times 3}.
$$

Entonces:

$$
Z = XW_1^\top + \mathbf{1}b_1^\top, \quad S = HW_2^\top + \mathbf{1}b_2^\top, \quad H = \operatorname{ReLU}(Z).
$$

Para:

$$
J = \frac{1}{B}\sum_n \ell_n,
$$

se tiene:

$$
G_S = \nabla_{S} J = \frac{P-Q}{B}.
$$

Además:

$$
\nabla_{W_2}J = G_S^\top H,
\quad
\nabla_{b_2}J = G_S^\top \mathbf{1},
\quad
G_H = G_S W_2,
\quad
G_Z = G_H \odot \mathbf{1}\{Z>0\},
$$

$$
\nabla_{W_1}J = G_Z^\top X,
\quad
\nabla_{b_1}J = G_Z^\top \mathbf{1}.
$$

**Relación matemática / PyTorch / Keras:**

- avance: $U W^\top + b^\top$ (sesgo broadcast),
- Keras: $U K + b^\top$ con $K = W^\top$,
- gradiente: $\nabla_W J$ y $\nabla_K J = (\nabla_W J)^\top$.

El factor de promedio ya está en $G_S$: no se divide de nuevo por $B$.

## Página 33 — Apéndice G. Diferencias finitas y kinks de ReLU

**Comprobación útil de gradientes:**

- usar un ejemplo pequeño y determinista,
- mantener fijas las demás coordenadas,
- preferir `float64` para la comprobación,
- probar varios tamaños de perturbación,
- comprobar lejos de los kinks de ReLU.

¿Qué ocurre en el kink?

$$
\frac{\operatorname{ReLU}(\varepsilon) - \operatorname{ReLU}(-\varepsilon)}{2\varepsilon} = \frac{1}{2}, \quad \varepsilon > 0.
$$

Esta estimación centrada no establece una derivada en cero.

La convención de retroceso de ReLU con valor 0 allí no tiene por qué coincidir con ella.

Cerca de cero, los gradientes necesitan una tolerancia absoluta. La error relativo solo puede ser engañoso. El código NumPy complementario comprueba cada parámetro y cada coordenada de entrada.

Las diferencias finitas verifican una implementación en puntos concretos; no prueban corrección global.

## Página 34 — Apéndice H. Lecturas y referencias API

**Conceptos y derivaciones**

- Christopher M. Bishop y Hugh Bishop (2024), *Deep Learning: Foundations and Concepts*. Cap. 8, *Backpropagation*; Cap. 7, *Gradient Descent* (puente a la optimización).
- Ian Goodfellow, Yoshua Bengio y Aaron Courville (2016), *Deep Learning*. Cap. 6, *Deep Feedforward Networks*; Cap. 4, *Numerical Computation*.

**Documentación oficial**

- PyTorch 2.14: `autograd` mechanics, `backward`, `zero_grad`, `CrossEntropyLoss`, `Linear`.
- TensorFlow: guía de diferenciación automática, API `GradientTape`, fuente del gradiente de ReLU.
- Keras: bucle de entrenamiento con backend TensorFlow, `Dense`, API de optimizadores.

**Material complementario reproducible**

- `session3_numeric.py`: valores hacia delante, todos los gradientes, una actualización, diferencias finitas.
- `session3_frameworks.py`: ejemplo completo en PyTorch o TensorFlow/Keras.

**Referencias API comprobadas el 17 de octubre de 2026.**

