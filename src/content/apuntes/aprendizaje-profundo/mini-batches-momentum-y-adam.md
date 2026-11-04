---
titulo: "Mini-batches, momentum y Adam"
asignatura: aprendizaje-profundo
fecha: 2026-10-24
orden: 4
tema: "Sesión 4"
tags: [mini-batches, SGD, momentum, Adam, optimización, gradientes, aprendizaje-profundo]
resumen: "Desde el gradiente del conjunto completo hasta el algoritmo de entrenamiento: estimación del gradiente con mini-batches, tasa de aprendizaje, ruido de gradiente, memoria del impulso y actualización de Adam."
---

## Página 1 — Mini-batches, momentum y Adam

**Tema:** Fundamentos matemáticos y computacionales del aprendizaje profundo.

**Del gradiente al algoritmo de entrenamiento**

1. Estimar un gradiente usando un mini-batch.
2. Explicar los roles de la tasa de aprendizaje y del estado del optimizador.
3. Seguir una actualización con momentum y otra con Adam.

**Docente:** Francisco R. Villatoro.

**Semana y sesión:** semana 2, sesión 4, jueves, 17:40–19:10.

**Supuesto:** composición de la red, entropía cruzada y retropropagación (sesiones 1–3).

## Página 2 — El gradiente y la actualización

La retropropagación calcula una derivada:

$$
g = \nabla_\theta \ell(\theta).
$$

El gradiente tiene las mismas componentes y formas que los parámetros. Describe la sensibilidad local en los parámetros actuales.

Un optimizador elige una actualización:

$$
\theta^+ = \theta - \eta g.
$$

La sesión 3 consideraba la misma red 2–4–3, con 27 parámetros y un único ejemplo.

$$
0.129109 \rightarrow 0.098396
$$

**Pérdida antes y después de un paso**, con $\eta = 0.1$.

Hoy: estimaremos el gradiente a partir de los datos y luego elegiremos la regla de actualización.

Continuación numérica de la red manual de las sesiones 1–3; no es un modelo entrenado.

## Página 3 — El objetivo de entrenamiento es una media

Los datos de entrenamiento son:

$$
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}, \quad y_i \in \{0,1,2\}.
$$

La pérdida por observación es:

$$
\ell_i(\theta) = -\log p_\theta(y_i \mid x_i),
$$

y la pérdida total es la media:

$$
J(\theta) = \frac{1}{N} \sum_{i=1}^{N} \ell_i(\theta).
$$

Una observación:

- $\ell_i$: cuán mal predice el modelo la etiqueta observada.
- $J$: pérdida media sobre las $N$ observaciones fijas de entrenamiento.

El gradiente es:

$$
\nabla J(\theta) = \frac{1}{N} \sum_{i=1}^{N} \nabla \ell_i(\theta).
$$

Reducir la pérdida de entrenamiento y generalizar a nuevos datos son preguntas distintas.

Aquí: $T=1$, logaritmos naturales, entropía cruzada sin ponderar, sin suavizado de etiquetas ni regularización.

## Página 4 — El gradiente de mini-batch

En la actualización $t = 1,2,\ldots$, se seleccionan índices:

$$
 i_{t,1}, \ldots, i_{t,B_t}.
$$

La estimación del mini-batch es:

$$
\hat{J}_t(\theta) = \frac{1}{B_t} \sum_{r=1}^{B_t} \ell_{i_{t,r}}(\theta), \qquad g_t = \nabla \hat{J}_t(\theta_{t-1}).
$$

Entonces:

$$
\theta_t = \theta_{t-1} - \eta_t g_t
$$

que es el **mini-batch SGD**.

- $B_t$ es el tamaño real del batch; el último batch puede ser más pequeño.
- Se evalúa cada componente de $g_t$ con los mismos parámetros antiguos.
- Si la pérdida ya devuelve una media, no se divide otra vez por $B_t$.
- Con reemplazo, los índices repetidos cuentan varias veces.

“SGD” suele incluir mini-batches; $B_t = 1$ es el caso de un solo ejemplo.

## Página 5 — Batch, paso y época

**Batch**: grupo de ejemplos usados juntos para calcular una pérdida y un gradiente.

**Paso**: una actualización del optimizador de los parámetros y del estado del optimizador.

**Época**: un pase completo por el conjunto de entrenamiento.

Si $N = 1000$ y se pide tamaño de batch 128, dejando el último batch:

$$
7 \times 128 + 104 = 1000 \Rightarrow 8 \text{ pasos por época.}
$$

Tres épocas dan 24 pasos si hay una actualización por batch.

Se deben especificar tanto el tamaño del batch como la unidad del presupuesto de entrenamiento.

Práctica típica: mezclar los ejemplos en cada época. Un muestreo nuevo e independiente con reemplazo es un esquema de muestreo distinto. No hay acumulación de gradiente aquí.

## Página 6 — Un paso de mini-batch en la misma red

Seis puntos etiquetados; el primer batch es $(1,2)$ con $y=1$ y $(3,1)$ con $y=0$.

$$
 g_1 = \frac{1}{2}(\nabla \ell_1(\theta_0) + \nabla \ell_2(\theta_0)), \qquad \theta_1 = \theta_0 - 0.1 g_1.
$$

| Media de pérdida | Antes | Después |
|---|---:|---:|
| Este batch de dos ejemplos | 0.089282 | 0.084092 |
| Todos los seis ejemplos de entrenamiento | 0.603837 | 0.606015 |

Un paso puede mejorar su propio mini-batch y empeorar el objetivo completo.

Se actualizan simultáneamente los 27 parámetros. Cálculo original en NumPy; conjunto completo de datos y pesos en el Apéndice A.

## Página 7 — Ruido del gradiente y muestreo no sesgado

Se congela $\theta$; se extraen ejemplos uniformemente e independientemente, con reemplazo.

$$
\mathbb{E}[g_B] = \nabla J(\theta), \qquad \mathrm{Cov}(g_B) = \Sigma/B,
$$

donde $\Sigma$ es la covarianza del gradiente de un ejemplo.

La figura muestra un ejemplo con dos coordenadas del sesgo de salida $b_2$.

- Un ejemplo: $B = 1$
- Media de dos: $B = 2$

El mismo gradiente completo; los parámetros están fijados.

Los batches ordenados pueden superponerse; el área del círculo representa probabilidad.

Cálculo original: 6 ejemplos y 36 pares ordenados, proyectados sobre dos coordenadas del sesgo de salida. “No sesgado” describe una esperanza, no cada actualización.

Ver Apéndice C.

## Página 8 — La tasa de aprendizaje controla el paso

$$
\theta_t = \theta_{t-1} - \eta_t g_t, \qquad \eta_t > 0.
$$

**Dirección**: para SGD ordinario, el paso apunta en la dirección de $-g_t$.

**Magnitud**:

$$
\|\theta_t - \theta_{t-1}\|_2 = \eta_t \|g_t\|_2.
$$

Una derivada es local. Un paso finito grande puede dejar la región donde la aproximación local es útil.

Una dirección de gradiente negativo no hace seguro cada tamaño del paso.

La igualdad de normas es para SGD plano, antes de añadir momentum o escalado adaptativo.

## Página 9 — Curvatura y pasos estables

Se muestra una variable $b$ y la exactitud de una función cuadrática:

$$
F(b) = 10b^2, \qquad F'(b) = 20b.
$$

Con recursión:

$$
 b_t = (1 - 20\eta) b_{t-1}, \qquad b_0 = 1.
$$

La convergencia a cero requiere:

$$
|1 - 20\eta| < 1 \quad \Longleftrightarrow \quad 0 < \eta < 0.1.
$$

- En $\eta = 0.1$: oscilación.
- En $\eta = 0.12$: divergencia.

Ejemplo cuadrático determinista. Este umbral exacto depende de la curvatura; no es una tasa universal de aprendizaje para redes neuronales.

## Página 10 — ¿Qué cambia con el tamaño del batch?

Aumentar $B$ cambia:

- **Variabilidad del gradiente**: menor covarianza para muestras independientes con parámetros fijos.
- **Uso de memoria**: más activaciones almacenadas, para el mismo modelo y precisión numérica.
- **Trabajo por paso**: más ejemplos; el rendimiento depende del hardware y de la implementación.
- **Pasos por época**: menos actualizaciones cuando cada ejemplo se visita una vez por época.

Cambiar $B$ también cambia el significado de “la misma cantidad de épocas”.

Las reglas de escalado de la tasa de aprendizaje son heurísticas de entrenamiento, no consecuencias de la fórmula de covarianza $1/B$. El rendimiento del hardware debe medirse.

## Página 11 — Tu turno: batches y pérdidas medias

**Ejercicio A**. Dos minutos en parejas, luego se discute.

Un conjunto de datos tiene $N = 10$ ejemplos. El tamaño del batch es 4, manteniendo el último batch.

1. ¿Cuántos pasos hay en una época? ¿Y en dos épocas?
2. En una evaluación separada, con parámetros congelados, las pérdidas medias por batch son:

| Tamaño de batch | Pérdida media |
|---|---:|
| 4 | 0.5 |
| 4 | 1.0 |
| 2 | 2.0 |

¿Qué pérdida media hay sobre los diez ejemplos? ¿Es $(0.5 + 1.0 + 2.0)/3$?

Suponiendo una actualización por batch de entrenamiento, sin ejemplos descartados ni acumulación de gradiente.

**Respuestas:** Apéndice B.

## Página 12 — Momentum recuerda gradientes

$$
 r_t = \mu r_{t-1} + g_t, \qquad \theta_t = \theta_{t-1} - \eta_t r_t.
$$

Con $r_0 = 0$ y $0 \le \mu < 1$. El buffer $r_t$ tiene las formas de los parámetros.

Puntos clave:

- Las direcciones repetidas se refuerzan entre sí.
- Las contribuciones opuestas pueden cancelarse parcialmente en el buffer.
- $\mu = 0$ recupera SGD ordinario.

La actualización depende ahora del gradiente actual y de su historia.

Momentum clásico, con buffer no normalizado; sin weight decay, dampening ni término de Nesterov. API de PyTorch SGD; otras convenciones en los Apéndices D–E.

## Página 13 — Momentum en un valle curvado

La figura muestra un ejemplo bidimensional con dos parámetros ficticios $a$ y $b$:

$$
F(a,b) = \frac{1}{2}(a^2 + 20b^2).
$$

Se parte de $(2,1)$ con gradientes completos.

- GD: $\eta = 0.08$
- Momentum: $\eta = 0.08$, $\mu = 0.8$

Las curvas de nivel muestran mayor curvatura en la dirección $b$.

Momentum lleva información entre pasos y puede rebasar el punto mínimo.

Cálculo original, 60 actualizaciones. $a$ y $b$ son parámetros de optimización, no características de entrada. Ilustración controlada, no una clasificación de optimizadores.

## Página 14 — Un ejemplo escalar de momentum

Gradientes prescritos:

$$
g_1 = 2, \quad g_2 = -1, \quad \mu = 0.9, \quad \eta = 0.1, \quad r_0 = 0.
$$

La recursión es:

$$
 r_t = 0.9 r_{t-1} + g_t,
\qquad
\theta_t = \theta_{t-1} - 0.1 r_t.
$$

| $t$ | $g_t$ | $r_t$ | $\Delta \theta_t$ | $\theta_t$ |
|---|---:|---:|---:|---:|
| 0 | — | 0 | — | 0.50 |
| 1 | 2 | 2 | -0.20 | 0.30 |
| 2 | -1 | 0.8 | -0.08 | 0.22 |

En el paso 2, el parámetro sigue bajando aunque el gradiente sea negativo.

Los gradientes se suministran para aislar la recursión; no se recalculan a partir del cuadrático previo. Aquí $\Delta \theta_t = \theta_t - \theta_{t-1}$.

## Página 15 — Escalado con gradientes al cuadrado

La idea de RMSProp es seguir la escala de gradientes recientes, coordenada a coordenada:

$$
 v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^{\odot 2}, \qquad v_0 = 0,
$$

$$
\theta_t = \theta_{t-1} - \eta_t \frac{g_t}{\sqrt{v_t}+\varepsilon}.
$$

Todas las potencias, raíces y divisiones son elementales; $\varepsilon > 0$.

Una coordenada con un gradiente cuadrado reciente grande recibe mayor escalado en el denominador.

Las estadísticas de gradiente al cuadrado describen la escala; no son una inversa del Hessiano.

Forma común de RMSProp, sin momentum ni centrado. Hinton, lecture 6e; PyTorch RMSprop. La ubicación de $\varepsilon$ varía entre implementaciones.

## Página 16 — Adam: dos medias móviles

$$
 m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,
$$

$$
 v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^{\odot 2}.
$$

- **Primer momento** $m_t$: media móvil exponencial de gradientes con signo. Lleva historia direccional.
- **Segundo momento** $v_t$: media móvil exponencial de gradientes al cuadrado. Rastrea la escala coordenada a coordenada.

Se parte de $m_0 = v_0 = 0$. Valores ilustrativos típicos: $\beta_1 = 0.9$, $\beta_2 = 0.999$.

$v_t$ estima un segundo momento no centrado, no una varianza.

Kingma y Ba, *Adam: A Method for Stochastic Optimization*, Algorithm 1.

## Página 17 — Corregir la inicialización a cero

Con un gradiente constante $g$, la primera media móvil es:

$$
 m_t = (1-\beta_1)\sum_{j=0}^{t-1} \beta_1^j g = (1-\beta_1^t)g.
$$

Sus pesos suman $1-\beta_1^t$, no 1.

Por eso se usa:

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}.
$$

En el ejemplo de gradiente constante, $\hat{m}_t = g$ y $\hat{v}_t = g^{\odot 2}$ desde la primera actualización.

La corrección de sesgo compensa el estado inicial cero.

Esto no convierte una historia cambiante en una estimación no sesgada de cada gradiente actual. Kingma y Ba, Algorithm 1 y Sección 3.

## Página 18 — La actualización de Adam

$$
\theta_t = \theta_{t-1} - \eta_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon}.
$$

1. Calcular el gradiente del mini-batch en $\theta_{t-1}$.
2. Actualizar ambas medias móviles y corregir su inicialización a cero.
3. Escalar el primer momento por la raíz cuadrada del segundo.

Las operaciones son elementales; $\eta_t$ sigue siendo una tasa de aprendizaje elegida.

Adam cambia la regla del paso. La retropropagación sigue proporcionando los gradientes.

Adam canónico, Algorithm 1. PyTorch usa esta convención; Keras usa una convención con “epsilon hat” (Apéndice E).

## Página 19 — Primera actualización de Adam

$$
\theta_0 = (1,-1), \quad g_1 = (0.2,-0.4), \quad \eta = 0.001, \quad \varepsilon = 10^{-8}.
$$

Con $\beta_1 = 0.9$, $\beta_2 = 0.999$, $m_0 = v_0 = 0$.

| Coordenada | $m_1$ | $v_1$ | $\hat{m}_1$ | $\hat{v}_1$ |
|---|---:|---:|---:|---:|
| 1 | 0.02 | 0.00004 | 0.2 | 0.04 |
| 2 | -0.04 | 0.00016 | -0.4 | 0.16 |

Entonces:

$$
\Delta \theta_1 \approx (-0.001, +0.001), \qquad \theta_1 \approx (0.999,-0.999).
$$

Primera actualización desde estados nulos; si $|g_{1,j}| \gg \varepsilon$, los tamaños del paso son parecidos a pesar de magnitudes diferentes del gradiente. Esto no es una fórmula para todas las actualizaciones posteriores.

## Página 20 — PyTorch: un bucle de entrenamiento con mini-batch

Se construye el optimizador una vez. El modelo devuelve logits de forma $(B,3)$.

```python
optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
model.train()
for Xb, yb in loader:
    optimizer.zero_grad(set_to_none=True)
    logits = model(Xb)
    loss = loss_fn(logits, yb)
    loss.backward()
    optimizer.step()
```

- $X_b$: punto flotante, forma $(B,2)$.
- $y_b$: `torch.long`, forma $(B,)`.

Para momentum, se sustituye el optimizador por:

```python
torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
```

El modelo y el loader ya están definidos; no se aplica softmax antes de esta pérdida. Adam, SGD y `CrossEntropyLoss`, API oficial.

## Página 21 — TensorFlow y Keras: un bucle de entrenamiento

Se usa el backend de TensorFlow; el modelo de Keras devuelve logits de forma $(B,3)$.

```python
optimizer = tf.keras.optimizers.Adam(
    learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="sum_over_batch_size")
for Xb, yb in dataset:
    with tf.GradientTape() as tape:
        logits = model(Xb, training=True)
        loss = loss_fn(yb, logits)
    variables = model.trainable_variables
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(grads, variables))
```

- $y_b$: índices de clase enteros, forma $(B,)$.
- La pérdida media se calcula una sola vez.

El modelo y el dataset ya están definidos. Bucle de entrenamiento personalizado de Keras; API de Adam y pérdidas. Convenciones de epsilon: Apéndice E.

## Página 22 — Hiperparámetros y estado del optimizador

| Cantidad | Ejemplos | Qué ocurre durante el entrenamiento |
|---|---|---|
| Parámetros | $W_1, b_1, W_2, b_2$ | Se actualizan con el optimizador |
| Hiperparámetros | $\eta$, $\mu$, $\beta_1$, $\beta_2$, $\varepsilon$ | Se especifican en la configuración o en un plan de entrenamiento |
| Estado del optimizador | $r_t$ o $m_t, v_t, t$ | Se calcula a partir del historial de actualizaciones |

`zero_grad()` borra los buffers de gradiente; no reinicia el momentum ni el estado de Adam.

Para reanudar exactamente la misma trayectoria del optimizador, hay que restaurar su estado junto con los pesos.

Para reproducibilidad completa, también hay que restaurar el estado del planificador, el orden de los datos y el generador aleatorio. Optimizers de PyTorch; optimizers de Keras.

## Página 23 — Cuatro errores típicos en el bucle de entrenamiento

| Error | Consecuencia |
|---|---|
| Crear Adam dentro de cada iteración del batch | Reinicia sus memorias y el contador de actualización |
| Olvidar `zero_grad()` en PyTorch | Acumula gradientes entre llamadas a `backward()` |
| Dividir una pérdida media por $B$ otra vez | Escala los gradientes por un factor adicional $1/B$ |
| Promediar medias de batch desiguales por igual | Da pesos distintos a ejemplos individuales |

Se debe leer el bucle como una secuencia de cambios de estado.

La acumulación intencional de gradientes es un algoritmo explícitamente escalado y distinto. Un registro de pérdida ponderado por tamaño mezcla pérdidas evaluadas en parámetros cambiantes; no es $J$ evaluado en los pesos finales.

## Página 24 — Tu turno: SGD y Adam

**Ejercicio B**. Dos minutos en parejas y luego comparación.

Dado el primer gradiente

$$
 g_1 = (3, 0.3)
$$

y $\eta = 0.01$:

1. Calcular el desplazamiento de SGD ordinario $\Delta \theta_1$.
2. Para Adam, partir de estados nulos y usar $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$. Calcular $m_1$, $v_1$, sus valores corregidos y $\Delta \theta_1$.
3. ¿Son iguales los desplazamientos porque las tasas de aprendizaje son iguales?

Compare vectores de actualización antes de comparar nombres de optimizadores.

Sin una función objetivo, estos gradientes solos no dicen qué actualización da una pérdida próxima menor.

**Respuestas:** Apéndice B.

## Página 25 — Comparación de ejecuciones de entrenamiento

1. **Fijar el problema**: mismo conjunto de datos, misma división, preprocesado y modelo; registrar inicialización y semillas aleatorias.
2. **Especificar el optimizador**: tasa de aprendizaje, momentum o coeficientes de Adam, epsilon y cualquier planificador.
3. **Declarar el presupuesto**: tamaño del batch, actualizaciones, ejemplos procesados y tiempo transcurrido responden preguntas distintas.
4. **Evaluar de forma consistente**: usar datos de validación para la selección; reservar el conjunto de prueba para la evaluación final.

La pérdida de entrenamiento mide la optimización del objetivo elegido. El rendimiento de validación mide el comportamiento en datos no vistos.

Una comparación útil informa tanto el resultado como las condiciones que lo produjeron.

Más discusión: Goodfellow, Bengio y Courville, capítulo 8; Bishop y Bishop, capítulo 7.

## Página 26 — Exit ticket

1. A parámetros fijos, ¿por qué dos mini-batches pueden dar gradientes distintos?
2. ¿Garantiza un estimador de gradiente no sesgado una disminución en cada paso?
3. ¿Es $v_t$ de Adam una estimación de la varianza del gradiente?
4. ¿Qué se pierde cuando Adam se construye otra vez dentro de cada iteración del batch?

Ahora ya deberías poder explicar:

- qué datos produjeron un gradiente,
- qué regla convirtió ese gradiente en un paso,
- qué cantidades persisten entre pasos.

Siguiente conexión: entrenar un objetivo y controlar la generalización.

**Respuestas** en el Apéndice B. Los apéndices son material de referencia, no ocho secciones extra temporizadas.

## Página 27 — Apéndice A. Dataset y red manual

| $i$ | $x_i^\top$ | $y_i$ |
|---|---|---:|
| 1 | $(1,2)$ | 1 |
| 2 | $(3,1)$ | 0 |
| 3 | $(-1,2)$ | 1 |
| 4 | $(1,-1)$ | 2 |
| 5 | $(2,1)$ | 0 |
| 6 | $(-1,-3)$ | 2 |

Índices de datos aquí: $1,\ldots,6$.

Índices en Python: $0,\ldots,5$.

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
\end{bmatrix}
$$

$$
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
\end{bmatrix}
$$

$$
 h_i = \operatorname{ReLU}(W_1 x_i + b_1), \quad s_i = W_2 h_i + b_2, \quad p_i = \operatorname{softmax}(s_i).
$$

Todos los 27 parámetros son entrenables. Las preactivaciones ocultas iniciales tienen $|z| \ge 1$ para estos datos.

En código, las filas del batch son:

$$
Z = XW_1^\top + \mathbf{1}b_1^\top, \qquad S = HW_2^\top + \mathbf{1}b_2^\top.
$$

Ilustración manual de las sesiones 1–3, ampliada a seis observaciones. El programa numérico del paquete adjunto reproduce la diapositiva 6.

## Página 28 — Apéndice B. Respuestas del ejercicio y del exit ticket

### Ejercicio A: pasos y promedio

$$
\left\lceil 10/4 \right\rceil = 3\ \text{pasos por época}; \quad 2\ \text{épocas} = 6\ \text{pasos}.
$$

La pérdida media global es:

$$
J = \frac{4(0.5) + 4(1) + 2(2)}{10} = 1.
$$

Ponderar por igual los tres promedios daría:

$$
\frac{0.5 + 1 + 2}{3} = \frac{7}{6} \approx 1.1667.
$$

### Ejercicio B: primera actualización

Para SGD:

$$
\Delta \theta_{\mathrm{SGD}} = (-0.03, -0.003).
$$

Para Adam con $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$:

$$
 m_1 = (0.3, 0.03), \quad v_1 = (0.009, 0.00009),
$$

$$
\hat{m}_1 = (3,0.3), \quad \hat{v}_1 = (9,0.09),
$$

$$
\Delta \theta_{\mathrm{Adam}} \approx (-0.01,-0.01).
$$

### Exit ticket

- Muestras distintas de ejemplos.
- No hay garantía de disminución por paso.
- $v_t$ es un segundo momento no centrado.
- Se pierden los historiales del impulso y el contador de actualizaciones.

No se especificó ninguna función objetivo para el Ejercicio B, así que los vectores desplazamiento no establecen cuál optimizador da una pérdida siguiente menor.

## Página 29 — Apéndice C. Muestreo y ruido del gradiente

A parámetros fijos $\theta$, defínase $g_i = \nabla \ell_i(\theta)$ y

$$
 g = \frac{1}{N} \sum_i g_i, \quad \Sigma = \frac{1}{N} \sum_i (g_i - g)(g_i-g)^\top.
$$

| Tipo de muestreo | Esperanza | Covarianza |
|---|---|---|
| Selección uniforme independiente, con reemplazo | $g$ | $\Sigma/B$ |
| Subconjunto uniforme, sin reemplazo | $g$ | $\frac{N-B}{B(N-1)}\Sigma$ |

Para $N=6$, $B=2$, los factores son $1/2$ y $2/5$ respectivamente.

Los batches posteriores dentro de una época mezclada no tienen por qué ser condicionalmente no sesgados para el gradiente completo cuando los parámetros cambian.

Aquí $\Sigma$ usa el denominador $N$, no $N-1$; $1 \le B \le N$ para muestreo sin reemplazo y $N>1$. La enumeración exacta se incluye en el programa numérico.

## Página 30 — Apéndice D. Estabilidad y memoria de momentum

Para una cuadrática unidimensional:

$$
F(u) = \frac{1}{2}\kappa u^2, \quad \kappa > 0,
$$

la recursión es:

$$
 u_t = (1-\eta\kappa)u_{t-1} = (1-\eta\kappa)^t u_0.
$$

Para $u_0 \neq 0$, la convergencia a cero se da cuando $0 < \eta\kappa < 2$. En $\eta\kappa = 2$, el signo alterna sin reducción de magnitud.

### Buffer no normalizado y media móvil normalizada

$$
 r_t = \mu r_{t-1} + g_t, \qquad m_t = \mu m_{t-1} + (1-\mu)g_t.
$$

Con estados iniciales nulos y el mismo $\mu$ fijo,

$$
 m_t = (1-\mu) r_t, \qquad \eta_{m,t} = \frac{\eta_{r,t}}{1-\mu},
$$

y por tanto,

$$
-\eta_{m,t}m_t = -\eta_{r,t}r_t.
$$

El factor de tasa de aprendizaje importa al comparar ecuaciones de momentum. Adam añade un segundo momento y corrección de sesgo; no es solo esta regla de momentum reescalada.

## Página 31 — Apéndice E. Convenciones de optimizadores entre librerías

### Momentum: buffer de gradiente frente a desplazamiento

- PyTorch:

$$
 r_t = \mu r_{t-1} + g_t, \qquad \theta_t = \theta_{t-1} - \eta_t r_t.
$$

- Keras:

$$
 d_t = \mu d_{t-1} - \eta_t g_t, \qquad \theta_t = \theta_{t-1} + d_t.
$$

Para $\eta$ constante y estados nulos, $d_t = -\eta r_t$. Con tasa variable, las dos recursiones no son equivalentes en general.

### Adam: dónde se coloca epsilon

- Canonical / PyTorch:

$$
\Delta \theta_t = -\eta_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon_P}.
$$

- Keras:

$$
\Delta \theta_t = -\eta_t \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}\frac{m_t}{\sqrt{v_t}+\varepsilon_K}.
$$

La regla de Keras tiene un epsilon corregido en el denominador $\varepsilon_K/\sqrt{1-\beta_2^t}$.

Ejemplos aquí: $\varepsilon_P = 10^{-8}$, $\varepsilon_K = 10^{-7}$. Coincidir en los valores visibles no garantiza trayectorias idénticas.

Actualizaciones densas ordinarias, sin weight decay, dampening, Nesterov ni AMSGrad. PyTorch SGD/Adam y Keras SGD/Adam oficiales.

## Página 32 — Apéndice F. Adam: algoritmo y segunda actualización

**Adam canónico**, con $t = 1,2,\ldots$

$$
 g_t = \nabla \hat{J}_t(\theta_{t-1}),
$$

$$
 m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,
$$

$$
 v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^{\odot 2},
$$

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t},
$$

$$
\theta_t = \theta_{t-1} - \eta_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon}.
$$

Se inicializa con $m_0 = v_0 = 0$ y se aplica elemento a elemento.

Se mantiene el estado entre batches.

Se continúa la primera coordenada de la diapositiva 19, suministrando $g_1 = 0.2$ y luego $g_2 = 0.1$.

Con $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\eta = 0.001$, $\varepsilon = 10^{-8}$ y $\theta_0 = 1$:

$$
 m_2 = 0.028,
\quad
v_2 = 0.00004996,
$$

$$
\hat{m}_2 \approx 0.147368421,
\quad
\hat{v}_2 \approx 0.024992496,
$$

$$
\Delta \theta_2 \approx -0.000932180,
\quad
\theta_2 \approx 0.998067820.
$$

El segundo desplazamiento difiere de $-\eta$.

El ejemplo numérico usa gradientes prescritos; no se suministra ninguna función de pérdida. Kingma y Ba, Algorithm 1.

## Página 33 — Apéndice G. Planificadores y clipping de gradiente

Un planificador debe especificar su unidad temporal:

$$
\eta_t = \eta_0 \gamma^{\lfloor (t-1)/S \rfloor}, \quad t = 1,2,\ldots
$$

Aquí $S$ cuenta actualizaciones del optimizador, con $0 < \gamma < 1$. Un planificador basado en épocas tiene otro reloj.

Se puede recortar la norma global del gradiente antes de la actualización del optimizador:

$$
\tilde{g}_t = 
\begin{cases}
 g_t, & \|g_t\|_2 \le c, \\
 c\, g_t / \|g_t\|_2, & \|g_t\|_2 > c,
\end{cases}
\quad c > 0.
$$

Ejemplo: si $g = (3,4)$ y $c = 1$, entonces $\tilde{g} = (0.6, 0.8)$.

Se alimenta el gradiente recortado al estado del optimizador. Con momentum o Adam, esto no limita en general la norma final del paso por $\eta_t c$.

PyTorch: `clip_grad_norm_`; Keras: planificador basado en pasos y callback de horario basado en épocas.

## Página 34 — Apéndice H. Lecturas y referencias de la API

### Libros y algoritmos originales

- C. M. Bishop y H. Bishop (2024), *Deep Learning: Foundations and Concepts*. Capítulo 7: *Gradient Descent*.
- I. Goodfellow, Y. Bengio y A. Courville (2016), *Deep Learning*. Capítulo 8: *Optimization for Training Deep Models*.
- D. P. Kingma y J. Ba, *Adam: A Method for Stochastic Optimization*, Algorithm 1.
- G. Hinton, *Neural Networks for Machine Learning*, lecture 6e: RMSProp.

### Referencias oficiales de implementación

- PyTorch: SGD, Adam, RMSprop, CrossEntropyLoss, optimizadores y planificadores.
- Keras: SGD, Adam, losses, custom training loop con TensorFlow.
- Enlaces de clipping y schedule en el Apéndice G.

Exposición docente original, ejemplos numéricos y figuras vectoriales. Enlaces de referencia consultados el 17 de octubre de 2026. Los enlaces de PyTorch identifican la versión 2.14; la documentación de Keras puede evolucionar.
