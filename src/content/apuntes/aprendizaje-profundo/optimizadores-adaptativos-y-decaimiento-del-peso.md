---
titulo: "Optimizadores adaptativos y decaimiento del peso"
asignatura: aprendizaje-profundo
fecha: 2026-10-29
orden: 5
tema: "Sesión 5"
tags: [RMSProp, Adam, AdamW, decaimiento, regularización, optimización, aprendizaje-profundo]
resumen: "Control de la escala por coordenada, memoria del historial del gradiente, adaptación de la tasa de aprendizaje y separación entre regularización L2 y decaimiento explícito del peso."
---

## Página 1 — Optimizadores adaptativos y decaimiento del peso

**Tema:** Fundamentos matemáticos y computacionales del aprendizaje profundo.

**RMSProp, Adam y AdamW**

Al final de esta sesión podrás:

- explicar qué recuerda un optimizador adaptativo;
- distinguir una penalización L2 del decaimiento desacoplado;
- relacionar una regla de actualización con la configuración de la librería.

**Semana 2, sesión 5**

**Horario:** jueves, 19:20–20:20.

**Duración:** 60 minutos.

**Docente:** Francisco R. Villatoro.

“Del gradiente a la actualización: tanto el historial como la regla importan.”

Se basa en la sesión 4: mini-batches, SGD, momentum y la receta de Adam.

## Página 2 — Escalas por coordenada y pasos adaptativos

Para la coordenada $j$, el SGD plano usa una sola tasa de aprendizaje global:

$$
\Delta w_{t,j} = -\eta_t g_{t,j}.
$$

Un método adaptativo puede usar también un factor positivo para cada coordenada:

$$
\Delta w_{t,j} = -\eta_t d_{t,j} g_{t,j}, \qquad d_{t,j} > 0.
$$

### Desajuste de escala

Si

$$
 g_t = (0.2, 2) \quad \text{y} \quad \eta_t = 0.01,
$$

entonces

$$
\Delta \theta_t = (-0.002, -0.02).
$$

### Respuesta basada en el historial

Se estima la escala reciente del gradiente al cuadrado de cada coordenada y se usa para ajustar el denominador de la actualización.

La magnitud del gradiente contiene información, pero no determina por sí sola la elección del tamaño del paso.

**Referencia:** Goodfellow, Bengio y Courville, capítulo 8, tasas de aprendizaje adaptativas.

## Página 3 — RMSProp: un segundo momento móvil

Se guarda una media móvil exponencial de los gradientes al cuadrado:

$$
 v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^{\odot 2}.
$$

Luego cada parámetro se actualiza con su propio denominador:

$$
\theta_t = \theta_{t-1} - \eta_t \frac{g_t}{\sqrt{v_t} + \varepsilon}.
$$

- $\beta_2 \in [0,1)$ controla la memoria; $v_0 = 0$.
- Los cuadrados, raíces cuadradas y divisiones son elementales.
- Un valor mayor de $v_{t,j}$ implica un multiplicador menor para la coordenada actual $g_{t,j}$.

**Variante usada aquí:** sin momentum, sin centrado, sin corrección de sesgo; $\varepsilon > 0$ fuera de la raíz cuadrada.

**Referencias:** PyTorch RMSprop; GBC §8.5.2 usa un estabilizador dentro de la raíz. Ver Apéndice A.

## Página 4 — RMSProp recuerda cambios de escala

Un gradiente prescrito: $g_t = 1$ para los pasos 1–10, $g_t = 4$ para los pasos 11–15 y luego vuelve a 1.

La figura muestra la escala del gradiente al cuadrado a lo largo del tiempo para dos valores de $\beta_2$.

- $\beta_2$ más pequeño: responde más rápido y olvida más deprisa.
- $\beta_2$ más grande: retiene más historia anterior.

Se trata de estadísticas del gradiente al cuadrado, no de curvas de pérdida.

## Página 5 — Adam combina dirección y escala

### Primer momento móvil

$$
 m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

Retiene los signos y la dirección de gradientes recientes.

### Segundo momento móvil

$$
 v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^{\odot 2}
$$

Retiene la escala al cuadrado en cada coordenada.

Se corrige el sesgo inicial:

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}.
$$

Y la actualización es:

$$
\theta_t = \theta_{t-1} - \eta_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}.
$$

Adam adapta la escala por coordenada; aun así, sigue eligiéndose la tasa de aprendizaje.

Se recuerda de la sesión 4: $m_0 = v_0 = 0$, $t = 1,2,\ldots$; convención canónica de Kingma y Ba, *Adam*, Algorithm 1.

## Página 6 — Adam tras un cambio de gradiente

Se toma una historia prescrita:

$$
 g_1 = (0.2,-0.4), \quad g_2 = (-0.1,-0.2).
$$

Con $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\eta = 0.001$, $\varepsilon = 10^{-8}$.

### Primera coordenada

$$
 m_{1,1} = 0.9\cdot 0 + 0.1(0.2) = 0.02,
$$

$$
 m_{2,1} = 0.9(0.02) + 0.1(-0.1) = 0.008 > 0.
$$

El gradiente actual es negativo, pero el primer momento móvil sigue siendo positivo.

### Segundo paso

$$
\hat{m}_{2,1} \approx 0.0421053,\qquad \hat{v}_{2,1} \approx 0.0249925,
$$

$$
\Delta w_{2,1} \approx -0.000266337.
$$

La variable sigue disminuyendo.

El gradiente actual por sí solo no determina la dirección de Adam.

## Página 7 — Epsilon controla denominadores pequeños

En el primer paso canónico de Adam, con momentos iniciales nulos:

$$
\hat{m}_1 = g_1, \qquad \hat{v}_1 = g_1^2, \qquad \Delta w_1 = -\eta \frac{g_1}{|g_1| + \varepsilon}.
$$

Para $\varepsilon = 10^{-8}$ y $g_1 > 0$:

| $g_1$ | $|\Delta w_1|/\eta$ | Interpretación |
|---|---:|---|
| $10^{-1}$ | $0.9999999$ | $\varepsilon$ es despreciable |
| $10^{-6}$ | $0.9900990$ | efecto pequeño |
| $10^{-9}$ | $0.0909091$ | $\varepsilon$ domina el denominador |

$\varepsilon$ afecta a la actualización cuando las escalas del gradiente son pequeñas.

## Página 8 — Qué preserva el reescalado del gradiente

Se reescala toda la historia prescrita:

$$
 g_s' = c g_s \quad \text{para todo } s \le t, \quad c > 0.
$$

Con momentos iniciales nulos y los mismos hiperparámetros:

$$
\hat{m}'_t = c\hat{m}_t,\qquad \hat{v}'_t = c^2\hat{v}_t.
$$

Por tanto:

$$
\Delta w_t' = -\eta_t \frac{c\hat{m}_t}{c\sqrt{\hat{v}_t} + \varepsilon}
= -\eta_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon/c}.
$$

- El mismo paso si $\varepsilon = 0$ y los denominadores son positivos.
- Aproximadamente el mismo si los términos de $\varepsilon$ son insignificantes.

Propiedad útil de escalado con supuestos explícitos.

## Página 9 — Gradientes de datos y penalizaciones de parámetros

Nuestro clasificador 2→4→3 tiene parámetros:

$$
\theta = (w,b).
$$

| Conjunto | Número de escalares | Elección en esta sesión |
|---|---:|---|
| $W_1\in\mathbb{R}^{4\times2}, W_2\in\mathbb{R}^{3\times4}$ | 20 | Penalizar o decaer |
| $b_1\in\mathbb{R}^4, b_2\in\mathbb{R}^3$ | 7 | No penalizar ni decaer |

Se aplana la matriz de pesos en $w$.

La pérdida del batch es:

$$
J_{\mathrm{data}}(w,b) = \frac{1}{B} \sum_{i\in B} -\log p_\theta(y_i\mid x_i),
$$

$$
 g_t = \nabla_w J_{\mathrm{data}}(\theta_{t-1}).
$$

Primero hay que especificar qué parámetros y qué objetivo se están considerando.

## Página 10 — Regularización L2 y su gradiente

Se añade una penalización al objetivo, con $\lambda \ge 0$:

$$
J_\lambda(w,b) = J_{\mathrm{data}}(w,b) + \frac{\lambda}{2}\|w\|_2^2.
$$

El optimizador ahora recibe:

$$
\tilde{g}_t = g_t + \lambda w_{t-1}.
$$

### Comprobación con dos pesos

Si $w = (1,2)$ y $\lambda = 0.1$:

- penalización: $0.25$,
- gradiente de la penalización: $(0.1,0.2)$.

Para la red:

$$
\|w\|_2^2 = \|W_1\|_F^2 + \|W_2\|_F^2.
$$

Los gradientes de sesgo no reciben un término adicional en este ejemplo.

El factor 1/2 hace que el gradiente sea $\lambda w$.

## Página 11 — SGD plano: L2 y decaimiento del peso

Aplicando SGD sin momentum a la función penalizada:

$$
 w_t = w_{t-1} - \eta_t(g_t + \lambda w_{t-1})
= (1-\eta_t\lambda)w_{t-1} - \eta_t g_t.
$$

- Estos son exactamente los mismos pasos con esta convención de coeficientes.
- Si $0 < \eta_t\lambda < 1$, el término de decaimento en sí mismo reduce cada peso hacia cero.
- Esta álgebra no establece la misma equivalencia para momentum o Adam.

Para SGD plano, una penalización L2 puede reescribirse como decaimiento del peso.

## Página 12 — Adam con penalización L2

La penalización se incorpora al gradiente antes de cualquiera de las medias móviles:

$$
\tilde{g}_t = g_t + \lambda w_{t-1}.
$$

Entonces:

$$
 m_t = \beta_1 m_{t-1} + (1-\beta_1)\tilde{g}_t,
$$

$$
 v_t = \beta_2 v_{t-1} + (1-\beta_2)\tilde{g}_t^{\odot 2}.
$$

Se bias-corrigen estos momentos y se aplica Adam estándar.

- La penalización influye tanto en la dirección como en la escala por coordenada.
- También entra en la historia que usan las actualizaciones futuras.

El término $\lambda w$ pasa a través de la maquinaria adaptativa.

## Página 13 — AdamW separa el decaimiento de los momentos

Se calculan $m_t$ y $v_t$ solo a partir del gradiente de datos $g_t$.

La actualización es:

$$
 w_t = (1-\eta_t\lambda)w_{t-1} - \eta_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon}.
$$

- $\lambda w_{t-1}$ no se añade directamente a ninguno de los momentos.
- El decaimiento usa los pesos previos, antes de la actualización adaptativa.
- En nuestra red, solo los 20 pesos de las matrices reciben decaimiento.

**Adam + L2 y AdamW definen algoritmos distintos.**

## Página 14 — Contraejemplo: gradiente de datos nulo

Sean:

$$
 w_0 = (1,2), \quad g_{\mathrm{data}} = 0, \quad m_0 = v_0 = 0,
$$

$$
 \eta = 0.01, \quad \lambda = 0.1.
$$

Para Adam + L2, el optimizador ve el término penalizador $(0.1,0.2)$.

Entonces:

$$
\Delta w_1 \approx (-0.01,-0.01).
$$

Para AdamW, los momentos iniciales son cero, así que el paso adaptativo es nulo:

$$
\Delta w_1 = (-0.001,-0.002).
$$

Mismas condiciones iniciales y mismo coeficiente, pero actualizaciones distintas.

## Página 15 — Tasa de aprendizaje y decaimiento acumulado

Aislar el decaimiento de AdamW: todos los gradientes de datos y momentos iniciales son cero.

Entonces:

$$
 w_t = (1 - \eta_t\lambda)w_{t-1}
\quad \Longrightarrow \quad
w_K = \left[\prod_{t=1}^{K}(1-\eta_t\lambda)\right] w_0.
$$

Con tasa constante $\eta = 0.01$ y $\lambda = 0.1$, cada paso multiplica por 0.999.

| Número de actualizaciones $K$ | Fracción restante | Reducción aproximada |
|---:|---:|---:|
| 100 | 0.904792 | 9.52% |
| 1000 | 0.367695 | 63.23% |

El mismo coeficiente de decaimiento puede producir reducciones acumuladas distintas.

## Página 16 — PyTorch: selección de qué pesos decaen

Para un `Sequential(Linear(2,4), ReLU(), Linear(4,3))`:

```python
eta, lam = 0.01, 0.1
weights = [model[0].weight, model[2].weight]
biases = [model[0].bias, model[2].bias]

optimizer = torch.optim.AdamW(
    [{"params": weights, "weight_decay": lam},
     {"params": biases, "weight_decay": 0.0}],
    lr=eta, betas=(0.9, 0.999), eps=1e-8
)
```

- Construir este optimizador una vez y conservar su estado entre batches.
- Usar la pérdida de datos: `CrossEntropyLoss()(logits, labels)`.
- Añadir un término L2 adicional implementaría ambos mecanismos.

## Página 17 — TensorFlow/Keras: decaimiento y L2

El modelo y sus capas ocultas/salida ya están construidos.

```python
eta, lam = 0.01, 0.1
optimizer = keras.optimizers.AdamW(
    learning_rate=eta, weight_decay=lam,
    beta_1=0.9, beta_2=0.999, epsilon=1e-7
)
optimizer.exclude_from_weight_decay(
    var_list=[hidden.bias, output.bias]
)
```

- Excluir variables antes de construir el optimizador o de aplicarlo por primera vez.
- Para L2 acoplado: usar `L2(lam / 2)` e incluir `model.losses`.

En Keras, el `weight_decay` de Adam también es desacoplado.

## Página 18 — Tu turno: una actualización de AdamW

En un paso, los momentos del gradiente de datos ya están bias-corrected:

$$
 w_{t-1} = (1,2), \quad \hat{m}_t = (0.2,-0.4), \quad \hat{v}_t = (0.04,0.16).
$$

Usa $\eta = 0.01$, $\lambda = 0.1$ y, solo para este cálculo, $\varepsilon = 0$.

1. Calcula el paso adaptativo de los datos.
2. Calcula el paso de decaimiento usando $w_{t-1}$.
3. Suma ambos pasos y encuentra $w_t$.
4. ¿Se mueve la segunda coordenada más cerca de cero?

## Página 19 — Solución del ejercicio e interpretación

Paso adaptativo:

$$
\Delta w_{\mathrm{adaptive}} = -0.01\left(\frac{0.2}{0.2},\frac{-0.4}{0.4}\right) = (-0.01,+0.01).
$$

Paso de decaimiento:

$$
\Delta w_{\mathrm{decay}} = -0.01(0.1)(1,2) = (-0.001,-0.002).
$$

Paso total:

$$
\Delta w_{\mathrm{total}} = (-0.011,+0.008),
$$

$$
 w_t = (0.989, 2.008).
$$

- La segunda coordenada aumenta: su paso adaptativo supera al decaimiento.
- El decaimiento por sí solo la lleva hacia cero.

El decaimiento desacoplado es una contribución al paso total, no el único mecanismo.

## Página 20 — Exit ticket

1. ¿Qué cantidad guarda la media móvil de RMSProp?
2. ¿Dónde entra $\lambda w$ en Adam con penalización L2?
3. ¿Debe moverse todo peso más cerca de cero tras un paso de AdamW?
4. ¿Qué coeficiente de Keras representa $\lambda \|W\|_F^2 / 2$?

La siguiente sesión: generalización, validación, regularización y dropout.

**Respuestas y recordatorios de implementación:** Apéndice E.

## Página 21 — Apéndice A. Fórmulas del optimizador entre librerías

RMSProp sin momentum ni centrado: la misma recurrencia para $v_t$, pero estabilizadores distintos.

| Librería | Parámetro de memoria | Denominador |
|---|---|---|
| PyTorch | $\alpha = \rho$ | $\sqrt{v_t} + \varepsilon_P$ |
| Keras / TensorFlow | $\rho = \rho$ | $\sqrt{v_t} + \varepsilon_K$ |

Adam / AdamW: Keras usa la forma con epsilon-hat:

$$
\alpha_t = \eta_t \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t},
\qquad
\Delta w_{\mathrm{adaptive}} = -\alpha_t\frac{m_t}{\sqrt{v_t}+\varepsilon_K}.
$$

En el Adam canónico de PyTorch el último denominador es

$$
\sqrt{\hat{v}_t} + \varepsilon_P.
$$

Igualar los valores numéricos de $\varepsilon$ no implica reglas de actualización idénticas.

## Página 22 — Apéndice B. Adam tras un cambio de signo

Con:

$$
 w_0 = (1,-1),\quad m_0 = v_0 = 0,\quad \eta = 0.001,
\quad (\beta_1,\beta_2) = (0.9,0.999),\quad \varepsilon = 10^{-8},
$$

se tienen:

| Cantidad | Paso $t=1$ | Paso $t=2$ |
|---|---|---|
| $g_t$ | $(0.2,-0.4)$ | $(-0.1,-0.2)$ |
| $m_t$ | $(0.02,-0.04)$ | $(0.008,-0.056)$ |
| $v_t$ | $(0.00004,0.00016)$ | $(0.00004996,0.00019984)$ |
| $\hat{m}_t$ | $(0.2,-0.4)$ | $(0.042105263,-0.294736842)$ |
| $\hat{v}_t$ | $(0.04,0.16)$ | $(0.024992496,0.099969985)$ |
| $\Delta w_t$ | $(-0.001,+0.001)$ | $(-0.000266337,+0.000932180)$ |
| $w_t$ | $(0.999,-0.999)$ | $(0.998733663,-0.998067820)$ |

En el paso 2, la primera coordenada muestra una respuesta retrasada al cambio de signo del gradiente.

## Página 23 — Apéndice C. L2 acoplado en PyTorch y Keras

Ambas construcciones diferencian:

$$
J_{\mathrm{data}} + \frac{\lambda}{2}(\|W_1\|_F^2 + \|W_2\|_F^2).
$$

### PyTorch: penalización explícita

```python
weights = [model[0].weight, model[2].weight]
optimizer = torch.optim.Adam(
    model.parameters(), lr=eta,
    weight_decay=0.0, eps=1e-8
)

optimizer.zero_grad()
logits = model(xb)
data_loss = loss_fn(logits, yb)
penalty = 0.5 * lam * sum(w.square().sum() for w in weights)
loss = data_loss + penalty
loss.backward()
optimizer.step()
```

### Keras: regularizadores de kernel

```python
reg = keras.regularizers.L2(lam / 2)
model = keras.Sequential([
    keras.Input(shape=(2,)),
    Dense(4, activation="relu", kernel_regularizer=reg),
    Dense(3, kernel_regularizer=reg),
])

with tf.GradientTape() as tape:
    s = model(xb, training=True)
    loss = loss_fn(yb, s) + sum(model.losses)
```

No dividir de nuevo la penalización del peso por $B$.

## Página 24 — Apéndice D. Reescalado y decaimiento acumulado

Para reescalado completo de la historia,

$$
 g_s' = c g_s, \quad c>0,
$$

con momentos iniciales nulos:

$$
 m_t' = c m_t, \qquad v_t' = c^2 v_t.
$$

La corrección de sesgo hace que el término de $\varepsilon$ se convierta en $\varepsilon/c$; la invariancia exacta requiere $\varepsilon = 0$ y denominadores positivos.

Para decaimiento de AdamW con gradientes de datos nulos:

$$
 w_K = \left[\prod_{t=1}^{K}(1-\eta_t\lambda)\right] w_0.
$$

Si $0 \le \eta_t\lambda < 1$, tomando logaritmos se obtiene una aproximación exponencial:

$$
 w_K \approx e^{-\lambda \sum_t \eta_t} w_0.
$$

## Página 25 — Apéndice E. Respuestas y estado del optimizador

**Respuestas del ejercicio y del exit ticket**

- Paso adaptativo: $(-0.01, +0.01)$.
- Decaimiento: $(-0.001,-0.002)$.
- Nuevos pesos: $(0.989,2.008)$.
- RMSProp promedia $g_{t,j}^2$: un segundo momento no centrado.
- L2 entra en los momentos de Adam a través del gradiente penalizado.
- Un paso total de AdamW no tiene por qué reducir cada peso.
- Coeficiente de Keras: `L2(lam / 2)`; incluir la penalización en la pérdida diferenciada.

**Recordatorios de implementación**

- 20 pesos de matrices seleccionados; 7 sesgos excluidos en la red 2–4–3.
- Un tensor de gradiente nulo es distinto de un gradiente ausente (`None`).
- Un gradiente actual nulo no borra momentos acumulados previamente.
- Para reanudar un optimizador, hay que restaurar sus momentos y contador de actualizaciones, así como los parámetros del modelo y la configuración de entrenamiento.

## Página 26 — Apéndice F. Lecturas y referencias de la API

### Libros y algoritmos originales

- C. M. Bishop y H. Bishop (2024), *Deep Learning: Foundations and Concepts*, capítulo 7: *Gradient Descent*.
- Goodfellow, Bengio y Courville (2016), *Deep Learning*, §8.5.2 RMSProp y §8.5.3 Adam.
- Kingma y Ba, *Adam: A Method for Stochastic Optimization*, ICLR 2015, Algorithm 1.
- Loshchilov y Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019, Algorithm 2.

### Contratos de implementación

- PyTorch 2.14: RMSprop, Adam, AdamW.
- Keras: AdamW, weight regularizers, custom loops con TensorFlow; fuente del optimizador Keras 3.15.1.

Fuentes primarias consultadas el 17 de octubre de 2026. Exposición docente original, figuras y ejemplos.

Las versiones numéricas opcionales distinguen las convenciones de epsilon canónicas/PyTorch y Keras.
