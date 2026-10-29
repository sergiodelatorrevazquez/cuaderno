---
titulo: "Optimizadores adaptativos: RMSProp, Adam y AdamW"
asignatura: aprendizaje-profundo
fecha: 2026-10-29
tema: "Semana 2 - Sesión 5"
tags: [rmsprop, adam, adamw, weight-decay, l2-regularization, optimizadores-adaptativos]
resumen: "Optimizadores adaptativos por coordenada: RMSProp, Adam, diferencia entre regularización L2 y weight decay desacoplado (AdamW)."
borrador: false
---

## Objetivo de la sesión

Al finalizar esta sesión deberías ser capaz de:

- Explicar qué recuerda un optimizador adaptativo
- Distinguir entre una penalización L2 y *weight decay* desacoplado
- Relacionar cada regla de actualización con su configuración en las bibliotecas

## ¿Por qué optimizadores adaptativos?

En **SGD** usamos una única tasa de aprendizaje global $\eta$ para todos los parámetros:

$$
\theta_{t,j} = \theta_{t-1,j} - \eta\, g_{t,j}, \qquad g_t = \nabla_\theta \hat{J}_t(\theta_{t-1}).
$$

Sin embargo, los gradientes pueden tener **escalas muy distintas** entre coordenadas. Un optimizador **adaptativo por coordenada** aplica un factor distinto a cada parámetro:

$$
\theta_{t,j} = \theta_{t-1,j} - \eta_t\, d_{t,j}\, g_{t,j}, \qquad d_{t,j} > 0.
$$

La idea es **ajustar el tamaño del paso** según la historia reciente de cada coordenada: usar gradientes pasados para estimar su escala típica y dividir por ella.

## RMSProp

**RMSProp** (*Root Mean Square Propagation*) fue propuesto por Hinton en sus cursos. Mantiene una **media móvil exponencial** del cuadrado de los gradientes.

Definimos:

$$
v_0 = 0, \qquad v_t = \rho\, v_{t-1} + (1-\rho)\, (g_t \odot g_t),
$$

donde $\odot$ es el producto elemento a elemento y $\rho \in [0,1)$ controla la memoria (típicamente $\rho = 0.9$).

Actualización:

$$
\theta_t = \theta_{t-1} - \eta \frac{g_t}{\sqrt{v_t} + \epsilon}.
$$

### Intuición

- Si una coordenada ha tenido **gradientes grandes** de forma consistente, $\sqrt{v_t}$ será grande => **paso menor**
- Si una coordenada ha tenido **gradientes pequeños**, $\sqrt{v_t}$ será pequeño => **paso mayor**
- Se adapta al **escala** local de cada parámetro sin necesidad de ajustar $\eta$ por coordenada manualmente
- Muy efectivo en redes recurrentes y problemas con curvaturas muy distintas

## Adam (repaso con enfoque adaptativo)

Como vimos en la sesión 4, **Adam** combina *momentum* (media del gradiente) y RMSProp (media del cuadrado del gradiente), con corrección de sesgo.

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad m_0=0
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) (g_t \odot g_t), \qquad v_0=0
$$

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

$$
\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

Valores típicos: $\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}, \eta=10^{-3}$.

## Regularización: L2 vs Weight Decay

Un mecanismo habitual para evitar *overfitting* es penalizar pesos grandes. Existen dos formas que, **en SGD**, son equivalentes, pero **en optimizadores adaptativos NO lo son**.

### Regularización L2

Se añade un término a la función de pérdida:

$$
\tilde{J}(\theta) = J(\theta) + \frac{\lambda}{2}\|\theta\|_2^2 = J(\theta) + \frac{\lambda}{2}\sum_j \theta_j^2.
$$

El gradiente se modifica:

$$
\nabla \tilde{J}(\theta) = \nabla J(\theta) + \lambda \theta.
$$

En SGD esto equivale a:

$$
\theta_t = \theta_{t-1} - \eta (g_t + \lambda \theta_{t-1}) = (1 - \eta\lambda)\theta_{t-1} - \eta g_t.
$$

Es decir, **se encoge (*decay*) el peso multiplicativamente y se suma al gradiente de la pérdida**.

### Weight Decay desacoplado

**Weight decay** resta directamente una fracción del peso, **independiente** del gradiente y de la adaptación por coordenada:

$$
\theta_t = \theta_{t-1} - \eta (d_{t}\, g_t^{(loss)}) - \eta \lambda\, \theta_{t-1} = (1 - \eta\lambda)\theta_{t-1} - \eta d_t\, g_t^{(loss)}.
$$

La clave: **se aplica el *decay* por separado de la actualización adaptativa**. No se añade $\lambda\theta$ al gradiente antes de calcular $m_t, v_t$.

## AdamW

**AdamW** (*Adam with Weight Decay*) implementa **weight decay desacoplado** en lugar de regularización L2. Esto es especialmente importante con optimizadores adaptativos.

### Comparativa

| Aspecto | **Adam + L2 (equivocado en práctica)** | **AdamW (correcto)** |
| --- | --- | --- |
| **Dónde se aplica** | $\lambda\theta$ se suma a $g_t$ antes de $m_t/v_t$ | $\lambda\theta$ se resta **directamente** tras la actualización adaptativa |
| **Efecto con adaptación** | El *weight decay* se ve afectado por $1/\sqrt{\hat{v}_t}$ | El *weight decay* se aplica **uniformemente** a todos los pesos, independientemente de su escala de gradiente |
| **Rendimiento** | Menor en muchos casos | Mejor generalización, muy usado hoy en día |
| **Uso en bibliotecas** | A veces llamado erróneamente "Adam + weight decay" | `torch.optim.AdamW`, `transformers.AdamW` |

### Algoritmo AdamW (resumido)

Para cada iteración:

$$
g_t = \nabla_\theta J(\theta_{t-1})
$$

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)(g_t \odot g_t)
$$

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

$$
\theta_t = \theta_{t-1} - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon + \lambda \theta_{t-1}}\right)
$$

*(En la implementación práctica, el *weight decay* se resta directamente: $\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_{t-1}$)*

## Comparativa entre optimizadores

| Optimizador | Adaptativo | Momentum | Estado extra | Comentario |
 &| --- | --- | --- | --- | --- |
| **SGD** | No &| No | Ninguno | Simple, buen generalizador con buen *learning rate* </| --- | --- | --- | --- |
| **SGD + Momentum** | No | Sí | $v_t |$ Más rápido y estable que SGD |
| **RMSProp** | Sí (por coordenada) | No | $v_t$ | Muy bueno, especialmente en RNNs |
| **Adam** | Sí (por coordenada) | Sí | $m_t, v_t$ | **Punto de partida por defecto** en la mayoría de proyectos |
| **AdamW** | Sí (por coordenada) | Sí | $m_t, v_t$ | **Recomendado hoy en día** para la gran mayoría de tareas con *weight decay* |

## Consideraciones prácticas

- **L2 vs weight decay**: Si usas `optim.Adam` en PyTorch, añadir `weight_decay` como argumento **NO equivale** a usar un término L2 correctamente. Para eso usa **AdamW**.
- **Learning rate**: Con Adam/AdamW, $\eta = 3\times 10^{-4}$ o $10^{-3}$ suele funcionar muy bien como punto de partida.
- **Tasa de weight decay**: Valores típicos entre $10^{-2}$ y $10^{-4}$ según arquitectura y dataset.
- **Warmup**: En Transformers es común usar *learning rate warmup* durante las primeras iteraciones.
- **Gradient clipping**: Útil para estabilizar entrenamientos con gradientes grandes.

## Ejercicios rápidos

1. **Adaptativo**: ¿Por qué dividir por $\sqrt{v_t}$ en RMSProp/Adam?
2. **Memoria**: ¿Qué información recuerda RMSProp? ¿Y Adam?
3. **L2 vs WD**: ¿Por qué Adam + L2 no es lo mismo que AdamW?
4. **Escalas**: Si un parámetro tiene gradientes muy grandes, ¿aumenta o disminuye su paso efectivo?
5. **Cuándo usar AdamW**: ¿Cuándo recomendarías usar AdamW frente a Adam?

**Respuestas:**
1. Para **normalizar por la escala histórica** de cada coordenada. Así se iguala el tamaño relativo del paso entre parámetros con magnitudes distintas.
2. **RMSProp** recuerda una media móvil del **cuadrado** del gradiente ($v_t$). **Adam** recuerda media móvil del gradiente ($m_t$) y media móvil de su cuadrado ($v_t$).
3. Porque con L2 el término $\lambda\theta$ entra en el cálculo de $m_t$ y $v_t$, siendo escalado por $1/\sqrt{\hat{v}_t}$. Con *weight decay* desacoplado se aplica **directamente**, sin esa adaptación.
4. **Disminuye**. Cuanto mayor sea $\sqrt{v_t}$, menor será $\eta / \sqrt{v_t}$.
5. Siempre que quieras usar **regularización** con un optimizador adaptativo. **AdamW** es el estándar actual en visión, NLP y transformers.

## Apéndices

### A1. RMSProp: sin corrección de sesgo

A diferencia de Adam, RMSProp **no suele usar corrección de sesgo**. En la práctica funciona bien con los valores por defecto.

### A2. Comparación numérica intuitiva

Para una coordenada con gradientes grandes y constantes: RMSProp/Adam reducirán el paso con el tiempo. Para coordenadas esporádicas con gradientes grandes puntuales, también se adaptarán rápidamente.

### A3. Referencias

- Tieleman, T. & Hinton, G. (2012). Lecture 6.5 - RMSProp. *COURSERA: Neural Networks for Machine Learning*.
- Kingma, D. P. & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.
- Loshchilov, I. & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR*.
- Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*. MIT Press. Capítulo 8.
- PyTorch Docs: [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
- PyTorch Docs: [RMSprop](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html)
