---
titulo: "Mini-batches, momentum y Adam"
asignatura: aprendizaje-profundo
fecha: 2026-10-28
tema: "Semana 2 - Sesión 4"
tags: [mini-batch, sgd, momentum, adam, optimizadores, descenso-por-gradiente, learning-rate]
resumen: "Estimación del gradiente con mini-batches, descenso por gradiente estocástico, momentum y Adam: intuición, actualizaciones y consideraciones prácticas."
borrador: false
---

## Objetivo de la sesión

Al finalizar esta sesión deberías ser capaz de:

1. Estimar un gradiente usando un *mini-batch*
2. Explicar el papel de la tasa de aprendizaje y el estado del optimizador
3. Seguir paso a paso una actualización con *momentum* y una con *Adam*

## Del gradiente a un algoritmo de entrenamiento

La **retropropagación** calcula la derivada $g = \nabla_\theta \ell(\theta)$ (o $\nabla_\theta J$). El gradiente tiene la misma forma y dimensiones que los parámetros y nos indica la **sensibilidad local** de la pérdida respecto a cada parámetro.

Un **optimizador** elige la actualización:

$$
\theta_{t} = \theta_{t-1} - \eta_t\, \Delta_t,
$$

donde $\eta_t > 0$ es la **tasa de aprendizaje** (*learning rate*) y $\Delta_t$ es la **dirección/paso** que el optimizador propone (basado en gradientes y estado acumulado).

En las sesiones anteriores vimos el caso con un solo ejemplo. Ahora pasamos a **estimar el gradiente con datos reales** usando *mini-batches*.

## Función objetivo: media sobre el conjunto de entrenamiento

Dado un conjunto de entrenamiento $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$:

$$
\ell_i(\theta) = -\log p_\theta(y_i \mid x_i), \qquad J(\theta) = \frac{1}{N}\sum_{i=1}^{N}\ell_i(\theta).
$$

El **gradiente verdadero** (sobre todo el conjunto) es:

$$
\nabla J(\theta) = \frac{1}{N}\sum_{i=1}^{N}\nabla \ell_i(\theta).
$$

Si calculásemos $\nabla J(\theta)$ completo en cada paso, obtendríamos **Batch Gradient Descent** (BGD). Para $N$ grande esto es computacionalmente inviable.

## Mini-batch Gradient Descent

En lugar de usar todo el conjunto, en cada iteración $t$ seleccionamos un **mini-batch** de índices $I_t = \{i_{t,1}, ..., i_{t,B_t}\}$ uniformemente (sin reemplazo por época):

$$
\hat{J}_t(\theta) = \frac{1}{B_t}\sum_{r=1}^{B_t}\ell_{i_{t,r}}(\theta), \qquad g_t = \nabla \hat{J}_t(\theta_{t-1}).
$$

La actualización de **Mini-batch SGD** es:

$$
\theta_t = \theta_{t-1} - \eta_t\, g_t.
$$

### Propiedades clave

- **Eficiente**: Usa $B$ ejemplos en lugar de $N$ ($B \ll N$ en la práctica)
- **Estocástico**: $g_t$ es una **estimación ruidosa** pero **insesgada** de $\nabla J(\theta)$: $\mathbb{E}[g_t \mid \theta_{t-1}] = \nabla J(\theta_{t-1})$
- **Rápido y regularizador implícito**: El ruido del mini-batch suele ayudar a escapar de mínimos locales/sillas y favorece soluciones con mejor generalización
- **Vectorizable**: Se aprovecha al máximo el paralelismo de la GPU
- **Épocas**: Una **época** equivale a recorrer todos los ejemplos del dataset una vez (usando $\lceil N/B \rceil$ iteraciones)

**Tamaños típicos de batch**: 32, 64, 128, 256. Cuanto mayor el batch, más estable el gradiente pero mayor uso de memoria y menor ruido estocástico.

## El papel de la tasa de aprendizaje

La **tasa de aprendizaje** $\eta$ controla el tamaño del paso:

- **$\eta$ muy pequeña**: Convergencia lenta, pero estable
- **$\eta$ muy grande**: Puede divergir (la pérdida aumenta) o oscilar
- **$\eta$ adecuado**: Descenso progresivo de la pérdida

En la práctica se usan **schedules** de learning rate (*step decay*, *cosine annealing*, *warmup*, etc.).

## Momentum

El **SGD con momentum** acumula una "velocidad" basada en gradientes pasados. Esto ayuda a acelerar en direcciones consistentes y a amortiguar oscilaciones en direcciones ruidosas.

Definimos el **momento** $v_t$:

$$
v_0 = 0,
$$

$$
v_t = \beta\, v_{t-1} + g_t, \qquad \beta \in [0,1).
$$

Actualización:

$$
\theta_t = \theta_{t-1} - \eta\, v_t.
$$

### Intuición

- $\beta = 0$: Equivale a **SGD estándar**
- $\beta = 0.9$ (muy común): Promedia gradientes de los últimos ~10 pasos aproximadamente (ventana exponencial)
- **Efecto bola rodando**: Gana velocidad en direcciones con gradiente consistente
- **Amortigua oscilaciones**: Reduce el ruido en valles estrechos

También existe la variante **Nesterov Momentum** (*NAG*), que calcula el gradiente en una posición "adelantada" ($\theta_{t-1} - \eta \beta v_{t-1}$) obteniendo, habitualmente, una convergencia más rápida.

## Adam (*Adaptive Moment Estimation*)

**Adam** combina dos ideas clave:

1. **Momentum**: Media exponencial del gradiente ($m_t$)
2. **RMSProp/Adaptativo por coordenada**: Media exponencial del cuadrado del gradiente ($v_t$) para escalar individualmente cada parámetro

### Algoritmo Adam

Inicialización: $m_0 = 0, v_0 = 0, t=0$

Para cada iteración $t \ge 1$:

1) Calcular gradiente: $g_t = \nabla_\theta \hat{J}_t(\theta_{t-1})$

2) **Primer momento** (media con decaimiento):

$$
m_t = \beta_1\, m_{t-1} + (1-\beta_1)\, g_t
$$

3) **Segundo momento** (media de cuadrados con decaimiento):

$$
v_t = \beta_2\, v_{t-1} + (1-\beta_2)\, (g_t \odot g_t)
$$

4) **Corrección de sesgo** (*bias correction*) (importante en las primeras iteraciones):

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

5) **Actualización**:

$$
\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

### Hiperparámetros típicos

| Parámetro | Valor típico | Comentario |
| --- | --- | --- |
| $\eta$ | $10^{-3}$ | Tasa de aprendizaje por defecto muy robusta |
| $\beta_1$ | $0.9$ | Decaimiento para el primer momento |
| $\beta_2$ | $0.999$ | Decaimiento para el segundo momento |
| $\epsilon$ | $10^{-8}$ | Estabilización numérica para evitar división por cero |

### Por qué funciona Adam

- **Adaptativo por parámetro**: Parámetros con gradientes grandes obtienen pasos más pequeños; parámetros con gradientes pequeños/frecuentes obtienen pasos relativamente mayores
- **Rápido en la práctica**: Suele converger rápido y requiere menos ajuste de $\eta$
- **Robusto**: Funciona bien con ruido estocástico de mini-batches
- **Eficiente en memoria**: Solo almacena $m_t$ y $v_t$ (dos vectores del mismo tamaño que $\theta$)

## Comparativa práctica

| Optimizador | Velocidad | Estabilidad | Ajuste de $\eta$ | Uso típico |
| --- | --- | --- | --- | --- |
| **SGD** | Más lento | Menor (más ruidoso) | Requiere cuidado/schedule | Cuando se busca buena generalización |
| **SGD + Momentum** | Más rápido | Más estable | Menor ajuste | Muy común, buen compromiso |
| **Adam** | Muy rápido | Muy estable | Muy poco ajuste ($10^{-3}$ suele funcionar) | **Punto de partida** en la gran mayoría de casos |

## Consideraciones prácticas

- **Tamaño de batch**: Batch pequeño => más ruido pero mejor generalización. Batch grande => gradiente más estable pero mayor consumo de memoria
- **Learning rate scheduling**: Reducir $\eta$ con el tiempo suele mejorar la convergencia final
- **AdamW**: Variante muy extendida hoy en día que separa correctamente el *weight decay* del gradiente adaptativo
- **Gradient clipping**: Útil en RNNs/transformers para evitar explosión de gradientes
- **Zero gradients**: En frameworks como PyTorch hay que llamar `optimizer.zero_grad()` antes de cada *backward*

## Ejercicios rápidos

1. **Mini-batch**: ¿Por qué usamos mini-batches en lugar de todo el dataset o un solo ejemplo?
2. **Momentum**: ¿Qué efecto tiene $\beta$ al aumentar hacia 0.99?
3. **Adam**: ¿Por qué hacemos *bias correction* ($1-\beta^t$)?
4. **SGD vs Adam**: ¿Cuándo podrías preferir SGD con momentum sobre Adam?
5. **Insesgadez**: ¿Es $g_t$ un estimador insesgado de $\nabla J(\theta)$?

**Respuestas:**
1. Equilibra **eficiencia computacional** (vectorización/GPU), **ruido útil** para generalización y **viabilidad** con grandes datasets.
2. Mayor acumulación de gradientes pasados: **más inercia**, convergencia más rápida pero mayor riesgo de sobrepasar mínimos.
3. Porque $m_t, v_t$ están inicializados a 0, por lo que en las **primeras iteraciones** están sesgados hacia 0. La corrección compensa ese sesgo.
4. En algunos casos para obtener **mejor generalización final**. Con buen *learning rate schedule*, SGD+Momentum puede superar a Adam en test.
5. **Sí**, $\mathbb{E}[g_t \mid \theta_{t-1}] = \nabla J(\theta_{t-1})$ si el muestreo del mini-batch es uniforme e independiente.

## Apéndices

### A1. Mini-batch vs Batch vs Stochastic

| Variante | $B$ | Características |
| --- | --- | --- |
| **Batch GD** | $B = N$ | Gradiente exacto, estable, lento e inviable para N grande |
| **Stochastic GD** | $B = 1$ | Muy ruidoso, actualizaciones frecuentes, puede oscilar |
| **Mini-batch SGD** | $1 < B < N$ | **Compromiso óptimo**: eficiente, estable y con ruido beneficioso |

### A2. Efecto del tamaño de batch

- **Batch pequeño (32)**: Más ruido estocástico, mayor exploración, suele generalizar mejor
- **Batch mediano (64-128)**: Buen compromiso entre velocidad y estabilidad
- **Batch grande (256+)**: Gradiente más preciso, mejor aprovechamiento de GPU, menor ruido. Suele requerir **mayor learning rate** o técnicas como *linear scaling rule*

### A3. Fórmulas resumidas

**SGD:**
$$
\theta_t = \theta_{t-1} - \eta\, g_t
$$

**Momentum:**
$$
v_t = \beta v_{t-1} + g_t, \qquad \theta_t = \theta_{t-1} - \eta v_t
$$

**Adam:**
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\quad v_t = \beta_2 v_{t-1} + (1-\beta_2)(g_t^2),\quad \hat{m}_t=\frac{m_t}{1-\beta_1^t},\quad \hat{v}_t=\frac{v_t}{1-\beta_2^t},\quad \theta_t=\theta_{t-1}-\eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

### A4. Referencias

- Robbins, H. & Monro, S. (1951). A stochastic approximation method. *Ann. Math. Stat.*, 22, 400–407.
- Polyak, B. T. (1964). Some methods of speeding up the convergence of iteration methods. *USSR Comput. Math. Math. Phys.*, 4, 1–17.
- Sutskever, I. et al. (2013). On the importance of initialization and momentum in deep learning. *ICML*.
- Kingma, D. P. & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.
- Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*. MIT Press. Capítulo 8.
- PyTorch Docs: [Optimizers](https://pytorch.org/docs/stable/optim.html)
