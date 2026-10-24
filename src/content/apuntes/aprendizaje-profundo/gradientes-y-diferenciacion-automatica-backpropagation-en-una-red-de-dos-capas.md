---
titulo: "Gradientes y diferenciación automática: backpropagation en una red de dos capas"
asignatura: aprendizaje-profundo
fecha: 2026-10-23
tema: "Semana 1 - Sesión 3"
tags: [gradientes, backpropagation, diferenciación-automática, autograd, regla-de-la-cadena, relu, softmax, entropía-cruzada, descenso-por-gradiente]
resumen: "Cálculo de gradientes mediante la regla de la cadena. Retropropagación en la red 2-4-3, formas de los gradientes, autograd y comprobación con diferencias finitas."
borrador: false
---

## Objetivo de la sesión

Al finalizar, deberías ser capaz de:

- Trazar un *backward pass* paso a paso
- Comprobar las formas de los gradientes
- Explicar una llamada a *autograd*
- Entender cómo la pérdida indica a cada peso qué cambio aplicar

**Duración:** 90 minutos. Diapositivas en inglés, discusión en español.

## La red y la pérdida (punto de partida)

Partimos de la misma arquitectura que en las sesiones 1 y 2:

$$
z = W^{(1)}x + b^{(1)},\quad h = \operatorname{ReLU}(z),\quad s = W^{(2)}h + b^{(2)},\quad p = \operatorname{softmax}(s),\quad \ell = -\log p_y.
$$

**Ejemplo en marcha:**

- $x = \begin{bmatrix}1\\2\end{bmatrix} \in \mathbb{R}^2$
- Etiqueta verdadera $y = 1$
- Tras el *forward pass*: $s = \begin{bmatrix}1.0\\ 3.0\\ -1.0\end{bmatrix}$, $p_y = p_1 \approx 0.878878$, $\ell \approx 0.129109$ (nats)
- Parámetros entrenables: $\theta = \{W^{(1)}, b^{(1)}, W^{(2)}, b^{(2)}\}$ con **27** parámetros en total

La pregunta clave: **¿Qué tan sensible es $\ell$ a cada uno de esos parámetros?**

## Sensibilidad local: derivada parcial

Si variamos **únicamente** un parámetro, por ejemplo $b^{(2)}_1$ (sesgo de la clase 1 en la segunda capa), podemos ver cómo cambia $\ell$ localmente:

$$
\ell(b^{(2)}_1 + \epsilon) \approx \ell(b^{(2)}_1) + \frac{\partial \ell}{\partial b^{(2)}_1}\,\epsilon.
$$

En el ejemplo, $\frac{\partial \ell}{\partial b^{(2)}_1} \approx -0.121122$. Un pequeño aumento en $b^{(2)}_1$ **disminuye** la pérdida localmente. Ahí está la intuición: el gradiente indica la **dirección** (y magnitud) para reducir la pérdida.

## Gradiente y vector de derivadas parciales

Para una función escalar $f(\theta)$ con $\theta \in \mathbb{R}^D$, el **gradiente** es el vector columna

$$
\nabla_\theta f = \begin{bmatrix}
\frac{\partial f}{\partial \theta_1}\\
\vdots\\
\frac{\partial f}{\partial \theta_D}
\end{bmatrix}.
$$

**Descenso por gradiente (idea):**

$$
\theta \leftarrow \theta - \eta\, \nabla_\theta \ell(\theta),
$$

donde $\eta > 0$ es la tasa de aprendizaje (*learning rate*). El objetivo es dar un pequeño paso en la **dirección opuesta** al gradiente para reducir $\ell$.

## Regla de la cadena (*chain rule*)

Como $\ell$ se compone como una cadena de funciones:

$$
x \to z \to h \to s \to p \to \ell,
$$

usamos la regla de la cadena para propagar las derivadas desde $\ell$ hacia atrás hasta cada parámetro.

Recordatorio:

$$
\frac{d}{dx}f(g(x)) = f'(g(x))\,g'(x),\qquad \frac{\partial \ell}{\partial \theta} = \sum (\cdots)\frac{\partial \text{intermedio}}{\partial \theta}.
$$

## Derivación paso a paso (softmax + entropía cruzada)

Un resultado fundamental (muy útil): para **softmax** con **entropía cruzada** categórica, la derivada con respecto a los *logits* es sorprendentemente simple.

Definimos $q = p - y_{onehot}$, donde $y_{onehot}$ es el vector 1-en-$y$. Entonces

$$
\frac{\partial \ell}{\partial s_k} = p_k - \delta_{k,y},\qquad \nabla_s \ell = p - e_y.
$$

**En nuestro ejemplo:** $p \approx [0.013,\ 0.879,\ 0.108]$ (valores aproximados para este ejemplo concreto), $e_y = [0,1,0]$, por lo que $\nabla_s \ell \approx [-0.013,\ -0.121,\ 0.108]$.

## Retropropagación por capas

Sea $\delta^{(l)} = \nabla_{a^{(l)}} \ell$ (o equivalentemente el gradiente con respecto a las pre-activaciones/activaciones según convención). Seguimos el flujo hacia atrás.

### Capa 2 (salida): afín $s = W^{(2)}h + b^{(2)}$

Teniendo $\delta^{(s)} = \nabla_s \ell = p - e_y \in \mathbb{R}^{K}$:

$$
\nabla_{W^{(2)}} \ell = \delta^{(s)} (h)^\top \in \mathbb{R}^{K\times d_h},\qquad \nabla_{b^{(2)}} \ell = \delta^{(s)} \in \mathbb{R}^{K}.
$$

El gradiente se propaga a la capa oculta:

$$
\nabla_h \ell = (W^{(2)})^\top \delta^{(s)} \in \mathbb{R}^{d_h}.
$$

### Activación ReLU: $h = \operatorname{ReLU}(z)$

ReLU es elemento a elemento: $h_i = \max(0, z_i)$. Su derivada es

$$
\frac{d}{dz_i}\operatorname{ReLU}(z_i)=\begin{cases}
1, & z_i > 0,\\
0, & z_i < 0,\\
\text{indefinida (convención)}, & z_i == 0.
\end{cases}
$$

En la práctica, la mayoría de frameworks usan la **convención**: en $z_i=0$ devuelven 0 (o se define arbitrariamente, pero el efecto es negligible). 

Aplicamos **producto elemento a elemento** (*Hadamard*) con una máscara:

$$
\delta^{(z)} = \nabla_z \ell = \nabla_h \ell \odot \mathbf{1}\{z > 0\} \in \mathbb{R}^{d_h}.
$$

### Capa 1 (oculta): afín $z = W^{(1)}x + b^{(1)}$

Con $\delta^{(z)} = \nabla_z \ell$:

$$
\nabla_{W^{(1)}} \ell = \delta^{(z)} (x)^\top \in \mathbb{R}^{d_h\times d_{in}},\qquad \nabla_{b^{(1)}} \ell = \delta^{(z)} \in \mathbb{R}^{d_h}.
$$

## Formas y convenciones (importante)

Es fundamental mantener coherentes las formas para evitar errores. Con un **solo ejemplo** ($B=1$):

| Cantidad | Forma | Notas |
| --- | --- | --- |
| $x$ | $(d_{in},)$ o $(d_{in},1)$ | Entrada |
| $W^{(1)}$ | $(d_h, d_{in})$ | Pesos capa oculta |
| $z,h$ | $(d_h,)$ | Pre-activación/activación |
| $W^{(2)}$ | $(K, d_h)$ | Pesos capa salida |
| $s,p,\delta^{(s)}$ | $(K,)$ | Logits, probs, error salida |
| $\nabla_{W^{(1)}}\ell$ | $(d_h,d_{in})$ | Misma forma que $W^{(1)}$ |
| $\nabla_{b^{(1)}}\ell$ | $(d_h,)$ | Misma forma que $b^{(1)}$ |
| $\nabla_{W^{(2)}}\ell$ | $(K,d_h)$ | Misma forma que $W^{(2)}$ |
| $\nabla_{b^{(2)}}\ell$ | $(K,)$ | Misma forma que $b^{(2)}$ |

**Regla de oro:** $\nabla_W \ell$ **siempre tiene la misma forma** que $W$.

## Con lotes (*batches*)

Con $B$ ejemplos, $X \in \mathbb{R}^{B\times d_{in}}$ (filas = ejemplos). Definimos $J = \frac{1}{B}\sum_{i=1}^{B}\ell_i$.

Los gradientes se **acumulan** (suman) sobre el lote y se promedian:

$$
\nabla_{W^{(2)}} J = \frac{1}{B}\sum_{i=1}^{B} \delta^{(s)(i)} (h^{(i)})^\top = \frac{1}{B} (G^{(s)})^\top H,\qquad G^{(s)} = P - Y_{onehot}.
$$

$$
\nabla_{b^{(2)}} J = \frac{1}{B}\sum_{i=1}^{B} \delta^{(s)(i)} = \frac{1}{B} G^{(s)}\,\mathbf{1}.
$$

$$
G^{(h)} = G^{(s)} W^{(2)},\qquad G^{(z)} = G^{(h)} \odot \mathbf{1}\{Z > 0\},
$$

$$
\nabla_{W^{(1)}} J = \frac{1}{B} (G^{(z)})^\top X,\qquad \nabla_{b^{(1)}} J = \frac{1}{B} G^{(z)}\,\mathbf{1}.
$$

**Importante:** Muchas bibliotecas devuelven los gradientes ya promediados por $B$ (como `CrossEntropyLoss` con `reduction='mean'`). Otras veces se suman. Siempre hay to check the **contrato** de la biblioteca.

## Diferenciación automática (*autograd*)

En lugar de derivar a mano, los frameworks construyen un **grafo computacional** dinámico:

1. **Forward:** se registran operaciones y tensores intermedios.
2. **Backward:** se aplica la regla de la cadena mediante *autograd* para obtener $\nabla_\theta \ell$.
3. **Update:** se aplica un optimizador (p. ej., descenso por gradiente) con $\theta \leftarrow \theta - \eta\nabla_\theta \ell$.
4. **Zero grad:** hay que poner a cero los gradientes antes de cada paso (*`zero_grad()`*).

**Ejemplo en PyTorch:**

```python
import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(2,4), nn.ReLU(), nn.Linear(4,3))
criterion = nn.CrossEntropyLoss()  # espera logits + índices
opt = torch.optim.SGD(model.parameters(), lr=0.01)

x = torch.tensor([[1.0, 2.0]])
y = torch.tensor([1])

opt.zero_grad()
s = model(x)
loss = criterion(s, y)
loss.backward()  # calcula todos los gradientes
opt.step()       # actualiza parámetros
```

**Puntos clave:**
- Pasa **logits** a `CrossEntropyLoss` (no probabilidades)
- Llama a `zero_grad()` antes de `backward()`
- Los gradientes quedan en `param.grad`
- `backward()` solo funciona sobre escalares (o con un vector de gradientes)

## Comprobación con diferencias finitas (*gradient checking*)

Una forma fiable de verificar es comparar con **diferencias finitas centradas**:

$$
\frac{\partial \ell}{\partial \theta_i} \approx \frac{\ell(\theta + h e_i) - \ell(\theta - h e_i)}{2h}.
$$

Recomendaciones prácticas:

- Usar **float64** para la comprobación
- Elegir $h$ pequeño (p. ej., $1e-5$)
- Comprobar **todos** los parámetros y también coordenadas de entrada si se desea
- Tener cuidado cerca de **kinks** de ReLU ($z\approx 0$): la derivada teórica no está definida, pero la convención del framework es la que importa
- Usar **error absoluto/relativo** con tolerancia razonable (cerca de los *kinks* conviene relajar un poco)

**Código mental:** el *companion* `session3_numeric.py` hace esto para el ejemplo 2-4-3 completo.

## Un vistazo al *update* completo

Tras obtener todos los gradientes, damos un paso con descenso por gradiente:

$$
W^{(2)} \leftarrow W^{(2)} - \eta \nabla_{W^{(2)}}\ell,\quad b^{(2)} \leftarrow b^{(2)} - \eta \nabla_{b^{(2)}}\ell,\quad W^{(1)} \leftarrow W^{(1)} - \eta \nabla_{W^{(1)}}\ell,\quad b^{(1)} \leftarrow b^{(1)} - \eta \nabla_{b^{(1)}}\ell.
$$

Con los valores del ejemplo, los gradientes indican que **aumentar** ligeramente $b^{(2)}_1$ (y ajustar los pesos correspondientes) **reduciría** $\ell$ localmente. Después de muchos pasos, las regiones de decisión del modelo cambian (como se vio en la sesión 1).

## Errores comunes

- **Olvidar `zero_grad()`:** los gradientes se **acumulan** entre batches
- **Aplicar softmax antes de CrossEntropyLoss:** doble normalización
- **Confundir formas:** gradiente debe tener **misma forma** que el parámetro
- **Olvidar dividir por $B$:** si sumas y quieres media, divide por $B$
- **Olvidar la máscara de ReLU:** `z > 0`
- **Mirar solo el error relativo en *kinks*:** usar tolerancia absoluta

## Ejercicios rápidos

1. **Regla de la cadena:** ¿Por qué propagamos desde $\ell$ hacia atrás y no hacia delante?
2. **Formas:** Para $W^{(1)} \in \mathbb{R}^{4\times 2}$, ¿qué forma tiene $\nabla_{W^{(1)}}\ell$?
3. **ReLU:** Si $z_3 < 0$, ¿contribuye esa unidad al gradiente de las capas anteriores?
4. **Softmax+CE:** ¿Por qué $\nabla_s \ell = p - e_y$ es tan conveniente?
5. **Autograd:** ¿Por qué hay que llamar a `zero_grad()` antes de `backward()`?

**Respuestas:**
1. Porque queremos saber cómo afecta cada parámetro (lejano) a una pérdida **escalar** al final. La regla de la cadena permite propagar hacia atrás eficientemente.
2. $(4,2)$, **igual** que $W^{(1)}$.
3. **No**. Con la máscara $\mathbf{1}\{z>0\}$ el gradiente se multiplica por 0 y se corta.
4. Simplifica enormemente el *backward pass* (evita derivar softmax+log por separado).
5. Porque PyTorch **acumula** los gradientes en `grad` (suma). Si no los ponemos a cero, se suman con los del batch anterior.

## Apéndices

### A1. Valores numéricos aproximados del ejemplo

Con el ejemplo trabajado en clase (valores ilustrativos):

- $W^{(2)}$ relevante da $s=[1.0,3.0,-1.0]$
- $p \approx [0.117,\ 0.866,\ 0.016]$
- $\nabla_s \ell = p - e_1 \approx [-0.117,\ -0.134,\ 0.016]$ (valores aproximados)
- $\nabla_{b^{(2)}}\ell \approx \nabla_s \ell$
- $\nabla_{h}\ell = (W^{(2)})^\top \nabla_s \ell$
- Con máscara ReLU se obtiene $\nabla_z \ell$
- $\nabla_{W^{(1)}}\ell = \nabla_z \ell\, x^\top$, $\nabla_{b^{(1)}}\ell = \nabla_z \ell$

*(Los valores exactos varían ligeramente según los pesos ilustrativos; la metodología es lo importante.)*

### A2. Diferencia entre suma y media en el batch

- Si $J = \sum_{i}\ell_i$: $\nabla J = \sum_i \nabla \ell_i$
- Si $J = \frac{1}{B}\sum_{i}\ell_i$: $\nabla J = \frac{1}{B}\sum_i \nabla \ell_i$

En descenso por gradiente, usar la **media** hace que la tasa de aprendizaje $\eta$ sea menos sensible al tamaño de batch $B$. `CrossEntropyLoss(reduction='mean')` usa la media.

### A3. ReLU en $z=0$

En $z_i=0$, $\operatorname{ReLU}'(0)$ no está definido matemáticamente. Las convenciones más habituales:

- **Subgradiente:** 0 o 1
- **PyTorch:** usa 0 en la práctica para `ReLU` en ese punto
- **TensorFlow/Keras:** también suele tomar 0

Con datos reales, la probabilidad de caer exactamente en 0 es muy baja y el efecto es despreciable. Para *gradient checking*, conviene **evitar** ese punto o usar tolerancia mayor.

### A4. Lecturas y referencias

- Bishop, C. M. & Bishop, H. (2024). *Deep Learning: Foundations and Concepts*. Springer. Capítulo 8 (*Backpropagation*).
- Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*. MIT Press. Capítulos 6 y 8. [deeplearningbook.org](https://www.deeplearningbook.org/)
- Rumelhart, D. E. et al. (1986). Learning representations by back-propagating errors. *Nature*, 323, 533–536.
- PyTorch: [Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html), [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- Wikipedia: [Backpropagation](https://es.wikipedia.org/wiki/Retropropagaci%C3%B3n), [Automatic differentiation](https://es.wikipedia.org/wiki/Diferenciaci%C3%B3n_autom%C3%A1tica)
- 3Blue1Brown: [Neural networks](https://www.3blue1brown.com/topics/neural-networks)
- Material complementario del curso: `session3_numeric.py`, `session3_frameworks.py`
