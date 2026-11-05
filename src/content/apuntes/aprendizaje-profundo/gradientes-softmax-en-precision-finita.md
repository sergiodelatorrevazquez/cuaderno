---
titulo: "Gradientes softmax en precisión finita"
asignatura: aprendizaje-profundo
fecha: 2026-10-30
orden: 7
tema: "Sesión 6 — paper discussion"
tags: [softmax, cross-entropy, precision-finita, Adam, atención, BF16, arXiv, aprendizaje-profundo]
resumen: "Discusión de dos artículos sobre la estabilidad numérica del softmax y la entropía cruzada en precisión finita, así como sobre el deterioro del gradiente en atención BF16 y correcciones de backward."
---

## Página 1 — Softmax gradients in finite precision

**A Fix for Softmax and Cross-Entropy**

- J. T. Barron, D. Gnanapragasam y M. Tschannen
- arXiv:2610.31642v2, 26 de octubre de 2026

**Broken Symmetry in BF16 Attention**

- Why FlashAttention Gradients Blow Up Late in Training
- Junlin Chen et al.
- arXiv:2609.34272v1, 28 de septiembre de 2026

**Pregunta central:** ¿puede redondear un cálculo exacto y cambiar una actualización de entrenamiento?

**Después de la sesión 5**

- 20 minutos + apéndices

**Docente:** Francisco R. Villatoro.

**Prerequisitos:** softmax, entropía cruzada, autodiff y Adam.

Ambos trabajos son preprints de arXiv. La atención se introducirá como promedio ponderado.

## Página 2 — El gradiente que ya conocemos

Para un ejemplo con etiqueta $y$ y temperatura $T=1$:

$$
p_k = \frac{e^{s_k}}{\sum_j e^{s_j}}, \qquad \ell = -\log p_y,
$$

$$
 g_k = \frac{\partial \ell}{\partial s_k} = p_k - \mathbf{1}\{k=y\}.
$$

**Probabilidades suman uno**:

$$
\sum_k g_k = \sum_k p_k - 1 = 0.
$$

El gradiente objetivo equilibra todos los demás gradientes logit.

**Cambio común**:

$$
\ell(s + c\mathbf{1}, y) = \ell(s, y).
$$

La suma cero también es la derivada a lo largo de un desplazamiento común.

Estas son identidades matemáticas exactas. ¿Cómo se evalúan en práctica?

## Página 3 — Una predicción segura de tres clases

$$
s = (9,-9,-9), \quad y = 0,
$$

$$
s - \max(s) = (0,-18,-18).
$$

Sea:

$$
r = 2e^{-18} \approx 3.046 \times 10^{-8}.
$$

En `float32`, $r$ es representable, pero $1+r$ se redondea a 1: la contribución pequeña desaparece cerca de uno.

| Cantidad | Ruta convencional `float32` | Referencia matemática |
|---|---:|---:|
| Pérdida $\ell$ | 0 | $3.046\times10^{-8}$ |
| Gradiente objetivo $g_0$ | 0 | $-3.046\times10^{-8}$ |
| Cada uno de $g_1, g_2$ | $1.523\times10^{-8}$ | $1.523\times10^{-8}$ |
| Suma $g_0 + g_1 + g_2$ | $3.046\times10^{-8}$ | 0 |

El gradiente objetivo desaparece mientras sobreviven los otros gradientes.

## Página 4 — El precipicio de precisión

El margen agregado compara la clase objetivo frente a todas las alternativas:

$$
m = s_y - \log\sum_{j\ne y} e^{s_j}.
$$

En el ejemplo, $m = 18 - \log 2 \approx 17.31$.

**Cliffs aproximados:**

- `bfloat16`: 5.5
- `float32`: 16.6
- `float64`: 36.7

El desplazamiento por máximo controla el desbordamiento. Pero la suma

$$
1 + e^{-m}
$$

puede seguir redondeándose a 1.

## Página 5 — La misma pérdida, calculada de forma distinta

Sea:

$$
r = \sum_{j\ne y} e^{s_j-s_y} = e^{-m}.
$$

**Pérdida hacia delante**:

$$
\ell = \log(1+r) = \log1p(r).
$$

`log1p(r)` conserva la contribución pequeña.

**Gradiente objetivo**:

$$
 g_y = \frac{1}{1+r}-1 = -\frac{r}{1+r}.
$$

La pequeña numerador sobrevive al redondeo en $1+r$.

Forma general, incluyendo un objetivo no máximo:

$$
 a = \mathrm{LSE}_{j\ne y}(s_j - s_y), \qquad \ell = \mathrm{softplus}(a),
$$

$$
 g_y = -\sum_{j\ne y} g_j.
$$

`LSE` es log-sum-exp; `softplus(a) = log(1 + e^a)`. Hay que usar primitivas estables.

## Página 6 — Por qué Adam puede amplificar el efecto

Ilustración escalar: optimizar un logit directamente. Para un gradiente constante,

$$
\hat{m} \approx g, \qquad \sqrt{\hat{v}} \approx |g|,
$$

así que la actualización normalizada tiene magnitud aproximada:

$$
\frac{|\Delta \theta|}{\eta} \approx \frac{|g|}{|g| + \varepsilon}.
$$

Al variar el margen logit:

- con $\varepsilon = 10^{-8}$,
- $g = 0$ da 0,
- $|g| = 3.046\times10^{-8}$ da aproximadamente 0.753.

Gradientes muy pequeños pueden producir actualizaciones normalizadas apreciables.

Es una simplificación de gradiente constante, no una trayectoria completa de Adam.

## Página 7 — Entropía cruzada: evidencia de entrenamiento

Se estudia un clasificador para aprender $a + b \bmod 31$: 31 clases, `float32`, 5 semillas.

**AdamW:** $\eta = 10^{-3}$, $\varepsilon = 10^{-8}$, regularización por peso $\lambda = 1$.

- **Pérdida de entrenamiento**
- **Precisión de prueba**

La CE corregida elimina los picos recurrentes de pérdida en este experimento.

La figura de los autores muestra solo la fila superior.

## Página 8 — La atención como promedio ponderado

Una consulta $q$ se compara con claves candidatas $k_j$.

Cada clave tiene un valor $v_j$ que lleva información para combinar.

$$
 s_j = \alpha q^\top k_j, \qquad \alpha = \frac{1}{\sqrt{d}},
$$

$$
 p_j = \frac{e^{s_j}}{\sum_i e^{s_i}},
\qquad
 o = \sum_j p_j v_j,
$$

con $p_j \ge 0$ y $\sum_j p_j = 1$.

**Operaciones familiares:** producto escalar, softmax y suma ponderada.

**Dónde aparece:** dentro de capas de atención. Una pérdida posterior depende de $o$.

**FlashAttention** calcula esta operación eficientemente en GPU.

**BF16** es un formato de punto flotante con menos bits significativos que `float32`.

## Página 9 — La misma identidad del gradiente dentro de la atención

Para una pérdida posterior $L$, escribimos:

$$
 g_j = \frac{\partial L}{\partial s_j}.
$$

Entonces:

$$
\sum_j g_j = 0,
\qquad
G_q = \frac{\partial L}{\partial q} = \alpha \sum_j g_j k_j.
$$

Un desplazamiento común de todas las claves deja el softmax sin cambios.

Si se suma $b$ a todas las $k_j$, se añade el mismo término $\alpha q^\top b$ a cada puntuación.

La contribución extra del gradiente del query es:

$$
\alpha b \sum_j g_j = 0.
$$

**Pero** el redondeo puede romper esta cancelación.

Para coeficientes redondeados fijos $t_j$, sea $\rho = \sum_j t_j$. Entonces:

$$
\hat{G}_q(K+b) - \hat{G}_q(K) = \alpha \rho b.
$$

Estas son gradientes de puntuación de atención, no $p - \mathbf{1}\{y\}$.

## Página 10 — Un cast BF16 convierte un cero exacto en −24

Cuatro claves, atención uniforme, cálculo exacto hacia delante.

Sea:

$$
 h = 3/1024.
$$

El gradiente exacto de puntuación y su redondeo BF16 son:

$$
 g = (1+h, -1, -h, 0),
$$

$$
\sum_j g_j = 0,
$$

$$
 t = \mathrm{BF16}(g) = (1,-1,-h,0),
\quad \rho = -h.
$$

En una coordenada, las cuatro claves son iguales a 65536. La escala es $\alpha = 1/8$.

**Derivada exacta del query**:

$$
\frac{1}{8\cdot 65536}\sum_j g_j = 0.
$$

**Después del cast BF16**:

$$
\frac{1}{8\cdot 65536}\sum_j t_j = -24.
$$

Acumular el producto en `float32` no restaura el término perdido.

## Página 11 — Reparación hacia delante y proyección del gradiente

**SBS forward:** restar el máximo antes de escalar.

**GProj backward:** corregir la suma residual después del cast BF16.

Sea:

$$
\rho = \sum_j t_j,
\qquad
M = \sum_j r_j > 0,
$$

$$
 t_j^{\perp} = t_j - \frac{\rho}{M} r_j.
$$

Entonces, en aritmética exacta:

$$
\sum_j t_j^{\perp} = \rho - \frac{\rho}{M}M = 0.
$$

- La masa almacenada $M$ no tiene que ser 1.
- El kernel corrige las contracciones sin recast de $t^{\perp}$ a BF16.
- Otros errores de gradiente pueden permanecer.

La corrección sigue $r$, no la dirección uniforme.

## Página 12 — Precisión del gradiente sobre entradas capturadas de entrenamiento

Se compara la mediana del error relativo sobre ocho entradas capturadas, con referencia en FP64.

| Método | Gradiente de query $dQ$ | Gradiente de clave $dK$ |
|---|---:|---:|
| FA3 | 773% | 85.6% |
| FA3-SBS | 219% | 13.3% |
| GProj | 0.342% | 0.371% |
| Atención FP32 | 0.385% | 0.371% |

FA3-SBS y GProj tienen salidas hacia delante idénticas.

La diferencia aislada está en el backward.

**Coste medido:** GProj añade un 4.73% a un paso completo de entrenamiento en un modelo de 450M, 4096 tokens y un H200.

## Página 13 — Experimento de entrenamiento con atención (450M)

La figura muestra pérdida de entrenamiento en función de tokens de entrenamiento.

- FA3: 1.866
- SBS: 1.679
- GProj: 1.665
- FP32: 1.665

SBS elimina ya la principal explosión de gradiente. GProj mejora el hueco restante.

El gradiente pre-clipping muestra la misma tendencia: FA3 y FA3-SBS difieren claramente de GProj y FP32.

## Página 14 — Lo que enseñan los dos artículos

### En entropía cruzada

- Términos pequeños pueden desaparecer en $1+r$ o en $p_y-1$.
- Fórmulas equivalentes pueden conservar esos términos.

### En atención

- Un gradiente correcto de puntuación puede perder su suma cero en un cast BF16.
- GProj elimina este modo residual de suma.

**Diagnóstico:** comparar identidades con una referencia de mayor precisión.

**Preguntas de discusión**

- ¿Una pérdida finita, sin NaN, prueba que los gradientes son precisos?
- ¿Hacer que la suma del gradiente sea cero recupera cada componente?

## Página 15 — Apéndice A: contrato de entrada de la pérdida

El contrato de entrada sigue siendo familiar:

- suministrar logits sin procesar a la pérdida;
- mantener etiquetas enteras para el caso objetivo duro;
- preservar cantidades pequeñas en las rutas hacia delante y hacia atrás.

Ejemplo de implementación auxiliar en PyTorch:

```python
from paper_numeric_demo import fixed_cross_entropy_torch

per_example = fixed_cross_entropy_torch(logits, y)
loss = per_example.mean()
loss.backward()
```

`logits`: forma `[B, C]`; `y`: índices de clase enteros de longitud `[B]`.

## Página 16 — Apéndice B: implementación transparente en PyTorch

```python
import torch
import torch.nn.functional as F

def fixed_cross_entropy_torch(logits, y):
    target_logit = logits.gather(-1, y.unsqueeze(-1))
    target_mask = F.one_hot(y, logits.shape[-1]).bool()
    others = (logits - target_logit).masked_fill(target_mask, float("-inf"))
    a = torch.logsumexp(others, dim=-1)
    return F.softplus(a)
```

- La clase objetivo se excluye explícitamente de la suma exponencial.
- Autodiff llega a ella a través de las $C-1$ diferencias restantes.
- No hay ponderación de clases, suavizado de etiquetas ni etiquetas ignoradas.

## Página 17 — Apéndice C: emparejar un objetivo suave

Un profesor fijo suministra probabilidades; un estudiante aprende a emparejarlas.

$$
 q = \mathrm{softmax}(s_T), \qquad p = \mathrm{softmax}(s_S),
\qquad
\ell = -\sum_k q_k \log p_k.
$$

**Requisito exacto**:

$$
\nabla_{s_S} \ell = p - q.
$$

Cuando $s_S = s_T$, el gradiente es cero.

La pérdida es generalmente positiva: $\ell = H(q)$.

**Dificultad numérica:** probabilidades casi iguales pierden precisión relativa al restarlas.

Diferentes fórmulas hacia delante y hacia atrás pueden dejar un residuo incluso en un emparejamiento perfecto.

El enfoque del paper: empezar con $s_S - s_T$, y usar `expm1(x)` para $e^x - 1$ y `log1p(x)` para $\log(1+x)$.

## Página 18 — Apéndice D: la diferencia para objetivos suaves

Profesor congelado:

$$
 q = \mathrm{softmax}(s_T),
\qquad
\delta_k = s_{S,k} - s_{T,k},
\qquad
 d_k = \delta_k - \sum_i q_i \delta_i.
$$

Se define:

$$
 a_k = d_k - \log\left(1 + \sum_i q_i \, \mathrm{expm1}(d_i)\right),
$$

$$
 u_k = q_k \, \mathrm{expm1}(a_k).
$$

- Cerca de un emparejamiento, $u_k$ evalúa $p_k - q_k$ sin restar dos probabilidades redondeadas.
- Para $|a_k| \ge 1$, se usa $u_k = p_k - q_k$ como respaldo.
- Su corrección hacia atrás es:

$$
 g_k = u_k - p_k\sum_i u_i.
$$

En logits idénticos, $\delta = d = a = u = 0$.

## Página 19 — Apéndice E: equilibrio y weight decay

En un punto estacionario con gradientes constantes, AdamW da:

$$
\frac{g_i}{|g_i| + \varepsilon} + \lambda \theta_i = 0.
$$

Cuando $|g_i| \ll \varepsilon$:

$$
 g_i \approx -\varepsilon \lambda \theta_i.
$$

La contribución del gradiente equilibra el decaimiento desacoplado.

El margen de escala del paper es:

$$
 m^* \approx \log(1/\varepsilon).
$$

Esto es una tendencia principal apoyada por experimentos controlados.

La relación depende también de $\lambda$, de la magnitud de los parámetros y de su efecto sobre los logits.

## Página 20 — Apéndice F: la proyección dentro de productos matriciales

Para una fila, si $u = \partial L / \partial o$, y $a_j = u^T v_j$, entonces:

$$
 g_j = p_j \left(a_j - \sum_i p_i a_i\right),
$$

con $\sum_j g_j = 0$.

A lo largo de filas de consulta, sean $T$ gradientes de puntuación redondeados y $R$ probabilidades redondeadas. Sea:

$$
\gamma_i = \frac{\sum_j T_{ij}}{\sum_j R_{ij}}.
$$

La corrección ideal es:

$$
 dQ = \alpha\bigl(TK - \gamma \odot (RK)\bigr),
$$

$$
 dK = \alpha\bigl(T^T Q - RT^T(\gamma \odot Q)\bigr).
$$

- $\gamma \odot A$ escala la fila $i$ de $A$ por $\gamma_i$.
- Los productos BF16 y la acumulación en FP32 mantienen la ruta aritmética rápida.
- La corrección clave requiere un segundo pase.
- `dV` no cambia.

## Página 21 — Apéndice G: fuentes y reproducibilidad

**Barron, Gnanapragasam y Tschannen (2026)**

- *A Fix for Softmax and Cross-Entropy*
- arXiv:2610.31642v2, 26 de octubre de 2026.

**Chen et al. (2026)**

- *Broken Symmetry in BF16 Attention: Why FlashAttention Gradients Blow Up Late in Training*
- arXiv:2609.34272v1, 28 de septiembre de 2026.

**Reproducibilidad en CPU**:

```bash
python paper_numeric_demo.py
python attention_bf16_demo.py
```

Los scripts de Python y los resultados JSON están incluidos.

Las figuras de entrenamiento son de los propios artículos. Las demos en CPU reproducen solo la aritmética.
