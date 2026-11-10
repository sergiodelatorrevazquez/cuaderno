---
titulo: "Convolución bidimensional y pooling"
asignatura: aprendizaje-profundo
fecha: 2026-11-05
orden: 9
tema: "Semana 3 — Sesión 8"
tags:
  - convolucion-2d
  - pooling
  - receptive-field
  - canales
  - rgb
  - conv2d
  - max-pooling
  - average-pooling
resumen: "Extensión de la operación local a dos dimensiones: ventanas 2D, correlación cruzada, canales, filtros, pooling y campos receptivos en CNNs."
---

## Página 1 — Convolución y pooling bidimensionales

**Tema:** Two-dimensional convolution and pooling

**Semana y sesión:** Week 3, Session 8 — Thursday, 19:20–20:20

**Pregunta central:** ¿Cómo convierte una operación local una imagen en mapas de características?

| Minutos | Enfoque |
|---|---|
| 0–15 | La ventana 2D |
| 15–30 | Canales y filtros |
| 30–45 | Pooling y campo receptivo |
| 45–60 | Código y ejercicio RGB |

**Objetivos de la sesión:**

- Calcular una capa convolucional 2D a mano.
- Rastrear canales y contar parámetros.
- Comparar `max` y `average pooling`.
- Reconstruir la región de entrada de una unidad.

Las ideas de la dimensión 1D se extienden por dos ejes espaciales.

**Mathematical and Computational Foundations** — Francisco R. Villatoro. Diapositivas principales 1–20; apéndices de referencia 21–26.

## Página 2 — Una ventana bidimensional

Una entrada con un único canal, un filtro, stride 1, sin padding y $b=0$.

$$
X \in \mathbb{R}^{5\times 5},\quad W \in \mathbb{R}^{2\times 2}.
$$

La misma ventana de $2 \times 2$ se mueve a lo largo de filas y columnas.

- En cada posición: multiplicar las entradas correspondientes, sumar y añadir el sesgo.
- Reutilizar los mismos cuatro pesos en todas las ubicaciones.
- Los índices empiezan en cero.

**Ejemplo:**

$$
X=
\begin{bmatrix}
2 & 2 & 0 & 1 & 3 \\
3 & 1 & 2 & 2 & 0 \\
0 & 1 & 3 & 1 & 2 \\
2 & 2 & 1 & 0 & 1 \\
1 & 0 & 2 & 3 & 1
\end{bmatrix},
\quad
W=
\begin{bmatrix}
1 & -1 \\
1 & -1
\end{bmatrix}.
$$

La suma local para la primera ventana es:

$$
Z_{0,0} = 2(1)+2(-1)+3(1)+1(-1)=2.
$$

Localidad y sharing: Goodfellow, Bengio & Courville (GBC), Ch. 9, §§9.1–9.2.

## Página 3 — La operación de correlación cruzada 2D

Para un kernel de tamaño $k_H \times k_W$, con stride 1 y sin padding:

$$
Z_{i,j} = b + \sum_{u=0}^{k_H-1}\sum_{v=0}^{k_W-1} W_{u,v} X_{i+u,j+v}.
$$

- $i,j$: posición de la ventana.
- $u,v$: desplazamiento dentro del kernel.

Ejemplo:

$$
W = \begin{bmatrix} 1 & -1 \\ 1 & -1 \end{bmatrix}.
$$

La primera ventana produce:

$$
Z_{0,0} = 2 - 2 + 3 - 1 = 2.
$$

Y, desplazada un paso hacia abajo y a la derecha:

$$
Z_{1,1} = 1 - 2 + 1 - 3 = -3.
$$

Las bibliotecas llaman a esto convolución, pero la operación real es correlación cruzada. PyTorch `Conv2d`; GBC, §9.1.

Los índices válidos cumplen:

$$
0 \le i \le H_{in} - k_H,\quad 0 \le j \le W_{in} - k_W.
$$

## Página 4 — Un mapa de características y su activación

La imagen de $5 \times 5$ produce un mapa de puntuaciones local de $4 \times 4$.

### Preactivación

$$
Z=
\begin{bmatrix}
2 & 1 & -1 & 0 \\
1 & -3 & 2 & 1 \\
-1 & -1 & 3 & -2 \\
1 & -1 & 0 & 1
\end{bmatrix}
$$

### Activación puntual

$$
H = \mathrm{ReLU}(Z) =
\begin{bmatrix}
2 & 1 & 0 & 0 \\
1 & 0 & 2 & 1 \\
0 & 0 & 3 & 0 \\
1 & 0 & 0 & 1
\end{bmatrix}
$$

- ReLU cambia los valores y conserva la forma del tensor.
- Cada posición reutiliza los mismos cuatro pesos y un sesgo.
- Un mapa de características registra dónde responde este filtro.

Ejemplo original. Aquí el sesgo vale cero, pero sigue siendo un parámetro entrenable.

## Página 5 — Altura y ancho de salida

Se cuentan los inicios válidos de la ventana de forma independiente en cada eje espacial:

$$
H_{out} = \left\lfloor \frac{H_{in}+p_T+p_B-k_H}{s_H} \right\rfloor + 1,
$$

$$
W_{out} = \left\lfloor \frac{W_{in}+p_L+p_R-k_W}{s_W} \right\rfloor + 1.
$$

- $p_T,p_B$: padding superior e inferior.
- $p_L,p_R$: padding izquierdo y derecho.
- $s_H,s_W$: strides vertical y horizontal.

En el ejemplo:

$$
H_{out} = W_{out} = \left\lfloor \frac{5 - 2}{1} \right\rfloor + 1 = 4.
$$

La altura y el ancho pueden usar tamaños, strides y padding distintos.

Dilatación 1; strides positivos; padding no negativo; debe caber al menos una ventana completa en la entrada paddeada. PyTorch `Conv2d`.

## Página 6 — Imágenes con varios canales

Una imagen RGB tiene dos ejes espaciales y tres componentes por píxel.

$$
\text{pixel }(i,j): (X_{R,i,j}, X_{G,i,j}, X_{B,i,j}).
$$

**Canales de entrada:** $C_{in}=3$.

**Un filtro espacial:** un filtro de $3 \times 3$ lee un parche del mismo punto en cada canal.

Para RGB, cada posición a lo largo de una imagen de 3 canales requiere:

$$
3 \times 3 \times 3 = 27 \text{ valores}.
$$

En capas posteriores, los canales son características aprendidas.

La convolución 2D se desliza en altura y anchura, y suma sobre canales.

Usamos mezcla densa ordinaria de canales con `groups=1`. El eje de canales no es un tercer eje espacial. GBC, §9.5; PyTorch Conv2d.

## Página 7 — Un filtro combina todos los canales de entrada

Para un canal de salida $o$, se suman las contribuciones de cada canal y se añade el sesgo una sola vez:

$$
Z_{n,o,i,j} = b_o + \sum_{c=0}^{C_{in}-1} \mathrm{corr2D}(X_{n,c}, W_{o,c})_{i,j}.
$$

Ejemplo de dos canales y una sola posición de salida: $n=0$, $o=0$, $i=0$, $j=1$.

**Canal 0**

$$
\begin{bmatrix} 2 & 0 \\ 1 & 2 \end{bmatrix},\quad
W_{0} = \begin{bmatrix} 1 & -1 \\ 1 & -1 \end{bmatrix}
$$

Contribución:

$$
2 - 0 + 1 - 2 = 1.
$$

**Canal 1**

$$
\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix},\quad
W_1 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
$$

Contribución:

$$
1 + 0 + 0 + 1 = 2.
$$

Total:

$$
Z_{0,0,0,1} = 1 + 2 + 1 = 4.
$$

El 1 final es el sesgo $b_0$.

## Página 8 — Un banco de filtros

Cada canal de salida tiene su propio filtro completo y su propio sesgo.

| Canal de salida | Slice para entrada 0 | Slice para entrada 1 | Bias |
|---|---|---|---|
| $o=0$ | $\begin{bmatrix}1 & -1 \\ 1 & -1\end{bmatrix}$ | $\begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix}$ | $b_0$ |
| $o=1$ | $\begin{bmatrix}1 & 1 \\ 1 & 1\end{bmatrix}$ | $\begin{bmatrix}-1 & 0 \\ 0 & -1\end{bmatrix}$ | $b_1$ |

Si la entrada es $X:(1,2,5,5)$, la salida es $Z:(1,2,4,4)$ en el layout NCHW.

En la posición $(0,1)$, el vector del canal de salida es $(4,1)$.

Dos filtros producen dos mapas, y cada mapa combina ambos canales de entrada.

## Página 9 — Parámetros entrenables en Conv2D

Para $C_{out}$ filtros, cada uno cubre $C_{in}$ canales de entrada:

$$
P = C_{out} C_{in} k_H k_W + 1 \quad \text{con bias.}
$$

**Ejemplo de dos canales y dos filtros**

| Cantidad | Valor |
|---|---:|
| Pesos del kernel | $2 \cdot 2 \cdot 2 \cdot 2 = 16$ |
| Biases | $2$ |
| Total | $18$ |

- El tamaño del batch y la resolución de la imagen no entran en este recuento.
- Imágenes más grandes producen más activaciones y más cálculo.

`groups=1`. Sin bias, se omite el $+1$.

## Página 10 — Layouts de tensores entre bibliotecas

La misma operación, distinto orden de ejes.

| Tensor | PyTorch | Keras / TensorFlow |
|---|---|---|
| Entrada $X$ | $(B,C_{in},H,W)$ | $(B,H,W,C_{in})$ |
| Kernel $W$ | $(C_{out},C_{in},k_H,k_W)$ | $(k_H,k_W,C_{in},C_{out})$ |
| Salida $Z$ | $(B,C_{out},H',W')$ | $(B,H',W',C_{out})$ |

Para NumPy:

```python
x_nhwc = x_nchw.transpose(0, 2, 3, 1)
w_hwio = w_oihw.transpose(2, 3, 1, 0)
```

Permutar los ejes conserva su significado. Un `reshape` no basta.

## Página 11 — Max pooling y average pooling

Volvemos a $H = \mathrm{ReLU}(Z)$ de la diapositiva 4. Usamos ventanas de $2 \times 2$ y stride 2.

$$
H =
\begin{bmatrix}
2 & 1 & 0 & 0 \\
1 & 0 & 2 & 1 \\
0 & 0 & 3 & 0 \\
1 & 0 & 0 & 1
\end{bmatrix}
$$

### Max pooling

La ventana más fuerte se usa como salida:

$$
\max(2,1,1,0)=2.
$$

### Average pooling

Se usa la media:

$$
\frac{2+1+1+0}{4}=1.
$$

El pooling espacial se realiza independientemente en cada canal. Aquí no hay padding.

**Diferencia clave**:

- Max pooling: respuesta más fuerte.
- Average pooling: respuesta media.

## Página 12 — Formas del pooling y la información

Para una ventana de pooling $q_H \times q_W$, sin padding y con tamaño suelo:

$$
H' = \left\lfloor \frac{H - q_H}{s_H} \right\rfloor + 1,
\qquad
W' = \left\lfloor \frac{W - q_W}{s_W} \right\rfloor + 1.
$$

Con `2x2`, stride 2:

$$
(B,C,4,4) \rightarrow (B,C,2,2).
$$

- El pooling conserva el tamaño del batch y el número de canales.
- Max y average pooling tienen cero parámetros entrenables.
- Cuatro valores pasan a convertirse en uno: se pierde detalle espacial.
- El tamaño de la ventana y el stride son hiperparámetros distintos.

Un mapa más pequeño reduce cómputo y cambia la información que se conserva.

## Página 13 — Pooling y translaciones

Se mueve una respuesta un píxel a la vez; se aplica max pooling con ventana $2\times 2$ y stride 2.

- Un movimiento dentro de la misma ventana puede conservar el valor pooled.
- Cruzar un borde de ventana puede cambiar el mapa de salida.

Esto muestra que la tolerancia local no implica invariancia exacta a la traslación.

En otras palabras: el pooling estratificado no garantiza insensibilidad frente a todas las traducciones pequeñas.

## Página 14 — Campo receptivo y espaciado de salida

Ahora usamos una convolución $3\times 3$, stride 1, padding 1, seguida de pooling $2\times 2$.

- Entrada $X$: $8 \times 8$
- Activación $H$: $8 \times 8$
- Pooled: $4 \times 4$

**Campo receptivo $r$**: extensión de entrada que puede influir en una salida.

**Spacing $j$**: distancia entre salidas adyacentes en coordenadas de entrada.

Aquí, para una unidad interna:

- campo receptivo: $4 \times 4$
- espaciado de salida: 2 píxeles

A diferencia del actual gradiente para una imagen concreta, esto representa dependencia geográfica potencial.

## Página 15 — Campos receptivos a través de una pila

Se calcula un eje espacial a la vez. Empezando con

$$
r_0 = j_0 = 1.
$$

La recurrencia es:

$$
r_\ell = r_{\ell-1} + (k_\ell - 1) j_{\ell-1},
\qquad
j_\ell = s_\ell j_{\ell-1}.
$$

| Operación | $k$ | $s$ | $(r,j)$ | Extensión de entrada |
|---|---:|---:|---|---|
| Entrada | – | – | $(1,1)$ | $1\times 1$ |
| Conv $3\times 3$, padding 1 | 3 | 1 | $(3,1)$ | $3\times 3$ |
| ReLU | 1 | 1 | $(3,1)$ | sin cambios |
| Pool $2\times 2$, sin padding | 2 | 2 | $(4,2)$ | $4\times 4$ |

¿Por qué 4 y no 6? Porque las regiones adyacentes de $3\times 3$ se solapan.

## Página 16 — Un bloque convolucional en código

Entrada RGB: Conv $3\times 3$, ocho canales de salida, ReLU, luego max pooling.

**PyTorch: NCHW**

```python
from torch import nn
block = nn.Sequential(
    nn.Conv2d(3, 8, 3, stride=1, padding=1, bias=True),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2, padding=0)
)
```

**Keras: NHWC, TensorFlow backend**

```python
import keras
L = keras.layers
block = keras.Sequential([
    keras.Input((32, 32, 3)),
    L.Conv2D(8, 3, strides=1, padding="same", use_bias=True, data_format="channels_last"),
    L.ReLU(),
    L.MaxPooling2D(2, strides=2, padding="valid", data_format="channels_last")
])
```

Para este kernel impar y stride 1, `Keras same` equivale a un cero a cada lado.

## Página 17 — Ejercicio: un lote de imágenes RGB

Trabaja en parejas durante 3 minutos y luego compara la lógica.

**Lote de dos imágenes RGB, cada una de $32\times 32$**:

$$
\mathrm{Conv}(3\rightarrow 8, 3\times3, s=1, p=1, bias) \rightarrow ReLU \rightarrow MaxPool(2\times2, s=2).
$$

1. Da las formas de entrada, preactivación, activación y salida del pooling en PyTorch.
2. Escribe las mismas cuatro formas en Keras con canales al final.
3. Da las formas del kernel y del sesgo y cuenta los parámetros entrenables.
4. Encuentra el campo receptivo final y el espaciado de salida en píxeles de entrada.

**Explicación:** ¿Qué cantidades dependen del tamaño del batch o de la resolución?

## Página 18 — Solución: formas y parámetros

| Etapa | PyTorch: NCHW | Keras: NHWC | Parámetros |
|---|---|---|---:|
| Entrada $X$ | $(2,3,32,32)$ | $(2,32,32,3)$ | – |
| Conv salida $Z$ | $(2,8,32,32)$ | $(2,32,32,8)$ | 224 |
| ReLU salida $H$ | $(2,8,32,32)$ | $(2,32,32,8)$ | 0 |
| Pool salida $P$ | $(2,8,16,16)$ | $(2,16,16,8)$ | 0 |

Kernel en PyTorch: $(8,3,3,3)$; en Keras: $(3,3,3,8)$.

Bias: $(8,)$.

Total:

$$
8(3\cdot 3\cdot 3 + 1)=224.
$$

**Tamaño espacial**:

$$
\frac{32+2-3}{1}+1=32,\qquad \frac{32-2}{2}+1=16.
$$

El recuento de parámetros es independiente del número de imágenes y de la resolución $32\times32$. Los mismos pesos aprendidos se reutilizan en todas las ventanas y en todos los elementos del batch.

## Página 19 — Solución: la región de entrada

Para el bloque `Conv3 -> ReLU -> Pool2`:

$$
(r,j): (1,1) \rightarrow (3,1) \rightarrow (3,1) \rightarrow (4,2).
$$

Para un eje, la posición de salida y los índices de entrada posibles son:

$$
2i - 1, 2i, 2i + 1, 2i + 2.
$$

Por ejemplo:

- índice interno $i=1$: $1,2,3,4$
- margen $i=0$: $-1,0,1,2$; solo los índices reales $0,1,2$ son válidos

Un píxel interior de salida tiene un campo receptivo nominal de $4\times4$. En la esquina superior izquierda, solo hay $3\times3$ píxeles reales disponibles.

La aplicación es por filas y columnas, y a través de los tres canales RGB.

Este es un cálculo de dependencia potencial; los pesos nulos, ReLU y el máximo seleccionado pueden reducir la influencia real. El padding añade constantes, no información de imagen.

## Página 20 — Lista de comprobación de la sesión y próximos pasos

1. Canales: ¿un filtro ordinario lee un canal de entrada o todos ellos?
2. Parámetros: ¿duplicar la anchura de la imagen duplica los pesos aprendidos?
3. Pooling: ¿qué dimensiones cambian y qué información puede desaparecer?
4. Campo receptivo: ¿por qué `Conv3` seguido de `Pool2` da $4\times4$?

**Puente al bloque de Enrique** — Siguientes fundamentos: supuestos de Markov y secuencias probabilísticas. Lo que sigue: la conexión con n-gramas.

Antes de irse: justifica una respuesta usando forma, fórmula o una pequeña ventana.

## Página 21 — El operador Conv2D completo

**Referencia** | índices a partir de cero, mezcla ordinaria de canales (`groups=1`).

$$
Z_{n,o,i,j} = b_o + \sum_{c=0}^{C_{in}-1}\sum_{u=0}^{k_H-1}\sum_{v=0}^{k_W-1} W_{o,c,u,v} X_{n,c, \, i s_H - p_T + u d_H, \, j s_W - p_L + v d_W}.
$$

$X$ coincide con la imagen y es cero fuera de ella.

- $n$: elemento del batch.
- $o,c$: canales de salida y entrada.
- $i,j$: coordenadas de salida.
- $u,v$: desplazamientos del kernel.
- $s_H,s_W$: strides.
- $d_H,d_W$: dilataciones del kernel.

La fórmula de anchura es análoga.

El sesgo se añade una vez por valor de salida.

**Cuenta**:

$$
C_{out}(C_{in}k_Hk_W + 1).
$$

La dilatación cambia las posiciones muestreadas, pero no añade coeficientes al kernel.

PyTorch Conv2d; GBC, §9.5. Las diapositivas principales usan $d_H=d_W=1$.

## Página 22 — Pooling en los bordes de la imagen

**Referencia** | producir formas iguales no implica producir valores iguales.

Media de una ventana $3\times3$, stride 1, con un padding implícito en cada lado:

$$
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
$$

| Configuración | Cada valor de salida |
|---|---|
| PyTorch `AvgPool2d(3, stride=1, padding=1)` | $10/9$ |
| PyTorch con `count_include_pad=False` | $10/4 = 2.5$ |
| TensorFlow / Keras–TensorFlow, `padding="SAME"` | $10/4 = 2.5$ |

- Max pooling excluye el padding del máximo (conceptualmente $-\infty$).
- Rellenar manualmente una imagen con ceros negativos puede cambiar el resultado.
- TensorFlow `SAME`: tamaño de salida $\lceil n/s \rceil$; el padding puede ser asimétrico.
- PyTorch `ceil_mode=True` y TensorFlow `SAME` no son la misma configuración.

## Página 23 — Campo receptivo, salto y alineación

**Referencia** | usar la recurrencia en cada eje espacial.

$$
k_{eff,\ell} = d_\ell(k_\ell - 1) + 1,
$$

$$
r_\ell = r_{\ell-1} + (k_{eff,\ell} - 1) j_{\ell-1},
\qquad
j_\ell = s_\ell j_{\ell-1}.
$$

Sea $a_\ell$ el centro del índice cero de salida en coordenadas de entrada:

$$
a_\ell = a_{\ell-1} + \left(\frac{k_{eff,\ell}-1}{2} - p^{-}_\ell\right) j_{\ell-1},
\quad r_0=j_0=1,
\quad a_0=2.
$$

| Operación | $r$ | $j$ | $a$ |
|---|---:|---:|---:|
| Conv 3, stride 1, padding 1 | 3 | 1 | 0.5 |
| Pool 2, stride 2, sin padding | 4 | 2 | 1 |
| Conv 3 adicional, stride 1, padding 1 | 8 | 2 | 1 |

La salida de índice $t$ tiene centro $a_\ell + t j_\ell$; $p^{-}_\ell$ es el padding antes de la entrada.

**Nota:** los intervalos nominales pueden incluir posiciones rellenadas o huecos; la influencia real también depende de los valores. Recurrencias adaptadas de Araujo, Norris & Sim (2019).

## Página 24 — Gradientes a través del pooling

**Referencia** | conectar el pooling local a la regla de la cadena.

$$
A = \begin{bmatrix} 2 & 1 \\ 1 & 0 \end{bmatrix},\quad p_{max}=2,\quad p_{avg}=1,\quad \frac{\partial L}{\partial p}=6.
$$

### Max pooling

El máximo es único y está en la esquina superior izquierda.

$$
\frac{\partial L}{\partial A} = \begin{bmatrix} 6 & 0 \\ 0 & 0 \end{bmatrix}
$$

### Average pooling

Cada entrada tiene derivada $1/4$.

$$
\frac{\partial L}{\partial A} = \begin{bmatrix} 1.5 & 1.5 \\ 1.5 & 1.5 \end{bmatrix}
$$

El gradiente va al valor seleccionado. Para ventanas superpuestas, las contribuciones se suman. Cuando hay máximos empatados, la función max no es diferenciable y la implementación elige una regla de retropropagación.

## Página 25 — Reconciliar pesos entre librerías

**Referencia** | arrays `X`, `W` y `b` usan `NCHW`, `OIHW` y `(C_out,)`.

```python
x = X.transpose(0, 2, 3, 1)    # NCHW -> NHWC
w = W.transpose(2, 3, 1, 0)    # OIHW -> HWIO
```

**Keras 3, TensorFlow backend**

```python
layer = keras.layers.Conv2D(
    2, 2, strides=1, padding="valid",
    use_bias=True, activation=None,
    data_format="channels_last", dtype="float64")
layer.build(x.shape)
layer.set_weights([w, b])
z_keras = layer(x).numpy()
```

**TensorFlow: operación**

```python
z_tf = tf.nn.conv2d(x, w, strides=(1,1), padding="VALID", data_format="NHWC")
z_tf = tf.nn.bias_add(z_tf, b, data_format="NHWC")
z_nchw = z_tf.numpy().transpose(0, 3, 1, 2)
```

Usa los mismos valores, dtype, padding y activación para comparar resultados. El ejemplo de dos canales tiene $Z_{0,0,0,1}=4$ en NCHW.

## Página 26 — Lecturas y ejemplos reproducibles

**Conceptos y explicaciones matemáticas**

- Goodfellow, Bengio & Courville (2016), Deep Learning, Ch. 9: §9.1 convolución, §9.2 motivación, §9.3 pooling, §9.5 variantes.
- Christopher M. Bishop & Hugh Bishop (2024), Deep Learning: Foundations and Concepts, Ch. 10, Convolutional Networks.
- Araujo, Norris & Sim (2019), Computing Receptive Fields of Convolutional Neural Networks, Distill.

**Traducir la notación a una API**

- PyTorch `Conv2d`, `MaxPool2d`, `AvgPool2d`
- Keras `Conv2D` y capas de pooling
- TensorFlow `conv2d`

**Programas acompañantes**

```bash
python session8_numeric.py
python session8_frameworks.py --backend numpy
python session8_frameworks.py --backend torch       # opcional
python session8_frameworks.py --backend tensorflow  # opcional
```

Las versiones de ejemplo originales. Documentación oficial consultada el 18 de octubre de 2026.
