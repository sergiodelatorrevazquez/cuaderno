---
titulo: "Patrones locales en señales 1D"
asignatura: aprendizaje-profundo
fecha: 2026-11-05
orden: 8
tema: "Sesión 7"
tags:
  - patrones-locales
  - convolucion-1d
  - pesos-compartidos
  - padding
  - stride
  - canales
  - dilatacion
resumen: "Introducción a la convolución 1D como operador local con pesos compartidos, padding, stride y canales, además de la conexión con la capa convolucional y la covarianza por traslación."
---

## Página 1 — Patrones locales en señales 1D

**Tema:** Local patterns in 1D signals

**Semana y sesión:** Week 3, Session 7 — Thursday, 17:40–19:10

**Pregunta central:** ¿Cómo puede una operación aprendible detectar un patrón en muchas posiciones?

| Minutos | Enfoque |
|---|---|
| 0–10 | Patrones locales |
| 10–30 | Productos deslizantes y convenciones |
| 30–45 | Localidad y pesos compartidos |
| 45–65 | Padding, stride y longitud |
| 65–80 | Canales y notación de librerías |
| 80–90 | Ejercicio y ticket de salida |

**Objetivos de la sesión:**

- Calcular a mano una capa 1D.
- Explicar localidad y pesos compartidos.
- Derivar la longitud de salida.
- Rastrear canales y contar parámetros.

**Mathematical and Computational Foundations** — Francisco R. Villatoro. Diapositivas principales 1–26; apéndices de referencia 27–34.

## Página 2 — Un patrón útil puede aparecer en cualquier lugar

Un evento breve puede desplazarse dentro de una grabación de sensores.

- Examinar un vecindario local en cada posición.
- Reutilizar un detector a lo largo de la secuencia.
- Mantener la posición de cada respuesta.

**Pregunta:** ¿Qué cambia cuando el patrón de entrada se desplaza?

**Referencia original de esquemas:** Locality and parameter sharing. Goodfellow, Bengio & Courville (GBC), Ch. 9, §9.2; Bishop & Bishop, Ch. 10.

## Página 3 — Una puntuación local en cada posición

Nuestra señal y los valores iniciales del núcleo son:

$$
x = (1, 2, 0, -1, 3, 1), \quad w = (1, 2, -1), \quad b = 0.
$$

| Objeto | Significado | En este ejemplo |
|---|---|---|
| $x_t$ | Una medición real en la posición $t$ | $t = 0, \ldots, 5$ |
| $w$ | Pesos aplicados a una ventana local | $K = 3$ pesos |
| $b$ | Un desplazamiento reutilizado en cada posición | Inicialmente cero |
| $z_t$ | Una respuesta local por ventana | Una secuencia de puntuaciones |

**Entrenable:** valores del núcleo y sesgo. **Elegido:** tamaño del núcleo y cómo se colocan las ventanas.

Todos los índices empiezan en cero. Comenzamos con un canal de entrada, un canal de salida, sin padding y stride 1. Los valores son fijos para la aritmética; el entrenamiento actualizaría las mismas variables.

## Página 4 — Correlación cruzada: un producto escalar deslizante

Para un núcleo de longitud $K$, sin padding y stride 1:

$$
z_t = b + \sum_{r=0}^{K-1} w_r x_{t+r}, \quad t = 0, \ldots, L_{in} - K.
$$

**Leemos los índices**

- $t$: inicio de la ventana actual.
- $r$: desplazamiento dentro del núcleo.

**Leemos la operación**

- Multiplicar las entradas correspondientes.
- Sumar los $K$ productos.
- Añadir el sesgo compartido.

Las bibliotecas de aprendizaje profundo llaman a esta operación convolución.

La operación real es correlación cruzada: PyTorch `Conv1d` y TensorFlow `nn.convolution`. GBC, §9.1.

## Página 5 — Recorremos la primera ventana

Usamos los primeros tres valores, $x_0, x_1, x_2$, y los mismos tres pesos.

$$
\begin{array}{c|cccccc}
          x[n] & 1 & 2 & 0 & -1 & 3 & 1 \\
          w[r] & 1 & 2 & -1
\end{array}
$$

Con la suma y añadiendo $b=0$:

$$
z_0 = 1\cdot 1 + 2\cdot 2 + (-1)\cdot 0 = 5.
$$

**Comprobación:** ¿Qué muestras y pesos contribuirán a $z_1$?

Ejemplo aritmético original. Un peso negativo cambia el signo de una contribución; no invierte el núcleo.

## Página 6 — Desliza los mismos pesos

$$
w = (1,2,-1), \quad b=0, \quad L_{in}=6, \quad K=3.
$$

| Posición $t$ | Ventana de entrada | Producto escalar | $z_t$ |
|---|---|---:|---:|
| 0 | $(1,2,0)$ | $1+4+0$ | 5 |
| 1 | $(2,0,-1)$ | $2+0+1$ | 3 |
| 2 | $(0,-1,3)$ | $0-2-3$ | -5 |
| 3 | $(-1,3,1)$ | $-1+6-1$ | 4 |

$$
z = (5,3,-5,4), \quad L_{out}=6-3+1=4.
$$

Hay cuatro posiciones de salida que reutilizan tres pesos y un sesgo.

Se calcula una ventana antes de comprobar su fila. Hay 12 productos peso–entrada, pero solo 3 parámetros distintos del núcleo.

## Página 7 — La convolución invierte el núcleo

La convolución discreta matemática es

$$
(x * w)[n] = \sum_{r=0}^{K-1} w_r x[n-r].
$$

Para la parte válida, escribimos $n = t + K - 1$ y reindexamos:

$$
(x * w)[t+K-1] = \sum_{r=0}^{K-1} w_{K-1-r} x[t+r].
$$

| Correlación cruzada | Convolución |
|---|---|
| $(1,2,-1) \cdot (1,2,0)=5$ | $(-1,2,1) \cdot (1,2,0)=3$ |

Se invierte el orden de los coeficientes; se mantienen los signos de los coeficientes.

GBC, §9.1; NumPy convolve. Se usan secuencias reales y $K \le L_{in}$. La convolución válida aquí tiene $n = 2, \ldots, 5$.

## Página 8 — Compara las dos operaciones

La entrada y los valores numéricos del núcleo son idénticos.

| Operación | Pesos de la ventana | Salida válida |
|---|---|---|
| Correlación cruzada | $(1,2,-1)$ | $(5,3,-5,4)$ |
| Convolución | $(-1,2,1)$ | $(3,-3,1,8)$ |

```python
import numpy as np
x = np.array([1., 2., 0., -1., 3., 1.])
w = np.array([1., 2., -1.])
np.correlate(x, w, mode="valid")  # [ 5.,  3., -5.,  4.]
np.convolve(x, w, mode="valid")   # [ 3., -3.,  1.,  8.]
```

Un núcleo aprendido sin restricciones puede absorber la inversión por reparametrización. Cuando se reutilizan pesos específicos, la convención debe coincidir.

NumPy correlate; NumPy convolve; GBC, §9.1.

## Página 9 — De un operador local a una capa neuronal

El operador convolucional es afín en la entrada cuando sus parámetros son fijos.

$$
x \mapsto z = C_{w,b}(x) \mapsto h = \mathrm{ReLU}(z).
$$

| Entrada | Operador afín local | $z$ | ReLU puntual |
|---|---|---|---|
| $x$ | pesos y sesgo compartidos | $h_t = \max(0,z_t)$ | $(5,3,0,4)$ |

La secuencia resultante es un mapa de características. Capas posteriores y una salida específica para la tarea pueden transformar las características en predicciones.

**Conexión con la sesión 1:** mapas afines, activación y composición. Estas cuatro puntuaciones locales no son una distribución de probabilidad de cuatro clases; no se aplica softmax a través de posiciones aquí.

## Página 10 — La localidad elimina conexiones distantes

Compare mapas de seis entradas a cuatro salidas.

| Conexión densa | Conexión local, $K=3$ |
|---|---|
| Cada salida puede usar las seis entradas. | Cada salida usa solo su ventana de tres muestras. |

**Localidad:** especifica qué conexiones pueden existir.

GBC, §9.2: sparse interactions. Una capa localmente conectada puede seguir usando pesos distintos en distintas posiciones; el sharing es una restricción adicional.

## Página 11 — El peso compartido une las conexiones locales

En cada posición, reutiliza las mismas variables $w_0, w_1, w_2$ y $b$.

$$
z_0 = w_0x_0 + w_1x_1 + w_2x_2 + b, \quad z_1 = w_0x_1 + w_1x_2 + w_2x_3 + b.
$$

| Tipo | Pesos | Sesgos | Total |
|---|---:|---:|---:|
| Densa | $4 \times 6 = 24$ | 4 | 28 |
| Local, no compartido ($K=3$) | $4 \times 3 = 12$ | 4 | 16 |
| Convolucional ($K=3$) | 3 | 1 | 4 |

El modelo asume que la misma regla local es útil en todas las posiciones.

La reutilización de un peso no crea nuevos parámetros.

GBC, §9.2: parameter sharing; Bishop & Bishop, Ch. 10. Los recuentos incluyen sesgos entrenables, incluso si se inicializan a cero.

## Página 12 — La convolución como matriz estructurada

Para nuestro ejemplo válido con stride 1:

$$
z = A(w)x + b\mathbf{1}_4,
$$

con

$$
A(w) =
\begin{bmatrix}
 w_0 & w_1 & w_2 & 0 & 0 & 0 \\
 0 & w_0 & w_1 & w_2 & 0 & 0 \\
 0 & 0 & w_0 & w_1 & w_2 & 0 \\
 0 & 0 & 0 & w_0 & w_1 & w_2
\end{bmatrix} \in \mathbb{R}^{4\times 6}.
$$

- Los ceros codifican conectividad local.
- Las diagonales constantes codifican pesos compartidos: una matriz de Toeplitz.
- Solo tres entradas de la matriz son entrenables de forma independiente.

La convolución es una familia restringida de transformaciones afines.

GBC, §9.1–9.2. Las librerías no necesitan construir explícitamente esta matriz. Entradas más largas producen más respuestas, con el mismo número de parámetros del núcleo pero más cálculo y activaciones.

## Página 13 — Desplaza la entrada, desplaza la respuesta

Una regla local compartida da equivarianza por traslación.

$$
C_w(T_\delta x) = T_\delta(C_w x).
$$

Con un stride 1 sobre secuencias definidas en todos los enteros, la respuesta se mueve con el patrón de entrada.

**Equivarianza:** la respuesta se desplaza junto con la entrada.

Los bordes y las elecciones de stride requieren precaución; ver Apéndice D.

GBC, §9.2. $(T_\delta x)[t] = x[t-\delta]$. Un sesgo compartido y una activación puntual preservan la identidad bajo las mismas suposiciones de dominio. La equivarianza no significa una salida invariable.

## Página 14 — El padding especifica los valores del borde

Con padding cero, se extiende la entrada fuera de su dominio observado:

$$
\bar{x}_i =
\begin{cases}
 x_i, & 0 \le i < L_{in}, \\
 0, & \text{en otro caso,}
\end{cases}
\quad
L_p = L_{in} + p_L + p_R.
$$

Para $p_L = p_R = 1$, el array rellenado es:

$$
(0, 1, 2, 0, -1, 3, 1, 0).
$$

- $p_L$ y $p_R$ especifican el número de posiciones añadidas a cada lado.
- Los ceros añadidos son valores fijos, no parámetros entrenables.
- `valid` significa sin padding: $p_L = p_R = 0$.

GBC, §9.5; PyTorch Conv1d. El padding es una convención de borde; esta sesión usa ceros.

## Página 15 — El stride especifica las posiciones de la ventana

El stride $s$ es la distancia entre inicios de ventanas consecutivos.

$$
z_t = b + \sum_{r=0}^{K-1} w_r \bar{x}_{ts-p_L+r}.
$$

Para $L_p = 8$, $K=3$:

| Stride $s$ | Inicios en el array padded | Número de ventanas |
|---|---|---:|
| 1 | $0,1,2,3,4,5$ | 6 |
| 2 | $0,2,4$ | 3 |

$t$ indexa la salida. $ts-p_L$ es el índice del primer valor original de entrada.

El stride selecciona qué ventanas locales se evalúan.

El núcleo sigue teniendo $K$ pesos adyacentes. La dilatación se fija a 1 en la sesión principal; se trata por separado en el Apéndice A.

## Página 16 — Deriva la longitud de salida

Sea $L_p = L_{in} + p_L + p_R$, y supongamos que $L_p \ge K$.

La ventana de salida $t$ empieza en $ts$ en el array padded. Encaja si:

$$
ts + K \le L_p \iff t \le \frac{L_p-K}{s}.
$$

El mayor índice entero admisible y el número de salidas son:

$$
 t_{max} = \left\lfloor \frac{L_p-K}{s} \right\rfloor,
\qquad
L_{out} = t_{max}+1 = \left\lfloor \frac{L_{in} + p_L + p_R - K}{s} \right\rfloor + 1.
$$

**Floor:** la última ventana debe encajar. **+1:** los índices de salida empiezan en cero.

Se deriva contando los inicios de ventana; dilatación 1, tamaño de kernel entero positivo y stride, padding no negativo. Una configuración sin ventana válida no es una salida Conv1D útil.

## Página 17 — Comprueba la fórmula con ventanas reales

$L_{in}=6$, $K=3$, $p_L=p_R=1$, $s=2$, $w=(1,2,-1)$, $b=0$.

$$
\bar{x}[n] = (0, 1, 2, 0, -1, 3, 1, 0).
$$

Los inicios válidos son $0,2,4$.

- Inicio 0: $z[0]=0$
- Inicio 2: $z[1]=3$
- Inicio 4: $z[2]=4$

$$
L_{out} = \left\lfloor \frac{6 + 1 + 1 - 3}{2} \right\rfloor + 1 = 3,
\quad z = (0,3,4).
$$

El último inicio padded admisible es 4. Un inicio en 6 necesitaría otra muestra.

Ejemplo original. Los inicios padded 0,2,4 corresponden a inicios originales $-1,1,3$; los tres productos escalares son $0+2-2=0$, $2+0+1=3$, $-1+6-1=4$.

## Página 18 — Diferente padding, diferente alineación

Mantén la misma señal y pesos, con $K=3$ y $s=1$.

| Padding $(p_L,p_R)$ | $L_{out}$ | Salida |
|---|---:|---|
| `valid`: $(0,0)$ | 4 | $(5, 3, -5, 4)$ |
| Zero padding: $(1,1)$ | 6 | $(0,5,3,-5,4,5)$ |

Para este kernel impar a stride 1, un cero a cada lado conserva la longitud de entrada.

- `same` es una convención de padding de la librería.
- Con stride 1, conserva la longitud; un kernel par puede requerir lados desiguales.
- Con strides mayores, comprueba la API y la alineación explícitamente.

Keras Conv1D; PyTorch Conv1d. En PyTorch, `padding="same"` solo admite `stride=1`. El Apéndice B compara longitudes de salida iguales con padding diferente.

## Página 19 — Predice la forma antes de ejecutar código

Longitud de entrada $L_{in}=6$. Todos los núcleos usan dilatación 1.

$$
L_{out} = \left\lfloor \frac{L_{in} + p_L + p_R - K}{s} \right\rfloor + 1.
$$

| Caso | Núcleo $K$ | Stride $s$ | Padding $(p_L,p_R)$ | $L_{out}$ |
|---|---:|---:|---|---:|
| A | 5 | 1 | $(0,0)$ |  |
| B | 3 | 2 | $(0,0)$ |  |
| C | 3 | 2 | $(1,1)$ |  |

Para cada caso, lista los inicios en el array padded antes de contarlos.

¿Qué decisiones arquitectónicas cambian el número de parámetros?

**Control oral breve.** Supón un canal de entrada, un canal de salida y un sesgo entrenable. Las respuestas y el razonamiento esperado están en las notas de enseñanza en español.

## Página 20 — Un filtro lee todos los canales de entrada

Un eje deslizante puede tener varios canales de entrada. Con `groups=1`:

$$
Z_{n,o,t} = b_o + \sum_{c=0}^{C_{in}-1}\sum_{r=0}^{K-1} W_{o,c,r}\,\bar{X}_{n,c,ts-p_L+r}.
$$

Ejemplo: un canal de entrada, dos canales.

$$
X_0 = \begin{bmatrix}
1 & 2 & 0 & -1 & 3 & 1 \\
0 & 1 & 2 & 1 & 0 & -1
\end{bmatrix},
\quad
W_0 = \begin{bmatrix}
1 & 2 & -1 \\
0 & 1 & 0
\end{bmatrix}.
$$

Con `valid`, stride 1:

$$
Z_{0,0,0} = (1+4+0) + (0+1+0) + 0.5 = 6.5.
$$

- $n$: índice del batch.
- $o$: canal de salida.
- $c$: canal de entrada.

Un filtro tiene $C_{in}K$ pesos y un sesgo compartido. PyTorch `Conv1d` y Keras `Conv1D`, configuración ordinaria con `groups=1`.

## Página 21 — Varios filtros producen varios canales

Mantén la misma entrada de dos canales. Usa dos filtros independientes:

$$
W_0 = \begin{bmatrix}1&2&-1\\0&1&0\end{bmatrix},
\quad
W_1 = \begin{bmatrix}0&1&0\\1&0&-1\end{bmatrix},
\quad b=(0.5,-1).
$$

Con $K=3$, padding válido y stride 1:

$$
Z_0 =
\begin{bmatrix}
6.5 & 5.5 & -3.5 & 4.5 \\
-1 & -1 & 0 & 4
\end{bmatrix}.
$$

- Cada filtro lee ambos canales de entrada y produce un canal de salida.
- Ambos canales de salida tienen la misma longitud, $L_{out}=4$.
- Estos valores son preactivaciones; cualquier ReLU sería un paso separado.

Ejemplo original reproducido en los programas NumPy y opcionales del entorno. Aquí $B=1$, $C_{in}=2$, $C_{out}=2$; el tensor completo tiene forma $(1,2,4)$ en el layout de PyTorch.

## Página 22 — Cuenta los parámetros entrenables

Un filtro: $C_{in}K$ pesos y un sesgo.

Para $C_{out}$ filtros:

$$
N_{params} = C_{out}(C_{in}K + 1).
$$

Para nuestro ejemplo de dos canales y dos filtros:

$$
2(2\cdot 3 + 1) = 14.
$$

```python
import torch
layer = torch.nn.Conv1d(
    in_channels=2, out_channels=2,
    kernel_size=3, bias=True)
```

```python
import keras
layer = keras.layers.Conv1D(
    filters=2, kernel_size=3,
    use_bias=True, activation=None)
```

El tamaño de batch y la longitud de secuencia no multiplican el recuento de parámetros.

Convolución ordinaria, `groups=1`; omite los sesgos de salida si están desactivados. Keras infiere $C_{in}$ cuando la capa se construye. Estos constructores definen la arquitectura; el Apéndice E asigna los pesos numéricos del ejemplo.

## Página 23 — Ejes del tensor: batch, canales y longitud

Declara el layout antes de pasar un tensor a una librería.

| Objeto | PyTorch | Keras / TensorFlow, canales al final |
|---|---|---|
| Entrada | $(B,C_{in},L_{in})$ | $(B,L_{in},C_{in})$ |
| Núcleo | $(C_{out},C_{in},K)$ | $(K,C_{in},C_{out})$ |
| Sesgo | $(C_{out},)$ | $(C_{out},)$ |
| Salida | $(B,C_{out},L_{out})$ | $(B,L_{out},C_{out})$ |

Ejemplo de salida: $(1,2,4)$ en PyTorch; $(1,4,2)$ con canales al final.

- Permuta los ejes de la entrada y el núcleo para convertir entre convenciones.
- Conserva el orden de los coeficientes a lo largo del núcleo.

PyTorch Conv1d; Keras Conv1D; Keras ops.conv; `tf.nn.conv1d`. La función de bajo nivel de TensorFlow devuelve la convolución; añade el sesgo por separado.

## Página 24 — Tu turno: especifica una capa Conv1D

Trabaja en parejas. Tienes cinco minutos antes de la solución.

$$
B=4, \quad C_{in}=2, \quad L_{in}=8, \quad C_{out}=3, \quad K=3, \quad s=2.
$$

Usa padding cero $p_L=p_R=1$, dilatación 1, `groups=1` y un sesgo entrenable.

1. Deriva $L_{out}$; muestra la operación floor.
2. Da la forma de salida en PyTorch y en Keras con canales al final.
3. Cuenta todos los pesos y sesgos entrenables.
4. Lista los índices originales de entrada usados por la primera y la última ventana.

**Explica:** Si el tamaño del batch se duplica, ¿qué respuestas cambian?

No se especifican pesos numéricos: la tarea es determinar la estructura del operador. Marca los índices fuera de la entrada original como ceros rellenados.

## Página 25 — Solución: ventanas, formas y parámetros

$$
L_{out} = \left\lfloor \frac{8+1+1-3}{2} \right\rfloor + 1 = \lfloor 3.5 \rfloor + 1 = 4.
$$

| Cantidad | Resultado |
|---|---|
| Salida de PyTorch | $(B,C_{out},L_{out})=(4,3,4)$ |
| Salida de Keras, canales al final | $(B,L_{out},C_{out})=(4,4,3)$ |
| Parámetros entrenables | $3(2\cdot 3 + 1)=18+3=21$ |

Índices originales de las cuatro ventanas:

$$
(-1,0,1), \quad (1,2,3), \quad (3,4,5), \quad (5,6,7).
$$

Duplicar $B$ cambia solo la primera dimensión de salida y la cantidad de trabajo. El núcleo, la longitud de salida y el recuento de parámetros se mantienen.

El índice $-1$ corresponde a un cero añadido. Un padding simétrico de 1 es explícito aquí; `Keras padding="same"` con stride 2 no sustituye esa alineación. Ver Apéndice G.

## Página 26 — Qué ya debes saber explicar

**Ticket de salida:** una frase por pregunta.

1. ¿La Conv1D de una librería invierte el núcleo numérico?
2. ¿Cómo restringen la localidad y el peso compartido un mapa afín?
3. ¿Qué ejes determinan la forma de salida y el recuento de parámetros entrenables?

**Tres comprobaciones antes de ejecutar una capa convolucional:**

- posiciones de la ventana,
- ejes del tensor,
- $C_{out}(C_{in}K + 1)$.

**Siguiente sesión, 19:20–20:20:** ampliar la operación local a dos dimensiones; después estudiar pooling y receptive fields.

**Fin del bloque de 90 minutos.** Las diapositivas 27–34 son apéndices de referencia opcionales; no añaden ocho secciones temporales más a esta clase.

## Página 27 — Dilatación: mayor alcance, mismos pesos

**Apéndice A.** La dilatación $d$ separa las posiciones muestreadas dentro de una ventana.

| $d=1$ | $d=2$ |
|---|---|
| $K_{eff}=3$ | $K_{eff}=5$ |

$$
z_t = b + \sum_{r=0}^{K-1} w_r \bar{x}_{ts-p_L+dr}, \quad K_{eff}=d(K-1)+1.
$$

$$
L_{out} = \left\lfloor \frac{L_{in}+p_L+p_R-K_{eff}}{s} \right\rfloor + 1.
$$

Para la señal de ejemplo, $K=3$, $d=2$, $s=1$, válido: $K_{eff}=5$, $L_{out}=2$, $z=(-2,-1)$.

El mismo núcleo $w=(1,2,-1)$ tiene tres coeficientes. La dilatación cambia las posiciones muestreadas.

**Extensión opcional;** la sesión cronometrada fija $d=1$. TensorFlow `nn.convolution`; Keras `Conv1D`: `strides>1` y `dilation_rate>1` son incompatibles en esta capa.

## Página 28 — Padding igual y alineación

**Apéndice B.** TensorFlow/Keras `SAME`, con dilatación 1:

$$
L_{out}=\left\lceil \frac{L_{in}}{s} \right\rceil, \quad P = \max((L_{out}-1)s + K - L_{in}, 0), \quad p_L = \lfloor P/2 \rfloor, \quad p_R = P-p_L.
$$

Para $L_{in}=6$, $K=3$, $s=2$, $w=(1,2,-1)$, $b=0$:

| Elección | $(p_L,p_R)$ | $L_{out}$ | Salida |
|---|---|---:|---|
| TensorFlow/Keras `SAME` | $(0,1)$ | 3 | $(5,-5,5)$ |
| Padding simétrico explícito | $(1,1)$ | 3 | $(0,3,4)$ |

- Un padding total impar coloca la posición extra a la derecha.
- `PyTorch padding="same"` solo admite stride 1.
- Un padding explícito seguido de `valid` hace que la alineación sea portable.

Longitudes de salida iguales pueden ocultar ventanas muestreadas distintas.

TensorFlow `nn.convolution`; notas oficiales de padding de TensorFlow; PyTorch Conv1d. Sustituye $K$ por $K_{eff}$ para la dilatación.

## Página 29 — Propagación hacia atrás a través de pesos compartidos

**Apéndice C.** Para $z_t = b + \sum_r w_r x_{t+r}$, sea $g_t = \partial J / \partial z_t$.

$$
\frac{\partial J}{\partial w_r} = \sum_t g_t x_{t+r},
\qquad
\frac{\partial J}{\partial b} = \sum_t g_t,
\qquad
\nabla_x J = A(w)^\top g.
$$

Usando $J = 12 \sum_t z_t^2 = 37.5$, se tiene $g = z = (5,3,-5,4)$.

| Posición | Contribución a $\nabla_w J$ |
|---|---|
| 0 | $(5,10,0)$ |
| 1 | $(6,0,-3)$ |
| 2 | $(0,5,-15)$ |
| 3 | $(-4,12,4)$ |

$$
\nabla_w J = (7,27,-14),\qquad \frac{\partial J}{\partial b} = 7.
$$

$$
\nabla_x J = (5,13,-4,-9,13,-4).
$$

La transpuesta propaga un gradiente. Es una adjunta, no generalmente una inversa.

Diferencias finitas centrales con $\varepsilon = 10^{-6}$ verifican estas derivadas.

Conexión con la Sesión 3. Verificación NumPy original en `session7_numeric.py`; este objetivo cuadrático ilustrativo está elegido por su aritmética transparente. El peso compartido recibe una suma, no una actualización independiente por posición.

## Página 30 — Equivarianza: supuestos y demostración

**Apéndice D.** Sea $x$ definido sobre $\mathbb{Z}$ y defínase

$$
(T_\delta x)[t] = x[t-\delta], \qquad (C_w x)[t] = \sum_{r=0}^{K-1} w_r x[t+r].
$$

A stride 1:

$$
C_w(T_\delta x)[t] = \sum_r w_r x[t+r-\delta] = (T_\delta (C_w x))[t].
$$

- Un sesgo compartido y una activación puntual preservan esta igualdad.
- Recorte finito y padding de borde pueden cambiar las muestras disponibles.
- Con stride $s$, los desplazamientos alineados satisfacen $C_{w,s}T_{sm}=T_m C_{w,s}$.

Un desplazamiento de una muestra no necesita corresponder a un desplazamiento entero de salida cuando $s>1$.

GBC, §9.2. El programa numérico suministrado incluye una comprobación exacta del interior y un contraejemplo finito de borde. Los supuestos del dominio y de la rejilla de muestreo forman parte del enunciado.

## Página 31 — Un operador en PyTorch y Keras

**Apéndice E.** Usa los arrays fijos del programa suministrado:

```python
from session7_frameworks import X_MAIN, W_MAIN, B_MAIN
X, W, b = X_MAIN, W_MAIN, B_MAIN  # NumPy float64: BCL, OIK, O
```

**PyTorch: entrada BCL**

```python
import torch
conv_pt = torch.nn.Conv1d(2, 2, 3, bias=True).double()
with torch.no_grad():
    conv_pt.weight.copy_(torch.from_numpy(W))
    conv_pt.bias.copy_(torch.from_numpy(b))
z_pt = conv_pt(torch.from_numpy(X))
```

**Keras con TensorFlow: entrada BLC**

```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
X_k = X.transpose(0, 2, 1)
conv_kr = keras.layers.Conv1D(
    2, 3, activation=None, dtype="float64",
    data_format="channels_last")
conv_kr.build(X_k.shape)
conv_kr.set_weights([W.transpose(2, 1, 0), b])
z_kr = conv_kr(X_k)
```

Tras convertir los ejes de salida, ambos implementan la misma matriz numérica de la diapositiva 21.

Ejecuta cada fragmento en su entorno correspondiente. Establece `KERAS_BACKEND` antes de importar Keras en un proceso nuevo. Comprobaciones completas: `session7_frameworks.py`. Convención del núcleo de Keras; PyTorch Conv1d.

## Página 32 — Opcional: un operador 1D causal

**Apéndice F.** En una tarea temporal, una salida en $t$ puede necesitar usar solo entradas de tiempos iguales o anteriores a $t$.

Mantén nuestra convención de correlación cruzada y stride 1. Elige $p_L = d(K-1)$, $p_R = 0$. Entonces

$$
z_t = b + \sum_{r=0}^{K-1} w_r \bar{x}_{t-d(K-1)+dr}.
$$

El mayor índice de entrada es:

$$
t - d(K-1) + d(K-1) = t.
$$

Para $d=1$, $K=3$, se usa $(\bar{x}_{t-2}, \bar{x}_{t-1}, x_t)$ con pesos $(w_0,w_1,w_2)$.

- Keras ofrece `Conv1D(..., padding="causal")`.
- En PyTorch, rellena explícitamente a la izquierda y usa `padding=0`.
- Escribir $\sum_r w_r x[t-r]$ invierte el orden de los coeficientes.

La fórmula de padding sigue del cálculo de soporte anterior. Keras Conv1D documenta la propiedad de no usar entradas futuras. Enlace opcional a modelado de secuencias; no se introduce una hipótesis de Markov aquí.

## Página 33 — Padding explícito y concordancia numérica

**Apéndice G.** Reutiliza `X`, `W`, `b`, `Xk` del Apéndice E. Elige $p_L = p_R = 1$, $s=2$.

**PyTorch: pad al último eje**

```python
import torch.nn.functional as F
x_pad = F.pad(torch.from_numpy(X), (1, 1))
z_pt = F.conv1d(x_pad, torch.from_numpy(W), torch.from_numpy(b), stride=2, padding=0)
```

**Keras: pad el eje de la secuencia**

```python
pad = keras.layers.ZeroPadding1D((1, 1), dtype="float64", data_format="channels_last")
x_pad = pad(X_k)
conv = keras.layers.Conv1D(2, 3, strides=2, padding="valid", dtype="float64", data_format="channels_last")
conv.build(x_pad.shape)
conv.set_weights([W.transpose(2, 1, 0), b])
z_kr = conv(x_pad)
```

Usa una tolerancia para comparar `z_pt.detach().numpy()` con `z_kr.numpy().transpose(0,2,1)`.

Para reproducir `SAME` para este caso $L=6$, $K=3$, $s=2$, usa $(0,1)$ en su lugar.

PyTorch `F.pad`; Keras `ZeroPadding1D`. `tf.nn.conv1d` usa un núcleo $(K,C_{in},C_{out})$ y necesita una suma de sesgo separada; una comprobación completa de TensorFlow está en el programa.

## Página 34 — Lecturas y material reproducible

**Apéndice H.** Lecturas que coinciden con esta sesión.

- C. M. Bishop and H. Bishop (2024). Deep Learning: Foundations and Concepts. Ch. 10: Convolutional Networks.
- I. Goodfellow, Y. Bengio and A. Courville (2016). Deep Learning. Ch. 9, §9.1–9.2 and §9.5: the operation, motivation and variations.

**Referencias oficiales consultadas 2026-10-18**

- PyTorch `Conv1d` y formas de tensores
- Keras `Conv1D` y padding
- TensorFlow `tf.nn.conv1d`
- Keras `ops.conv` y ejes del núcleo
- NumPy `correlate`
- NumPy `convolve`

**En el ZIP del material**

```bash
python session7_numeric.py --output numeric_results.json
python session7_frameworks.py --backend numpy
# Entornos opcionales: --backend torch o --backend tensorflow
```

Las dos fuentes TEX compilan sin los programas Python. Se ejecutaron las comprobaciones de aritmética NumPy y diferencias finitas; las ramas opcionales de frameworks se revisaron frente a las API oficiales y se comprobó su sintaxis, pero no se ejecutaron en el entorno de redacción.

## Cierre

El apunte cubre las 34 páginas del PDF, desde la portada hasta los apéndices de referencia, con la traducción al español de la sesión sobre patrones locales en señales 1D, convolución, padding, stride, canales y pesos compartidos.
