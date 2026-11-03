---
titulo: "Introducción a CUDA, arquitectura de GPU y programación paralela"
asignatura: sistemas-graficos-y-multimedia
fecha: 2026-10-20
orden: 1
tema: "Tema 1 - Sistemas gráficos y multimedia: GPU y CUDA"
tags: ["CUDA", "GPU", "GPGPU", "arquitectura", "memoria", "paralelismo", "programación"]
resumen: "Introducción a la computación heterogénea, la arquitectura de GPUs y CUDA, desde la jerarquía de memoria y los multiprocesadores hasta el modelo de programación, el stencil y el producto de matrices."
---

# Sistemas Gráficos y Multimedia

## Sesión 1 — Introducción a CUDA y arquitectura de GPU

### Datos de la sesión

- Profesor: Manuel Ujaldón
- Curso académico: 2026/27
- Máster: Ingeniería Informática
- Enfoque: GPU, GPGPU, CUDA, computación heterogénea, paralelismo masivo

## Contenidos de la sesión

1. Introducción
2. Arquitectura de la GPU y del modelo hardware de CUDA
3. Evolución de generaciones: Pascal, Volta, Turing, Ampere, Hopper
4. Programación CUDA
5. Sintaxis y gestión de memoria
6. Compilación
7. Ejemplos: VectorAdd, stencil y producto de matrices
8. Los diez mandamientos del buen cudista
9. Bibliografía y recursos

## Prerrequisitos

La sesión parte de una base mínima:

- Familiaridad con el lenguaje C
- No es necesario saber programación paralela de antemano
- No es necesario tener experiencia previa en arquitectura GPU
- No hace falta manejar shaders, vértices ni texturas; CUDA hace transparente esas capas de renderizado

El objetivo es entender la GPU como un coprocesador de propósito general.

## I. Introducción

### 1. Bienvenido al mundo de las GPUs

La GPU ha pasado de ser un acelerador exclusivamente gráfico a una unidad de cálculo de alto rendimiento, con un papel central en:

- Deep Learning
- HPC
- análisis de datos
- renderizado por ray-tracing
- inteligencia artificial
- vehículos autónomos
- robótica
- análisis de video y señales en streaming

Se menciona el ecosistema de CUDA y de la GPU como un conjunto de herramientas, librerías y plataformas que convierten la potencia hardware en capacidad de cálculo general.

### 2. Modelos comerciales: GeForce y Tesla

Se distinguen dos familias:

- GeForce: orientada a usuarios domésticos y gaming
  - prioridad al precio y disponibilidad
  - memoria de vídeo menor
  - buena base para desarrollo que luego se replica entorno HPC
- Tesla: orientada a HPC y computación científica
  - fiabilidad, trabajo 24/7, clusters
  - más memoria de vídeo
  - GPUDirect, soporte para clusters y alta disponibilidad

El objetivo de la clase es enfatizar que la GPU no es solo una tarjeta de juegos; es un procesador masivo muy útil para el cálculo científico y para aceleración de software.

### 3. Los tres rasgos que hicieron a la GPU un procesador único

La GPU destaca por tres cualidades:

- Simplicidad: el control de un hilo se amortiza con muchos otros; el modelo trabaja con warps de 32 hilos.
- Escalabilidad: la GPU maneja enormes cantidades de datos con una estrategia paralela sostenible.
- Productividad: cuando un hilo debe esperar memoria o datos, otro hilo ocupa su lugar; hay ocultación de latencia y conmutación de contexto casi gratis.

Palabras clave esenciales:

- Warp
- SIMT
- ocultación de latencia
- conmutación de contexto gratuita

### 4. ¿Qué es CUDA?

CUDA significa Compute Unified Device Architecture.

Es una plataforma software-hardware diseñada para explotar la potencia de la GPU en aplicaciones de propósito general.

Se compone de:

- Software: extensiones mínimas sobre C para operar con la GPU
- Firmware y drivers: acceso a memoria, dispositivos y ejecución
- Hardware: multiprocesadores con cientos o miles de núcleos

La idea central es que el programador escribe código como si trabajara con un único hilo, y el sistema instancia ese código sobre muchos hilos del hardware.

## II. Lo esencial de CUDA C

CUDA no es un lenguaje completamente nuevo: es C con extensiones mínimas.

Elementos clave:

- `__global__` indica una función que se ejecuta en la GPU y se invoca desde la CPU
- `threadIdx`, `blockIdx`, `blockDim`, `gridDim` son variables intrínsecas del modelo
- `__shared__` define memoria compartida dentro de un bloque
- `__syncthreads()` sincroniza hilos de un bloque

El objetivo es:

- construir código escalable a miles de núcleos
- permitir computación heterogénea entre CPU y GPU
- definir un modelo de memoria más explícito

## III. Computación heterogénea

### 1. Host vs device

Se distinguen dos niveles:

- Host: CPU + memoria principal (DDR4)
- Device: GPU + memoria de vídeo (GDDR5/HBM)

La CPU se encarga de la parte secuencial, la toma de decisiones, el control del flujo y la gestión de la ejecución. La GPU se usa como coprocesador, ideal para datos estructurados y patrón de paralelismo masivo.

### 2. Flujo de procesamiento típico

Un flujo típico es:

1. Copiar los datos de entrada desde la CPU a la GPU
2. Cargar y ejecutar el programa en la GPU
3. Ejecutar el kernel sobre miles de hilos
4. Transferir los resultados desde GPU a CPU

Se señala que el código reescrito en CUDA puede ser muy pequeño (inferior al 5% del código), pero si no se migra a la GPU, puede consumir más del 50% del tiempo total de la aplicación.

### 3. Ejemplo de “Hola mundo” en CUDA

```cpp
__global__ void mikernel(void)
{
    printf("Hola mundo desde la GPU\n");
}

int main(void)
{
    mikernel<<<1, 1>>>();
    return 0;
}
```

Los símbolos `<<< ... >>>` son la forma de lanzar un kernel desde CPU al device. El número de bloques y hilos se indica en ese punto:

```cpp
mikernel<<<1000, 100>>>();
```

Cada kernel se ejecuta sobre una malla de bloques y hilos.

## IV. Arquitectura del hardware de CUDA

### 1. Modelo hardware

La GPU está formada por varios multiprocesadores. Cada multiprocesador tiene:

- varios cores o procesadores streaming
- registros
- memoria compartida
- cachés de instrucciones y de datos
- unidades de control

La CPU es mejor en bifurcaciones, control de flujo y tareas irregulares. La GPU es mejor en procesamiento masivo sobre datos, con paralelismo SIMD/SIMT.

### 2. Jerarquía de memoria

La jerarquía es esencial en CUDA:

- registros: acceso muy rápido, locales a cada hilo
- memoria compartida: compartida por los hilos de un bloque, de rapidez alta
- caché L1 / L2
- memoria global: VRAM / memoria de vídeo

Se destaca la diferencia brutal entre memoria compartida y memoria global:

- memoria compartida: de tipo SRAM, muy rápida
- memoria global: DRAM, mucho más lenta

Esto implica que un buen diseño de kernels debe minimizar los accesos a memoria global y maximizar la reutilización mediante memoria compartida y registros.

## V. Evolución de generaciones de GPU

La sesión repasa la evolución hacia arquitecturas más potentes:

### 1. Pascal (2016-2017)

- incremento sustancial en rendimiento
- uso de HBM2 / memoria 3D
- GPU para HPC y aprendizaje profundo
- arquitectura con multiprocesadores muy densos

### 2. Volta (GV100)

- mayor rendimiento y capacidad para entrenamiento de redes neuronales
- aumento en caché L1 y memoria compartida por SM
- soporte mejorado para computación de alto rendimiento
- introducción de tensor cores (primeras versiones más relevantes en Turing/Ampere)

### 3. Turing (TUxxx)

- rendimiento de la familia RTX
- diseño orientado a gráficos, IA y cómputo científico
- soporte a precisiones INT8 e INT4 en los tensor cores

### 4. Ampere (Axxx)

- GA100 con 128 SMs
- HBM2 con mayor ancho de banda
- aumento enorme de rendimiento y memoria para IA
- A100 se convierte en referencia para entrenamientos de deep learning

### 5. Hopper (Hxxx)

- GH100 con 144 SMs y HBM3
- ancho de banda mucho más alto
- soporte de IA y computación infraestructura de alto rendimiento
- más memoria y mejor escalabilidad

### 6. Síntesis generacional

Se observa una tendencia clara:

- más núcleos por SM
- más memoria por dispositivo
- más ancho de banda
- más capacidad para computación de IA y HPC
- mayor rendimiento por vatio y por operación

La tabla comparativa de la sesión muestra un crecimiento enorme desde Fermi, Kepler, Maxwell, Pascal, Volta, Turing, Ampere y Hopper.

## VI. Programación CUDA

### 1. Modelo de programación

CUDA ofrece una visión del device como coprocesador altamente paralelo:

- La GPU tiene su propia DRAM
- Los hilos eran extremadamente ligeros
- La conmutación de contexto es muy barata
- Se crean muchos hilos para explotar el rendimiento

La malla o grid es un conjunto de bloques. Cada bloque contiene varios hilos. Los bloques se organizan en una estructura 1D, 2D o 3D. Cada hilo obtiene un identificador para operar sobre un subconjunto del problema.

### 2. Conceptos clave

- Dispositivo: GPU
- Multiprocesador: conjunto de cores y memoria compartida
- Kernel: función ejecutada en la GPU
- Malla: conjunto de bloques que ejecuta un kernel
- Bloque: conjunto de hilos que cooperan
- Warp: conjunto de 32 hilos gestionado por el hardware

La ejecución es escalable: el mismo kernel puede ejecutarse con cualquier número de cores sin recompilar.

### 3. Relación hardware-software y memoria

Se enfatiza la relación entre:

- threads
- bloques
- multiprocesadores
- memoria compartida
- memoria global

El problema real del rendimiento en CUDA no es solo la cantidad de cores, sino el patrón de acceso a memoria. Un kernel muy bien paralelizado pero con mucho tráfico a memoria global puede quedarse limitado por el ancho de banda.

## VII. Sintaxis de CUDA

### 1. Un ejemplo preliminar: SAXPY

```cpp
__global__ void saxpy_paralelo(int n, float a, float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

int numero_de_bloques = (n + 255) / 256;
saxpy_paralelo<<<numero_de_bloques, 256>>>(n, 2.0, x, y);
```

La idea es clara: cada hilo escribe un elemento del vector de salida.

### 2. Modificadores y variables intrínsecas

- `__global__` para funciones lanzadas desde CPU
- `__device__` para funciones solo ejecutadas en GPU
- `__shared__` para memoria compartida
- `__constant__` para memoria constante
- `threadIdx`, `blockIdx`, `blockDim`, `gridDim`
- `__syncthreads()` para sincronizar hilos de un bloque

### 3. Gestión de memoria de vídeo

```cpp
cudaMalloc(&d_A, numBytes);
cudaFree(d_A);
cudaMemcpy(d_A, h_A, numBytes, cudaMemcpyHostToDevice);
cudaMemcpy(h_A, d_A, numBytes, cudaMemcpyDeviceToHost);
```

La convención habitual es usar prefijos como:

- `h_` para memoria host / principal
- `d_` para memoria device / GPU

### 4. Transferencias asíncronas

Se menciona `cudaMemcpyAsync()`, que permite solapar comunicación y computación, con retorno inmediato al host y mejor aprovechamiento del bus PCIe y de los streams.

## VIII. Ejemplo 1: incremento de un vector

El primer ejemplo funcional es sumar un valor `b` a cada elemento de un vector.

```cpp
void incremento_en_cpu(float *a, float b, int N)
{
    for (int idx = 0; idx < N; ++idx)
        a[idx] = a[idx] + b;
}
```

La versión CUDA hace que cada hilo procese un elemento concreto:

```cpp
__global__ void incremento_en_gpu(float *a, float b, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        a[idx] = a[idx] + b;
}
```

Con `N = 16` y `blockDim = 4`, la malla se configura con cuatro bloques y cada bloque tiene cuatro hilos. Cada hilo procesa un índice del vector.

### ¿Qué aporta esto?

- paralelismo de grano fino
- cada hilo trabaja sobre un dato distinto
- el patrón es perfectamente regular

## IX. Ejemplo 2: stencil 1D

Un stencil es un patrón de vecinos; el valor de salida depende del valor de entrada de elementos cercanos.

### Idea

Cada valor de salida requiere datos del entorno inmediato. En una versión simple, el resultado depende del valor actual y de vecinos en ambos lados.

```cpp
__global__ void stencil_1d(int *d_in, int *d_out)
{
    __shared__ int temp[BLOCKSIZE + 2 * RADIO];
    int gindex = blockIdx.x * blockDim.x + threadIdx.x;
    int lindex = threadIdx.x + RADIO;

    temp[lindex] = d_in[gindex];
    if (threadIdx.x < RADIO) {
        temp[lindex - RADIO] = d_in[gindex - RADIO];
        temp[lindex + blockDim.x] = d_in[gindex + blockDim.x];
    }

    __syncthreads();

    int result = 0;
    for (int offset = -RADIO; offset <= RADIO; offset++)
        result += temp[lindex + offset];

    d_out[gindex] = result;
}
```

Las claves del enfoque son:

- mover los datos a memoria compartida
- sincronizar con `__syncthreads()`
- reutilizar elementos del vector para reducir acceso a memoria global
- evitar conflictos de memoria o accesos no alineados

### Optimización

La sesión muestra una mejora del algoritmo original al cargar los datos con una estrategia mejorada y evitar algunos accesos redundantes. En general, la regla es: si el patrón requiere vecinos, la memoria compartida reduce el coste de los accesos globales y permite una ejecución más eficiente.

## X. Ejemplo 3: producto de matrices

El producto de matrices es uno de los ejemplos clásicos del paralelismo en GPU.

### Versión secuencial

```cpp
for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
        float sum = 0;
        for (int k = 0; k < N; ++k)
            sum += A[i*N + k] * B[k*N + j];
        C[i*N + j] = sum;
    }
```

### Versión CUDA inicial

Cada hilo calcula un elemento de la matriz `C` y se aprovecha la regularidad del problema.

```cpp
__global__ void MxMonGPU(float *A, float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0.0f;

    for (int k = 0; k < N; ++k)
        sum += A[i * N + k] * B[k * N + j];

    C[i * N + j] = sum;
}
```

### Problema principal

Este enfoque requiere mucho ancho de banda a memoria global. En una matriz de tamaño grande, la memoria se convierte en el cuello de botella.

### Mejor solución: tiling (mosaicos)

Se divide la matriz en bloques de 32x32 y se llevan a memoria compartida para reutilizar datos. La idea:

- cada bloque de hilos calcula una submatriz `Csub`
- cada bloque carga un mosaico de `A` y otro de `B`
- el acceso a memoria compartida es mucho más barato que el acceso a memoria global
- `__syncthreads()` garantiza que todas las cargas de cada mosaico se completan antes de la fase de computación

La sesión también menciona la necesidad de evitar conflictos en bancos de memoria compartida, y recomienda transponer los datos al almacenarlos para acceder sin colisiones.

### Unrolling

Además, se comenta la optimización del compilador mediante *loop unrolling*:

- se desenrollan los bucles de multiplicación de mosaicos
- se reduce el overhead de control de flujo
- se mejora el rendimiento

## XI. Los diez mandamientos del buen cudista

La sesión culmina con una lista de principios prácticos para optimizar kernels CUDA:

### 1. El volumen de datos

Si el problema es demasiado pequeño, no merece la pena moverlo a la GPU. El umbral mínimo para aspirar a aceleración es del orden de 10 millones de datos.

### 2. Aprovecha la ubicuidad de CUDA

Busca librerías, ejemplos o soluciones previas en repositorios. Muchas veces hay implementaciones de referencia que resuelven el problema antes de reinventar la rueda.

### 3. Detectar dependencias de datos

Las dependencias RAW o carry dependencies pueden destruir el paralelismo. Si hay dependencia entre iteraciones, conviene factorizar el problema o reestructurar el algoritmo.

### 4. Aplicar el paralelismo adecuado

Hay que escoger bien entre paralelismo de datos y de tareas, y lanzar suficientes hilos como para aprovechar la GPU. En muchos casos la estrategia adecuada es asociar una iteración por dato o usar tiling.

### 5. Owner’s computer rule

Cuando el problema puede paralelizarse como un `forall`, cada hilo debe ser responsable del dato que produce: “cada iteración la computa el hilo propietario del dato resultante”.

### 6. Delimitar transferencias y sincronizaciones

No es rentable mover demasiados datos entre CPU y GPU. También hay que controlar bien la sincronización, especialmente si se usa memoria unificada o streams.

### 7. Analizar si el kernel es memory-bound o compute-bound

El modelo Roofline ayuda a decidir si tenemos un problema limitado por memoria o por cálculo.

- Si el kernel es memory-bound, conviene fomentar la reutilización de datos con memoria compartida y registros.
- Si es compute-bound, conviene maximizar el uso de unidades especializadas.

### 8. La fuerza bruta es caballo ganador

Cuando se puede, es mejor favorecer un patrón regular y predecible, evitar ramificaciones complejas y manejar condiciones con padding o estructura uniforme.

### 9. Solapar computación y comunicación

Se debe aprovechar `cudaMemcpyAsync()` y la gestión de streams para superponer transferencia y cálculo, especialmente en kernels memory-bound.

### 10. Elegir la mejor configuración de lanzamiento

La elección de bloques por malla y hilos por bloque es crítica. Se suele empezar con una configuración de referencia y luego ajustar: 64, 128, 256, 512 o 1024 hilos por bloque.

## XII. Bibliografía y herramientas

La sesión recomienda:

- CUDA Zone: documentación y recursos principales
- CUDA C Programming Guide
- CUDA C Best Practices Guide
- documentación y herramientas del SDK de NVIDIA

Además, destaca la idea de que la práctica y la intuición son esenciales: muchas mejoras CUDA no vienen de un teorema, sino de la experiencia y del ajuste empírico a los patrones de ejecución de la GPU.

## Resumen final

La sesión presenta la GPU como un coprocesador altamente paralelo, la arquitectura CUDA como elemento de software y hardware integrados, y la programación paralela como un modelo basado en grids, blocks, threads y jerarquías de memoria. Los conceptos esenciales son:

- CPU y GPU cooperan en computación heterogénea
- la GPU se beneficia del paralelismo masivo
- la memoria compartida y la sincronización son pieza clave
- los kernels deben ser regulares y diseñados para reducir accesos a memoria global
- la optimización real pasa por patrones bien elegidos, no por intentar “forzar” la GPU sin un diseño claro

## Conclusión

Esta primera sesión sienta las bases para entender cómo CUDA transforma el problema de cálculo en un modelo de paralelismo masivo, y cómo los principios de arquitectura, memoria y sincronización condicionan tanto el diseño como el rendimiento de cualquier programa que se ejecute en GPU.
