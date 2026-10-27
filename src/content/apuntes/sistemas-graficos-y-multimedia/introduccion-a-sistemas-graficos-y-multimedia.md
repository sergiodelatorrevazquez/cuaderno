---
titulo: "Introducción a Sistemas Gráficos y Multimedia"
asignatura: sistemas-graficos-y-multimedia
fecha: 2026-10-20
tema: "Semana 1 - Sesión 1"
tags: [cuda, gpu, sistemas-gráficos, arquitectura-gpu, nvidia, simt, pascal, volta, turing, ampere, hopper]
resumen: "Introducción a los Sistemas Gráficos y Multimedia, arquitecturas GPU NVIDIA, modelo CUDA, jerarquía de ejecución, memoria y evolución generacional desde Pascal hasta Hopper."
borrador: false
---

## I. Introducción

Los **Sistemas Gráficos y Multimedia** abordan la generación, procesamiento, visualización y aceleración de contenido gráfico e interactivo. El curso se centra en la computación paralela con GPUs, con especial énfasis en **CUDA** sobre arquitecturas NVIDIA.

### Prerrequisitos

- Familiaridad con el lenguaje **C**
- No se requiere experiencia previa en programación paralela
- No se requieren conocimientos sobre arquitectura GPU (se introducen desde cero)
- No es necesario tener experiencia en programación gráfica (CUDA es transparente a vértices, píxeles o texturas)

### ¿Por qué usar GPUs para cómputo general?

Las GPUs se han convertido en procesadores fundamentales para **HPC**, **IA/Deep Learning**, **análisis de datos**, **gráficos en tiempo real**, **vehículos autónomos**, **robótica**, **sanidad** y mucho más.

Las tres cualidades que hacen única a la GPU:

1. **Simplicidad**: El control de un hilo se amortiza sobre otros 31 (tamaño de *warp* = 32)
2. **Escalabilidad**: Aprovechando el gran volumen de datos, el modelo de paralelización es sostenible
3. **Productividad**: Cuando un hilo no puede ejecutarse rápidamente, otro toma inmediatamente el procesador ocultando la latencia (*context switching* gratuito)

Las tres cualidades que han atraído a más usuarios:

- **Coste**: Muy favorable por el gran volumen de ventas (se venden ~3 GPUs por cada CPU)
- **Ubicuidad**: Presente en casi cualquier sistema y fácilmente adquirible
- **Consumo**: Alta eficiencia energética. La relación GFLOPS/W ha mejorado drásticamente: Fermi (~5-6 GFLOPS/W FP32), Kepler (~15-17), Maxwell (~40), llegando a valores mucho mayores en generaciones posteriores

## II. ¿Qué es CUDA?

**CUDA** (*Compute Unified Device Architecture*) es una plataforma diseñada conjuntamente a nivel hardware y software para aprovechar la potencia de la GPU en aplicaciones de propósito general (*GPGPU*).

Se articula en tres niveles:

- **Software**: Extensiones mínimas pero potentes al lenguaje C para lograr una ejecución eficiente y escalable
- **Firmware/Driver**: Proporciona las APIs para gestionar dispositivos, memoria, transferencias, etc.
- **Hardware**: Habilita el paralelismo masivo mediante **Multiprocesadores en Streaming** (*Streaming Multiprocessors*, SMs) con jerarquía de memoria optimizada

### CUDA C: Lo esencial

- Es esencialmente **C con extensiones CUDA**
- El programador escribe código para **un solo hilo** (*thread*); el runtime instancia automáticamente ese código sobre miles/millones de hilos
- Se distingue entre código **host** (CPU) y **device** (GPU)
- Funciones con `__global__` se ejecutan en GPU y son llamadas desde CPU
- Funciones con `__device__` se ejecutan y llaman desde GPU
- Funciones con `__host__` se ejecutan en CPU

## III. Modelo de ejecución: Jerarquía de hilos

CUDA sigue un modelo **SIMT** (*Single Instruction, Multiple Threads*). La organización jerárquica es clave para la escalabilidad:

| Nivel | Descripción | Notas |
| --- | --- | --- |
| **Thread** | Unidad mínima de ejecución | Ejecuta el mismo kernel con índices distintos |
| **Warp** | Grupo de 32 hilos | Unidad de ejecución física. Todos los hilos de un warp ejecutan la misma instrucción simultáneamente (o quedan inactivos por divergencia) |
| **Block (Thread Block)** | Grupo de hilos (1D/2D/3D) | Pueden cooperar vía **memoria compartida** y sincronizarse con `__syncthreads()`. Tamaño limitado por recursos del SM |
| **Grid** | Conjunto de bloques (1D/2D/3D) | Ejecuta todo el kernel. Escalable a múltiples SMs |

**Palabras clave**: *Warp*, *SIMT*, *ocultación de latencia*, *conmutación de contexto gratuita*.

### Índices de hilos

Para identificar cada hilo se utilizan variables intrínsecas:

```c
blockIdx.x, blockIdx.y, blockIdx.z  // Índice del bloque en el grid
threadIdx.x, threadIdx.y, threadIdx.z // Índice del hilo en el bloque
blockDim.x, blockDim.y, blockDim.z     // Dimensiones del bloque
gridDim.x, gridDim.y, gridDim.z       // Dimensiones del grid
```

El índice global típico para 1D es:

```c
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

## IV. Arquitectura GPU NVIDIA: Modelo hardware CUDA

La GPU se organiza en torno a **Streaming Multiprocessors (SMs)**. Cada SM contiene:

- **CUDA Cores**: Unidades aritmético-lógicas para punto flotante e entero
- **Warp Schedulers**: Programadores que seleccionan warps listos para ejecutar
- **Registro file**: Gran banco de registros compartido por los hilos del SM
- **Memoria compartida / L1 cache**: Configurable según carga de trabajo
- **Tensor Cores** (en generaciones recientes): Aceleración para multiplicación-acumulación de matrices
- **RT Cores** (Turing+): Aceleración para *ray tracing*
- **Load/Store Units**: Acceso a memoria

La ejecución se basa en **conmutación de warps**: cuando un warp se detiene esperando memoria, el scheduler cambia inmediatamente a otro warp listo, **ocultando la latencia** sin coste de cambio de contexto.

## V. Evolución generacional NVIDIA

### 1) Pascal (2016-2017) - 5ª generación

Arquitecturas: **GP100**, **GP104**

- Soporte para **NVLink** (en versión datacenter)
- Mejora importante en eficiencia energética
- Soporte nativo para **FP16**
- Mejor rendimiento en HPC y cómputo general
- Base sólida para la adopción masiva de GPGPU

### 2) Volta (2018-2020) - 6ª generación

Arquitectura: **GV100**

- **Tensor Cores** introducidos (aceleración masiva para GEMM)
- Arquitectura optimizada para entrenamiento/inferencia de redes neuronales
- Mejor utilización de recursos y mayor rendimiento por vatio
- Importante salto para IA y Deep Learning

### 3) Turing (2019-2020)

Arquitecturas: **TU102/TU104/TU106**

- **RT Cores** para *ray tracing* en tiempo real
- **Tensor Cores** de 2ª generación
- Orientado a estaciones de trabajo, creación de contenido y gaming profesional
- Soporte para técnicas gráficas modernas
- Mejoras en inferencia con precisión mixta

### 4) Ampere (2020-2022) - 7ª generación

Arquitecturas: **GA100** (datacenter), **GA102** (consumo/profesional)

- **Tensor Cores** de 3ª generación
- Soporte para **TF32**, **BF16**, **FP64** con mayor eficiencia
- Mayor rendimiento por vatio
- Mejor escalado con **NVLink**
- Muy fuerte en HPC, IA y simulaciones

### 5) Hopper (2022-presente) - 8ª generación

Arquitectura: **GH100**

- **Tensor Cores** de 4ª generación
- **Transformer Engine** optimizado para modelos *transformer*
- Soporte nativo para **FP8**
- Mayor ancho de banda y conectividad
- Optimizado para **LLMs** e IA generativa
- Grandes mejoras en escalabilidad para clusters

### Síntesis generacional

| Generación | Énfasis | Novedad clave |
| --- | --- | --- |
| **Pascal** | Eficiencia / GPGPU general | FP16 nativo, NVLink |
| **Volta** | IA/Deep Learning | Tensor Cores |
| **Turing** | Gráficos + IA | RT Cores + Tensor Cores 2ª gen. |
| **Ampere** | HPC + IA escalable | TF32/BF16, mejor NVLink |
| **Hopper** | IA generativa / Supercomputación | Transformer Engine, FP8 |

## VI. Jerarquía de memoria CUDA

La elección correcta de memoria es crítica para obtener buen rendimiento. El acceso coalescente y el uso de memoria compartida son fundamentales.

| Memoria | Alcance | Visibilidad | Latencia | Notas |
| --- | --- | --- | --- | --- |
| **Registros** | Por hilo | Privada | Mínima | Máxima prioridad. Limitados por SM |
| **Memoria compartida (*shared*)** | Por bloque | Todos los hilos del bloque | Muy baja | Ideal para reutilización de datos. Sincronizable |
| **Memoria local (*local*)** | Por hilo | Privada | Alta (va a DRAM) | Usada por desbordamiento de registros o arrays grandes |
| **Memoria constante (*constant*)** | Global | Todos los hilos, solo lectura | Baja (cacheada) | Óptima para datos uniformes/accesos broadcast |
| **Memoria de textura (*texture*)** | Global | Todos los hilos, solo lectura | Baja (cacheada) | Optimizada para acceso 2D/espacial |
| **Memoria global (*global*)** | Grid completo | Todos los hilos | Alta | Principal. Requiere **acceso coalescente** para ser eficiente |

## VII. Conceptos clave para programación eficiente

- **SIMT**: Modelo de ejecución por warps
- **Divergencia de warps**: Si hilos de un warp toman ramas distintas (`if/else`), se ejecutan secuencialmente penalizando el rendimiento
- **Coalescencia**: Accesos consecutivos de hilos de un warp a direcciones consecutivas de memoria global
- **Ocultación de latencia**: Cambiar a otro warp listo mientras se espera una transferencia/memoria
- **Ocupación (*occupancy*)**: Proporción de warps activos respecto al máximo soportado por el SM
- **Sincronización**: `__syncthreads()` para coordinar hilos dentro de un bloque
- **Transferencias Host-Device**: `cudaMemcpy` para copiar datos entre CPU y GPU (costosas si frecuentes)

## VIII. Modelos comerciales: GeForce vs Tesla

| Aspecto | **GeForce** | **Tesla / Datacenter** |
| --- | --- | --- |
| **Objetivo** | Gaming | HPC, IA, clusters |
| **Precio** | Más accesible | Mayor (orientado a datacenter) |
| **Garantía/fiabilidad** | Orientado a consumidor | Mayor fiabilidad, pensado para 24/7 |
| **Memoria** | Generalmente menor | Mayor capacidad (ideal para grandes volúmenes) |
| **Funcionalidades** | Enfocado a gráficos | Soporte para NVLink, GPUDirect RDMA, mayor soporte para clusters |
| **Uso desarrollo** | Perfecto para desarrollar y probar | Despliegue en producción/HPC |

## IX. Bibliografía y recursos

- **Kirk, D.B. & Hwu, W-m.W.** *Programming Massively Parallel Processors: A Hands-on Approach*. Morgan Kaufmann.
- **Sanders, J. & Kandrot, E.** *CUDA by Example: An Introduction to General-Purpose GPU Programming*. Addison-Wesley.
- **NVIDIA** - [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- **NVIDIA Developer** - [https://developer.nvidia.com/](https://developer.nvidia.com/)
- **NVIDIA DLI** - Formación oficial NVIDIA Deep Learning Institute
