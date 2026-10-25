---
titulo: "Reglas de asociación y Market Basket Analysis"
asignatura: ingenieria-y-ciencia-de-datos-i
fecha: 2026-10-21
tema: "Semana 1 - Sesión 3"
tags: [reglas-de-asociación, market-basket-analysis, apriori, soporte, confianza, lift, itemsets-frecuentes, arules]
resumen: "Descubrimiento de reglas de asociación en bases de datos transaccionales: soporte, confianza, lift, principio Apriori, poda de reglas y visualización."
borrador: false
---

## ¿Qué es el *Association Rule Mining*?

El *Association Rule Mining* (ARM) o **minería de reglas de asociación** es una técnica de descubrimiento de patrones no supervisada cuyo objetivo es encontrar afinidades, co-ocurrencias recurrentes e implicaciones probabilísticas ocultas en grandes bases de datos transaccionales.

Un caso clásico es el **Análisis de Cesta de la Compra (*Market Basket Analysis*)**: identificar qué productos se compran juntos con frecuencia para tomar decisiones sobre ubicación de productos, promociones cruzadas, recomendaciones, etc.

## Fundamentos: transacciones, ítems e itemsets

Sea $\mathcal{I} = \{i_1, i_2, ..., i_m\}$ el catálogo universal de **ítems** distintos. 

Una **base de datos transaccional** $\mathcal{D} = \{t_1, ..., t_n\}$ está formada por $n$ transacciones, donde cada transacción $t_j \subseteq \mathcal{I}$ es el conjunto de ítems comprados en esa operación.

- **Itemset**: Cualquier subconjunto $X \subseteq \mathcal{I}$.
- **$k$-itemset**: Itemset formado por exactamente $k$ ítems.
- **Soporte en conteo (*support count*)**: Número de transacciones que contienen el itemset $X$.
  
$$
\sigma(X) = |\{t_j \in \mathcal{D} \mid X \subseteq t_j\}|
$$

## Reglas de asociación

Una **regla de asociación** es una implicación de la forma

$$
A \implies B,
$$

donde $A \subset \mathcal{I}$, $B \subset \mathcal{I}$ y $A \cap B = \emptyset$. 

- $A$: **antecedente** (*left-hand side*, LHS)
- $B$: **consecuente** (*right-hand side*, RHS)

El objetivo es encontrar reglas con valores elevados en ciertas métricas de interés.

## Métricas de calidad

### Soporte (*support*)

El **soporte** mide la frecuencia con la que aparece el itemset $\{A \cup B\}$ en la base de datos. Se interpreta como la probabilidad empírica de que una transacción contenga ambos conjuntos.

$$
\text{soporte}(A \implies B) = \frac{\sigma(A \cup B)}{n} = P(A \cup B)
$$

Suele establecerse un **umbral mínimo de soporte** ($min\_sup$) para filtrar patrones poco frecuentes.

### Confianza (*confidence*)

La **confianza** mide la fiabilidad de la regla: dado que la transacción contiene $A$, ¿con qué probabilidad contiene también $B$?

$$
\text{confianza}(A \implies B) = \frac{\sigma(A \cup B)}{\sigma(A)} = \frac{\text{soporte}(A \cup B)}{\text{soporte}(A)} = P(B \mid A)
$$

Se filtran las reglas con **confianza mínima** ($min\_conf$).

### Lift (*elevación*)

El **lift** compara la probabilidad observada de que $A$ y $B$ aparezcan juntos con la probabilidad esperada si fueran independientes.

$$
\text{lift}(A \implies B) = \frac{\text{confianza}(A \implies B)}{\text{soporte}(B)} = \frac{P(B \mid A)}{P(B)} = \frac{P(A \cup B)}{P(A)P(B)}
$$

- $\text{lift} > 1$: $A$ y $B$ aparecen juntos **más** de lo esperado por azar (*correlación positiva*)
- $\text{lift} = 1$: $A$ y $B$ son **independientes** (*sin asociación*)
- $\text{lift} < 1$: $A$ y $B$ aparecen juntos **menos** de lo esperado por azar (*correlación negativa*)

### Leverage

El **leverage** mide la diferencia absoluta entre la frecuencia observada conjunta y la esperada bajo independencia.

$$
\text{leverage}(A \implies B) = \text{soporte}(A \cup B) - \text{soporte}(A)\,\text{soporte}(B)
$$

- $\text{leverage} > 0$: Mayor frecuencia observada que la esperada
- $\text{leverage} = 0$: Independencia
- $\text{leverage} < 0$: Menor frecuencia observada que la esperada

### Conviction (*convicción*)

La **conviction** evalúa la fuerza de la implicación, penalizando fuertemente cuando $B$ aparece con frecuencia incluso sin $A$.

$$
\text{conviction}(A \implies B) = \frac{1 - \text{soporte}(B)}{1 - \text{confianza}(A \implies B)}
$$

- $\text{conviction} \to +\infty$: La regla es **muy fuerte** (cuando $B$ casi nunca ocurre sin $A$)
- $\text{conviction} = 1$: La regla aporta **poca o ninguna información** sobre $B$

## El principio Apriori

El gran problema del *Association Rule Mining* es el **espacio combinatorio**: con $m$ ítems hay $2^m$ posibles itemsets, lo cual es inviable para catálogos grandes.

El algoritmo **Apriori** resuelve esto gracias al **principio de anti-monotonía del soporte**:

> **Si un itemset $X$ es infrecuente ($\text{soporte}(X) < min\_sup$), entonces cualquier superconjunto $Y \supset X$ también será infrecuente.**

Equivalente: **Si un itemset es frecuente, todos sus subconjuntos son frecuentes**.

Este principio permite **podar** grandes ramas del retículo de itemsets (*itemset lattice*) sin necesidad de evaluarlas explícitamente.

### Algoritmo Apriori (pasos)

1. **Generar itemsets de tamaño 1**: Contar frecuencia de cada ítem individual.
2. **Filtrar**: Conservar solo los ítems con $\sigma(i) >= min\_sup$.
3. **Generar candidatos**: A partir de $k$-itemsets frecuentes, generar candidatos $(k+1)$-itemsets por unión.
4. **Podar por subconjuntos**: Eliminar candidatos que tengan algún $k$-subconjunto infrecuente.
5. **Contar soporte**: Escanear la base de datos para contar $\sigma$ de candidatos restantes.
6. **Filtrar y repetir**: Conservar frecuentes, volver al paso 3 hasta que no se puedan generar nuevos itemsets.
7. **Generar reglas**: A partir de cada itemset frecuente $L = A \cup B$, generar reglas $A \implies B$ y comprobar $min\_conf$.

## Generación de reglas desde itemsets frecuentes

Dado un itemset frecuente $Z = A \cup B$ con $A \cap B = \emptyset$, podemos dividirlo en antecedente $A$ y consecuente $B$ para formar una regla. 

Para todo posible particionado, se calcula la confianza:

$$
\text{confianza}(A \implies B) = \frac{\text{soporte}(Z)}{\text{soporte}(A)}
$$

Solo se mantienen aquellas reglas que cumplan $\text{confianza}(A \implies B) >= min\_conf$.

**Nota**: Diferentes particionados del mismo itemset pueden generar reglas con distinta confianza.

## Poda de reglas redundantes

Cuando se generan muchas reglas, es común encontrar **reglas redundantes**. Una regla $r: A \implies B$ se considera **redundante** si existe una regla más general o más fuerte $r': A' \implies B'$ tal que:

- $A' \subset A$ (antecedente más general/más corto)
- $r'$ es al menos tan "buena" como $r$ en las métricas principales
- $r'$ se cumple con mayor o igual cobertura

En R, el paquete `arules` permite detectar y eliminar reglas redundantes con `is.redundant()`.

## Implementación práctica con `arules` (R)

El flujo típico con el paquete `arules` es el siguiente:

```r
library(arules)
library(arulesViz)

# Leer transacciones (formato basket)
trans <- read.transactions("cesta.csv", format = "basket", sep = ",")

# Inspección
summary(trans)
inspect(head(trans))

# Minería con Apriori
reglas <- apriori(trans, 
                  parameter = list(supp = 0.01, conf = 0.3, minlen = 2))

# Ordenar por lift
reglas_ordenadas <- sort(reglas, by = "lift")

# Inspeccionar mejores reglas
inspect(head(reglas_ordenadas, 10))

# Eliminar redundantes
reglas_filtradas <- reglas_ordenadas[!is.redundant(reglas_ordenadas)]

# Visualización
plot(reglas_filtradas, method = "scatterplot", measure = c("support", "lift"), shading = "confidence")
plot(reglas_filtradas, method = "graph")
plot(reglas_filtradas, method = "grouped")
```

**Parámetros clave:**
- `supp`: Soporte mínimo
- `conf`: Confianza mínima
- `minlen`: Longitud mínima de la regla (antecedente + consecuente)
- `maxlen`: Longitud máxima permitida

## Visualización de reglas

`arulesViz` ofrece varias visualizaciones para interpretar los resultados:

| Método | Uso |
| --- | --- |
| `scatterplot` | Relación entre soporte, confianza y lift. Muy útil para explorar el espacio de reglas. |
| `graph` | Grafo bipartito (ítems y reglas). Ideal para ver conexiones entre productos. |
| `grouped` | Agrupa reglas por antecedente. Excelente para visualización compacta. |
| `paracoord` | Coordenadas paralelas para ver patrones entre múltiples ítems. |

## Buenas prácticas

- **Elegir umbrales con criterio**: Empezar con valores conservadores ($min\_sup$ bajo/moderado, $min\_conf$ razonable) e ir ajustando según el tamaño de la base de datos.
- **Priorizar lift**: Una regla puede tener alta confianza pero $lift \approx 1$ (producto muy popular). Es preferible buscar reglas con **lift > 1**.
- **Eliminar redundantes**: Reducir el número de reglas sin perder información relevante.
- **Interpretar en contexto**: Las correlaciones encontradas **no implican causalidad**. Son asociaciones estadísticas.
- **Cuidado con catálogos grandes**: Ajustar umbrales para controlar la explosión combinatoria.
- **Considerar la rareza**: Algunos patrones interesantes pueden tener soporte bajo pero gran valor comercial (*nicho*).

## Ejercicios rápidos

1. **Conceptos**: ¿Cuál es la diferencia entre itemset y regla de asociación?
2. **Métricas**: Si $\text{soporte}(A)=0.2$, $\text{soporte}(B)=0.3$, $\text{soporte}(A\cup B)=0.1$. Calcula confianza y lift.
3. **Apriori**: ¿Por qué es fundamental el principio de anti-monotonía?
4. **Interpretación**: Un regla con $\text{lift}=2.5$, ¿qué significa?
5. **Redundancia**: ¿Cuándo decimos que una regla es redundante?

**Soluciones:**
1. Un **itemset** es un conjunto de ítems que aparecen juntos. Una **regla** establece una implicación $A \implies B$ entre dos disjuntos.
2. $\text{confianza}(A\implies B)=0.1/0.2=0.5$ (50%). $\text{lift}=0.5/0.3 \approx 1.666... > 1$.
3. Permite **podar** itemsets infrecuentes y todos sus superconjuntos, reduciendo drásticamente el espacio de búsqueda.
4. Que $A$ y $B$ aparecen juntos **2.5 veces más** de lo que se esperaría si fueran independientes.
5. Cuando existe otra regla más general con antecedente más corto que es al menos tan buena en las métricas y cubre la misma relación.

## Apéndices

### A1. Ejemplo numérico sencillo

Supongamos 5 transacciones:

| Transacción | Items |
| --- | --- |
| 1 | {Pan, Leche, Mantequilla} |
| 2 | {Pan, Leche} |
| 3 | {Pan, Mantequilla} |
| 4 | {Leche, Mantequilla} |
| 5 | {Pan, Leche, Mantequilla} |

Para la regla $\{\text{Pan}\} \implies \{\text{Leche}\}$:
- $\sigma(\{\text{Pan}\})=4$, $\text{soporte}(\{\text{Pan}\})=4/5=0.8$
- $\sigma(\{\text{Pan},\text{Leche}\})=3$, $\text{soporte}(\{\text{Pan},\text{Leche}\})=3/5=0.6$
- $\text{confianza}=0.6/0.8=0.75$
- $\text{soporte}(\{\text{Leche}\})=4/5=0.8$, $\text{lift}=0.75/0.8=0.9375 < 1$ (*ligera asociación negativa*)

### A2. Lecturas recomendadas

- Agrawal, R., Imieliński, T. & Swami, A. (1993). Mining association rules between sets of items in large databases. *SIGMOD '93*, 207–216.
- Han, J., Kamber, M. & Pei, J. (2011). *Data Mining: Concepts and Techniques* (3ª ed.). Morgan Kaufmann.
- Hahsler, M., Grün, B. & Hornik, K. (2005). arules – A computational environment for mining association rules and frequent item sets. *JSS*, 14(15).
- Hahsler, M. & Chelluboina, S. (2017). Visualizing association rules in hierarchical groups. *JSS*, 77(1).
- Tan, P.-N., Steinbach, M. & Kumar, V. (2018). *Introduction to Data Mining* (2ª ed.). Pearson.
- CRAN: [arules package](https://cran.r-project.org/package=arules), [arulesViz package](https://cran.r-project.org/package=arulesViz)
