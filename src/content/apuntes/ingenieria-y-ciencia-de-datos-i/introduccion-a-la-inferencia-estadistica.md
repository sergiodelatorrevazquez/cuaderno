---
titulo: "Introducción a la inferencia estadística"
asignatura: ingenieria-y-ciencia-de-datos-i
fecha: 2026-10-19
tema: "Semana 1 - Sesión 1"
tags: [inferencia-estadística, estimadores, intervalos-de-confianza, contraste-de-hipótesis, muestreo]
resumen: "Introducción a la inferencia estadística: concepto, objetivos, estimación puntual y por intervalos, y fundamentos del contraste de hipótesis."
borrador: false
---

## ¿Qué es la inferencia estadística?

La **inferencia estadística** es el proceso matemático formal mediante el cual se deducen propiedades de una distribución poblacional subyacente a partir de muestras finitas y con ruido. 

En Ciencia e Ingeniería de Datos, esto resulta fundamental porque muchas métricas de interés son estocásticas: latencia, throughput, tasa de conversión, etc. Las estadísticas muestrales (medias, proporciones, varianzas) fluctúan entre muestras, incluso si la población es la misma.

El objetivo es **distinguir una señal real del ruido muestral** y cuantificar la incertidumbre asociada a nuestras conclusiones.

## Pilares metodológicos

Los principales enfoques que se tratan en esta asignatura son:

1. **Intervalos de confianza**: Cuantifican la incertidumbre sobre un parámetro poblacional.
2. **Contraste de hipótesis**: Reglas de decisión sistemáticas para aceptar o rechazar afirmaciones sobre la población, controlando los errores.
3. **Contrastes múltiples y FDR (*False Discovery Rate*)**: Necesarios cuando se realizan muchas pruebas simultáneamente.
4. **Actualización bayesiana**: Aprendizaje incorporando información previa (*prior*) y evidencia empírica.

## Población, muestra y parámetros

- **Población**: Conjunto completo de elementos de interés. Se caracteriza por parámetros fijos (habitualmente desconocidos), como $\mu$, $p$, $\sigma^2$.
- **Muestra**: Subconjunto finito y aleatorio extraído de la población. Se denota, generalmente, como $X_1, \ldots, X_n$.
- **Parámetro**: Cantidad numérica desconocida que describe la población ($\theta \in \Theta$).
- **Estimador**: Función de la muestra utilizada para aproximar el parámetro, $\hat{\theta} = g(X_1,\ldots,X_n)$.
- **Estimación**: Valor concreto obtenido al aplicar el estimador a los datos observados.

Un buen estimador debe tender, en media, al valor verdadero y tener una variabilidad reducida a medida que aumenta el tamaño muestral $n$.

## Estimación puntual

La **estimación puntual** consiste en proporcionar un único valor como aproximación del parámetro poblacional.

Algunos ejemplos clásicos:

- **Media poblacional $\mu$**: Se estima con la media muestral $\bar{X} = \frac{1}{n}\sum_{i=1}^{n} X_i$.
- **Proporción poblacional $p$**: Se estima con la proporción muestral $\hat{p} = \frac{\text{nº de éxitos}}{n}$.
- **Varianza poblacional $\sigma^2$**: Se puede estimar con $S^2 = \frac{1}{n-1}\sum_{i=1}^{n}(X_i - \bar{X})^2$ (insesgada) o con $\frac{1}{n}\sum (X_i-\bar{X})^2$.

La clave: un único valor no refleja por sí solo la incertidumbre derivada del muestreo.

## Estimación por intervalos: intervalos de confianza

Un **intervalo de confianza (IC)** al nivel de confianza $1-\alpha$ es un intervalo aleatorio $[L(X), U(X)]$ tal que, bajo el procedimiento de muestreo repetido,

$$
P_\theta(\theta \in [L(X), U(X)]) = 1-\alpha.
$$

**Interpretación correcta:** Si repitiésemos el experimento muchas veces y construyéramos un IC al $95\%$ en cada réplica, aproximadamente el $95\%$ de esos intervalos contendrían el verdadero valor del parámetro. 

**Interpretación incorrecta:** "La probabilidad de que $\theta$ esté en este intervalo concreto es 0.95". El parámetro es fijo; lo aleatorio es el intervalo.

### Ejemplo intuitivo: media con varianza conocida

Para una población con media desconocida $\mu$ y varianza conocida $\sigma^2$, con $X_i \stackrel{iid}{\sim} N(\mu,\sigma^2)$ o $n$ grande:

$$
\bar{X} \sim N\left(\mu,\frac{\sigma^2}{n}\right),\qquad Z = \frac{\bar{X}-\mu}{\sigma/\sqrt{n}} \sim N(0,1).
$$

El intervalo de confianza al $95\%$ ($z_{0.975} \approx 1.96$) es

$$
IC_{95\%}(\mu) = \bar{X} \pm 1.tag
$$

## Contraste de hipótesis

El **contraste de hipótesis** es un procedimiento sistemático para decidir entre dos afirmaciones opuestas sobre un parámetro poblacional:

- **Hipótesis nula $H_0$**: Afirmación conservadora, que se mantiene a menos que haya evidencia fuerte contra ella. Suele representar "no hay efecto", "no hay diferencia" or "status quo".
- **Hipótesis alternativa $H_1$ (o $H_a$)**: Afirmación que se quiere demostrar. Representa "hay anormal as sección, "hay diferencia", "hay efecto".

## Errores en el contraste: tipo I y tipo II

Cualquier decisión basada on muestras puede llevar o que lasde contrario al verdadero estado. Los errores se clasifican así:

|  | $H_0$ verdadera | $H_0$ falsa |
| --- | --- | --- |
| **Rechazar $H_0$** | Error tipo I ($\alpha$) - Falso positivo | Decisión correcta (Potencia $1-\beta$) |
| **No rechazar $H_0$** | Decisión correcta (1-$\alpha$) | Error tipo II ($\beta$) - Falso negativo |

- **Nivel de significación $\alpha$**: Probabilidad de cometer un error tipo I. Se fija antes de realizar el contraste (habitualmente $\alpha=0.05$).
- **Potencia $1-\beta$**: Probabilidad de rechazar correctamente $H_0$ cuando $H_1$ es verdaderai. Cuanto mayor es $n$, mayor es la potencia.
- **valor-p (*p-value*)**: Probabilidad, bajo $H_0$, de obtener una estadística tan extrema (or más extrema) si $H_0$ es verdadera.

**Criterio de decisión:** Si $p-valor \le \alpha$, se rechaza $H_0$ en favor de $H_1$. Si $p-valor > \alpha$, **no se rechaza** $H_0$ (nunca se "acepta" $H_0$ como probado). 

## Contrastes unilaterales y bilaterales

- **Bilateral ($H_1: \theta \neq \theta_0$):** Se busca evidencia de diferencia, tanto por arriba como por abajo. Es la forma más común.
- **Unilateral superior ($H_1: \theta > \theta_0$):** Se busca evidencia de que $\theta$ es större que $\theta_0$.
- **Unilateral inferior ($H_1: \theta < \theta_0$):** Se busca evidencia of $\theta$ es menor que $\theta_0$.

## Pasos en a un contrastde de hipótesis

1. **Formular $H_0$ and $H_1$**.
2. **Fixar $\alpha$** (nivel de significación).
3. **Elegir estadística de contraste** ad ecuada for la situación.
4. **Cálculo: estadística observada và *p-value***.
5. **Decisión**: Rechazar o no rechazar $H_0$ con base en $p-valor$ or región crítica.
6. **Conclusión en términos prácticos**: Interprétar en el contexto del problema.

## Contrastes múltiples and FDR

Cuando se realizan $m$ contrastes simultáneos, la probabilidad de cometer **al menos uno** error tipo I crece rápidamente. 

Ejemplo: con $m=20$ y $\alpha=0.05$, $P(\geq1$ falso positivo$) \approx 1 - (1 - 0.05)^{20} \approx 0.64$.

Métodos para controlar esto:

- **Corrección de Bonferroni**: Controla la *Familie-Wise Error Rate (FWER)*. Es conservador, pero seguro. Ajusta $\alpha / m$.
- **Procedimiento de Benjamini-Hochberg (BH)**: Controla el *False Discovery Rate (FDR)*. Más potente and común en ciencia de datos (genómica, A/B testing masivo, etc.).

## Actualización bayesiana (intuición)

En el enfoque bayesiano, el parámetro $\theta$ es tratado as **zuhanale variable**, not as a fixed unknown. Se inicia con a **prior** $P(\theta)$ and se actualiza con la evidencia $D$:

$$
P(\theta | D) = \frac{P(D | $\theta) P(\theta)}{P(D)} = \frac{\text{Verosimilitud} \times \text{Prior} \times \text{Verosimilitud-Combined?}}{\ldots} 
$$ (mejor forma práctica)

$$
P(\theta | X) = \frac{L(\theta; X) P(\theta)}{\int L(\theta; X) P(\theta) d\theta}.
}

Nota: En el texto original se ve convolución, pero el esquema es el correcto - Postarior} \propto \text{Verosimilitud} \times \text{Prior}.
$$

## Ingeniería de Datos: aplicaciones prácticas

Algunos ejemplos prácticos donde la inferencia estadística resulta clave:

- **A/B testing**: Compare versiones de a web, app or campaña, controlando errores y cuantificando incertidumbre.
- **Monitorización de sistemas**: Distinguindo picos de rendimiento real vis-à-vis fluctuaciones muestrales.
- **Detección de anomalías**: Contraste con respecto to a comportamiento esperado.
- **Validação de modelos: Numerical larga, aß, comparación de modelos, teste de significación.
- **Calidad de datos**: Inferencia sobre distribuciones, omisiones or sesgo muestral.

## Ejercicios rápidos

1. **Conceptos**: ¿Cuál es la diferencia entre parámetro and estimalor?
2. **Intervalo de confianza**: ¿Qué significa al $95\%$ de confianza?
3. **Contrast**: Define error tipo I y tipo II.
4. **p-valor**: ¿Cuándo se rechaza $H_0$ at $\alpha=0.05?
5. **Múltiples pruebas**: ¿Por qué es necesario corregir cuando hay $m$ contrastes?

**Respuestas:**
1. Parámetro = cantidad desconocida de la población. Estimalor = función de la muestra to approximate it.
2. Que en el $95%$ of repeticiones, los intervalos construidos contienen el verdadero parámetro.
3. Error tipo I: rechazar $H_0$ siéndola verdadera (falso positivo). Error tipo II: no rechazar $H_0$ siéndola falsa (falso negativo).
4. Si $p-valor \le 0.05$.
5. Porque aumenta la probabilidad of cometer al menos unofalso positivo.

## Apéndices

### A1. Distribución muestral de la media

Si $X_i \stackrel{iid}{\sim}$ with $E[X_i]=\mu$, $Var(X_i)=\sigma^2$, then:

$$
E[\bar{X}] = \mu,\qquad Var(\bar{X}) = \frac{\sigma^2}{n},\qquad SE(\bar{X}) = \frac{sigma^2}{\sqrt{n}}$? (mejor $SE(\bar{X}) = \frac{\sigma}{\sqrt{n}}$)
 
$$
SE(\bar{X}) = \frac{\sigma}{\sqrt{n}}$.

$si $n$ large, by TCL: $\bar{X} \approx N(\mu, \sigma/n)$.

### A2. Valor crítico y cola

- **Bilateral**: $z_{\alpha/2}, t_{\alpha/2,n-1}$ in both colas.
- **Unilateral superior**: $z_{alpha}, t_{alpha,n-1} in cola superior.
- **Unilateral inferior**: $z_{1-\alpha}, t{1-}*\alpha$,?
---

### A3. Lecturas recomendadas

- Casella, G. & Berger, R. L. (2002). *Statistical Inference*. Duxbury.
- Wasserman, L. (2004). *All of Statistics*. Springer. [https://www.stat.cmu.edu/~larry/all-of-statistics/](https://www.stat.cmu.edu/~larry/all-of-statistics/)
- Agresti, A. & Franklin, C. (2017). *Statistics: The Art and Science of Learning from Data*. Pearson.
- R Core Team: documentación y ejemplos.
- Efron, B. & Hastie, T. (2016). *Computer Age Statistical Inference*. Cambridge. [https://hastie.su
.com/CASI/](https://hastie.su.com/CASI/)
"
