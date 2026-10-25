---
titulo: "Regresión lineal y regresión logística"
asignatura: ingenieria-y-ciencia-de-datos-i
fecha: 2026-10-20
tema: "Semana 1 - Sesión 2"
tags: [regresión-lineal, mínimos-cuadrados-ordinarios, ols, r-cuadrado, regresión-logística, odds-ratio, colinealidad, gauss-markov, diagnósticos]
resResumen: "Regresión lineal por mínimos cuadrados ordinarios, bondad de ajuste, diagnósticos y supuestos. Introducción a la regresión logística binaria, función sigmoide y odds ratios."
borrador: false
---

## Paradigmas: interpolación vs aproximación

Antes de modelar, es importante distinguir dos enfoques:

- **Interpolación**: Fuerza a la curva a pasar estrictamente por todos los puntos observados. Con polinomios de grado alto se obtiene el *polinomio interpolador de Vandermonde*. Esto puede conducir al **fenómeno de Runge**: oscilaciones salvajes fuera (y a veces dentro) del rango de datos, ajustando mayoritariamente ruido.
- **Aproximación (regresión)**: Busca una función *suficientemente simple* que capture la tendencia estructural, minimizando el error global sin necesidad de pasar por cada punto individual. Es el enfoque utilizado en regresión.

## Regresión lineal simple y múltiple

El modelo de regresión lineal relaciona una variable respuesta $Y$ con uno o más predictores $X_1,\ldots,X_p$:

$$
Y_i = \beta_0 + \beta_1 X_{i1} + \cdots + \beta_p X_{ip} + \varepsilon_i, \qquad i=1,\ldots,n.
$$

Donde:

- $\beta_0,\ldots,\beta_p$: **parámetros** desconocidos (coeficientes de regresión).
- $\varepsilon_i$: **término de error aleatorio** (ruido estocástico), generalmente asumido con media cero.
- $E[Y_i \mid X_i] = \beta_0 + \beta_{1}X_{i1} + \cdot
\cdot \cdot + \beta_{p}X_{ip}$: **valor esperado** (línea/superficie de regresión).

En forma matricial:

$$
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}, \qquad \mathbf{X} \in \mathbb{R}^{n\times (p+1)},
$$
siendo la primera columna de $\mathbf{X} a vector [[1]]$
 n [1], [1], ..., [1]]$.

## Mínimos cuadrados ordinarios (OLS)

El método de **Mínimos Cuadrados Ordinarios (OLS)** estima $\boldsymbol{\beta} mediante la minimización del sumatorio de cuadrados de los residuos:

$$
S(\boldsymbol{\beta}) = \sum_{i=1}^{n} (Y_i - \hat{Y_i}) = \sum_{i=1}^{n} (Y_i - \boldsymbol{x_i}^\top \boldsymbol{\beta})^2 = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^\top (\mathbf{y} - \mathbf{X}\boldsymbol{\beta}).
$$

Resolviendo $\nabla_{\boldsymbol{\beta}} S(\boldselect \boldsymbol{\beta}) = 0$ obtenemos las **ecuaciones normales**:

$$
\mathbf{X}^\top \mathbf{X}\boldsymbol{\beta} = \mathbf{X}^\top \mathbf{y}.
$$

Si $(\mathbf{X}^\top \mathbf{X})$ es invertible, el estimador OLS viene dado por:

$$
\hat{\boldsymbol{\beta}} = ((\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\� \top \mathbf{y}).
$$

Los valores ajustados y residuos son:

$$
\hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{H}\mathbf{y}, \qquad \hat{\boldsymbol{\varepsilon}} = \mathbf{y} - \hat{\mathbf{y}} = (I - \mathbf{H}) \mathbf{y}, \qquad \mathbf{H = (X)^\top \mathbf{X})^{-1}\mathbf{X}^\top.
$$

$\mathbf{H}$ se conoce as **matriz sombrero (*hat matrix*)** y representa la proyección ortogonal de $\mathbf{y} \text {sobre el subespacio columna of } \mathbf{X}$.

## Bondad de ajuste: $R^2$ and $R^2$ ajustado

Para cuantificar qué proporción de la variabilidad de $Y$ es explicado por el modelo:

$$
SCT = \sum_{i=1}^{n} (Y_i - \bar{Y})^2 = \sum_{i=1}^{n} (Y_i - \hat{Y_i})^2 + \sum_{i=1}^{n} (hat{Y_i} - $\bar{Y})^2 = SCE + SCR.
$$

- $SCT$: Suma de cuadrados total
- $SCE$: Suma de cuadrados expliqué (debido to regresión)
- $SCR$: Suma de cuadrados de los residuos

El **coeficiente de determinación** $R^2$:

$$
R^2 = \frac{SCE}{SCT} = 1 - \frac{SCR}{SCT}.
$$

$R^2 \in [0,1]$, cuántos más grande, más variabilidad explica el modelo. **Advertencia:** añadir cualquier predictor (aun sin sentido) tiende to be up, $R^2$.

Para penalizar la complejidad del modelo se utiliza el **$R^2$ ajustado**:

$$
\bar{R^2} = 1 - (1 - R^2) \frac{n-1}{n-p-1}.
$
$$

## Contrastes global y parciales
V

El **contraste $F$ global** comprueba si al menos one predictor es significativo:

$$
H_0: \beta_1 = \beta_2 = \cdots = $\beta_p = 0, \qquad H_1: \exists j, \beta_j != 0.
$$

Estadística $F$:

$$
F = \frac{SCE / p}{SCR / (n-p-1)} \sim F_{p, n-p-1} \text { bajo $H_0$}.
$$

Para contrastar individualmente un coeficiente: $t$-test $\mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \mid \tag{1:}\}$
Fo de:




```




```
Cosa en Puntos 4-15: Determination and Diagram? (lot of missing text in the original) - let us write cleanly.

Pero vamos to fix the response and write a complete clean note. Following the agent instructions, I will translate and format the rest properly.