---
titulo: "Regresión lineal y regresión logística"
asignatura: ingenieria-y-ciencia-de-datos-i
fecha: 2026-10-20
tema: "Semana 1 - Sesión 2"
tags: [regresion-lineal, minimos-cuadrados-ordinarios, ols, r-cuadrado, regresion-logistica, odds-ratio, colinealidad, gauss-markov, diagnosticos]
resumen: "Regresión lineal por mínimos cuadrados ordinarios, bondad de ajuste, supuestos y diagnósticos. Introducción a la regresión logística binaria, la función sigmoide y las razones de momios."
borrador: false
---

## Interpolación y regresión

- **Interpolación**: construye una función que pasa por todos los puntos observados. Un polinomio de grado alto puede ajustarse demasiado a los datos y oscilar, especialmente cerca de los extremos del intervalo (fenómeno de Runge).
- **Regresión**: estima una relación que describa la tendencia de los datos, sin exigir que la función pase por cada observación. El ajuste busca equilibrar la fidelidad a los datos y la complejidad del modelo.

## Regresión lineal simple y múltiple

La regresión lineal relaciona una variable respuesta cuantitativa $Y$ con uno o más predictores $X_1,\ldots,X_p$. Para la observación $i$:

$$
Y_i = \beta_0 + \beta_1 X_{i1} + \cdots + \beta_p X_{ip} + \varepsilon_i,\qquad i=1,\ldots,n.
$$

- $\beta_0,\ldots,\beta_p$ son los coeficientes o parámetros del modelo.
- $\varepsilon_i$ es el error aleatorio, que recoge la variabilidad no explicada por los predictores.
- La media condicional del modelo es:

$$
E[Y_i \mid X_{i1},\ldots,X_{ip}] = \beta_0 + \beta_1 X_{i1} + \cdots + \beta_p X_{ip}.
$$

En regresión simple hay un predictor; en regresión múltiple hay varios. El modelo es lineal en los coeficientes, aunque se pueden incluir transformaciones de los predictores.

En forma matricial:

$$
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon},
$$

donde $\mathbf{y}\in\mathbb{R}^{n}$, $\boldsymbol{\beta}\in\mathbb{R}^{p+1}$ y $\mathbf{X}\in\mathbb{R}^{n\times(p+1)}$. Si el modelo incluye intercepto, la primera columna de $\mathbf{X}$ está formada por unos.

## Mínimos cuadrados ordinarios

Los **mínimos cuadrados ordinarios (MCO; OLS, por sus siglas en inglés)** estiman los coeficientes minimizando la suma de los cuadrados de los residuos:

$$
S(\boldsymbol{\beta}) = \sum_{i=1}^{n}(Y_i-\mathbf{x}_i^\top\boldsymbol{\beta})^2
= (\mathbf{y}-\mathbf{X}\boldsymbol{\beta})^\top(\mathbf{y}-\mathbf{X}\boldsymbol{\beta}).
$$

Al igualar el gradiente de esta función a cero se obtienen las **ecuaciones normales**:

$$
\mathbf{X}^\top\mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}^\top\mathbf{y}.
$$

Si las columnas de $\mathbf{X}$ son linealmente independientes, la solución es:

$$
\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}.
$$

Si $\mathbf{X}^\top\mathbf{X}$ no es invertible, los coeficientes no se pueden determinar de forma única mediante esta expresión; puede haber predictores redundantes o colinealidad perfecta.

Los valores ajustados y los residuos son:

$$
\hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{H}\mathbf{y},
\qquad
\hat{\boldsymbol{\varepsilon}} = \mathbf{y}-\hat{\mathbf{y}},
\qquad
\mathbf{H} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top.
$$

La matriz $\mathbf{H}$ se denomina **matriz sombrero** (*hat matrix*): proyecta las observaciones sobre el espacio generado por las columnas de $\mathbf{X}$.

## Bondad de ajuste

Con intercepto en el modelo, la variabilidad total se descompone en variabilidad explicada y residual:

$$
\underbrace{\sum_{i=1}^{n}(Y_i-\bar{Y})^2}_{SCT}
=
\underbrace{\sum_{i=1}^{n}(\hat{Y}_i-\bar{Y})^2}_{SCE}
+
\underbrace{\sum_{i=1}^{n}(Y_i-\hat{Y}_i)^2}_{SCR}.
$$

- $SCT$: suma de cuadrados total.
- $SCE$: suma de cuadrados explicada por la regresión.
- $SCR$: suma de cuadrados residual.

El **coeficiente de determinación** mide la proporción de la variabilidad muestral de $Y$ explicada por el modelo:

$$
R^2 = \frac{SCE}{SCT} = 1-\frac{SCR}{SCT}.
$$

En una regresión OLS con intercepto, $R^2$ está entre 0 y 1. Un valor alto no demuestra que el modelo sea correcto ni que exista una relación causal. Además, al añadir predictores, $R^2$ no disminuye, aunque estos aporten poco.

El **$R^2$ ajustado** penaliza la inclusión de predictores mediante los grados de libertad:

$$
R^2_{\mathrm{ajustado}} = 1-(1-R^2)\frac{n-1}{n-p-1},
$$

donde $n$ es el número de observaciones y $p$ el número de predictores, sin contar el intercepto. Puede disminuir al añadir un predictor que no mejore suficientemente el ajuste.

## Contrastes de hipótesis

El **contraste $F$ global** evalúa si todos los coeficientes de los predictores son simultáneamente cero:

$$
H_0:\beta_1=\cdots=\beta_p=0,
\qquad
H_1:\text{al menos un }\beta_j\ne 0.
$$

Bajo $H_0$ y los supuestos del modelo, el estadístico es:

$$
F = \frac{SCE/p}{SCR/(n-p-1)} \sim F_{p,n-p-1}.
$$

Un resultado significativo indica evidencia de que al menos un predictor está asociado con la respuesta; no implica que todos lo estén ni demuestra causalidad.

Para contrastar un coeficiente individual, por ejemplo $H_0:\beta_j=0$, se utiliza un estadístico $t$:

$$
t_j = \frac{\hat{\beta}_j}{SE(\hat{\beta}_j)}.
$$

Su interpretación se realiza manteniendo constantes los demás predictores del modelo. Los valores $p$ y los intervalos de confianza dependen de los supuestos usados para calcular los errores estándar.

## Supuestos y diagnóstico del modelo lineal

Para las propiedades clásicas de OLS se consideran los siguientes supuestos:

1. **Linealidad en los parámetros**: la media condicional está correctamente representada por la forma del modelo.
2. **Exogeneidad**: el error tiene media condicional cero dado el conjunto de predictores, $E[\varepsilon_i\mid\mathbf{X}]=0$.
3. **Ausencia de dependencia entre errores**: las observaciones y los errores no presentan una dependencia no modelada.
4. **Homoscedasticidad**: la varianza condicional del error es constante, $Var(\varepsilon_i\mid\mathbf{X})=\sigma^2$.
5. **Ausencia de colinealidad perfecta**: ningún predictor es combinación lineal exacta de los demás.

El teorema de **Gauss-Markov** establece que, bajo las condiciones lineales, exogeneidad, homoscedasticidad y ausencia de colinealidad perfecta, OLS es el estimador lineal insesgado de varianza mínima (BLUE). La normalidad de los errores no es necesaria para este resultado, pero se suele asumir para obtener inferencia exacta en muestras pequeñas.

Herramientas habituales de diagnóstico:

- **Residuos frente a valores ajustados**: patrones curvos pueden señalar falta de linealidad; una dispersión en forma de abanico puede indicar heteroscedasticidad.
- **Gráfico Q-Q de residuos**: ayuda a evaluar si la distribución de los errores se aproxima a una normal.
- **Residuos frente al orden temporal o espacial**: puede revelar dependencia.
- **Colinealidad**: correlaciones altas o factores de inflación de la varianza elevados pueden volver inestables los coeficientes y aumentar sus errores estándar.
- **Observaciones influyentes**: residuos atípicos, alto apalancamiento o medidas de influencia como la distancia de Cook ayudan a detectar observaciones que afectan mucho al ajuste.

Un diagnóstico debe guiar una revisión del modelo y de los datos; no justifica eliminar observaciones automáticamente.

## Regresión logística binaria

La regresión logística se utiliza cuando la respuesta $Y$ es binaria, por ejemplo, $Y\in\{0,1\}$. Modela la probabilidad condicional de la clase 1:

$$
p(\mathbf{x}) = P(Y=1\mid\mathbf{X}=\mathbf{x})
= \frac{1}{1+\exp(-\eta)},
\qquad
\eta = \beta_0+\beta_1x_1+\cdots+\beta_px_p.
$$

La función sigmoide transforma cualquier valor real de $\eta$ en una probabilidad entre 0 y 1. De forma equivalente, el **logit** de la probabilidad es lineal en los predictores:

$$
\log\left(\frac{p(\mathbf{x})}{1-p(\mathbf{x})}\right)
= \beta_0+\beta_1x_1+\cdots+\beta_px_p.
$$

Los coeficientes se estiman normalmente por **máxima verosimilitud**, no minimizando la suma de cuadrados. Para un predictor $X_j$, un aumento de una unidad, manteniendo los demás constantes, cambia el logaritmo de las *odds* en $\beta_j$ y multiplica las *odds* por:

$$
\text{OR}_j = \exp(\beta_j).
$$

Esta cantidad es la **razón de momios** (*odds ratio*). Si $\text{OR}_j>1$, las *odds* aumentan; si $\text{OR}_j<1$, disminuyen. Un *odds ratio* no es directamente un cambio en la probabilidad: el cambio de probabilidad depende de la probabilidad inicial.

Para clasificar, se puede comparar la probabilidad estimada con un umbral, a menudo 0.5. Ese umbral no es obligatorio: debe elegirse según el coste de falsos positivos y falsos negativos y el objetivo de la aplicación. La evaluación puede incluir matriz de confusión, precisión, exhaustividad (*recall*), especificidad, ROC-AUC y, cuando hay desbalance de clases, precisión-exhaustividad.

## Regresión lineal y logística: diferencias clave

| Aspecto | Regresión lineal | Regresión logística binaria |
|---|---|---|
| Tipo de respuesta | Cuantitativa | Binaria |
| Cantidad modelada | Media condicional de $Y$ | Probabilidad de $Y=1$ |
| Función de enlace | Identidad | Logit |
| Estimación habitual | Mínimos cuadrados | Máxima verosimilitud |
| Interpretación de coeficientes | Cambio esperado en $Y$ por unidad del predictor | Cambio en log-*odds*; $\exp(\beta_j)$ es el *odds ratio* |
