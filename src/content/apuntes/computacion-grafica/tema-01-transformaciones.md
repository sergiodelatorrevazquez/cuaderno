---
titulo: "Transformaciones geométricas en 2D y 3D"
asignatura: computacion-grafica
fecha: 2026-10-06
tema: Tema 1
tags: [matrices, coordenadas-homogéneas, transformaciones]
resumen: Traslación, escala y rotación con coordenadas homogéneas, y por qué el orden de composición importa.
borrador: false
---

## Por qué coordenadas homogéneas

Una traslación no es una transformación lineal en coordenadas cartesianas: no se puede expresar como una matriz de 2×2. Añadiendo una coordenada extra obtenemos una transformación lineal en 3D que, restringida al plano `w = 1`, equivale a una traslación.

Un punto 2D `(x, y)` se representa como `(x, y, 1)`. Un vector se representa como `(x, y, 0)`, de modo que las traslaciones no le afectan.

## Matrices básicas en 2D

### Traslación

```text
T(tx, ty) = | 1  0  tx |
            | 0  1  ty |
            | 0  0   1 |
```

### Escala

```text
S(sx, sy) = | sx  0  0 |
            |  0 sy  0 |
            |  0  0  1 |
```

### Rotación

Giro de ángulo `θ` en sentido antihorario:

```text
R(θ) = | cos θ  -sin θ  0 |
       | sin θ   cos θ  0 |
       |   0       0    1 |
```

## Composición

Las transformaciones se componen multiplicando matrices. El orden importa:

```text
p' = T · R · S · p
```

Primero se aplica la escala, después la rotación y por último la traslación. Si cambiamos el orden, el resultado cambia.

> Regla práctica: se lee de derecha a izquierda, porque cada matriz se aplica al vector resultante de la anterior.

Ejemplo en JavaScript con una clase mínima de matrices 3×3:

```js
function multiplicar(a, b) {
  const c = Array.from({ length: 3 }, () => [0, 0, 0]);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      for (let k = 0; k < 3; k++) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return c;
}

function trasladar(tx, ty) {
  return [
    [1, 0, tx],
    [0, 1, ty],
    [0, 0, 1],
  ];
}
```

## Caso 3D

En 3D usamos matrices 4×4. La traslación y la escala son análogas, pero hay tres rotaciones básicas: alrededor de X, Y y Z.

| Transformación | Grados de libertad | Matriz |
| --- | --- | --- |
| Traslación | 3 | 4×4 |
| Escala | 3 | 4×4 |
| Rotación | 3 | 4×4 |
| Shear | 6 | 4×4 |

Las rotaciones en 3D no conmutan: aplicar primero X y luego Y no es lo mismo que aplicar Y y luego X.

## Para repasar

- [ ] Deducir la matriz de rotación a partir de trigonometría.
- [ ] Componer una rotación alrededor de un punto arbitrario.
- [ ] Explicar por qué los vectores tienen `w = 0`.
- [ ] Comprobar numéricamente que `R · T ≠ T · R`.
