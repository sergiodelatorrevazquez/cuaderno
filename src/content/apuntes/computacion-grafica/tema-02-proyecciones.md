---
titulo: "Proyecciones y cámara virtual"
asignatura: computacion-grafica
fecha: 2026-10-13
tema: Tema 2
tags: [proyecciones, cámara, perspectiva]
resumen: Apunte en construcción sobre proyecciones ortográficas y en perspectiva.
borrador: true
---

## Proyección ortográfica

Conserva las proporciones y no introduce profundidad. Es útil en vistas técnicas y en juegos con estética isométrica.

## Proyección en perspectiva

Los objetos lejanos se ven más pequeños. La matriz de perspectiva introduce una división por la coordenada `w` al final del pipeline.

```text
x_ndc = x_clip / w_clip
y_ndc = y_clip / w_clip
```

## Pendiente

- [ ] Desarrollar la matriz de perspectiva.
- [ ] Explicar el volumen de visión.
- [ ] Añadir un ejemplo con `gl-matrix`.
