---
titulo: "Búsqueda no informada y búsqueda A*"
asignatura: sistemas-inteligentes
fecha: 2026-09-29
tema: Tema 2
tags: [búsqueda, heurísticas, grafos]
resumen: Comparativa de BFS, DFS, coste uniforme y A*, con las condiciones que garantizan optimalidad.
borrador: false
---

## Formulación del problema

Un problema de búsqueda se define con cinco componentes:

1. Estado inicial.
2. Acciones disponibles.
3. Modelo de transición.
4. Test de objetivo.
5. Coste de cada camino.

La solución es una secuencia de acciones que lleva del estado inicial a un estado objetivo.

## Búsqueda no informada

| Algoritmo | Estructura | Completa | Óptima | Complejidad temporal |
| --- | --- | --- | --- | --- |
| BFS | Cola FIFO | Sí | Solo con costes iguales | `O(b^d)` |
| DFS | Pila LIFO | No en grafos con ciclos | No | `O(b^m)` |
| Coste uniforme | Cola de prioridad por `g(n)` | Sí | Sí | `O(b^(1+⌊C*/ε⌋))` |

donde `b` es el factor de ramificación, `d` la profundidad de la solución, `m` la profundidad máxima y `C*` el coste de la solución óptima.

## Búsqueda informada

Una **heurística** `h(n)` estima el coste desde el nodo `n` hasta el objetivo. Para que A* sea óptimo, `h` debe ser:

- **Admisible**: nunca sobreestima el coste real, `h(n) ≤ h*(n)`.
- **Consistente**: `h(n) ≤ c(n, n') + h(n')` para todo sucesor `n'`.

La consistencia implica admisibilidad, pero no al revés.

### Algoritmo A*

A* expande el nodo con menor valor de:

> **f(n) = g(n) + h(n)**

Siendo `g(n)` el coste acumulado y `h(n)` la estimación restante.

```python
import heapq

def a_estrella(inicio, objetivo, vecinos, h):
    abiertos = [(h(inicio), 0, inicio)]
    coste = {inicio: 0}
    padre = {}

    while abiertos:
        _, g, nodo = heapq.heappop(abiertos)

        if nodo == objetivo:
            return reconstruir(padre, nodo)

        if g > coste.get(nodo, float("inf")):
            continue

        for siguiente, c in vecinos(nodo):
            nuevo = g + c
            if nuevo < coste.get(siguiente, float("inf")):
                coste[siguiente] = nuevo
                padre[siguiente] = nodo
                heapq.heappush(abiertos, (nuevo + h(siguiente), nuevo, siguiente))

    return None
```

## Ideas clave

> A* es completo y óptimo si la heurística es admisible y el factor de ramificación es finito.

- Con `h(n) = 0` degenera en coste uniforme.
- Cuanto más informada sea la heurística, menos nodos se expanden.
- Una heurística consistente evita reabrir nodos ya cerrados.

## Para repasar

- [ ] Construir un ejemplo donde DFS no encuentra la solución óptima.
- [ ] Comprobar si una heurística es admisible y consistente.
- [ ] Calcular el orden de expansión de A* en un grafo pequeño.
- [ ] Explicar la diferencia entre completitud y optimalidad.
