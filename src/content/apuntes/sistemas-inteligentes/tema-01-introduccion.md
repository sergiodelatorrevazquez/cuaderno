---
titulo: "Introducción a los sistemas inteligentes"
asignatura: sistemas-inteligentes
fecha: 2026-09-22
tema: Tema 1
tags: [agentes, entornos, fundamentos]
resumen: Qué es un agente inteligente, cómo se relaciona con su entorno y qué significa actuar de forma racional.
borrador: false
---

## Qué es un sistema inteligente

Un **agente** es cualquier entidad que percibe su entorno mediante sensores y actúa sobre él mediante actuadores. La elección de la acción depende de la *función de agente*, que transforma la secuencia de percepciones en una acción.

> Un agente racional es aquel que, para cada secuencia de percepciones, elige la acción que maximiza el rendimiento esperado.

## Elementos de un agente

| Elemento | Descripción | Ejemplo |
| --- | --- | --- |
| Percepción | Información que recibe del entorno | Lectura de un sensor de temperatura |
| Acción | Intervención sobre el entorno | Abrir una válvula |
| Entorno | Todo lo que rodea al agente | Una habitación, una carretera |
| Medida de rendimiento | Criterio de éxito | Consumo energético mínimo |

Un agente está formado por:

1. **Arquitectura**: hardware y software que ejecuta el programa.
2. **Programa**: lógica que decide la acción a partir de las percepciones.

## Tipos de entornos

- **Totalmente observable** o parcialmente observable.
- **Determinista** o estocástico.
- **Episódico** o secuencial.
- **Estático** o dinámico.
- **Discreto** o continuo.

La dificultad de diseño crece cuando el entorno es parcialmente observable, estocástico y dinámico a la vez.

## Tipos de agentes

### Agente reactivo simple

Selecciona la acción a partir de la percepción actual, sin memoria. Funciona bien en entornos totalmente observables.

### Agente reactivo basado en modelo

Mantiene una representación interna del mundo. Puede manejar observabilidad parcial, pero no planifica a largo plazo.

### Agente basado en objetivos

Elige acciones que lo acerquen a un estado objetivo. Necesita un modelo del entorno y un mecanismo de búsqueda o planificación.

### Agente basado en utilidad

Generaliza el anterior: en lugar de un objetivo binario, maximiza una función de utilidad. Es la base de gran parte de la inteligencia artificial moderna.

```python
class AgenteReactivo:
    def __init__(self, reglas):
        self.reglas = reglas

    def actuar(self, percepcion):
        for condicion, accion in self.reglas:
            if condicion(percepcion):
                return accion
        return "esperar"
```

## Para repasar

- [ ] Diferenciar agente, arquitectura y programa.
- [ ] Clasificar un entorno según sus propiedades.
- [ ] Justificar por qué la racionalidad depende de la medida de rendimiento.
- [ ] Comparar los cuatro tipos básicos de agentes.
