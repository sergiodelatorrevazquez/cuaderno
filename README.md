# Cuaderno MUII

Cuaderno de apuntes del **Máster Universitario en Ingeniería Informática** de la Universidad de Málaga (UMA).

Sitio estático, rápido y sin base de datos: cada asignatura y cada apunte es un archivo Markdown.

## Stack

| Pieza | Elección | Por qué |
| --- | --- | --- |
| Framework | [Astro 7](https://astro.build) | Genera HTML estático, ideal para contenido y para GitHub Pages o Vercel. |
| Contenido | Content Collections + Markdown | Frontmatter validado con Zod y rutas generadas automáticamente. |
| Estilos | CSS puro | Sin dependencias, con tema claro/oscuro y diseño adaptable. |
| JavaScript | Casi ninguno | Solo el botón de tema. El resto es HTML y CSS. |
| Despliegue | GitHub Actions / Vercel | Build automático en cada `push`. |

## Requisitos

- Node.js 22.23.4 o superior (recomendado 24).
- npm.

## Puesta en marcha

```bash
npm install
npm run dev
```

La web estará en `http://localhost:4321`.

### Scripts

| Comando | Qué hace |
| --- | --- |
| `npm run dev` | Servidor de desarrollo con recarga en caliente. |
| `npm run build` | Genera el sitio estático en `dist/`. |
| `npm run preview` | Sirve el resultado de `dist/` en local. |
| `npm run check` | Comprueba tipos y contenido con Astro Check. |

## Estructura

```text
cuaderno/
├─ public/                    # favicon, robots.txt y estáticos
├─ src/
│  ├─ content/
│  │  ├─ asignaturas/         # una ficha .md por asignatura
│  │  └─ apuntes/             # apuntes en .md, organizados por asignatura
│  ├─ layouts/
│  │  └─ BaseLayout.astro     # cabecera, tema y pie comunes
│  ├─ lib/                    # utilidades (URLs, fechas, lectura)
│  ├─ pages/
│  │  ├─ index.astro          # portada
│  │  ├─ asignaturas/[id].astro
│  │  ├─ apuntes/[...id].astro
│  │  └─ 404.astro
│  ├─ styles/global.css
│  └─ content.config.ts       # esquemas de las colecciones
├─ astro.config.mjs
└─ .github/workflows/deploy.yml
```

## Cómo añadir una asignatura

Crea un archivo en `src/content/asignaturas/`. El nombre del archivo será su identificador:

```yaml
---
nombre: Sistemas Inteligentes
codigo: MUII-SI
curso: 2026/2027
semestre: 1
descripcion: Agentes, búsqueda y aprendizaje automático.
color: "#6366f1"
orden: 1
---
```

| Campo | Obligatorio | Descripción |
| --- | --- | --- |
| `nombre` | Sí | Nombre visible de la asignatura. |
| `codigo` | No | Código del plan de estudios. |
| `curso` | No | Por defecto, `2026/2027`. |
| `semestre` | No | `1` o `2`. Por defecto, `1`. |
| `descripcion` | No | Texto corto para la tarjeta. |
| `color` | No | Color de acento en formato `#rrggbb`. |
| `orden` | No | Orden en la portada (menor primero). |

## Cómo añadir un apunte

Crea un `.md` dentro de `src/content/apuntes/<id-asignatura>/`:

```yaml
---
titulo: "Búsqueda no informada y búsqueda A*"
asignatura: sistemas-inteligentes
fecha: 2026-09-29
tema: Tema 2
tags: [búsqueda, heurísticas]
resumen: Comparativa de BFS, DFS, coste uniforme y A*.
borrador: false
---
```

| Campo | Obligatorio | Descripción |
| --- | --- | --- |
| `titulo` | Sí | Título del apunte. |
| `asignatura` | Sí | **ID** de la asignatura (nombre del archivo sin `.md`). |
| `fecha` | Sí | Fecha en formato `AAAA-MM-DD`. |
| `tema` | No | Por ejemplo, `Tema 2`. |
| `tags` | No | Lista de etiquetas. |
| `resumen` | No | Entradilla que aparece bajo el título. |
| `borrador` | No | Si es `true`, solo se muestra en desarrollo. |

El **ID** del apunte se construye con la ruta del archivo. Por ejemplo:

```text
src/content/apuntes/sistemas-inteligentes/tema-02-busqueda.md
→ /apuntes/sistemas-inteligentes/tema-02-busqueda/
```

El cuerpo admite GitHub Flavored Markdown: tablas, listas de tareas, bloques de código con resaltado, citas, etc.

## Borradores

Con `borrador: true` el apunte se ve en `npm run dev`, pero **no se publica** en el build de producción. Es útil para notas a medio hacer.

## Despliegue

### Vercel

1. Sube el repositorio a GitHub.
2. En Vercel, **Add New → Project** e importa el repositorio.
3. Vercel detecta Astro automáticamente. Build: `npm run build`. Output: `dist`.

No hace falta configurar nada más.

### GitHub Pages

1. Sube el repositorio a GitHub.
2. Ve a **Settings → Pages** y elige **GitHub Actions** como origen.
3. Haz `push` a `main`. El workflow de `.github/workflows/deploy.yml` compila y publica.

El workflow pasa automáticamente el `BASE_PATH` correcto (por ejemplo `/cuaderno`) y la URL del sitio. Si algún día usas un dominio propio, seguirá funcionando.

Para probar el build con la misma base que GitHub Pages:

```powershell
$env:BASE_PATH="/cuaderno"
npm run build
npm run preview
```

## Personalización

- **Nombre, descripción y autoría**: `src/lib/site.ts`.
- **Colores, tipografía y espaciados**: variables CSS al inicio de `src/styles/global.css`.
- **Tema inicial**: claro u oscuro según el sistema; el botón de la cabecera lo guarda en `localStorage`.
- **Contenido de ejemplo**: puedes borrar las asignaturas y apuntes de ejemplo y empezar con los tuyos.

## Ideas para más adelante

- Buscador local con [Pagefind](https://pagefind.app/).
- Soporte de fórmulas con KaTeX (plugin de Markdown).
- Página de etiquetas y RSS.
- Enlaces a PDFs o ejercicios resueltos.
