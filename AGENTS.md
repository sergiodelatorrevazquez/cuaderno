# Agentes para el Cuaderno MUII

Este archivo define dos agentes reutilizables para trabajar con este proyecto. Puedes invocarlos desde OpenCode para automatizar la creación de asignaturas y la conversión de PDFs a apuntes.

## Agente 1: Configurador de asignaturas

**ID:** `agente-configurador-asignaturas`  
**Tipo:** Agente principal/subagente  
**Propósito:** Crear la sección/ficha de una nueva asignatura en la aplicación, preguntando toda la información necesaria.

### Cuándo usarlo
Cuando quieras añadir una nueva asignatura al cuaderno (por ejemplo, al inicio de un cuatrimestre).

### Flujo de trabajo

1. **Recopilar información básica**
   Pregunta al usuario:
   - Nombre oficial de la asignatura (obligatorio). Ej: "Aprendizaje Automático"
   - Código de la asignatura si lo conoce (opcional). Ej: "MUII-AA"
   - ¿Pertenece al curso 2026/2027 o a otro? (por defecto: `2026/2027`)
   - ¿Semestre 1 o 2? (por defecto: `1`)
   - Color en formato `#rrggbb` para el acento (opcional, por defecto `#6366f1`)
   - Orden en la portada (opcional, numérico, menor = aparece primero)

2. **Generar identificador (slug)**
   - Convierte el nombre a minúsculas
   - Reemplaza espacios y acentos por guiones
   - Elimina caracteres no alfanuméricos
   - Usa ese slug como nombre de archivo: `src/content/asignaturas/<slug>.md`

3. **Crear la ficha de asignatura**
   Crea el archivo con el siguiente frontmatter (respetando el esquema de `src/content.config.ts`):
   
   ```yaml
   ---
   nombre: "Nombre de la Asignatura"
   codigo: "MUII-XX"  # opcional, omitir si no se indica
   curso: "2026/2027"
   semestre: 1
   color: "#6366f1"
   orden: 0
   ---
   ```
   
   - `nombre`: obligatorio
   - `codigo`: opcional
   - `curso`: string, por defecto `2026/2027`
   - `semestre`: número 1 o 2, por defecto `1`
   - `color`: regex `^#[0-9a-fA-F]{6}$`, por defecto `#6366f1`
   - `orden`: número, por defecto `0`

4. **Verificar y crear estructura**
   - Verifica que no existe ya un archivo con ese slug
   - Crea el directorio `src/content/apuntes/<slug>/` si no existe (aunque esté vacío, es útil para organizar)
   - Informa al usuario con la ruta creada y cómo se verá en la web

5. **Validación**
   - Ejecuta `npm run check` para verificar que el contenido es válido según los esquemas de Content Collections
   - Si hay errores, corrígelos y vuelve a verificar

### Reglas importantes
- Siempre pregunta antes de crear. No asumas valores sin confirmación.
- Respeta estrictamente el esquema definido en `src/content.config.ts`
- Usa nombres de archivo en minúsculas, con guiones, sin espacios ni tildes
- El ID de la asignatura es el nombre del archivo sin `.md`


## Agente 2: Transcriptor fiel de PDF a apunte página por página

**ID:** `agente-pdf-a-apunte`  
**Tipo:** Agente principal/subagente  
**Propósito:** Convertir un PDF de clase en un apunte exhaustivo en español, organizado página por página y fiel al documento, listo para añadir al cuaderno.

### Cuándo usarlo
Cuando tengas un PDF (transparencias, apuntes, resumen de clase) y quieras pasarlo a limpio como apunte en una asignatura existente.

### Requisitos críticos
- **Una sección por cada página**: El apunte debe tener una sección Markdown independiente por cada página del PDF, en el mismo orden y con el número de página original. Usa `## Página 1 — título de la página`, `## Página 2 — título de la página`, etc. Si una página no tiene título, usa `## Página N`.
- **Cobertura completa**: Transfiere todo el contenido informativo de cada página, no solo el texto extraído. Incluye títulos, subtítulos, párrafos, listas, definiciones, argumentos, teoremas, demostraciones, fórmulas, símbolos, ejemplos, ejercicios y soluciones, tablas, pies de figura, referencias, notas al pie, leyendas y anotaciones visibles. Conserva cifras, unidades, nombres, condiciones y relaciones entre elementos.
- **Figuras y gráficos**: Inspecciona visualmente cada página del PDF, además de extraer su texto. Describe cada imagen, diagrama, esquema, gráfico y captura con suficiente detalle para conservar la información que comunica: elementos, etiquetas, ejes, escalas, leyendas, relaciones, tendencias, valores legibles y conclusiones explícitas. Para tablas, conserva filas, columnas, encabezados, valores y unidades. No afirmes que has inspeccionado una figura si solo dispones de extracción de texto.
- **Fórmulas y código**: Transcribe las fórmulas con todos sus términos, subíndices, superíndices, límites y condiciones; conserva los bloques de código y su lenguaje cuando sea identificable. Comprueba visualmente las expresiones cuya extracción pueda haber alterado símbolos o estructura.
- **Traducción fiel**: Si el PDF no está en español, tradúcelo íntegramente. Mantén el significado, el grado de certeza y la terminología técnica; conserva nombres propios, métodos y términos originales entre paréntesis cuando ayuden a identificarlos. No sustituyas traducción por resumen.
- **Sin invenciones ni omisiones silenciosas**: No añadas explicaciones, ejemplos, conclusiones ni datos que no estén en el PDF como si formaran parte de él. Si un elemento es ilegible, ambiguo o no se puede recuperar, identifica la página y el elemento concreto y pregunta al usuario antes de dar el apunte por completo.
- **Fidelidad de organización**: Mantén el contenido en la sección de la página en que aparece. No fusiones páginas, no agrupes varias páginas en un único apartado y no reordenes la secuencia del PDF.
- **Formato compatible con el sitio**: Usa Markdown compatible con GitHub Flavored Markdown. El sitio tiene `remark-math` y KaTeX configurados: representa matemáticas con `$...$` y `$$...$$`, comprobando que cada delimitador y expresión estén correctamente cerrados.

### Flujo de trabajo

1. **Recopilar información inicial**
   Pregunta al usuario:
   - ¿A qué asignatura pertenece? (debe existir en `src/content/asignaturas/`). Si no existe, sugiere usar el Agente 1 primero.
   - Ruta o archivo PDF a convertir (ruta local)
   - ¿Nombre/título del apunte que quieres que tenga? (si no lo indica, propón uno basado en el contenido)
   - ¿Fecha del apunte? (formato `YYYY-MM-DD`, obligatorio para el frontmatter)
   - Asigna el siguiente índice entero libre de esa asignatura para que el apunte aparezca al final de su lista
   - ¿Tema/número de tema? (opcional, ej: "Tema 3")
   - ¿Etiquetas/tags? (opcional, lista separada por comas)
   - ¿Resumen/entradilla? (opcional)

2. **Leer e inventariar el PDF completo**
   - Abre el archivo local, comprueba el acceso a todas sus páginas y determina el número total antes de redactar.
   - Recorre todas las páginas en orden, incluidas portada, referencias, anexos y páginas que parezcan contener solo figuras.
   - Extrae el texto página por página e inspecciona visualmente cada página para identificar lo que la extracción no captura: imágenes, ecuaciones, diagramas, gráficos, tablas y anotaciones.
   - Si el PDF es largo, trabaja por bloques y registra qué páginas has leído y transcrito. Al final, comprueba la cobertura desde la primera hasta la última.
   - Identifica idioma, jerarquía, contenido y relación entre texto y elementos visuales de cada página. Traduce al español al redactar.
   - No empieces a redactar basándote únicamente en el nombre del archivo, el índice, una muestra de páginas o una extracción parcial.

3. **Preguntar aclaraciones si es necesario**
   Antes de escribir el apunte, pregunta por cada problema que impida una transcripción fiel:
   - Página y fragmento ilegible, cortado, ambiguo o ausente.
   - Símbolo, subíndice, superíndice, fórmula o valor que no se pueda distinguir con seguridad.
   - Información de una figura, gráfico, tabla o anotación que no se pueda recuperar visualmente.
   - No pidas al usuario que describa una figura que tú puedas inspeccionar abriendo o renderizando localmente la página del PDF.
   - Si no existe forma de acceder a un elemento, indícalo expresamente y no presentes el apunte como una transcripción completa.

4. **Generar el apunte**
   Crea el archivo en: `src/content/apuntes/<id-asignatura>/<slug-apunte>.md`
   
   - **ID de asignatura**: nombre del archivo sin `.md` de la asignatura elegida
   - **Slug del apunte**: generar desde el título (minúsculas, guiones, sin tildes). Si ya existe un archivo con ese nombre, añadir sufijo numérico o preguntar otro nombre
   - Asigna el siguiente `orden` libre dentro de la asignatura.
   - Después del frontmatter, crea una sección `## Página N — título` por cada página del PDF. No combines varias páginas en una sección.
   - Dentro de cada sección, transfiere el contenido de esa página en su orden original. Describe los elementos visuales en el mismo contexto e indica qué elemento estás describiendo.
   - Conserva los títulos y la jerarquía del original cuando sea posible. Si una frase, tabla, demostración o lista continúa en otra página, indícalo sin eliminar la separación obligatoria por páginas.
   - Traduce al español sin resumir ni reinterpretar lo que afirma el original.
   
   **Frontmatter obligatorio:**
   ```yaml
   ---
   titulo: "Título completo del apunte"
   asignatura: <id-asignatura>  # referencia al archivo de asignatura
   fecha: YYYY-MM-DD
   orden: 1  # entero positivo único dentro de la asignatura
   tema: "Tema X"  # opcional
   tags: [tag1, tag2]  # opcional, por defecto []
   resumen: "Entradilla breve"  # opcional
   ---
   ```
   
   Respetando el esquema de `src/content.config.ts`:
   - `titulo`: string, obligatorio
   - `asignatura`: referencia a colección `asignaturas`, obligatorio
   - `fecha`: coerce a date, obligatorio (formato YYYY-MM-DD)
   - `orden`: entero positivo obligatorio, único dentro de la asignatura; determina la posición en las listas
   - `tema`: string opcional
   - `tags`: array de strings, por defecto `[]`
   - `resumen`: string opcional

5. **Formatear el contenido Markdown**
   - Cada página debe comenzar con `## Página N — título original traducido`; usa `## Página N` si no tiene título
   - Mantén encabezados jerárquicos subordinados dentro de cada sección de página
   - Convierte tablas del PDF a tablas Markdown
   - Para listas, usa listas Markdown apropiadas
   - Para matemáticas, utiliza `$...$` y `$$...$$`, ya que el sitio renderiza las fórmulas con KaTeX. Conserva la expresión íntegra y comprueba que los delimitadores estén cerrados.
   - Describe las figuras con sus rótulos, valores y relaciones visibles; no las sustituyas por frases genéricas como «se muestra un gráfico».
   - Para código/algoritmos, usa bloques con lenguaje si se identifica (`python`, `java`, etc.)
   - Conserva el énfasis, las definiciones y las etiquetas que reflejen el original; no cambies el sentido al dar formato
   - Incluye ejemplos, ejercicios y soluciones completos, en la página donde aparecen
   - Usa listas de tareas `- [ ]` solo si aparecen explícitamente como tal

6. **Traducción y fidelidad**
   - Traduce TODO el contenido al español si el original está en otro idioma, página por página
   - Mantén terminología técnica estándar en español cuando exista; si es muy específica del área y no hay traducción clara, puedes mantener el término original entre paréntesis
   - No resumas, no suprimas repeticiones informativas y no añadas explicaciones propias dentro de la transcripción
   - Conserva nombres propios, nombres de algoritmos reconocidos (A*, BFS, etc.) tal como son

7. **Verificación final**
   - Comprueba que existen tantas secciones `## Página N` como páginas tenga el PDF, numeradas consecutivamente desde la primera hasta la última.
   - Compara el PDF con el apunte página por página. Revisa texto, fórmulas, código, tablas, imágenes, gráficos, leyendas, valores, notas y referencias.
   - Revisa expresamente las páginas iniciales y finales, las que tienen poco texto y las continuaciones entre páginas.
   - Comprueba que no quedan marcadores provisionales, contenido ilegible presentado como seguro ni material inventado.
   - Si alguna página o figura no se pudo verificar, describe la limitación y pide lo necesario; no afirmes que la cobertura es completa.
   - Verifica el frontmatter contra el esquema
   - Ejecuta `npm run check` para validar tipos y contenido
   - Ejecuta `npm run build` para asegurar que compila correctamente
   - Muestra al usuario la ruta del archivo creado y un resumen de lo generado

### Reglas importantes
- **Una página del PDF = una sección del apunte** - No combines ni omitas páginas; verifica la numeración completa
- **NUNCA resumas ni omitas contenido informativo** - Conserva los detalles textuales y visuales de todas las páginas
- **Inspecciona texto e imágenes** - La extracción textual por sí sola no basta para cubrir gráficos, fórmulas, diagramas, tablas o anotaciones
- **Pregunta siempre** si hay cualquier duda sobre contenido, formato o metadatos
- **KaTeX está configurado** - Usa delimitadores matemáticos Markdown y comprueba su renderizado
- **Un apunte por archivo** - Un PDF = un archivo `.md` (a menos que el usuario indique explícitamente lo contrario)
- **Todos los apuntes se publican** - No añadas un campo `borrador`
- **Orden por asignatura** - Asigna el siguiente índice libre para insertar el apunte al final; la fecha es solo un metadato visible


## Uso general

### Invocación
Desde OpenCode, puedes pedir:
- *"Usa el agente configurador de asignaturas para añadir [nombre asignatura]"*
- *"Usa el agente PDF a apunte para convertir [ruta.pdf] en apunte de [asignatura]"*

### Comandos útiles del proyecto
```bash
npm run dev      # Servidor de desarrollo
npm run build    # Build de producción
npm run check    # Verificación de tipos y contenido
npm run preview  # Previsualizar build
```

### Notas importantes
- Los IDs se generan automáticamente: asignatura = nombre archivo sin `.md`, apunte = ruta relativa sin `.md`
- Todos los apuntes se publican; no existe el campo `borrador`
- Las URLs respetan el `base` path para GitHub Pages
- El sitio está en español, mantén ese idioma en todo el contenido generado
- Sigue el estilo de los apuntes existentes (ver ejemplos en `src/content/apuntes/`)
