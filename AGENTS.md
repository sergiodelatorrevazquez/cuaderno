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
   - Descripción corta para la tarjeta de la portada (opcional)
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
   descripcion: "Descripción corta de la asignatura."
   color: "#6366f1"
   orden: 0
   ---
   ```
   
   - `nombre`: obligatorio
   - `codigo`: opcional
   - `curso`: string, por defecto `2026/2027`
   - `semestre`: número 1 o 2, por defecto `1`
   - `descripcion`: string, por defecto `""`
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


## Agente 2: Convertidor PDF a apunte detallado

**ID:** `agente-pdf-a-apunte`  
**Tipo:** Agente principal/subagente  
**Propósito:** Convertir un PDF de clase en un apunte detallado en español, sin perder información, listo para añadir al cuaderno.

### Cuándo usarlo
Cuando tengas un PDF (transparencias, apuntes, resumen de clase) y quieras pasarlo a limpio como apunte en una asignatura existente.

### Requisitos críticos
- **Traducción obligatoria**: Si el PDF está en inglés, tradúcelo completo y fielmente al español. Conserva terminología técnica correcta cuando sea necesario, pero prioriza claridad en español.
- **Máximo detalle**: Debes ser extremadamente minucioso. No omitas definiciones, teoremas, fórmulas, demostraciones, tablas, diagramas descritos, ejemplos, ejercicios, anotaciones del profesor o notas al pie. Cualquier omisión puede ser crítica.
- **Fiel al contenido**: No inventes información. Si algo no está claro en el PDF, pregunta.
- **Adaptado al formato**: Genera Markdown compatible con el sitio (GitHub Flavored Markdown). Ten en cuenta que **NO hay renderizador KaTeX/MathJax** instalado actualmente - usa código inline o bloques de código `text` para fórmulas matemáticas.

### Flujo de trabajo

1. **Recopilar información inicial**
   Pregunta al usuario:
   - ¿A qué asignatura pertenece? (debe existir en `src/content/asignaturas/`). Si no existe, sugiere usar el Agente 1 primero.
   - Ruta o archivo PDF a convertir (ruta local)
   - ¿Nombre/título del apunte que quieres que tenga? (si no lo indica, propón uno basado en el contenido)
   - ¿Fecha del apunte? (formato `YYYY-MM-DD`, obligatorio para el frontmatter)
   - ¿Tema/número de tema? (opcional, ej: "Tema 3")
   - ¿Etiquetas/tags? (opcional, lista separada por comas)
   - ¿Resumen/entradilla? (opcional)
   - ¿Es un borrador? (por defecto `false` - solo marcar como borrador si explícitamente se pide)

2. **Analizar el PDF**
   - Lee el PDF completo. Si es largo, analízalo por secciones.
   - Detecta el idioma (español/inglés/otros). Si es inglés, planifica traducción.
   - Identifica estructura: índices, títulos, subtítulos, listas, tablas, figuras, fórmulas, código, ejercicios
   - Extrae TODO el contenido sin resumir excesivamente. El objetivo es un apunte completo y detallado, no un resumen

3. **Preguntar aclaraciones si es necesario**
   Antes de generar el archivo, pregunta sobre cualquier ambigüedad:
   - Conceptos poco claros o con abreviaturas no definidas
   - Fórmulas que no se pueden transcribir correctamente al formato disponible
   - Diagramas/figuras: pide descripción o indica que se mencionen textualmente si es posible
   - Ejercicios/resoluciones incompletas
   - Cualquier información que falte para que los apuntes sean completos

4. **Generar el apunte**
   Crea el archivo en: `src/content/apuntes/<id-asignatura>/<slug-apunte>.md`
   
   - **ID de asignatura**: nombre del archivo sin `.md` de la asignatura elegida
   - **Slug del apunte**: generar desde el título (minúsculas, guiones, sin tildes). Si ya existe un archivo con ese nombre, añadir sufijo numérico o preguntar otro nombre
   
   **Frontmatter obligatorio:**
   ```yaml
   ---
   titulo: "Título completo del apunte"
   asignatura: <id-asignatura>  # referencia al archivo de asignatura
   fecha: YYYY-MM-DD
   tema: "Tema X"  # opcional
   tags: [tag1, tag2]  # opcional, por defecto []
   resumen: "Entradilla breve"  # opcional
   borrador: false  # opcional, por defecto false
   ---
   ```
   
   Respetando el esquema de `src/content.config.ts`:
   - `titulo`: string, obligatorio
   - `asignatura`: referencia a colección `asignaturas`, obligatorio
   - `fecha`: coerce a date, obligatorio (formato YYYY-MM-DD)
   - `tema`: string opcional
   - `tags`: array de strings, por defecto `[]`
   - `resumen`: string opcional
   - `borrador`: boolean, por defecto `false`

5. **Formatear el contenido Markdown**
   - Usa estructura clara con encabezados (`##`, `###`, `####`) jerárquicos
   - Convierte tablas del PDF a tablas Markdown
   - Para listas, usa listas Markdown apropiadas
   - Para **fórmulas matemáticas**: NO uses `$...$` o `$$...$$`. Usa código inline con backticks para fórmulas cortas o bloques de código `text` para fórmulas más complejas. Ejemplo:
     ```text
     f(n) = g(n) + h(n)
     ```
   - Para código/algoritmos, usa bloques con lenguaje si se identifica (`python`, `java`, etc.)
   - Conserva definiciones importantes, resaltándolas con **negrita** o blockquotes cuando tenga sentido
   - Incluye ejemplos tal como aparecen
   - Si hay ejercicios, inclúyelos completos
   - Usa listas de tareas `- [ ]` solo si aparecen explícitamente como tal

6. **Traducción y fidelidad**
   - Traduce TODO al español si el original estaba en inglés
   - Mantén terminología técnica estándar en español cuando exista; si es muy específica del área y no hay traducción clara, puedes mantener el término original entre paréntesis
   - No parafrasees para "resumir" - expande y clarifica solo si ayuda a la comprensión sin perder información
   - Conserva nombres propios, nombres de algoritmos reconocidos (A*, BFS, etc.) tal como son

7. **Verificación final**
   - Revisa que todo el contenido del PDF relevante ha sido transferido
   - Verifica el frontmatter contra el esquema
   - Ejecuta `npm run check` para validar tipos y contenido
   - Ejecuta `npm run build` para asegurar que compila correctamente
   - Muestra al usuario la ruta del archivo creado y un resumen de lo generado

### Reglas importantes
- **NUNCA resumas** - Si el PDF tiene 10 páginas con detalles, el apunte debe reflejar esos detalles
- **Pregunta siempre** si hay cualquier duda sobre contenido, formato o metadatos
- **Sin KaTeX** - Respeta la limitación actual del proyecto (no hay soporte matemático con LaTeX)
- **Un apunte por archivo** - Un PDF = un archivo `.md` (a menos que el usuario indique explícitamente lo contrario)
- **Respeta borradores** - Por defecto `borrador: false`. Solo usar `true` si se solicita expresamente


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
- En producción los borradores (`borrador: true`) no se muestran
- Las URLs respetan el `base` path para GitHub Pages
- El sitio está en español, mantén ese idioma en todo el contenido generado
- Sigue el estilo de los apuntes existentes (ver ejemplos en `src/content/apuntes/`)
