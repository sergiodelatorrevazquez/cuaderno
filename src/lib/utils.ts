import type { CollectionEntry } from 'astro:content';

const formatoFecha = new Intl.DateTimeFormat('es-ES', {
  day: 'numeric',
  month: 'long',
  year: 'numeric',
});

export function formatDate(fecha: Date): string {
  return formatoFecha.format(fecha);
}

export function ordenarApuntes(apuntes: CollectionEntry<'apuntes'>[]): CollectionEntry<'apuntes'>[] {
  const ordenesPorAsignatura = new Map<string, Set<number>>();

  for (const apunte of apuntes) {
    const asignaturaId = apunte.data.asignatura.id;
    const ordenes = ordenesPorAsignatura.get(asignaturaId) ?? new Set<number>();
    if (ordenes.has(apunte.data.orden)) {
      throw new Error(`El orden ${apunte.data.orden} está duplicado en la asignatura "${asignaturaId}".`);
    }
    ordenes.add(apunte.data.orden);
    ordenesPorAsignatura.set(asignaturaId, ordenes);
  }

  return [...apuntes].sort(
    (a, b) =>
      a.data.asignatura.id.localeCompare(b.data.asignatura.id) || a.data.orden - b.data.orden,
  );
}

export function readingTime(body?: string): number {
  if (!body) return 1;
  const palabras = body
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/\|/g, ' ')
    .trim()
    .split(/\s+/)
    .filter(Boolean).length;
  return Math.max(1, Math.round(palabras / 220));
}

export function pluralizar(n: number, singular: string, plural: string): string {
  return `${n} ${n === 1 ? singular : plural}`;
}
