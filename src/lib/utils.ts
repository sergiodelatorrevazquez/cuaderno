const formatoFecha = new Intl.DateTimeFormat('es-ES', {
  day: 'numeric',
  month: 'long',
  year: 'numeric',
});

export function formatDate(fecha: Date): string {
  return formatoFecha.format(fecha);
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
