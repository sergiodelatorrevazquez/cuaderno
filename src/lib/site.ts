export const SITE = {
  title: 'Cuaderno MUII',
  description:
    'Apuntes del Máster Universitario en Ingeniería Informática de la Universidad de Málaga.',
  author: 'Sergio',
};

/**
 * Construye una URL interna teniendo en cuenta el `base` de Astro.
 * Sirve tanto en local (`/`) como en GitHub Pages (`/cuaderno`).
 */
export function url(path = '/'): string {
  const base = import.meta.env.BASE_URL.replace(/\/+$/, '');
  const limpio = path.replace(/^\/+/, '');
  return limpio ? `${base}/${limpio}` : `${base}/`;
}
