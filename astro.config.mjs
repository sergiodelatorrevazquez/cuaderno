import { defineConfig } from 'astro/config';

// En GitHub Pages, BASE_PATH lo inyecta el workflow (por ejemplo "/cuaderno").
// En Vercel o en local se queda en "/".
const base = process.env.BASE_PATH || '/';

export default defineConfig({
  site: process.env.SITE_URL,
  base,
  trailingSlash: 'ignore',
  compressHTML: true,
  build: {
    format: 'directory',
  },
  markdown: {
    shikiConfig: {
      theme: 'github-dark',
    },
  },
});
