import { defineConfig } from 'astro/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

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
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    shikiConfig: {
      theme: 'github-dark',
    },
  },
});
