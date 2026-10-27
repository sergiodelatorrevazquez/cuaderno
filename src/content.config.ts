import { defineCollection, reference } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const asignaturas = defineCollection({
  loader: glob({ base: './src/content/asignaturas', pattern: '**/*.md' }),
  schema: z.object({
    nombre: z.string(),
    codigo: z.string().optional(),
    curso: z.string().default('2026/2027'),
    semestre: z.number().int().min(1).max(2).default(1),
    color: z
      .string()
      .regex(/^#[0-9a-fA-F]{6}$/, 'El color debe tener el formato #rrggbb')
      .default('#6366f1'),
    orden: z.number().default(0),
  }),
});

const apuntes = defineCollection({
  loader: glob({ base: './src/content/apuntes', pattern: '**/*.md' }),
  schema: z.object({
    titulo: z.string(),
    asignatura: reference('asignaturas'),
    fecha: z.coerce.date(),
    tema: z.string().optional(),
    tags: z.array(z.string()).default([]),
    resumen: z.string().optional(),
    borrador: z.boolean().default(false),
  }),
});

export const collections = { asignaturas, apuntes };
