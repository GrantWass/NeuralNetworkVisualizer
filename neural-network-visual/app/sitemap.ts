import type { MetadataRoute } from 'next'

export default function sitemap(): MetadataRoute.Sitemap {
  return [
    {
      url: 'https://nn-visual.com',
      lastModified: new Date(),
      changeFrequency: 'weekly',
      priority: 1,
    },
    {
      url: 'https://nn-visual.com/attention',
      lastModified: new Date('2026-05-20'),
      changeFrequency: 'weekly',
      priority: 0.8,
    },
    {
      url: 'https://nn-visual.com/transformers',
      lastModified: new Date('2026-05-21'),
      changeFrequency: 'weekly',
      priority: 0.8,
    },
    {
      url: 'https://nn-visual.com/about',
      lastModified: new Date('2026-05-21'),
      changeFrequency: 'monthly',
      priority: 0.5,
    },
  ]
}
