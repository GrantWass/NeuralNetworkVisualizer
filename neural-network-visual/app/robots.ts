import type { MetadataRoute } from "next";

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      {
        userAgent: "*",
        allow: "/",
      },
    ],
    sitemap: "https://www.nn-visual.com/sitemap.xml",
    host: "https://www.nn-visual.com",
  };
}
