import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Turbopack (dev server): shim Node-only modules so default imports return {} not undefined.
  // @xenova/transformers checks isEmpty(fs) via Object.keys — throws if fs is undefined.
  turbopack: {
    resolveAlias: {
      "sharp": { browser: "./shims/empty.js" },
      "onnxruntime-node": { browser: "./shims/empty.js" },
      "fs": { browser: "./shims/empty.js" },
      "path": { browser: "./shims/empty.js" },
      "url": { browser: "./shims/empty.js" },
    },
  },
  // Webpack (production build): exclude Node-only modules from the browser bundle
  webpack: (config) => {
    config.resolve.alias = {
      ...config.resolve.alias,
      "sharp$": false,
      "onnxruntime-node$": false,
    };
    return config;
  },
};

export default nextConfig;
