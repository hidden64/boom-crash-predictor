import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));

/** @type {import('next').NextConfig} */
const nextConfig = {
  // Fixe la racine workspace pour Turbopack (évite la confusion avec un lockfile parent).
  turbopack: {
    root: __dirname,
  },
};

export default nextConfig;
