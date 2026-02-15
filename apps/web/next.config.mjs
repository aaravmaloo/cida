import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rawApiBase = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000").trim();
const apiBaseWithProtocol = /^https?:\/\//i.test(rawApiBase) ? rawApiBase : `https://${rawApiBase}`;
const apiBase = apiBaseWithProtocol.replace(/\/+$/, "");

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  typedRoutes: true,
  outputFileTracingRoot: path.join(__dirname, "../.."),
  async rewrites() {
    return [
      {
        source: "/v1/:path*",
        destination: `${apiBase}/v1/:path*`,
      },
      {
        source: "/healthz",
        destination: `${apiBase}/healthz`,
      },
      {
        source: "/readyz",
        destination: `${apiBase}/readyz`,
      },
    ];
  },
};

export default nextConfig;

