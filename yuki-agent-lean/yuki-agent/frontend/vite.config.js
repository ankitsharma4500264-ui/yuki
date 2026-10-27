import { defineConfig } from "vite";

// JSX is compiled by Vite's built-in esbuild, so no extra React plugin is needed.
// Dev: /api is proxied to the local backend, so no CORS setup is needed.
export default defineConfig({
  esbuild: { jsx: "automatic" },
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8000" } },
});
