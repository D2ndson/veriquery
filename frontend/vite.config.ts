import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` proxies the API to the FastAPI server on :8080
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://localhost:8080", "/results-files": "http://localhost:8080" } },
  build: { chunkSizeWarningLimit: 4000 },
});
