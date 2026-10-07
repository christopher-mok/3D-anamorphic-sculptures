import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// FastAPI backend (see docs/api.md). `/api` includes the job WebSocket.
const BACKEND = "http://localhost:8000";

const proxy = {
  "/api": { target: BACKEND, changeOrigin: true, ws: true },
  "/outputs": { target: BACKEND, changeOrigin: true },
};

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy },
  preview: { port: 4173, proxy },
  build: { chunkSizeWarningLimit: 2000 },
});
