import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";
import tailwindcss from "@tailwindcss/vite";
import process from "node:process";

export default defineConfig(({ command }) => ({
  base: command === "build" ? "/static/dragon-next/" : "/",
  plugins: [react(), tailwindcss()],
  build: {
    outDir: "../static/dragon-next",
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": process.env.DRAGON_API_TARGET || "http://127.0.0.1:20203",
      "/ws": {
        target: process.env.DRAGON_API_TARGET || "http://127.0.0.1:20203",
        ws: true,
      },
    },
  },
  test: {
    include: ["src/**/*.test.{ts,tsx}"],
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
  },
}));
