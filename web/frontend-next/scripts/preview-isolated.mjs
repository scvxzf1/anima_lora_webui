import { chromium } from "@playwright/test";
import { createServer } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { fileURLToPath } from "node:url";
import { mockWorkspace } from "../e2e/fixtures.ts";

const root = fileURLToPath(new URL("../", import.meta.url));
const port = Number(process.env.DRAGON_PREVIEW_PORT || 20521);
if (!Number.isInteger(port) || port < 1024 || port > 65535) throw new Error("Invalid preview port");
const routes = [];
const browser = await chromium.launch({ channel: "chrome", headless: true });
try {
  const page = await browser.newPage();
  await mockWorkspace({
    evaluate: (callback) => page.evaluate(callback),
    routeWebSocket: async () => {},
    route: async (matches, handler) => routes.push({ matches, handler }),
  });
} finally {
  await browser.close();
}

const server = await createServer({
  root,
  configFile: false,
  plugins: [react(), tailwindcss(), {
    name: "isolated-readonly-fixtures",
    configureServer(vite) {
      vite.middlewares.use(async (request, response, next) => {
        const url = new URL(request.url || "/", `http://127.0.0.1:${port}`);
        if (!url.pathname.startsWith("/api/") && !url.pathname.startsWith("/ws/")) return next();
        response.setHeader("Cache-Control", "no-store");
        const fulfill = async ({ json, body, status = 200, contentType = "application/json" }) => {
          response.writeHead(status, { "Content-Type": contentType });
          response.end(json === undefined ? body : JSON.stringify(json));
        };
        if (request.method !== "GET") return fulfill({ status: 403, json: { error: "Isolated preview: writes are disabled" } });
        const route = routes.find(({ matches }) => matches(url));
        if (!route) return fulfill({ status: 404, json: { error: "No preview fixture" } });
        try {
          await route.handler({ request: () => ({ url: () => url.href, method: () => "GET" }), fulfill });
        } catch {
          if (!response.headersSent) await fulfill({ status: 500, json: { error: "Preview fixture failed" } });
          else response.end();
        }
      });
      vite.httpServer.on("upgrade", (request, socket) => {
        if (request.url?.startsWith("/ws/")) socket.destroy();
      });
    },
  }],
  server: { host: "127.0.0.1", port, strictPort: true },
});
await server.listen();
console.log(`Readonly synthetic preview: http://127.0.0.1:${port}/next/history/fixture-run`);
for (const signal of ["SIGINT", "SIGTERM"]) process.once(signal, async () => { await server.close(); process.exit(0); });
