import { defineConfig } from "@playwright/test";

const port = Number(process.env.DRAGON_E2E_PORT || 5174);
if (!Number.isInteger(port) || port < 1024 || port > 65535) {
  throw new Error("DRAGON_E2E_PORT must be an integer between 1024 and 65535");
}
const origin = `http://127.0.0.1:${port}`;

export default defineConfig({
  testDir: "./e2e",
  outputDir: "./test-results",
  timeout: 45000,
  workers: 2,
  use: {
    baseURL: origin,
    channel: "chrome",
    headless: true,
    trace: "retain-on-failure",
  },
  webServer: {
    command: `pnpm dev --port ${port} --strictPort`,
    url: `${origin}/next`,
    reuseExistingServer: false,
  },
});
