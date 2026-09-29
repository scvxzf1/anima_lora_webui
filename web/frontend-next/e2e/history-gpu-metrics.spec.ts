import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

const start = Date.UTC(2026, 8, 27, 8, 0, 0) / 1000;
const system = [
  { ts: start, vram_used_gb: 6, gpu_util: 90, gpu_temp: 60, per_gpu: [
    { index: 0, uuid: "GPU-a", name: "GPU Alpha", vram_used_gb: 4, gpu_util: 90, gpu_temp: 60 },
    { index: 1, uuid: "GPU-b", name: "GPU Beta", vram_used_gb: 2, gpu_util: 40, gpu_temp: 51 },
  ] },
  { ts: start + 2, vram_used_gb: 7, gpu_util: 85, gpu_temp: 63, per_gpu: [
    { index: 0, uuid: "GPU-a", name: "GPU Alpha", vram_used_gb: 5, gpu_util: 85, gpu_temp: 63 },
    { index: 1, uuid: "GPU-b", name: "GPU Beta", vram_used_gb: 2, gpu_util: 0, gpu_temp: 49 },
  ] },
];

for (const width of [1440, 390]) {
  test(`history GPU charts switch between aggregate and physical devices at ${width}px`, async ({ page }, info) => {
    await page.setViewportSize({ width, height: 900 });
    const mocks = await mockWorkspace(page);
    await page.route((url) => url.pathname === "/api/training/history/fixture-run", (route) => route.fulfill({ json: {
      task: { id: "fixture-run", job: "training", state: "done", name: "Multi-GPU run", gpu_whitelist: [0, 1] },
      metrics: [{ step: 1, loss: 0.2 }, { step: 2, loss: 0.18 }], system,
      limits: { system_total: 2 },
    } }));
    await page.goto("/next/history/fixture-run?view=metrics");
    const region = page.getByRole("region", { name: "GPU 资源历史" });
    await expect(region.getByRole("group", { name: "GPU 资源范围" }).getByRole("button")).toHaveCount(3);
    await expect(region.getByRole("button", { name: "汇总" })).toHaveAttribute("aria-pressed", "true");
    await expect(region.getByRole("group", { name: "显存 (GB)，2 个点" })).toBeVisible();
    await region.getByRole("button", { name: "GPU 1" }).click();
    await expect(region).toContainText("GPU 1 · GPU Beta · 任务已选");
    const utilization = region.getByRole("group", { name: "GPU 利用率 (%)，2 个点" });
    await expect(utilization).toBeVisible();
    await utilization.focus();
    await page.keyboard.press("ArrowRight");
    await expect(region.getByText(/检查点 1\/2 · 时间: .* · GPU 利用率 \(%\): 40\.00/, { exact: false })).toBeVisible();
    await page.keyboard.press("ArrowRight");
    await expect(region.getByText(/检查点 2\/2 · 时间: .* · GPU 利用率 \(%\): 0\.000/, { exact: false })).toBeVisible();
    await expect(region.getByRole("group", { name: "GPU 温度 (°C)，2 个点" })).toBeVisible();
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.screenshot({ path: info.outputPath(`history-gpu-${width}.png`), fullPage: true });
    expect(mocks.writes).toEqual([]);
    expect(mocks.unhandled).toEqual([]);
  });
}

test("legacy history keeps its aggregate chart without inventing GPU identities", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  await page.route((url) => url.pathname === "/api/training/history/fixture-run", (route) => route.fulfill({ json: {
    task: { id: "fixture-run", job: "training", state: "done", gpu_whitelist: [1] },
    metrics: [], system: [{ ts: start, gpu_index: 1, vram_used_gb: 11.2 }, { ts: start + 2, gpu_index: 1, vram_used_gb: 12 }],
  } }));
  await page.goto("/next/history/fixture-run?view=metrics");
  const region = page.getByRole("region", { name: "GPU 资源历史" });
  await expect(region).toContainText("仅有汇总记录");
  await expect(region.getByRole("group", { name: "GPU 资源范围" })).toHaveCount(0);
  await expect(region.getByRole("group", { name: "显存 (GB)，2 个点" })).toBeVisible();
  await expect(region.getByRole("group", { name: /GPU 利用率/ })).toHaveCount(0);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("history GPU charts expose an empty state when no system metrics were recorded", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  await page.route((url) => url.pathname === "/api/training/history/fixture-run", (route) => route.fulfill({ json: {
    task: { id: "fixture-run", job: "training", state: "done" },
    metrics: [{ step: 1, loss: 0.2 }],
    system: [],
  } }));
  await page.goto("/next/history/fixture-run?view=metrics");
  const region = page.getByRole("region", { name: "GPU 资源历史" });
  await expect(region.getByText("暂无 GPU 资源记录")).toBeVisible();
  await expect(region.getByRole("group", { name: "GPU 资源范围" })).toHaveCount(0);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});
