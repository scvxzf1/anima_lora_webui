import { expect, test, type Page } from "@playwright/test";
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
    await mockSystemPages(page, system);
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
  await mockSystemPages(page, [{ ts: start, vram_used_gb: 11.2 }, { ts: start + 2, vram_used_gb: 12 }]);
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
  await mockSystemPages(page, []);
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

async function mockSystemPages(page: Page, points: Record<string, unknown>[]) {
  await page.route((url) => url.pathname === "/api/training/history/fixture-run/system", (route) => {
    const offset = Number(new URL(route.request().url()).searchParams.get("offset") || 0);
    const end = Math.min(points.length, offset + 2000);
    return route.fulfill({ json: { system: points.slice(offset, end), indices: Array.from({ length: end - offset }, (_, i) => offset + i),
      total: points.length, next_offset: end, has_more_after: end < points.length } });
  });
}

test("long GPU history offers exact 2000-point segments and adaptive all on narrow screens", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 900 });
  const mocks = await mockWorkspace(page);
  const points = Array.from({ length: 7889 }, (_, i) => ({ ts: start + i, vram_used_gb: i === 2111 ? 999 : i }));
  await mockSystemPages(page, points);
  await page.route((url) => url.pathname === "/api/training/history/fixture-run", (route) => route.fulfill({ json: {
    task: { id: "fixture-run", job: "training", state: "done" }, metrics: [], system: points.slice(-1000), limits: { system_total: points.length },
  } }));
  await page.goto("/next/history/fixture-run?view=metrics");
  const region = page.getByRole("region", { name: "GPU 资源历史" });
  await expect(region).toContainText("已读取 7889 / 7889 个采样点");
  const window = region.getByLabel("显存 (GB)数据窗口");
  await expect(window.locator("option")).toHaveText(["全部已读取", "0–2000 点", "2000–4000 点", "4000–6000 点", "6000–7889 点"]);
  await window.selectOption("2000");
  const chart = region.getByRole("group", { name: "显存 (GB)，2000 个点" });
  await expect(chart).toBeVisible();
  await chart.focus(); await page.keyboard.press("ArrowRight");
  await expect(region).toContainText("显存 (GB): 2000");
  await window.selectOption("6000");
  await expect(region.getByRole("group", { name: "显存 (GB)，1889 个点" })).toBeVisible();
  await window.selectOption("all");
  const allChart = region.getByRole("group", { name: /显存 \(GB\)，\d+ 个点/ });
  await expect(allChart).toBeVisible();
  const label = await allChart.getAttribute("aria-label");
  expect(Number(label?.match(/，(\d+) 个点/)?.[1])).toBeLessThanOrEqual(2000);
  await expect(region).toContainText("7889 个有效点");
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(mocks.writes).toEqual([]); expect(mocks.unhandled).toEqual([]);
});
