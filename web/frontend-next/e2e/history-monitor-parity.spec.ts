import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

test("history loss chart exposes labeled tooltip text on pointer hover", async ({ page }, info) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  const mocks = await mockWorkspace(page);
  await page.route((url) => url.pathname === "/api/training/history/fixture-run", (route) => route.fulfill({ json: {
    task: { id: "fixture-run", name: "Tooltip fixture", job: "training", state: "done" },
    metrics: [
      { step: 0, loss: 0.42 },
      { step: 20, loss: 0.28 },
      { step: 40, loss: 0.16 },
      { step: 60, loss: 0.09 },
    ],
    system: [],
  } }));

  await page.goto("/next/history/fixture-run?view=metrics");
  const chart = page.locator(".chart-section[aria-label='Loss 趋势'] .metric-chart");
  await expect(chart).toBeVisible();
  await expect(chart.locator("canvas")).toBeVisible();
  const canvas = chart.locator("canvas");
  const box = await chart.boundingBox();
  expect(box).not.toBeNull();
  await page.mouse.move(box!.x + box!.width * 0.65, box!.y + box!.height * 0.5);
  await expect(page.getByText("STEP: 40", { exact: true })).toBeVisible();
  await expect(page.getByText("Loss: 0.1600", { exact: true })).toBeVisible();
  await page.screenshot({ path: info.outputPath("history-loss-hover.png"), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  const mobileBox = await chart.boundingBox();
  expect(mobileBox).not.toBeNull();
  await page.mouse.move(mobileBox!.x + mobileBox!.width * 0.65, mobileBox!.y + mobileBox!.height * 0.5);
  await expect(page.getByText("STEP: 40", { exact: true })).toBeVisible();
  await expect(page.getByText("Loss: 0.1600", { exact: true })).toBeVisible();
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await chart.focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.locator(".chart-inspection")).toContainText("STEP: 0");
  await page.screenshot({ path: info.outputPath("history-loss-hover-mobile.png"), fullPage: true });
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

for (const viewport of [{ width: 1280, height: 800 }, { width: 390, height: 844 }]) {
  test(`monitor warning and queue navigation remain readable at ${viewport.width}px`, async ({ page }, info) => {
    await page.setViewportSize(viewport);
    if (viewport.width > 1000) await page.addInitScript(() => localStorage.setItem("dragon-next-ui-v1-theme", "light"));
    const mocks = await mockWorkspace(page);
    await page.route((url) => url.pathname === "/api/training/status", (route) => route.fulfill({ json: {
      status: "running",
      task_id: "hot-fixture",
      job: "training",
      latest_progress: { current: 840, total: 1600, loss: 0.094, lr: 0.00002, rate: "2.73s/it" },
      latest_system: { vram_used_gb: 11.3, vram_total_gb: 24, gpu_temp: 84, gpu_util: 98 },
    } }));

    await page.goto("/next/monitor");
    await expect(page.getByText("高温预警", { exact: true })).toBeVisible();
    await expect(page.getByRole("region", { name: "实时指标" })).toContainText("84°C");
    await expect(page.getByRole("progressbar", { name: "任务进度" })).toBeVisible();
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.screenshot({ path: info.outputPath(`monitor-warning-${viewport.width}.png`), fullPage: true });

    if (viewport.width < 600) await page.getByRole("button", { name: "切换导航" }).click();
    await page.getByRole("link", { name: "训练队列" }).click();
    await expect(page).toHaveURL(/\/next\/queue$/);
    await expect(page.getByRole("heading", { name: "训练队列" })).toBeVisible();
    await expect(page.locator(".queue-card")).toHaveCount(3);
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.screenshot({ path: info.outputPath(`queue-after-monitor-${viewport.width}.png`), fullPage: true });
    expect(mocks.writes).toEqual([]);
    expect(mocks.unhandled).toEqual([]);
  });
}
