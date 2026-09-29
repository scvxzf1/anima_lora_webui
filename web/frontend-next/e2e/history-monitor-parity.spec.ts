import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

test("history loss and learning-rate charts expose labeled metric values", async ({ page }, info) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  const mocks = await mockWorkspace(page);
  await page.route((url) => url.pathname === "/api/training/history/fixture-run", (route) => route.fulfill({ json: {
    task: { id: "fixture-run", name: "Tooltip fixture", job: "training", state: "done" },
    metrics: [
      { step: 0, loss: 0.42, lr: 0.0001 },
      { step: 20, loss: 0.28, lr: 0.00008 },
      { step: 40, loss: 0.16, lr: 0.00006 },
      { step: 60, loss: 0.09, lr: 0.00004 },
    ],
    system: [],
  } }));

  await page.goto("/next/history/fixture-run?view=metrics");
  const lossSection = page.getByRole("region", { name: "Loss 趋势" });
  const lossChart = lossSection.getByRole("group", { name: "Loss 趋势，4 个点" });
  await expect(lossChart).toBeVisible();
  await expect(lossChart.locator("canvas")).toBeVisible();
  const desktopBox = await lossChart.boundingBox();
  expect(desktopBox).not.toBeNull();
  await page.mouse.move(desktopBox!.x + desktopBox!.width * 0.65, desktopBox!.y + desktopBox!.height * 0.5);
  await expect(page.getByText("STEP: 40", { exact: true })).toBeVisible();
  await expect(page.getByText("Loss: 0.1600", { exact: true })).toBeVisible();
  await lossChart.focus();
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("ArrowRight");
  await expect(lossSection.getByText("检查点 3/4 · STEP: 40 · Loss: 0.1600", { exact: true })).toBeVisible();

  const lrSection = page.getByRole("region", { name: "学习率趋势" });
  const lrChart = lrSection.getByRole("group", { name: "学习率趋势，4 个点" });
  await lrChart.focus();
  await page.keyboard.press("ArrowRight");
  await expect(lrSection.getByText("检查点 1/4 · STEP: 0 · 学习率: 1.000e-4", { exact: true })).toBeVisible();
  await page.screenshot({ path: info.outputPath("history-loss-inspection.png"), fullPage: true });

  await page.setViewportSize({ width: 390, height: 844 });
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  const tooltip = lossChart.locator(':scope > div[style*="z-index: 9999999"]');
  await expect(tooltip).toBeHidden();
  const mobileBox = await lossChart.boundingBox();
  expect(mobileBox).not.toBeNull();
  await page.mouse.move(mobileBox!.x + mobileBox!.width * 0.65, mobileBox!.y + mobileBox!.height * 0.5);
  await expect(page.getByText("STEP: 40", { exact: true })).toBeVisible();
  await expect(page.getByText("Loss: 0.1600", { exact: true })).toBeVisible();
  await expect.poll(() => tooltip.evaluate((element) => {
    const tip = element.getBoundingClientRect();
    const chart = element.parentElement!.getBoundingClientRect();
    return tip.left >= chart.left && tip.right <= chart.right;
  })).toBe(true);
  await lrChart.scrollIntoViewIfNeeded();
  await expect(lrChart.locator("canvas")).toBeVisible();
  const mobileLrBox = await lrChart.boundingBox();
  expect(mobileLrBox).not.toBeNull();
  await page.mouse.move(mobileLrBox!.x + mobileLrBox!.width * 0.65, mobileLrBox!.y + mobileLrBox!.height * 0.5);
  const lrTooltip = lrChart.locator(':scope > div[style*="z-index: 9999999"]');
  await expect(lrTooltip).toBeVisible();
  await expect(lrTooltip).toContainText("学习率:");
  await expect.poll(() => lrTooltip.evaluate((element) => {
    const tip = element.getBoundingClientRect();
    const chart = element.parentElement!.getBoundingClientRect();
    return tip.left >= chart.left && tip.right <= chart.right;
  })).toBe(true);
  await lossChart.focus();
  await page.keyboard.press("ArrowLeft");
  await page.keyboard.press("ArrowLeft");
  await expect(lossSection.getByText("检查点 1/4 · STEP: 0 · Loss: 0.4200", { exact: true })).toBeVisible();
  await page.screenshot({ path: info.outputPath("history-loss-inspection-mobile.png"), fullPage: true });
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
