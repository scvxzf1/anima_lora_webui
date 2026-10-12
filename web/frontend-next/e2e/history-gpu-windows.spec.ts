import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";
for (const width of [1440, 390]) test(`history GPU intervals at ${width}px`, async ({ page }) => {
  await page.setViewportSize({ width, height: 900 });
  const mocks = await mockWorkspace(page);
  const total = 78889;
  await page.route(url => url.pathname === "/api/training/history/fixture-run", route => route.fulfill({ json: { task: { id: "fixture-run", name: "GPU fixture", job: "training" }, system: [], metrics: [], limits: { system_total: total } } }));
  await page.route(url => url.pathname === "/api/training/history/fixture-run/system", route => {
    const offset = Number(new URL(route.request().url()).searchParams.get("offset"));
    const end = Math.min(total, offset + 2000);
    return route.fulfill({ json: { system: Array.from({ length: end - offset }, (_, i) => ({ ts: offset + i + 1, gpu_util: i % 100, gpu_temp: 60, vram_used_gb: 5 })), indices: Array.from({ length: end - offset }, (_, i) => offset + i), total, next_offset: end, has_more_after: end < total } });
  });
  await page.goto("/next/history/fixture-run?view=metrics");
  await expect(page.getByRole("status")).toContainText(`已读取 ${total} / ${total}`, { timeout: 30000 });
  const vram = page.getByRole("combobox", { name: "显存 (GB)数据窗口" });
  await expect(vram.locator("option")).toHaveCount(41);
  await vram.selectOption("2000"); await expect(vram).toHaveValue("2000");
  await expect(page.getByRole("combobox", { name: "GPU 利用率 (%)数据窗口" })).toHaveValue("all");
  await vram.selectOption("78000"); await expect(vram.locator("option:checked")).toHaveText("78000–78889 点");
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 2)).toBe(true);
  expect(mocks.writes).toEqual([]); expect(mocks.unhandled).toEqual([]);
});
