import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

for (const scenario of [
  { width: 1440, height: 900, scale: 100 },
  { width: 1440, height: 900, scale: 150 },
  { width: 390, height: 844, scale: 100 },
]) {
  test(`recent training fits viewport at ${scenario.width}px / ${scenario.scale}%`, async ({ page }) => {
    await page.setViewportSize(scenario);
    const mocks = await mockWorkspace(page);
    await page.route(url => url.pathname === "/api/settings/global", route => route.fulfill({ json: { ui_scale: scenario.scale } }));
    await page.route(url => url.pathname === "/api/training/history", route => route.fulfill({ json: {
      tasks: Array.from({ length: 20 }, (_, index) => ({ id: index ? `run-${index}` : "fixture-run", name: `Recent training ${index}`, job: "training", state: "completed", started_at: 2000 - index })),
    } }));
    await page.goto("/next/monitor");
    await page.getByRole("button", { name: "最近训练", exact: true }).click();
    await expect(page.locator(".monitor-recent-task")).toHaveCount(20);
    await expect.poll(() => page.locator(".monitor-history-workspace").evaluate(node => Math.abs(node.getBoundingClientRect().bottom - innerHeight))).toBeLessThan(2);
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollHeight <= innerHeight + 2)).toBe(true);
    await expect.poll(() => page.locator(".monitor-recent-list").evaluate(node => node.scrollHeight > node.clientHeight)).toBe(true);
    await page.locator(".monitor-recent-list").evaluate(node => { node.scrollTop = 120; });
    await expect.poll(() => page.locator(".monitor-recent-list").evaluate(node => node.scrollTop)).toBeGreaterThan(0);
    await page.locator(".monitor-recent-task").first().click();
    await expect(page.getByRole("button", { name: "返回当前监控" })).toBeVisible();
    expect(new URL(page.url()).pathname).toBe("/next/monitor");
    await page.getByRole("button", { name: "返回当前监控" }).click();
    await expect(page.getByRole("button", { name: "最近训练", exact: true })).toBeVisible();
    expect(mocks.writes).toEqual([]);
    expect(mocks.unhandled).toEqual([]);
  });
}
