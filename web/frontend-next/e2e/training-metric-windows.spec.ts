import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

for (const path of ["/next/monitor", "/next/history/fixture-run?view=metrics"]) {
  for (const width of [1440, 390]) {
    test(`training metric windows retain STEP at ${path} ${width}px`, async ({ page }) => {
      await page.setViewportSize({ width, height: 900 });
      const mocks = await mockWorkspace(page);
      const metrics = Array.from({ length: 7889 }, (_, i) => ({ step: 100 + i * 3, loss: i === 2111 ? 999 : i, lr: i / 10000 }));
      const records = [{ ev: "preprocess", loss: 99999, lr: 9 }, ...metrics, { kind: "sample", loss: 99999, lr: 9 }];
      await page.route((url) => url.pathname === "/api/training/metrics", (route) => route.fulfill({ json: records }));
      await page.route((url) => url.pathname === "/api/training/history/fixture-run", (route) => route.fulfill({ json: {
        task: { id: "fixture-run", job: "training", state: "done" }, metrics: records, system: [], limits: { metrics_total: records.length },
      } }));
      await page.route((url) => url.pathname === "/api/training/history/fixture-run/system", (route) => route.fulfill({ json: { system: [], total: 0, next_offset: 0, has_more_after: false } }));
      await page.goto(path);
      for (const label of ["Loss 与学习率趋势"]) {
        const region = page.getByRole("region", { name: label, exact: true });
        const window = region.getByLabel(`${label}数据窗口`);
        await expect(window.locator("option")).toHaveText(["全部已读取", "0–2000 点", "2000–4000 点", "4000–6000 点", "6000–7889 点"]);
        await expect(region).toContainText("7889 个有效点");
        const all = region.getByRole("group", { name: new RegExp(`${label}，\\d+ 个点`) });
        expect(Number((await all.getAttribute("aria-label"))?.match(/，(\d+)/)?.[1])).toBeLessThanOrEqual(2000);
        await window.selectOption("2000");
        const chart = region.getByRole("group", { name: `${label}，2000 个点`, exact: true });
        await chart.focus(); await page.keyboard.press("ArrowRight");
        await expect(region).toContainText("检查点 1/2000 · STEP: 6100");
        await window.selectOption("6000");
        await expect(region).not.toContainText("检查点 1/2000");
        await region.getByRole("group", { name: `${label}，1889 个点`, exact: true }).focus();
        await page.keyboard.press("ArrowLeft");
        await expect(region).toContainText("检查点 1889/1889 · STEP: 23764");
      }
      await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      expect(mocks.writes).toEqual([]); expect(mocks.unhandled).toEqual([]);
    });
  }
}
