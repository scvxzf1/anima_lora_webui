import { expect, test, type Page } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

const taskId = "snapshot-layout-fixture";
const configToml = [
  'model_family = "krea2_raw"',
  ...Array.from({ length: 55 }, (_, index) => `parameter_${index} = ${index}`),
  'target_marker = "first target marker"',
  ...Array.from({ length: 35 }, (_, index) => `later_parameter_${index} = ${index}`),
  'second_marker = "target marker again"',
  `long_value = "${"readable-long-toml-value-".repeat(24)}"`,
].join("\n");

async function historySnapshotFixture(page: Page) {
  const mocks = await mockWorkspace(page);
  await page.route(
    (url) => url.pathname === `/api/training/history/${taskId}`,
    (route) => route.fulfill({ json: {
      task: { id: taskId, job: "training", state: "done", name: "Snapshot layout fixture", started_at_text: "2026-09-30 10:00" },
      metrics: [],
      system: [],
      logs: [],
      config_toml: configToml,
    } }),
  );
  await page.route(
    (url) => url.pathname === `/api/training/history/${taskId}/artifacts`,
    (route) => route.fulfill({ json: { artifacts: [
      { key: "config-snapshot", state: "available", name: "config.snapshot.toml", size_bytes: configToml.length },
      { key: "logs", state: "available", name: "logs.jsonl", size_bytes: 128 },
    ] } }),
  );
  return mocks;
}

for (const viewport of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
  for (const theme of ["light", "dark"] as const) {
    test(`history configuration snapshot search and layout ${viewport.width}px ${theme}`, async ({ page }, info) => {
      await page.setViewportSize(viewport);
      const mocks = await historySnapshotFixture(page);
      await page.addInitScript((value) => localStorage.setItem("dragon-next-ui-v1-theme", value), theme);
      await page.goto(`/next/history/${taskId}?view=config`);

      const snapshot = page.locator(".history-snapshot");
      const code = page.getByLabel("配置快照代码");
      const search = page.getByRole("searchbox", { name: "搜索配置快照" });
      await expect(snapshot.getByRole("heading", { name: "配置快照" })).toBeVisible();
      await expect(page.getByRole("link", { name: "config.snapshot.toml", exact: true }))
        .toHaveAttribute("href", `/api/training/history/${taskId}/artifacts/config-snapshot?download=1`);

      await search.fill("target marker");
      await expect(snapshot.locator(".history-snapshot-match")).toHaveText("1 / 2");
      const firstHit = code.locator(".history-snapshot-hit-current");
      await expect(firstHit).toContainText("target marker");
      await expect.poll(() => code.evaluate((node) => {
        const hit = node.querySelector<HTMLElement>(".history-snapshot-hit-current")!;
        const box = hit.getBoundingClientRect();
        const parent = node.getBoundingClientRect();
        return node.scrollTop > 0 && box.top >= parent.top && box.bottom <= parent.bottom;
      })).toBe(true);
      await snapshot.getByRole("button", { name: "下一个匹配项" }).click();
      await expect(snapshot.locator(".history-snapshot-match")).toHaveText("2 / 2");
      await expect(code.locator(".history-snapshot-hit-current").locator("xpath=ancestor::span[contains(@class, 'history-snapshot-line')]")).toContainText("target marker again");

      const visual = await page.evaluate(() => {
        const key = document.querySelector<HTMLElement>(".history-snapshot-key")!;
        const codeBox = document.querySelector<HTMLElement>(".history-snapshot-code")!;
        const keyColor = getComputedStyle(key).color;
        const background = getComputedStyle(codeBox).backgroundColor;
        return {
          documentWidth: document.documentElement.scrollWidth,
          viewportWidth: innerWidth,
          codeClientWidth: codeBox.clientWidth,
          codeScrollWidth: codeBox.scrollWidth,
          keyColor,
          background,
          theme: document.documentElement.dataset.theme,
        };
      });
      expect(visual.documentWidth).toBeLessThanOrEqual(visual.viewportWidth);
      expect(visual.codeScrollWidth).toBeLessThanOrEqual(visual.codeClientWidth + 1);
      expect(visual.keyColor).not.toBe(visual.background);
      expect(visual.theme).toBe(theme);
      await page.screenshot({ path: info.outputPath("configuration-snapshot.png"), fullPage: true });
      expect(mocks.writes).toEqual([]);
      expect(mocks.unhandled).toEqual([]);
    });
  }
}
