import { expect, test, type Page } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

async function historyAssetsFixture(page: Page) {
  const mocks = await mockWorkspace(page);
  const reads: URL[] = [];
  await page.route((url) => url.pathname === "/api/training/history/fixture-run", (route) => route.fulfill({ json: {
    task: { id: "fixture-run", job: "training", state: "error", name: "Krea-2 Portrait / checkpoint failure", last_step: 6400, final_loss: 0.08, metric_count: 73, started_at: 100, finished_at: 3700, message: "CUDA out of memory while saving checkpoint", output_dir: "output/runs/portrait-study/long-result-directory/checkpoints" },
    metrics: [], config_toml: 'model_family="krea2_raw"\nnetwork_dim=32\nbase_compute="nf4"',
  } }));
  await page.route((url) => url.pathname.endsWith("/artifacts"), (route) => route.fulfill({ json: {
    artifacts: [
      { key: "config-snapshot", state: "available", name: "config.snapshot.toml", size_bytes: 96 },
      { key: "logs", state: "available", name: "logs.jsonl", size_bytes: 128 },
      { key: "runtime-config", state: "missing", message: "未保存或文件已不存在" },
    ],
  } }));
  await page.route((url) => ["/api/preview/images", "/api/preview/weights"].includes(url.pathname), (route) => {
    const url = new URL(route.request().url());
    reads.push(url);
    const images = url.pathname.endsWith("images");
    const total = images ? 125 : 505;
    const offset = Number(url.searchParams.get("offset"));
    const limit = Number(url.searchParams.get("limit"));
    return route.fulfill({ json: {
      total, offset, next_offset: offset + limit < total ? offset + limit : null,
      [images ? "images" : "weights"]: Array.from({ length: Math.min(limit, Math.max(0, total - offset)) }, (_, index) => ({
        file: `${images ? "image" : "weight"}-${offset + index}`, name: `${images ? "image" : "weight"}-${offset + index}`,
        width: 480, height: 480, sample: { step: offset + index, seed: 42, prompt: "Studio portrait" },
        size_bytes: 1048576, scope_label: "本任务",
      })),
    } });
  });
  return { ...mocks, reads };
}

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 720 }, { width: 768, height: 1024 }, { width: 390, height: 844 }]) {
  for (const theme of ["dark", "light"]) {
    test(`overview and paged assets ${viewport.width} ${theme}`, async ({ page }, info) => {
      await page.setViewportSize(viewport);
      const mocks = await historyAssetsFixture(page);
      await page.addInitScript((theme) => localStorage.setItem("dragon-next-ui-v1-theme", theme), theme);
      await page.goto("/next/history/fixture-run");
      await expect(page.getByText("CUDA out of memory while saving checkpoint")).toBeVisible();
      await expect(page.getByLabel("训练摘要")).toContainText("6400");
      await expect(page.getByLabel("训练摘要")).toContainText("0.0800");
      await expect(page.getByLabel("训练产物摘要")).toContainText("125 项");
      await expect(page.getByLabel("训练产物摘要")).toContainText("505 项");
      expect(mocks.reads.every((url) => url.searchParams.get("limit") === "1")).toBe(true);
      expect(new Set(mocks.reads.map((url) => url.pathname)).size).toBe(2);
      await expect(page.getByRole("link", { name: "config.snapshot.toml" })).toHaveAttribute("href", /config-snapshot\?download=1/);
      await expect(page.getByText("未保存或文件已不存在")).toBeVisible();
      await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await page.screenshot({ path: info.outputPath("overview.png"), fullPage: true });
      await page.getByRole("link", { name: "产物", exact: true }).click();
      await expect(page.locator(".history-image-grid img")).toHaveCount(60);
      await page.getByRole("button", { name: "下一页样张", exact: true }).click();
      await expect(page.locator(".history-image-grid button").first()).toHaveText("image-60");
      await page.getByRole("button", { name: "下一页样张", exact: true }).click();
      await expect(page.locator(".history-image-grid img")).toHaveCount(5);
      await expect(page.getByRole("button", { name: "下一页样张", exact: true })).toBeDisabled();
      await page.locator(".history-image-grid button").first().click();
      await expect(page.getByRole("dialog")).toContainText("Studio portrait");
      await expect.poll(() => page.locator(".history-asset-full").evaluate((node) => (node as HTMLImageElement).naturalWidth)).toBe(480);
      await page.screenshot({ path: info.outputPath("image-dialog.png") });
      await page.keyboard.press("Escape");
      for (let index = 1; index <= 5; index++) {
        await page.getByRole("button", { name: "下一页权重", exact: true }).click();
        await expect(page.locator(".weight-list a").first()).toHaveText(`weight-${index * 100}`);
      }
      await expect(page.locator(".weight-list a")).toHaveCount(5);
      await expect(page.getByRole("button", { name: "下一页权重", exact: true })).toBeDisabled();
      await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await page.screenshot({ path: info.outputPath("assets.png"), fullPage: true });
      expect(mocks.reads.some((url) => url.searchParams.get("offset") === "500")).toBe(true);
      expect(mocks.writes).toEqual([]);
      expect(mocks.unhandled).toEqual([]);
    });
  }
}
