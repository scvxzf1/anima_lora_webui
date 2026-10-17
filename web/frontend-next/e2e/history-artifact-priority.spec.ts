import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

for (const width of [1440, 390]) {
  test(`artifact priority and grouped evolution at ${width}px`, async ({ page }, info) => {
    await page.setViewportSize({ width, height: 900 });
    const mocks = await mockWorkspace(page);
    await page.route((url) => url.pathname === "/api/training/history/fixture-run", (route) => route.fulfill({ json: {
      task: { id: "fixture-run", name: "Qwen Edit / visual study", job: "training", state: "idle", started_at_text: "2026-10-03 16:23" },
      config_toml: 'model_family="qwen_image_2_1"\nnetwork_dim=16',
      metrics: [{ step: 200, loss: .31 }, { step: 1000, loss: .19 }, { step: 200, loss: .99, kind: "val" }],
    } }));
    await page.route((url) => url.pathname === "/api/preview/images", (route) => route.fulfill({ json: {
      images: [1000, 800, 600, 400, 200].flatMap((step) => [0, 1].map((index) => ({
        file: `${step}-${index}.png`, name: `${step}-${index}.png`,
        sample: { step, prompt_index: index, seed: 42 + index, prompt: "Japanese-style fantasy illustration" },
      }))), total: 10,
    } }));
    await page.route((url) => url.pathname === "/api/preview/weights", (route) => route.fulfill({ json: {
      weights: [200, 1000].map((steps) => ({ file: `step${steps}.safetensors`, name: `study-step${steps}.safetensors`,
        abs_path: `/home/user/training-output/very-long-study-name/study-step${steps}.safetensors`, steps,
        kind: steps === 1000 ? "final" : "step", size_bytes: 33608512, scope_label: "本任务" })), total: 2,
    } }));
    await page.route((url) => url.pathname.endsWith("/artifacts"), (route) => route.fulfill({ json: {
      artifacts: ["config-snapshot", "runtime-config", "original-config", "dataset-config", "logs", "metrics", "system"].map((key) => ({
        key, name: `${key}.toml`, state: "available", size_bytes: 4096,
      })),
    } }));
    await page.goto("/next/history/fixture-run?view=artifacts");
    await expect(page.locator(".history-detail-title .history-detail-state")).toHaveText("完成");
    await expect(page.getByRole("button", { name: "以此配置新建训练" })).toBeEnabled();
    await expect(page.getByRole("button", { name: "检查点续训", exact: true })).toBeVisible();
    await expect(page.locator(".history-image-grid img")).toHaveCount(10);
    await expect(page.locator(".weight-list")).toContainText("0.3100");
    await expect(page.locator(".weight-list")).not.toContainText("0.9900");
    await expect(page.getByText("FINAL MODEL", { exact: true })).toBeVisible();
    const sections = await page.locator(".history-assets > section").evaluateAll((nodes) => nodes.map((node) => node.getAttribute("aria-label")));
    expect(sections).toEqual(["训练样张", "权重", "结果文件"]);
    await expect(page.locator(".history-artifact-grid > div")).toHaveCount(7);
    if (width === 1440) expect(await page.locator(".history-artifact-grid").evaluate((node) => getComputedStyle(node).gridTemplateColumns.split(" ").length)).toBe(4);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.screenshot({ path: info.outputPath("artifacts.png"), fullPage: true });
    await page.getByRole("combobox", { name: "浏览方式" }).selectOption("grouped");
    await expect(page.locator(".history-sample-group")).toHaveCount(2);
    await expect(page.locator(".history-sample-group").first().locator(".history-sample-step").first()).toHaveText("Step 200");
    await page.locator(".history-sample-group").first().getByRole("button").first().click();
    await expect(page.getByRole("dialog")).toContainText("200-0.png");
    await page.keyboard.press("Escape");
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.getByRole("link", { name: "配置", exact: true }).click();
    await expect(page.getByRole("heading", { name: "配置快照" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "结果文件" })).toHaveCount(0);
    await page.getByRole("button", { name: "以此配置新建训练" }).click();
    await expect(page.getByRole("dialog", { name: "新建训练配置" })).toBeVisible();
    await expect(page.getByRole("textbox", { name: "TOML", exact: true })).toHaveValue(/qwen_image_2_1/);
    expect(mocks.writes).toEqual([]);
    expect(mocks.unhandled).toEqual([]);
  });
}
