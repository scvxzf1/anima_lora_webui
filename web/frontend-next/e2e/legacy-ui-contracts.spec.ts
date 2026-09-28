import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

const longTaskName =
  "Studio portrait editorial study with an intentionally long task name that must not cover training metrics";

async function mockLongHistory(page: import("@playwright/test").Page) {
  await mockWorkspace(page);
  const task = {
    id: "long-name-run",
    name: longTaskName,
    group: "Studio",
    history_group_key: "studio",
    history_source_config_file: "configs/studio-portrait.toml",
    history_group_label: "Portrait studies",
    job: "training",
    state: "completed",
    last_step: 1200,
    final_loss: 0.0432,
    metric_count: 120,
    log_count: 900,
    started_at: 1_757_412_600,
  };
  await page.route("**/api/training/history**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (route.request().method() !== "GET")
      return route.fulfill({
        status: 409,
        json: { error: "fixture is read-only" },
      });
    if (path.endsWith("/collections/settings"))
      return route.fulfill({
        json: { collection_order: ["Studio"], config_group_order: {} },
      });
    return route.fulfill({ json: { ok: true, tasks: [task] } });
  });
}

for (const width of [1440, 390]) {
  test(`history long task names keep metrics visible ${width}`, async ({
    page,
  }) => {
    await page.setViewportSize({ width, height: 900 });
    await mockLongHistory(page);
    await page.goto("/next/history?collection=Studio");
    await page.getByRole("button", { name: "平铺任务", exact: true }).click();

    const card = page.locator('[data-history-task="long-name-run"]');
    const title = card.locator("h3");
    const metrics = card.locator(".history-card-meta");
    await expect(title).toHaveText(longTaskName);
    await expect(metrics).toContainText("STEP 1200");
    await expect(metrics).toContainText("LOSS 0.0432");

    const boxes = await Promise.all([
      title.boundingBox(),
      metrics.boundingBox(),
    ]);
    expect(boxes[0]).not.toBeNull();
    expect(boxes[1]).not.toBeNull();
    expect(boxes[0]!.x + boxes[0]!.width).toBeLessThanOrEqual(boxes[1]!.x + 1);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
  });
}

for (const width of [1440, 390]) {
  test(`dataset save remains reachable ${width}`, async ({ page }) => {
    await page.setViewportSize({ width, height: 800 });
    const mocks = await mockWorkspace(page);
    await page.goto("/next/datasets");
    const source = page.getByLabel("原始图片目录", { exact: true });
    await source.fill("images/mobile-save-check");

    const save = page.getByRole("button", { name: "保存", exact: true });
    await expect(save).toBeVisible();
    await expect(save).toBeEnabled();
    const box = await save.boundingBox();
    expect(box).not.toBeNull();
    expect(box!.x).toBeGreaterThanOrEqual(0);
    expect(box!.x + box!.width).toBeLessThanOrEqual(width);
    await save.click();
    await expect(page.getByRole("alert")).toContainText(
      "Fixture blocked command",
    );
    expect(mocks.writes).toHaveLength(1);
  });
}
