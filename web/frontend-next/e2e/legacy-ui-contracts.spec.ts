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
  }, info) => {
    await page.setViewportSize({ width, height: 900 });
    await mockLongHistory(page);
    await page.goto("/next/history?collection=Studio");

    const configGroups = page.locator(".history-config-groups");
    await configGroups
      .getByRole("button", { name: "Portrait studies", exact: true })
      .click();
    const stack = page.locator(".history-task-stack");
    await expect(stack).toBeVisible();
    await stack.locator(".history-stack-toggle").click();

    const card = page.locator('[data-history-task="long-name-run"]');
    const state = card.locator(".history-state");
    const title = card.locator("h3");
    const configPath = card.locator(".history-card-link p");
    const metrics = card.locator(".history-card-meta");
    await expect(title).toHaveText(longTaskName);
    await expect(metrics).toContainText("STEP 1200");
    await expect(metrics).toContainText("LOSS 0.0432");

    const boxes = await Promise.all([
      state.boundingBox(),
      title.boundingBox(),
      configPath.boundingBox(),
      metrics.boundingBox(),
    ]);
    await card.scrollIntoViewIfNeeded();
    await page.screenshot({
      path: info.outputPath(`history-config-group-${width}.png`),
    });
    await expect(title).toBeVisible();
    await expect(configPath).toContainText("configs/studio-portrait.toml");
    expect(boxes.every(Boolean)).toBe(true);
    const [stateBox, titleBox, pathBox, metricsBox] = boxes;
    expect(titleBox!.width).toBeGreaterThanOrEqual(64);
    expect(pathBox!.width).toBeGreaterThanOrEqual(80);
    for (const [left, right] of [
      [stateBox!, titleBox!],
      [stateBox!, pathBox!],
      [titleBox!, pathBox!],
      [titleBox!, metricsBox!],
      [pathBox!, metricsBox!],
    ]) {
      const overlaps =
        left.x < right.x + right.width &&
        left.x + left.width > right.x &&
        left.y < right.y + right.height &&
        left.y + left.height > right.y;
      expect(overlaps).toBe(false);
    }
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
  });
}

for (const width of [1440, 390]) {
  test(`dataset save remains reachable ${width}`, async ({ page }, info) => {
    await page.setViewportSize({ width, height: 800 });
    const mocks = await mockWorkspace(page);
    await page.goto("/next/datasets");
    const source = page.getByLabel("原始图片目录", { exact: true });
    await source.fill("images/mobile-save-check");

    const save = page.getByRole("button", { name: "保存", exact: true });
    await expect(save).toBeVisible();
    await expect(save).toBeEnabled();
    await save.scrollIntoViewIfNeeded();
    const box = await save.boundingBox();
    expect(box).not.toBeNull();
    expect(box!.x).toBeGreaterThanOrEqual(0);
    expect(box!.x + box!.width).toBeLessThanOrEqual(width);
    if (width === 390) {
      expect(box!.y).toBeGreaterThanOrEqual(0);
      expect(box!.y + box!.height).toBeLessThanOrEqual(800);
      const commandBar = page.locator(".dataset-command-bar");
      const firstAction = commandBar.locator("button").first();
      const firstBox = await firstAction.boundingBox();
      expect(firstBox).not.toBeNull();
      expect(box!.y).toBe(firstBox!.y);
      expect(box!.width).toBeGreaterThanOrEqual(52);
      await expect(save).toHaveClass(/primary-command/);
      for (const secondary of await commandBar.locator("button").all()) {
        if (await secondary.getAttribute("type") === "submit") continue;
        const other = await secondary.boundingBox();
        if (!other || other.y !== box!.y) continue;
        const overlaps =
          box!.x < other.x + other.width && box!.x + box!.width > other.x;
        expect(overlaps).toBe(false);
      }
    }
    await page.screenshot({
      path: info.outputPath(`dataset-save-${width}.png`),
    });
    await save.click();
    await expect(page.getByRole("alert")).toContainText(
      "Fixture blocked command",
    );
    expect(mocks.writes).toHaveLength(1);
  });
}
