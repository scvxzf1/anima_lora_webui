import { expect, test, type Page } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

const preset = (name: string) => ({
  path: `configs/datasets/${name}.toml`,
  label: name,
  summary: { dataset_count: 1, repeat_total: 1 },
});
const group = (id: string, names: string[]) => ({
  id,
  label: id,
  kind: "dataset",
  movable: true,
  deletable: true,
  renamable: true,
  files: names.map(preset),
});
async function setup(page: Page, { fail = false, showControls = true, coverMissing = false } = {}) {
  await mockWorkspace(page);
  let groups = [
    group("Source", ["A"]),
    group("Target", ["B", "C"]),
    group("Empty", []),
  ];
  const writes: { file: string; group: string; order: string[] }[] = [];
  let release = () => {};
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  await page.route("**/api/config/dataset-presets", (route) =>
    route.fulfill({
      json: { ok: true, groups, presets: groups.flatMap((item) => item.files) },
    }),
  );
  await page.route("**/api/config/dataset-presets/read?*", (route) =>
    route.fulfill({
      json: {
        ok: true,
        file: new URL(route.request().url()).searchParams.get("file"),
        name: "A",
        content: "",
        readonly: false,
        datasets: [{ source_dir: "fixture/images", num_repeats: 1 }],
        defaults: { resolution: 1024, batch_size: 1 },
        summary: { dataset_count: 1 },
      },
    }),
  );
  await page.route("**/api/config/dataset-presets/cover?*", (route) =>
    route.fulfill({
      json: {
        image: coverMissing ? null : "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='48' height='48'%3E%3Crect width='48' height='48' fill='%233b82f6'/%3E%3C/svg%3E",
        reason: coverMissing ? "图片目录不存在或无法访问" : "fixture",
      },
    }),
  );
  await page.route("**/api/config/file-groups/place", async (route) => {
    const body = route.request().postDataJSON();
    writes.push(body);
    await gate;
    if (fail)
      return route.fulfill({ status: 400, json: { error: "Move failed" } });
    const entries = new Map(
      groups.flatMap((item) => item.files).map((file) => [file.path, file]),
    );
    groups = groups.map((item) => ({
      ...item,
      files:
        item.id === body.group
          ? body.order.map((path: string) => entries.get(path)!)
          : item.files.filter((file) => file.path !== body.file),
    }));
    await route.fulfill({ json: { ok: true, message: "Moved" } });
  });
  await page.goto("/next/datasets");
  if (showControls) {
    await page.getByText("详细管理", { exact: true }).click();
    await expect(
      page.getByRole("button", { name: "拖动排序预设 A", exact: true }),
    ).toBeVisible();
  }
  await expect(page.getByRole("button", { name: "添加子集", exact: true })).toBeVisible();
  return { writes, release };
}
const row = (page: Page, name: string) =>
  page.locator(
    `.dataset-preset-row[data-file="configs/datasets/${name}.toml"]`,
  );

test.skip('collapsed groups persist and drag hover only expands temporarily', async ({ page }, info) => {
  await page.setViewportSize({ width: 1440, height: 1300 });
  const state = await setup(page);
  const target = page.locator('[data-group-id="Target"]');
  await page.getByRole('button', { name: '折叠分组 Target', exact: true }).click();
  await expect(target).toHaveAttribute('data-collapsed', 'true');
  await page.reload();
  await expect(target).toHaveAttribute('data-collapsed', 'true');
  const handle = page.getByRole('button', { name: '拖动排序预设 A', exact: true });
  await handle.scrollIntoViewIfNeeded();
  const start = (await handle.boundingBox())!;
  await page.mouse.move(start.x + 10, start.y + 10);
  await page.mouse.down();
  const header = target.locator('header');
  const end = (await header.boundingBox())!;
  await page.mouse.move(end.x + 20, end.y + 15, { steps: 10 });
  await page.waitForTimeout(1000);
  await expect(target).toHaveAttribute('data-collapsed', 'true');
  await page.mouse.move(1100, 400);
  await page.waitForTimeout(1200);
  await expect(target).toHaveAttribute('data-collapsed', 'true');
  await page.mouse.move(end.x + 20, end.y + 15, { steps: 10 });
  await expect(target).toHaveAttribute('data-temporary-expanded', 'true', { timeout: 4000 });
  await expect(row(page, 'B')).toBeVisible();
  const inside = (await row(page, 'B').boundingBox())!;
  await page.mouse.move(inside.x + 50, inside.y + 20);
  await expect(target).toHaveAttribute('data-temporary-expanded', 'true');
  await page.screenshot({ path: info.outputPath('temporary-expanded.png') });
  await page.mouse.move(1100, 400);
  await expect(target).toHaveAttribute('data-collapsed', 'true');
  await page.keyboard.press('Escape');
  await page.mouse.up();
  expect(state.writes).toEqual([]);
  await expect(page.getByRole('button', { name: '展开分组 Target', exact: true })).toBeEnabled();
  expect(await page.evaluate(() => JSON.parse(localStorage.getItem('dragon-next:dataset-groups:collapsed:v1') || '[]'))).toContain('Target');
  await page.setViewportSize({ width: 390, height: 844 });
  await header.scrollIntoViewIfNeeded();
  await page.screenshot({ path: info.outputPath('collapsed-mobile.png') });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.getByRole('searchbox', { name: '搜索预设' }).fill('B');
  await expect(row(page, 'B')).toBeVisible();
  await page.getByRole('searchbox', { name: '搜索预设' }).fill('');
  await expect(target).toHaveAttribute('data-collapsed', 'true');
});

test.skip('dropping on collapsed header appends without changing saved collapse', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1300 });
  const state = await setup(page);
  await page.getByRole('button', { name: '折叠分组 Target', exact: true }).click();
  const handle = page.getByRole('button', { name: '拖动排序预设 A', exact: true });
  const start = (await handle.boundingBox())!;
  const end = (await page.locator('[data-group-id="Target"] > header').boundingBox())!;
  await page.mouse.move(start.x + 10, start.y + 10);
  await page.mouse.down();
  await page.mouse.move(end.x + 20, end.y + 15, { steps: 10 });
  await page.mouse.up();
  await expect.poll(() => state.writes.length).toBe(1);
  expect(state.writes[0].order).toEqual(['B', 'C', 'A'].map(name => preset(name).path));
  state.release();
  await expect(page.locator('[data-group-id="Target"]')).toHaveAttribute('data-collapsed', 'true');
});

test('short cover press selects while long press starts preset sorting', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1300 });
  const state = await setup(page);
  const source = row(page, 'B');
  const cover = source.locator('.dataset-cover');

  await cover.click();
  await expect(source.locator('.dataset-preset')).toHaveAttribute('data-selected', 'true');
  await expect(page.locator('.dataset-drag-overlay')).toHaveCount(0);
  expect(state.writes).toEqual([]);

  const start = (await cover.boundingBox())!;
  await page.mouse.move(start.x + start.width / 2, start.y + start.height / 2);
  await page.mouse.down();
  await page.waitForTimeout(100);
  await page.mouse.move(start.x + start.width / 2 + 18, start.y + start.height / 2 + 12);
  await expect(page.locator('.dataset-drag-overlay')).toBeVisible();
  await expect(source).toHaveAttribute('data-dragging', 'true');

  const destination = row(page, 'C');
  const end = (await destination.boundingBox())!;
  await page.mouse.move(end.x + end.width / 2, end.y + end.height * 0.75, { steps: 12 });
  await expect(destination).toHaveAttribute('data-drop-position', 'after');
  await page.mouse.up();

  await expect.poll(() => state.writes.length).toBe(1);
  expect(state.writes[0]).toEqual({
    target: 'file',
    file: 'configs/datasets/B.toml',
    group: 'Target',
    order: ['configs/datasets/C.toml', 'configs/datasets/B.toml'],
  });
  state.release();
  await expect(page.locator('[data-group-id="Target"] .dataset-preset-title strong')).toHaveText(['C', 'B']);
});

test('missing-image cover still drags when detailed management is off', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1300 });
  const state = await setup(page, { showControls: false, coverMissing: true });
  const source = row(page, 'A');
  const cover = source.locator('.dataset-cover');
  await expect(page.getByRole('switch', { name: '详细管理' })).not.toBeChecked();
  await expect(cover).toHaveAttribute('data-drag-enabled', 'true');
  await expect(cover).toContainText('无图像');
  await expect(page.getByRole('button', { name: '拖动排序预设 A' })).toHaveCount(0);

  const start = (await cover.boundingBox())!;
  await page.mouse.move(start.x + start.width / 2, start.y + start.height / 2);
  await page.mouse.down();
  await page.waitForTimeout(220);
  await page.mouse.move(start.x + start.width / 2 + 18, start.y + start.height / 2 + 12);
  await expect(page.locator('.dataset-drag-overlay')).toBeVisible();

  const target = row(page, 'B');
  const end = (await target.boundingBox())!;
  await page.mouse.move(end.x + end.width / 2, end.y + end.height * 0.25, { steps: 12 });
  await expect(target).toHaveAttribute('data-drop-position', 'before');
  await page.mouse.up();
  await expect.poll(() => state.writes.length).toBe(1);
  expect(state.writes[0]).toMatchObject({
    file: preset('A').path,
    group: 'Target',
    order: ['A', 'B', 'C'].map((name) => preset(name).path),
  });
  state.release();
});

async function drag(page: Page, position: "before" | "after" | "empty") {
  const handle = page.getByRole("button", {
    name: "拖动排序预设 A",
    exact: true,
  });
  const target =
    position === "empty"
      ? page.locator('[data-group-id="Empty"] .dataset-group-dropzone')
      : row(page, "B");
  await target.scrollIntoViewIfNeeded();
  const start = (await handle.boundingBox())!;
  const end = (await target.boundingBox())!;
  await page.mouse.move(start.x + start.width / 2, start.y + start.height / 2);
  await page.mouse.down();
  await page.mouse.move(
    end.x + end.width / 2,
    end.y + end.height * (position === "before" ? 0.25 : 0.75),
    { steps: 15 },
  );
  // Auto-scroll can move the target while the pointer travels between groups.
  const settled = (await target.boundingBox())!;
  await page.mouse.move(settled.x + settled.width / 2, settled.y + settled.height * (position === "before" ? 0.25 : 0.75));
  if (position === "empty")
    await expect(target).toHaveAttribute("data-over", "true");
  else await expect(target).toHaveAttribute("data-drop-position", position);
  await expect(page.locator(".dataset-drag-overlay")).toBeVisible();
}

for (const position of ["before", "after", "empty"] as const) {
  test(`cross-group ${position} shows exact insertion and immediate landing`, async ({
    page,
  }, info) => {
    await page.setViewportSize({ width: 1440, height: 1300 });
    const state = await setup(page);
    const danger = page.getByRole("button", {
      name: "删除分组 Source",
      exact: true,
    });
    const colors = await danger.evaluate((node) => ({
      color: getComputedStyle(node).color,
      background: getComputedStyle(node).backgroundColor,
    }));
    expect(colors.background).not.toBe("rgb(255, 255, 255)");
    const luminance = (value: string) => {
      const c = value
        .match(/[\d.]+/g)!
        .slice(0, 3)
        .map(Number)
        .map((v) => v / 255)
        .map((v) => (v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
      return c[0] * 0.2126 + c[1] * 0.7152 + c[2] * 0.0722;
    };
    const values = [luminance(colors.color), luminance(colors.background)].sort(
      (a, b) => b - a,
    );
    expect((values[0] + 0.05) / (values[1] + 0.05)).toBeGreaterThanOrEqual(4.5);
    await drag(page, position);
    await page.screenshot({ path: info.outputPath("drag-insertion.png") });
    await page.mouse.up();
    const destination = position === "empty" ? "Empty" : "Target";
    const expected =
      position === "before"
        ? ["A", "B", "C"]
        : position === "after"
          ? ["B", "A", "C"]
          : ["A"];
    await expect.poll(() => state.writes.length).toBe(1);
    expect(state.writes[0].order).toEqual(
      expected.map((name) => preset(name).path),
    );
    await expect(
      page.locator(
        `[data-group-id="${destination}"] .dataset-preset-title strong`,
      ),
    ).toHaveText(expected);
    state.release();
    await expect(
      page.getByRole("button", { name: "拖动排序预设 A", exact: true }),
    ).toBeEnabled();
    await page.screenshot({ path: info.outputPath("landed.png") });
  });
}

test("failed cross-group move rolls back and escape cancels without a write", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 1300 });
  const state = await setup(page, { fail: true });
  await drag(page, "after");
  await page.keyboard.press("Escape");
  expect(state.writes).toEqual([]);
  await page.mouse.up();
  await drag(page, "after");
  await page.mouse.up();
  await expect(
    page.locator('[data-group-id="Target"] .dataset-preset-title strong'),
  ).toHaveText(["B", "A", "C"]);
  state.release();
  await expect(page.getByRole("alert")).toHaveText("Move failed");
  await expect(
    page.locator('[data-group-id="Source"] .dataset-preset-title strong'),
  ).toHaveText(["A"]);
  expect(state.writes).toHaveLength(1);
});

test("keyboard cross-group placement and mobile themes remain usable", async ({
  page,
}, info) => {
  await page.setViewportSize({ width: 1440, height: 1300 });
  const state = await setup(page);
  const handle = page.getByRole("button", {
    name: "拖动排序预设 A",
    exact: true,
  });
  await handle.focus();
  await page.keyboard.press("Space");
  await expect(handle).toHaveAttribute("aria-pressed", "true");
  for (let step = 0; step < 4; step++) {
    await page.keyboard.press("ArrowDown");
    await page.evaluate(() => new Promise(requestAnimationFrame));
    if (await row(page, "B").getAttribute("data-drop-position")) break;
  }
  await expect(row(page, "B")).toHaveAttribute(
    "data-drop-position",
    /before|after/,
  );
  const position = await row(page, "B").getAttribute("data-drop-position");
  await page.keyboard.press("Space");
  await expect.poll(() => state.writes.length).toBe(1);
  expect(state.writes[0].order).toEqual(
    (position === "before" ? ["A", "B", "C"] : ["B", "A", "C"]).map(
      (name) => preset(name).path,
    ),
  );
  state.release();
  await expect(handle).toBeEnabled();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator('[data-group-id="Target"]').scrollIntoViewIfNeeded();
  for (const theme of ["dark", "light"]) {
    await page
      .locator("html")
      .evaluate((node, value) => node.setAttribute("data-theme", value), theme);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    await page.screenshot({ path: info.outputPath(`mobile-${theme}.png`) });
  }
});
