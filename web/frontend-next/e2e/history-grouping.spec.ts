import { expect, test, type Page, type Locator } from "@playwright/test";
import { mockWorkspace } from "./fixtures";
import type { HistoryCollections } from "../src/features/training-history/api";

async function historyFixture(page: Page) {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await mockWorkspace(page);
  let settings: HistoryCollections = {
    collection_order: ["Studio", "Archive"],
    config_group_order: { Studio: ["a", "b", "unloaded"], Archive: ["c"] },
  };
  const tasks = Array.from({ length: 14 }, (_, i) => ({
    id: `a-${i}`,
    name: `Run ${i}`,
    group: "Studio",
    history_group_key: "a",
    history_source_config_file: "configs/a.toml",
    history_group_label: "Portrait A",
    job: i % 2 ? "preprocess" : "training",
    state: "idle",
    archived: false,
  })).concat([
    {
      id: "b-0",
      name: "Other run",
      group: "Studio",
      history_group_key: "b",
      history_source_config_file: "configs/b.toml",
      history_group_label: "Portrait B",
      job: "training",
      state: "error",
      archived: false,
    },
    {
      id: "c-0",
      name: "Archive run",
      group: "Archive",
      history_group_key: "c",
      history_source_config_file: "configs/c.toml",
      history_group_label: "Portrait C",
      job: "training",
      state: "idle",
      archived: false,
    },
  ]);
  const writes: { path: string; body: Record<string, unknown> }[] = [];
  const control = { fail: false };
  await page.route(
    (url) => url.pathname.startsWith("/api/training/history"),
    async (route) => {
      const path = new URL(route.request().url()).pathname;
      const method = route.request().method();
      if (method === "GET")
        return route.fulfill({
          json: path.endsWith("settings") ? settings : { tasks },
        });
      const body = route.request().postDataJSON();
      writes.push({ path, body });
      if (control.fail)
        return route.fulfill({
          status: 500,
          json: { error: "Fixture: grouping save failed" },
        });
      if (method === "PUT") {
        settings = body;
        return route.fulfill({ json: settings });
      }
      expect(body.action).toBe("set_group");
      const keys = tasks
        .filter((task) => body.task_ids.includes(task.id))
        .map((task) => task.history_group_key);
      tasks
        .filter((task) => keys.includes(task.history_group_key))
        .forEach((task) => {
          task.group = body.group;
        });
      return route.fulfill({ json: { ok: true } });
    },
  );
  return { writes, control };
}

async function drag(page: Page, source: Locator, target: Locator) {
  await expect(source).toBeEnabled();
  await target.scrollIntoViewIfNeeded();
  await source.scrollIntoViewIfNeeded();
  const a = (await source.boundingBox())!;
  const b = (await target.boundingBox())!;
  await page.mouse.move(a.x + a.width / 2, a.y + a.height / 2);
  await page.mouse.down();
  await page.mouse.move(a.x + a.width / 2 + 8, a.y + a.height / 2, {
    steps: 3,
  });
  await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2, { steps: 20 });
  await page.mouse.up();
}

for (const width of [1440, 390]) {
  test(`history stacks expand with bounded rows ${width}`, async ({
    page,
  }, info) => {
    const { writes } = await historyFixture(page);
    await page.setViewportSize({ width, height: 900 });
    await page.goto("/next/history?collection=Studio");
    await expect(page.locator(".history-task-stack")).toHaveCount(2);
    await expect(page.locator(".history-card")).toHaveCount(0);
    const toggle = page.getByRole("button", {
      name: "Portrait A 14 条",
      exact: false,
    });
    await toggle.click();
    await expect(toggle).toHaveAttribute("aria-expanded", "true");
    await expect(page.locator(".history-card")).toHaveCount(10);
    await page.getByRole("button", { name: "下一页任务", exact: true }).click();
    await expect(page.locator(".history-card")).toHaveCount(4);
    await page
      .getByRole("checkbox", { name: "选择配置组 Portrait A", exact: true })
      .check();
    await expect(page.getByText("已选 14 项", { exact: true })).toBeVisible();
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({
      path: info.outputPath("history-stacked.png"),
    });
    await page.locator(".history-task-stack").first().scrollIntoViewIfNeeded();
    await page.screenshot({
      path: info.outputPath("history-stack-expanded.png"),
    });
    await page.getByRole("button", { name: "平铺任务", exact: true }).click();
    await expect(page.locator(".history-card")).toHaveCount(15);
    expect(writes).toEqual([]);
  });
}

test("history pointer sorting persists collections and config order", async ({
  page,
}) => {
  const { writes } = await historyFixture(page);
  await page.goto("/next/history?collection=Studio");
  await drag(
    page,
    page.getByRole("button", { name: "拖动集合 Studio", exact: true }),
    page.locator('[data-history-drag="collection:Archive"]'),
  );
  await expect.poll(() => writes.length).toBe(1);
  expect(writes[0].body.collection_order).toEqual(["Archive", "Studio"]);
  const nav = page.locator(".history-config-groups");
  await drag(
    page,
    nav.getByRole("button", { name: "拖动配置组 Portrait A", exact: true }),
    nav.locator("[data-history-drag]").filter({ hasText: "Portrait B" }),
  );
  await expect.poll(() => writes.length).toBe(2);
  expect(writes[1].body.config_group_order).toEqual({
    Studio: ["b", "a", "unloaded"],
    Archive: ["c"],
  });
  await page.reload();
  await expect(page.locator(".history-stack-toggle").first()).toContainText(
    "Portrait B",
  );
  const stacks = page.locator(".history-task-stack");
  await drag(
    page,
    stacks.getByRole("button", { name: "拖动配置组 Portrait B", exact: true }),
    stacks.locator("[data-history-drag]").filter({ hasText: "Portrait A" }),
  );
  await expect.poll(() => writes.length).toBe(3);
  expect(writes[2].body.config_group_order).toEqual({
    Studio: ["a", "b", "unloaded"],
    Archive: ["c"],
  });
});

test("history keyboard config sorting persists", async ({ page }) => {
  const { writes } = await historyFixture(page);
  await page.goto("/next/history?collection=Studio");
  const nav = page.locator(".history-config-groups");
  const source = nav.getByRole("button", {
    name: "拖动配置组 Portrait A",
    exact: true,
  });
  const target = nav
    .locator("[data-history-drag]")
    .filter({ hasText: "Portrait B" });
  await expect(source).toBeEnabled();
  await source.focus();
  await page.keyboard.press("Space");
  await expect(nav.locator('[data-dragging="true"]')).toHaveCount(1);
  await page.keyboard.press("ArrowDown");
  await expect(target).toHaveAttribute("data-over", "true");
  await page.keyboard.press("Space");
  await expect.poll(() => writes.length).toBe(1);
  expect(writes[0].body.config_group_order).toEqual({
    Studio: ["b", "a", "unloaded"],
    Archive: ["c"],
  });
});

test("history drag moves entire config and selected tasks, cancellation is inert", async ({
  page,
}) => {
  const { writes } = await historyFixture(page);
  await page.goto("/next/history?collection=Studio");
  const source = page
    .locator(".history-task-stack")
    .getByRole("button", { name: "拖动配置组 Portrait A", exact: true });
  const target = page.locator('[data-history-drag="collection:Archive"]');
  await drag(page, source, target);
  await expect(
    page.getByRole("dialog", { name: "移动历史记录" }),
  ).toBeVisible();
  await page
    .getByRole("dialog", { name: "移动历史记录" })
    .getByRole("button", { name: "取消", exact: true })
    .click();
  expect(writes).toHaveLength(0);
  await drag(page, source, target);
  await expect(
    page.getByRole("dialog", { name: "移动历史记录" }),
  ).toBeVisible();
  await page
    .getByRole("dialog", { name: "移动历史记录" })
    .getByRole("button", { name: "确定", exact: true })
    .click();
  await expect.poll(() => writes.length).toBe(1);
  expect(writes[0].body).toEqual({
    action: "set_group",
    group: "Archive",
    task_ids: Array.from({ length: 14 }, (_, i) => `a-${i}`),
  });
  await expect(page.locator(".history-task-stack")).toHaveCount(1);
  await page.getByRole("button", { name: "全部集合", exact: true }).click();
  await page.getByRole("button", { name: "平铺任务", exact: true }).click();
  await page.getByRole("checkbox", { name: "选择 Run 0", exact: true }).check();
  await page
    .getByRole("checkbox", { name: "选择 Other run", exact: true })
    .check();
  await drag(
    page,
    page.getByRole("button", { name: "拖动任务 Run 0", exact: true }),
    page.locator('[data-history-drag="collection:"]'),
  );
  await expect(
    page.getByRole("dialog", { name: "移动历史记录" }),
  ).toBeVisible();
  await page
    .getByRole("dialog", { name: "移动历史记录" })
    .getByRole("button", { name: "确定", exact: true })
    .click();
  await expect.poll(() => writes.length).toBe(2);
  expect(writes[1].body).toEqual({
    action: "set_group",
    group: "",
    task_ids: ["a-0", "b-0"],
  });
});

test("history keyboard drag supports cancel and failed saves retain order", async ({
  page,
}) => {
  const { writes, control } = await historyFixture(page);
  await page.goto("/next/history?collection=Studio");
  const handle = page.getByRole("button", {
    name: "拖动集合 Studio",
    exact: true,
  });
  await handle.focus();
  await page.keyboard.press("Space");
  await expect(
    page.locator('[data-history-drag="collection:Studio"]'),
  ).toHaveAttribute("data-dragging", "true");
  await page.keyboard.press("ArrowDown");
  await expect(
    page.locator('[data-history-drag="collection:Archive"]'),
  ).toHaveAttribute("data-over", "true");
  await page.keyboard.press("Escape");
  await expect(
    page.locator('[data-history-drag="collection:Studio"]'),
  ).toHaveAttribute("data-dragging", "false");
  expect(writes).toHaveLength(0);
  control.fail = true;
  await handle.focus();
  await page.keyboard.press("Space");
  await expect(
    page.locator('[data-history-drag="collection:Studio"]'),
  ).toHaveAttribute("data-dragging", "true");
  await page.keyboard.press("ArrowDown");
  await expect(
    page.locator('[data-history-drag="collection:Archive"]'),
  ).toHaveAttribute("data-over", "true");
  await page.keyboard.press("Space");
  await expect(page.getByRole("alert")).toContainText("grouping save failed");
  expect(writes).toHaveLength(1);
  await expect(
    page.locator(".object-library [data-history-drag]").nth(1),
  ).toHaveAttribute("data-history-drag", "collection:Studio");
});
