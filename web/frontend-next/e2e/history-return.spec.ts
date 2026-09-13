import { expect, test, type Page } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

async function fixture(page: Page, total = 600) {
  const mocks = await mockWorkspace(page);
  const cursors: number[] = [];
  const task = (index: number) => ({
    id: `run-${index}`, name: `Run ${index}`, group: "Studio", history_group_key: "a",
    history_group_label: "Portrait A", history_source_config_file: "configs/a.toml",
    state: "idle", job: "training", started_at: 1000 - index,
  });
  await page.route((url) => url.pathname === "/api/training/history", (route) => {
    const url = new URL(route.request().url());
    const offset = Number(url.searchParams.get("cursor") || 0);
    cursors.push(offset);
    const limit = Number(url.searchParams.get("limit") || 200);
    return route.fulfill({ json: { total, next_cursor: offset + limit < total ? offset + limit : null,
      tasks: Array.from({ length: Math.min(limit, total - offset) }, (_, index) => task(offset + index)),
    } });
  });
  await page.route(/\/api\/training\/history\/run-\d+$/, (route) => route.fulfill({ json: {
    task: task(Number(new URL(route.request().url()).pathname.split("-").at(-1))), metrics: [], logs: [],
  } }));
  await page.route((url) => /\/run-\d+\/artifacts$/.test(url.pathname), (route) => route.fulfill({ json: { artifacts: [] } }));
  await page.route((url) => /\/run-\d+\/logs$/.test(url.pathname), (route) => route.fulfill({
    json: { logs: [], indices: [], offset: 0, total: 0 },
  }));
  return { ...mocks, cursors };
}

test("a deep detail URL restores filters, loaded depth, outer page and scroll anchor", async ({ page }) => {
  const mocks = await fixture(page);
  await page.goto("/next/history?collection=Studio&q=Run&layout=list&archived=all");
  await page.getByRole("button", { name: "载入更多记录", exact: true }).click();
  await expect(page.getByLabel("历史任务统计")).toContainText("400");
  for (let index = 0; index < 3; index++) {
    await page.getByRole("button", { name: "下一页", exact: true }).click();
    await expect(page).toHaveURL(new RegExp(`page=${index + 1}`));
  }
  const target = page.locator('[data-history-task="run-350"]');
  await target.locator(".history-card-link").click();
  await page.getByRole("link", { name: "指标", exact: true }).click();
  await page.getByRole("link", { name: "配置", exact: true }).click();
  const deepUrl = page.url();
  // Drop both router state and QueryClient memory, as when opening a shared detail URL.
  await page.evaluate(() => history.replaceState(null, ""));
  await page.goto(deepUrl);
  const before = mocks.cursors.length;
  await page.getByRole("link", { name: "返回历史", exact: true }).click();
  await expect(target).toBeInViewport();
  await expect(page).toHaveURL(/depth=2/);
  await expect(page).toHaveURL(/page=3/);
  await expect(page.getByRole("searchbox", { name: "搜索历史记录" })).toHaveValue("Run");
  expect([...new Set(mocks.cursors.slice(before))]).toEqual([0, 200]);
  await page.goBack();
  await expect(page.getByRole("heading", { name: "Run 350", exact: true })).toBeVisible();
  await page.goForward();
  await expect(target).toBeInViewport();
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("stack expansion and its page survive detail refresh and return", async ({ page }) => {
  const mocks = await fixture(page, 24);
  await page.goto("/next/history?collection=Studio");
  await page.locator(".history-stack-toggle").click();
  await page.getByRole("button", { name: "下一页任务", exact: true }).click();
  const target = page.locator('[data-history-task="run-15"]');
  await target.locator(".history-card-link").click();
  await page.getByRole("link", { name: "日志", exact: true }).click();
  await page.reload();
  await page.getByRole("link", { name: "返回历史", exact: true }).click();
  await expect(target).toBeInViewport();
  await expect(page.locator(".history-stack-toggle")).toHaveAttribute("aria-expanded", "true");
  await expect(page.locator(".history-stack-tasks")).toContainText("第 2 / 3 页");
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("large restoration pauses after ten batches and resumes only on demand", async ({ page }) => {
  const mocks = await fixture(page, 3000);
  await page.goto("/next/history?layout=list&depth=12");
  await expect(page.getByRole("button", { name: "继续恢复位置", exact: true })).toBeVisible();
  // React StrictMode may cancel and repeat the initial request; page boundaries remain bounded.
  expect([...new Set(mocks.cursors)]).toEqual(Array.from({ length: 10 }, (_, index) => index * 200));
  await page.getByRole("button", { name: "继续恢复位置", exact: true }).click();
  await expect(page.locator(".history-card")).toHaveCount(100);
  expect([...new Set(mocks.cursors)]).toEqual(Array.from({ length: 12 }, (_, index) => index * 200));
  expect(mocks.cursors.at(-1)).toBe(2200);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("a deleted return anchor gives an explicit fallback without unbounded reads", async ({ page }) => {
  const mocks = await fixture(page, 24);
  await page.goto("/next/history?layout=list&depth=2&anchor=deleted-task");
  await expect(page.locator(".history-notice")).toContainText("原任务已不在当前列表位置");
  await expect(page.locator(".history-card")).toHaveCount(24);
  expect([...new Set(mocks.cursors)]).toEqual([0]);
  await page.getByRole("button", { name: "保留当前列表" }).click();
  await expect(page).not.toHaveURL(/anchor=/);
  await expect(page.getByText(/原任务已不在当前列表位置/)).toHaveCount(0);
  expect(mocks.writes).toEqual([]);
});

test("restoration failure remains retryable and hidden selections have an explicit scope", async ({ page }) => {
  const mocks = await fixture(page, 400);
  let fail = true;
  await page.route((url) => url.pathname === "/api/training/history" && url.searchParams.get("cursor") === "200", (route) => fail
    ? route.fulfill({ status: 503, json: { error: "restore offline" } }) : route.fallback());
  await page.goto("/next/history?layout=list&depth=2&q=Run&anchor=run-1");
  await expect(page.getByRole("alert")).toContainText("restore offline");
  await expect(page.getByRole("button", { name: "重试恢复位置" })).toBeVisible();
  fail = false;
  await page.getByRole("button", { name: "重试恢复位置" }).click();
  await expect(page.locator('[data-history-task="run-1"]')).toBeInViewport();
  await page.getByRole("checkbox", { name: "选择 Run 1", exact: true }).check();
  await page.getByRole("button", { name: "下一页", exact: true }).click();
  await expect(page.locator(".history-bulk-bar")).toContainText("其中 1 项不在当前展开页");
  await expect(page.locator(".history-filter-scope")).toContainText("已读取 400 / 400 条");
  await page.getByRole("button", { name: "清除选择" }).click();
  await expect(page.locator(".history-bulk-bar")).toHaveCount(0);
  await page.getByRole("button", { name: "清除筛选" }).click();
  await expect(page.getByRole("searchbox", { name: "搜索历史记录" })).toHaveValue("");
  expect(mocks.writes).toEqual([]);
});
