import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

function mockAiohttpRootRedirect(page: Parameters<typeof mockWorkspace>[0], fragment: string) {
  return page.route((url) => url.pathname === "/", (route) =>
    route.fulfill({ status: 302, headers: { location: `/next/#${fragment}` } }),
  );
}

test("Dragon dataset bookmark survives the aiohttp root redirect", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  // Vite does not implement the aiohttp compatibility redirect. Preserve the
  // browser's original fragment as an HTTP redirect would.
  await mockAiohttpRootRedirect(page, "dataset-editor");

  await page.goto("/?ui=dragon#dataset-editor");
  await expect(page).toHaveURL(/\/next\/datasets$/);
  await expect(page.getByRole("heading", { name: "数据集蓝图" })).toBeVisible();
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("dataset and live-monitor Dragon bookmarks map to their Next workspaces", async ({ page }) => {
  const mocks = await mockWorkspace(page);

  await page.goto("/next/training#dataset-editor");
  await expect(page).toHaveURL(/\/next\/datasets$/);
  await expect(page.getByRole("heading", { name: "数据集蓝图" })).toBeVisible();

  await mockAiohttpRootRedirect(page, "page/live-training");
  await page.goto("/next/training#page/live-training");
  await expect(page).toHaveURL(/\/next\/monitor$/);
  await expect(page.getByRole("heading", { name: "当前监控" })).toBeVisible();
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("encoded history bookmark selects its metrics view and survives refresh", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  await page.route((url) => url.pathname.includes("p2%20run%20one"), (route) =>
    route.fulfill({ json: {
      ok: true,
      task: { id: "p2 run one", name: "Encoded bookmark run", job: "training", state: "idle" },
      metrics: [{ step: 1, loss: 0.25 }],
      logs: [],
      system: [],
    } }),
  );

  await page.goto("/next/training#history/p2%20run%20one/metrics");
  await expect(page).toHaveURL(/\/next\/history\/p2%20run%20one\?view=metrics$/);
  await expect(page.locator(".history-detail-page")).toHaveAttribute("data-view", "metrics");
  await expect(page.getByRole("heading", { name: "Encoded bookmark run" })).toBeVisible();
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("unknown legacy hash stays in Next without overwriting the current route", async ({ page }) => {
  const mocks = await mockWorkspace(page);

  await page.goto("/next/training#old-unmapped-page");
  await expect(page.getByRole("heading", { name: "训练配置" })).toBeVisible();
  await expect(page).toHaveURL(/\/next\/training#old-unmapped-page$/);
  expect(page.url()).toContain("#old-unmapped-page");
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("Next history deep link refreshes without relying on a legacy hash", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  await page.goto("/next/history/fixture-run?view=metrics");
  await expect(page.locator(".history-detail-page")).toHaveAttribute("data-view", "metrics");
  await expect(page.getByRole("heading", { name: "Studio portrait / rank 32" })).toBeVisible();
  await page.reload();
  await expect(page.locator(".history-detail-page")).toHaveAttribute("data-view", "metrics");
  await expect(page.getByRole("heading", { name: "Studio portrait / rank 32" })).toBeVisible();
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("a failed lazy workspace chunk offers an in-Next recovery route", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  await page.route((url) => url.pathname.endsWith("/DatasetWorkspace.tsx"), (route) => route.abort());

  await page.goto("/next/datasets");
  await expect(page.getByRole("heading", { name: "工作区未能加载" })).toBeVisible();
  await page.getByRole("link", { name: "训练配置" }).click();
  await expect(page).toHaveURL(/\/next\/training$/);
  await expect(page.getByRole("heading", { name: "训练配置" })).toBeVisible();
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});
