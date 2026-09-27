import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

test("sample image failure can recover without losing the selected sample", async ({ page }, info) => {
  const mocks = await mockWorkspace(page);
  let missing = true;
  await page.route((url) => url.pathname === "/api/preview/image", (route) => missing
    ? route.fulfill({ status: 404, body: "missing sample" }) : route.fallback());
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/next/history/fixture-run?view=artifacts");
  await expect(page.locator(".history-thumbnail.image-fallback")).toBeVisible();
  await page.locator(".history-image-grid button").first().click();
  const dialog = page.getByRole("dialog");
  await expect(dialog).toContainText("无法读取图片：step-1000.png");
  const bounds = await dialog.boundingBox();
  expect(bounds!.y).toBeGreaterThanOrEqual(0);
  expect(bounds!.y + bounds!.height).toBeLessThanOrEqual(844);
  missing = false;
  await dialog.getByRole("button", { name: "重试图片" }).click();
  await expect.poll(() => dialog.locator("img").evaluate((node) => node.naturalWidth)).toBe(480);
  await page.screenshot({ path: info.outputPath("sample-recovered.png") });
  await page.keyboard.press("Escape");
  await expect(page.locator(".history-image-grid button").first()).toBeFocused();
  expect(mocks.writes).toEqual([]); expect(mocks.unhandled).toEqual([]);
});

test("caption logs retry locally and failed thumbnails do not obscure drafts", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let failed = true;
  await page.route((url) => url.pathname === "/api/captioning/logs", (route) => failed
    ? route.fulfill({ status: 503, json: { error: "log storage unavailable" } })
    : route.fulfill({ json: { lines: [{ level: "info", message: "logs recovered", sequence: 1 }] } }));
  await page.route((url) => url.pathname === "/api/config/dataset-presets/image", (route) => route.fulfill({ status: 404, body: "missing image" }));
  await page.goto("/next/captioning?job=caption-1");
  await expect(page.locator(".caption-thumbnail.image-fallback").first()).toBeVisible();
  const editor = page.getByRole("textbox", { name: "候选标注", exact: true });
  await editor.fill("Unsaved caption survives retry");
  await page.locator(".caption-logs summary").click();
  await expect(page.locator(".caption-logs")).toContainText("log storage unavailable");
  await expect(page.locator(".caption-logs")).not.toContainText("暂无日志");
  failed = false;
  await page.getByRole("button", { name: "重试打标日志" }).click();
  await expect(page.locator(".caption-logs pre")).toContainText("logs recovered");
  await expect(editor).toHaveValue("Unsaved caption survives retry");
  expect(mocks.writes).toEqual([]); expect(mocks.unhandled).toEqual([]);
});

test("overview preserves missing fields behind disclosure and follows task identity", async ({ page }, info) => {
  const mocks = await mockWorkspace(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.route((url) => url.pathname === "/api/training/history/fixture-run", (route) => route.fulfill({ json: {
    task: { id: "fixture-run", name: "Sparse training", job: "training", state: "running", last_step: 10 },
    metrics: [{ step: 10, loss: 0.2 }, { kind: "val", step: 10, loss: 0.91, cmmd: 0.91 }],
    config_toml: "max_train_steps = 100",
  } }));
  await page.goto("/next/history/fixture-run");
  await expect(page.getByLabel("训练摘要")).toContainText("0.2000");
  const fingerprint = page.locator(".history-overview-section").filter({ has: page.getByRole("heading", { name: "配置指纹" }) });
  await expect(fingerprint.getByText("100 步", { exact: true })).toBeVisible();
  await expect(fingerprint.getByText("Rank", { exact: true })).not.toBeVisible();
  await fingerprint.locator("summary").click();
  await expect(fingerprint.getByText("Rank", { exact: true })).toBeVisible();
  await fingerprint.locator("summary").click();
  await page.screenshot({ path: info.outputPath("sparse-overview.png"), fullPage: true });
  await page.getByRole("link", { name: "指标", exact: true }).click();
  await expect(page.getByRole("img", { name: /Loss 趋势，1 个点，最新 0.2000/ })).toBeVisible();
  await expect(page.getByRole("img", { name: /验证 CMMD，1 个点，最新 0.9100/ })).toBeVisible();
  await page.getByRole("link", { name: "概览", exact: true }).click();
  await page.getByRole("link", { name: "查看当前监控" }).click();
  await expect(page.getByText(/来源任务 fixture-run 已不是当前监控对象/)).toBeVisible();
  await expect(page.getByRole("link", { name: "返回来源任务" })).toHaveAttribute("href", "/next/history/fixture-run");
  await expect(page.locator(".monitor-identity")).toContainText("current-fixture");
  expect(mocks.writes).toEqual([]); expect(mocks.unhandled).toEqual([]);
});

test("overview result summary isolates failures and does not invent missing totals", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let failed = true;
  await page.route((url) => url.pathname === "/api/preview/images", (route) => route.fulfill({ json: { images: [], total: 0, directory_exists: true } }));
  await page.route((url) => url.pathname === "/api/preview/weights", (route) => failed
    ? route.fulfill({ status: 503, json: { error: "weights unavailable" } })
    : route.fulfill({ json: { weights: [{ file: "saved.safetensors", name: "saved.safetensors", size_bytes: 1 }] } }));
  await page.goto("/next/history/fixture-run");
  const summary = page.getByLabel("训练产物摘要");
  await expect(summary).toContainText("weights unavailable");
  await expect(summary.locator("dl")).toContainText("样张0 项");
  failed = false;
  await summary.getByRole("button", { name: "重试权重摘要" }).click();
  await expect(summary).toContainText("已发现 1 项（总数未记录）");
  await expect(summary).toContainText("saved.safetensors");
  expect(mocks.writes).toEqual([]); expect(mocks.unhandled).toEqual([]);
});

test("overview result summary keeps weights readable while images recover to a missing directory", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let imageFailure = true;
  let imageReads = 0;
  await page.route((url) => url.pathname === "/api/preview/images", (route) => {
    imageReads += 1;
    return imageFailure
      ? route.fulfill({ status: 503, json: { error: "sample directory unavailable" } })
      : route.fulfill({ json: { images: [], total: 0, directory_exists: false } });
  });
  await page.route((url) => url.pathname === "/api/preview/weights", (route) => route.fulfill({ json: {
    weights: [{ file: "saved.safetensors", name: "saved.safetensors", size_bytes: 1 }],
    total: 1,
    directory_exists: true,
  } }));

  await page.goto("/next/history/fixture-run");
  const summary = page.getByLabel("训练产物摘要");
  await expect(summary.getByRole("alert")).toContainText("sample directory unavailable");
  await expect(summary.locator("dl")).toContainText("权重1 项");
  await expect(summary).toContainText("saved.safetensors");
  const readsBeforeRetry = imageReads;

  imageFailure = false;
  await summary.getByRole("button", { name: "重试样张摘要", exact: true }).click();
  await expect(summary.getByRole("alert")).toHaveCount(0);
  await expect(summary.locator("dl")).toContainText("样张目录不存在");
  await expect(summary.locator("dl")).toContainText("权重1 项");
  await expect(summary).toContainText("saved.safetensors");
  expect(imageReads).toBe(readsBeforeRetry + 1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});
