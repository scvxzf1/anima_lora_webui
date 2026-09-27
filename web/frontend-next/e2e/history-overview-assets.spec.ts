import { expect, test, type Page } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

const historyDetail = {
  task: { id: "fixture-run", job: "training", state: "error", name: "Krea-2 Portrait / checkpoint failure", last_step: 6400, final_loss: 0.08, metric_count: 73, started_at: 100, finished_at: 3700, message: "CUDA out of memory while saving checkpoint", output_dir: "output/runs/portrait-study/long-result-directory/checkpoints" },
  metrics: [],
  config_toml: 'model_family="krea2_raw"\nnetwork_dim=32\nbase_compute="nf4"',
};

async function historyAssetsFixture(page: Page) {
  const mocks = await mockWorkspace(page);
  const reads: URL[] = [];
  await page.route((url) => url.pathname === "/api/training/history/fixture-run", (route) => route.fulfill({ json: historyDetail }));
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

test("history detail announces loading until its snapshot arrives", async ({ page }) => {
  const mocks = await historyAssetsFixture(page);
  let releaseDetail = () => {};
  const detailGate = new Promise<void>((resolve) => { releaseDetail = resolve; });
  let requests = 0;
  await page.route((url) => url.pathname === "/api/training/history/fixture-run", async (route) => {
    requests += 1;
    await detailGate;
    await route.fulfill({ json: historyDetail });
  });

  await page.goto("/next/history/fixture-run");
  await expect(page.getByRole("status")).toContainText("正在读取历史任务");
  expect(requests).toBeGreaterThanOrEqual(1);
  releaseDetail();
  await expect(page.getByRole("heading", { name: /Krea-2 Portrait/ })).toBeVisible();
  await expect(page.getByText("正在读取历史任务", { exact: true })).toHaveCount(0);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("history detail reports a failed snapshot read and recovers on explicit retry", async ({ page }) => {
  const mocks = await historyAssetsFixture(page);
  let requests = 0;
  let allowRecovery = false;
  await page.route((url) => url.pathname === "/api/training/history/fixture-run", (route) => {
    requests += 1;
    return allowRecovery
      ? route.fulfill({ json: historyDetail })
      : route.fulfill({ status: 503, json: { error: "历史任务服务暂不可用" } });
  });

  await page.goto("/next/history/fixture-run");
  const error = page.locator(".history-detail-error");
  await expect(error).toContainText("无法读取历史任务");
  await expect(error).toContainText("历史任务服务暂不可用");
  await expect(error.getByRole("button", { name: "重新读取" })).toBeEnabled();
  const failedRequests = requests;
  allowRecovery = true;
  await error.getByRole("button", { name: "重新读取" }).click();
  await expect(page.getByRole("heading", { name: /Krea-2 Portrait/ })).toBeVisible();
  await expect(error).toHaveCount(0);
  expect(requests).toBeGreaterThan(failedRequests);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("history detail explains empty read-only views for preprocessing tasks", async ({ page }) => {
  const mocks = await historyAssetsFixture(page);
  await page.route((url) => url.pathname === "/api/training/history/fixture-run", (route) => route.fulfill({ json: {
    task: { id: "fixture-run", job: "preprocess", state: "done", name: "Dataset preprocessing" },
    metrics: [],
    system: [],
    logs: [],
    config_toml: "",
  } }));

  await page.goto("/next/history/fixture-run");
  await expect(page.getByRole("heading", { name: "Dataset preprocessing" })).toBeVisible();
  await expect(page.getByRole("button", { name: "检查点续训" })).toHaveCount(0);
  await page.getByRole("link", { name: "指标", exact: true }).click();
  await expect(page.getByText("此任务不适用训练 Loss。", { exact: true })).toBeVisible();
  await page.getByRole("link", { name: "配置", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("此任务未保存配置快照。");
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("history success keeps summary, artifacts, and logs on one task identity", async ({ page }) => {
  const mocks = await historyAssetsFixture(page);
  const checkpoint = "output/runs/portrait-study/checkpoints/step-6400";
  const submissions: Array<{ task_id?: string; checkpoint?: string }> = [];
  await page.route((url) => url.pathname === "/api/training/history/fixture-run", (route) => route.fulfill({ json: {
    task: {
      ...historyDetail.task,
      state: "done",
      name: "Krea-2 Portrait / completed",
      message: "训练已完成",
      last_step: 6400,
      final_loss: 0.08,
      metric_count: 2,
      log_count: 1,
    },
    metrics: [{ step: 3200, loss: 0.12, lr: 0.00002 }, { step: 6400, loss: 0.08, lr: 0.00001 }],
    system: [{ timestamp: 100, vram_used_gb: 11.2 }],
    logs: [{ line: "completed step 6400" }],
    config_toml: historyDetail.config_toml,
  } }));
  await page.route((url) => url.pathname === "/api/training/history/fixture-run/logs", (route) => {
    const url = new URL(route.request().url());
    return route.fulfill({ json: {
      logs: [{ line: "completed step 6400" }],
      offset: Number(url.searchParams.get("offset") || 0),
      total: 1,
    } });
  });
  await page.route((url) => url.pathname === "/api/training/history/fixture-run/resume-options", (route) => route.fulfill({ json: {
    checkpoints: [{ path: checkpoint, name: "step-6400", step: 6400, target_total_steps: 8000, remaining_steps: 1600, resume_available: true, state_integrity: { ok: true } }],
    default_checkpoint: checkpoint,
    message: "",
  } }));
  await page.route((url) => url.pathname === "/api/training/resume", (route) => {
    submissions.push(route.request().postDataJSON() as { task_id?: string; checkpoint?: string });
    return route.fulfill({ json: { ok: true, message: "续训任务已创建" } });
  });

  await page.goto("/next/history/fixture-run");
  await expect(page.getByRole("heading", { name: "Krea-2 Portrait / completed", exact: true })).toBeVisible();
  await expect(page.getByRole("region", { name: "执行结果" })).toHaveAttribute("data-state", "done");
  await expect(page.getByRole("region", { name: "执行结果" })).toContainText("训练已完成");
  await expect(page.getByLabel("训练摘要")).toContainText("6400");
  await expect(page.getByLabel("训练产物摘要")).toContainText("125 项");
  await expect(page.getByRole("link", { name: "config.snapshot.toml" })).toBeVisible();
  await expect(page.getByRole("button", { name: "检查点续训", exact: true })).toBeVisible();

  await page.getByRole("link", { name: "产物", exact: true }).click();
  await expect(page.locator(".history-image-grid img")).toHaveCount(60);
  await page.getByRole("link", { name: "日志", exact: true }).click();
  await expect(page.getByRole("log")).toContainText("completed step 6400");
  await page.getByRole("link", { name: "概览", exact: true }).click();
  await page.getByRole("button", { name: "检查点续训", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "从历史检查点续训" });
  await expect(dialog.getByRole("checkbox")).toBeEnabled();
  await dialog.getByRole("checkbox").check();
  await dialog.getByRole("button", { name: "确认续训", exact: true }).click();
  await expect(dialog.getByRole("status")).toContainText("续训任务已创建");
  await expect(dialog.getByRole("link", { name: "查看监控", exact: true })).toHaveAttribute("href", "/next/monitor");
  expect(submissions).toEqual([{ task_id: "fixture-run", checkpoint }]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("history artifact sample and weight failures recover independently", async ({ page }) => {
  const mocks = await historyAssetsFixture(page);
  let imageFailure = true;
  let weightFailure = true;
  let imageReads = 0;
  let weightReads = 0;
  await page.route((url) => url.pathname === "/api/preview/images", (route) => {
    imageReads += 1;
    return imageFailure
      ? route.fulfill({ status: 503, json: { error: "样张目录不可读" } })
      : route.fulfill({ json: { images: [], total: 0, directory_exists: true } });
  });
  await page.route((url) => url.pathname === "/api/preview/weights", (route) => {
    weightReads += 1;
    return weightFailure
      ? route.fulfill({ status: 503, json: { error: "权重目录不可读" } })
      : route.fulfill({ json: { weights: [], total: 0, directory_exists: true } });
  });

  await page.goto("/next/history/fixture-run?view=artifacts");
  const assets = page.locator(".history-assets");
  const imageError = assets.getByRole("alert").filter({ hasText: "样张目录不可读" });
  const weightError = assets.getByRole("alert").filter({ hasText: "权重目录不可读" });
  await expect(imageError).toBeVisible();
  await expect(weightError).toBeVisible();

  const imageReadsBeforeRetry = imageReads;
  imageFailure = false;
  await imageError.getByRole("button", { name: "重试样张", exact: true }).click();
  await expect(assets).toContainText("本页暂无样张");
  await expect(imageError).toHaveCount(0);
  await expect(weightError).toBeVisible();
  expect(imageReads).toBe(imageReadsBeforeRetry + 1);

  const weightReadsBeforeRetry = weightReads;
  weightFailure = false;
  await weightError.getByRole("button", { name: "重试权重", exact: true }).click();
  await expect(assets).toContainText("本页暂无权重");
  await expect(weightError).toHaveCount(0);
  expect(weightReads).toBe(weightReadsBeforeRetry + 1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("history success keeps incomplete artifacts and missing directories explicit", async ({ page }) => {
  const mocks = await historyAssetsFixture(page);
  await page.route((url) => url.pathname === "/api/training/history/fixture-run", (route) => route.fulfill({ json: {
    task: {
      ...historyDetail.task,
      state: "done",
      name: "Krea-2 Portrait / partial artifacts",
      message: "训练已完成，但部分产物不可用",
      last_step: 6400,
      final_loss: 0.08,
      metric_count: 1,
    },
    metrics: [{ step: 6400, loss: 0.08, lr: 0.00001 }],
    system: [],
    logs: [{ line: "completed step 6400" }],
    config_toml: historyDetail.config_toml,
  } }));
  await page.route((url) => url.pathname.endsWith("/artifacts"), (route) => route.fulfill({ json: {
    artifacts: [
      { key: "config-snapshot", state: "available", name: "config.snapshot.toml", size_bytes: 96 },
      { key: "logs", state: "unreadable", message: "日志文件不可读" },
      { key: "runtime-config", state: "missing", message: "运行配置不存在" },
    ],
  } }));
  await page.route((url) => url.pathname === "/api/preview/images", (route) => route.fulfill({ json: {
    images: [], total: 0, offset: 0, next_offset: null, directory_exists: false, message: "样张目录不存在",
  } }));
  await page.route((url) => url.pathname === "/api/preview/weights", (route) => route.fulfill({ json: {
    weights: [], total: 0, offset: 0, next_offset: null, directory_exists: false, message: "权重目录不存在",
  } }));

  await page.goto("/next/history/fixture-run");
  await expect(page.getByRole("heading", { name: "Krea-2 Portrait / partial artifacts", exact: true })).toBeVisible();
  const summary = page.getByLabel("训练产物摘要");
  await expect(summary).toContainText("样张");
  await expect(summary).toContainText("权重");
  await expect(summary.getByText("目录不存在", { exact: true })).toHaveCount(2);

  await page.getByRole("link", { name: "产物", exact: true }).click();
  await expect(page.getByRole("link", { name: "config.snapshot.toml", exact: true })).toBeVisible();
  const logsRow = page.getByText("完整日志", { exact: true }).locator("..");
  await expect(logsRow).toContainText("日志文件不可读");
  await expect(logsRow.getByRole("link")).toHaveCount(0);
  const runtimeRow = page.getByText("运行配置", { exact: true }).locator("..");
  await expect(runtimeRow).toContainText("运行配置不存在");
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

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
