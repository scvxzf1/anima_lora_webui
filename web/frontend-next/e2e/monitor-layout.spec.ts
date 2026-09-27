import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

for (const viewport of [{ width: 1280, height: 720 }, { width: 390, height: 844 }]) {
  test(`idle monitor stays compact at ${viewport.width}px`, async ({ page }, info) => {
    await page.setViewportSize(viewport);
    if (viewport.width > 1000) await page.addInitScript(() => localStorage.setItem("dragon-next-ui-v1-theme", "light"));
    const mocks = await mockWorkspace(page);
    await page.route((url) => url.pathname === "/api/training/status", (route) =>
      route.fulfill({ json: { status: "idle" } }),
    );
    await page.goto("/next/monitor");
    await expect(page.getByRole("region", { name: "空闲状态" })).toContainText("暂无当前任务。");
    await expect(page.getByRole("navigation", { name: "空闲时操作" }).getByRole("link", { name: "训练队列" })).toBeVisible();
    await expect(page.locator(".monitor-log-panel, .monitor-chart-panel, .monitor-stop")).toHaveCount(0);
    await expect(page.getByRole("region", { name: "设备信息" })).toContainText("Fixture GPU");
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.screenshot({ path: info.outputPath(`monitor-idle-${viewport.width}.png`), fullPage: true });
    expect(mocks.writes).toEqual([]);
    expect(mocks.unhandled).toEqual([]);
  });
}

for (const viewport of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
  test(`monitor keeps progress, trends, devices and logs readable at ${viewport.width}px`, async ({ page }, info) => {
    await page.setViewportSize(viewport);
    if (viewport.width > 1000) await page.addInitScript(() => localStorage.setItem("dragon-next-ui-v1-theme", "light"));
    const mocks = await mockWorkspace(page);
    await page.goto("/next/monitor");

    await expect(page.getByRole("progressbar", { name: "任务进度" })).toHaveAttribute("aria-valuenow", "52.5");
    await expect(page.getByRole("region", { name: "训练趋势" })).toBeVisible();
    await expect.poll(() => page.locator(".metric-chart canvas").first().evaluate((canvas) => {
      const image = (canvas as HTMLCanvasElement).getContext("2d")?.getImageData(0, 0, (canvas as HTMLCanvasElement).width, (canvas as HTMLCanvasElement).height);
      if (!image) return 0;
      let visible = 0;
      for (let index = 3; index < image.data.length; index += 4) if (image.data[index] > 0) visible += 1;
      return visible;
    })).toBeGreaterThan(1000);
    await expect(page.getByRole("region", { name: "设备信息" })).toContainText("Fixture GPU");
    await expect(page.getByRole("region", { name: "实时日志" })).toContainText("step 840");
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    const bounds = await page.locator(".monitor-workspace").evaluate((node) => {
      const chart = node.querySelector(".monitor-chart-panel")!.getBoundingClientRect();
      const gpu = node.querySelector(".monitor-gpu-panel")!.getBoundingClientRect();
      const log = node.querySelector(".monitor-log-panel")!.getBoundingClientRect();
      return { chartRight: chart.right, chartBottom: chart.bottom, gpuLeft: gpu.left, gpuTop: gpu.top, gpuBottom: gpu.bottom, logTop: log.top };
    });
    if (viewport.width > 1100) expect(bounds.chartRight).toBeLessThan(bounds.gpuLeft);
    else expect(bounds.chartBottom).toBeLessThanOrEqual(bounds.gpuTop);
    expect(bounds.logTop).toBeGreaterThanOrEqual(bounds.gpuBottom);
    await page.screenshot({ path: info.outputPath(`monitor-${viewport.width}.png`), fullPage: true });
    expect(mocks.writes).toEqual([]);
    expect(mocks.unhandled).toEqual([]);
  });
}

test("monitor keeps status and dependent panels in loading states without stale data", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let announceStatusStarted = () => {};
  let releaseStatusResponse = () => {};
  let announceGpuStarted = () => {};
  let releaseGpuResponse = () => {};
  let announceMetricsStarted = () => {};
  let releaseMetricsResponse = () => {};
  let announceLogsStarted = () => {};
  let releaseLogsResponse = () => {};
  const statusStarted = new Promise<void>(resolve => { announceStatusStarted = resolve; });
  const statusGate = new Promise<void>(resolve => { releaseStatusResponse = resolve; });
  const gpuStarted = new Promise<void>(resolve => { announceGpuStarted = resolve; });
  const gpuGate = new Promise<void>(resolve => { releaseGpuResponse = resolve; });
  const metricsStarted = new Promise<void>(resolve => { announceMetricsStarted = resolve; });
  const metricsGate = new Promise<void>(resolve => { releaseMetricsResponse = resolve; });
  const logsStarted = new Promise<void>(resolve => { announceLogsStarted = resolve; });
  const logsGate = new Promise<void>(resolve => { releaseLogsResponse = resolve; });
  await page.route(url => url.pathname === "/api/training/status", async route => {
    if (route.request().method() !== "GET") return route.fallback();
    announceStatusStarted();
    await statusGate;
    return route.fallback();
  });
  await page.route(url => url.pathname === "/api/training/gpus", async route => {
    if (route.request().method() !== "GET") return route.fallback();
    announceGpuStarted();
    await gpuGate;
    return route.fallback();
  });
  await page.route(url => url.pathname === "/api/training/metrics", async route => {
    if (route.request().method() !== "GET") return route.fallback();
    announceMetricsStarted();
    await metricsGate;
    return route.fallback();
  });
  await page.route(url => url.pathname === "/api/training/logs", async route => {
    if (route.request().method() !== "GET") return route.fallback();
    announceLogsStarted();
    await logsGate;
    return route.fallback();
  });

  await page.goto("/next/monitor");
  await Promise.all([statusStarted, gpuStarted]);
  await expect(page.getByLabel("任务状态读取状态")).toContainText("正在读取任务状态");
  await expect(page.getByRole("status", { name: "当前任务状态" })).toHaveText("读取中");
  await expect(page.locator(".monitor-workspace")).toHaveCount(0);
  await expect(page.getByRole("region", { name: "设备信息" })).toContainText("正在读取GPU");

  releaseStatusResponse();
  await Promise.all([metricsStarted, logsStarted]);
  await expect(page.locator(".monitor-workspace")).toBeVisible();
  await expect(page.getByLabel("指标读取状态")).toContainText("正在读取指标");
  await expect(page.getByLabel("日志读取状态")).toContainText("正在读取日志");
  await expect(page.getByRole("region", { name: "实时日志" })).not.toContainText("step 840");
  await expect(page.locator(".metric-chart")).toHaveCount(0);

  releaseMetricsResponse();
  releaseLogsResponse();
  releaseGpuResponse();
  await expect(page.getByRole("progressbar", { name: "任务进度" })).toBeVisible();
  await expect(page.getByRole("log")).toContainText("step 840");
  await expect(page.locator(".monitor-gpu-card")).toHaveCount(1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("an idle snapshot is not presented as confirmed when status refresh fails", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let fails = false;
  await page.route((url) => url.pathname === "/api/training/status", (route) =>
    fails ? route.fulfill({ status: 503, json: { error: "status offline" } })
      : route.fulfill({ json: { status: "idle" } }),
  );
  await page.goto("/next/monitor");
  await expect(page.getByRole("region", { name: "空闲状态" })).toContainText("暂无当前任务。");
  fails = true;
  await expect(page.getByLabel("任务状态读取状态")).toContainText("status offline");
  await expect(page.getByRole("region", { name: "空闲状态" })).toContainText("任务状态待确认。");
  await expect(page.getByRole("navigation", { name: "空闲时操作" })).toHaveCount(0);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

for (const viewport of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
  test(`multi-GPU metrics and selection remain readable at ${viewport.width}px`, async ({ page }, info) => {
    await page.setViewportSize(viewport);
    const mocks = await mockWorkspace(page);
    await page.route((url) => url.pathname === "/api/training/status", (route) => route.fulfill({ json: {
      status: "running", task_id: "multi-run", job: "training", gpu_whitelist: [0],
      latest_system: { gpu_indices: [0], vram_used_gb: 6, vram_total_gb: 24, gpu_util: 72, gpu_temp: 62 },
    } }));
    await page.route((url) => url.pathname === "/api/training/gpus", (route) => route.fulfill({ json: {
      ok: true, sampled_at: Date.now() / 1000, stale: false,
      gpus: [
        { index: 0, uuid: "GPU-a", name: "GPU Alpha", memory_used_gb: 6, memory_total_gb: 24, gpu_util: 72, gpu_temp: 62 },
        { index: 1, uuid: "GPU-b", name: "GPU Beta", memory_used_gb: 3, memory_total_gb: 32, gpu_util: 18, gpu_temp: 47 },
      ],
    } }));
    await page.goto("/next/monitor");
    const cards = page.locator(".monitor-gpu-card");
    await expect(cards).toHaveCount(2);
    await expect(cards.nth(0)).toContainText("当前任务已选");
    await expect(cards.nth(0)).toContainText("6.0 GB / 24.0 GB");
    await expect(cards.nth(1)).toContainText("未选用");
    await expect(cards.nth(1)).toContainText("18%");
    await expect(page.getByRole("region", { name: "实时指标" })).toContainText("采样最高温度");
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.screenshot({ path: info.outputPath(`monitor-multi-gpu-${viewport.width}.png`), fullPage: true });
    expect(mocks.writes).toEqual([]);
    expect(mocks.unhandled).toEqual([]);
  });
}

test("GPU sampling failure does not present stale devices as available", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  await page.route((url) => url.pathname === "/api/training/gpus", (route) => route.fulfill({ json: {
    ok: true, gpus: [], sampled_at: null, stale: true,
  } }));
  await page.goto("/next/monitor");
  await expect(page.getByRole("region", { name: "设备信息" })).toContainText("GPU 采样暂不可用");
  await expect(page.locator(".monitor-gpu-card")).toHaveCount(0);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("a running task without an explicit GPU whitelist does not claim a participating card", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  await page.route((url) => url.pathname === "/api/training/status", (route) => route.fulfill({ json: {
    status: "running", task_id: "unresolved-run", job: "training", gpu_whitelist: [],
  } }));
  await page.goto("/next/monitor");
  await expect(page.locator(".monitor-gpu-card")).toContainText("参与状态未确认");
  await expect(page.locator(".monitor-gpu-card")).not.toContainText("当前任务已选");
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("task metrics and logs pause while the status snapshot is unconfirmed", async ({ page }, info) => {
  const mocks = await mockWorkspace(page);
  let fails = false;
  let statusReads = 0;
  let metricReads = 0;
  let logReads = 0;
  await page.route((url) => url.pathname === "/api/training/status", (route) => {
    statusReads += 1;
    return fails ? route.fulfill({ status: 503, json: { error: "status offline" } })
      : route.fulfill({ json: { task_id: "run-A", status: "running", job: "training" } });
  });
  await page.route((url) => url.pathname === "/api/training/metrics", (route) => {
    metricReads += 1;
    return route.fulfill({ json: [{ step: 1, loss: 0.2 }] });
  });
  await page.route((url) => url.pathname === "/api/training/logs", (route) => {
    logReads += 1;
    return route.fulfill({ json: { records: [{ id: 1, line: "last confirmed log" }] } });
  });
  await page.goto("/next/monitor");
  await expect(page.getByRole("log")).toContainText("last confirmed log");
  await expect(page.getByRole("img", { name: /Loss 趋势/ })).toBeVisible();
  fails = true;
  await expect(page.locator(".monitor-stale")).toBeVisible();
  await page.screenshot({ path: info.outputPath("monitor-stale.png"), fullPage: true });
  const reads = { metricReads, logReads };
  const failedStatusReads = statusReads;
  await page.waitForTimeout(5500);
  expect(statusReads).toBe(failedStatusReads);
  expect({ metricReads, logReads }).toEqual(reads);
  await expect(page.getByRole("log")).toContainText("last confirmed log");
  fails = false;
  await page.getByRole("button", { name: "重试任务状态" }).click();
  await expect.poll(() => statusReads).toBeGreaterThan(1);
  await expect(page.locator(".monitor-stale")).toHaveCount(0);
  await expect.poll(() => metricReads + logReads).toBeGreaterThan(reads.metricReads + reads.logReads);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("stop confirmation can be cancelled and closes when the monitored task changes", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let taskId = "run-A";
  await page.route((url) => url.pathname === "/api/training/status", (route) =>
    route.fulfill({ json: { task_id: taskId, status: "running", job: "training" } }),
  );
  await page.goto("/next/monitor");
  const stop = page.getByRole("button", { name: "停止训练", exact: true });
  await stop.click();
  const dialog = page.getByRole("dialog", { name: "停止训练" });
  await expect(dialog).toContainText("run-A");
  await dialog.getByRole("button", { name: "取消" }).click();
  await expect(dialog).toHaveCount(0);

  await stop.click();
  taskId = "run-B";
  await expect(page.getByRole("link", { name: "run-B", exact: true })).toBeVisible();
  await expect(dialog).toHaveCount(0);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});
