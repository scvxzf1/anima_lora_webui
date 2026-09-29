import { expect, test, type Request } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

test("REST recovery replays the latest snapshot and isolates logs by task", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let failNextStatus = false;
  let recoveredStatus = false;
  let recoveredLogs = false;
  const metricsTaskIds: string[] = [];

  await page.route((url) => url.pathname === "/api/training/status", (route) => {
    if (failNextStatus) {
      failNextStatus = false;
      return route.fulfill({ status: 503, json: { error: "status temporarily offline" } });
    }
    if (!recoveredStatus) return route.fulfill({ json: {
      status: "running", task_id: "run-A", job: "training",
      latest_progress: { current: 12, total: 100, loss: 0.81, lr: 0.0001, rate: "4s/it" },
      latest_metric: { step: 12, loss: 0.82, lr: 0.0001 },
      latest_system: { vram_used_gb: 4, vram_total_gb: 24, gpu_temp: 55, gpu_util: 40 },
    } });
    return route.fulfill({ json: {
      status: "running", task_id: "run-B", job: "training",
      latest_progress: { current: 76, total: 100, loss: 0.31, lr: 0.00002, rate: "2s/it" },
      latest_metric: { step: 76, loss: 0.42, lr: 0.00002 },
      latest_system: { vram_used_gb: 11.5, vram_total_gb: 24, gpu_temp: 72, gpu_util: 96 },
    } });
  });
  await page.route((url) => url.pathname === "/api/training/logs", (route) => {
    const taskId = new URL(route.request().url()).searchParams.get("task_id");
    if (taskId === "run-A") return route.fulfill({ json: { records: [{ id: 1, line: "run-A old task log" }] } });
    if (taskId === "run-B") {
      if (!recoveredLogs) return route.fulfill({ status: 503, json: { error: "logs temporarily offline" } });
      return route.fulfill({ json: { records: [{ id: 2, line: "run-B recovered task log" }] } });
    }
    return route.fulfill({ json: { records: [] } });
  });
  await page.route((url) => url.pathname === "/api/training/metrics", (route) => {
    const taskId = new URL(route.request().url()).searchParams.get("task_id");
    if (taskId) metricsTaskIds.push(taskId);
    return route.fulfill({ json: taskId === "run-B" ? [{ step: 76, loss: 0.53, lr: 0.00002 }] : [{ step: 12, loss: 0.83, lr: 0.0001 }] });
  });

  await page.goto("/next/monitor");
  await expect(page.getByRole("link", { name: "run-A", exact: true })).toBeVisible();
  await expect(page.getByRole("log")).toContainText("run-A old task log");
  await expect(page.getByRole("progressbar", { name: "任务进度" })).toHaveAttribute("aria-valuenow", "12");
  await expect(page.getByLabel("实时指标")).toContainText("0.81");
  await expect(page.getByLabel("实时指标")).toContainText("4.0 GB / 24.0 GB");
  await expect(page.getByLabel("实时指标")).toContainText("55°C");
  await expect(page.getByText("最新 Loss: 0.8300")).toBeVisible();

  failNextStatus = true;
  await expect(page.getByLabel("任务状态读取状态")).toContainText("status temporarily offline", { timeout: 7000 });
  await expect(page.locator(".monitor-stale")).toBeVisible();
  await expect(page.getByRole("progressbar", { name: "任务进度" })).toHaveAttribute("aria-valuenow", "12");
  await expect(page.getByLabel("实时指标")).toContainText("0.81");
  await expect(page.getByLabel("实时指标")).toContainText("4.0 GB / 24.0 GB");
  await expect(page.getByLabel("实时指标")).toContainText("55°C");
  await expect(page.getByRole("button", { name: "停止训练", exact: true })).toBeDisabled();
  recoveredStatus = true;
  await page.getByRole("button", { name: "重试任务状态" }).click();

  await expect(page.getByRole("link", { name: "run-B", exact: true })).toBeVisible();
  await expect(page.locator(".monitor-stale")).toHaveCount(0);
  await expect(page.getByLabel("任务状态读取状态")).toHaveCount(0);
  await expect(page.getByRole("progressbar", { name: "任务进度" })).toHaveAttribute("aria-valuenow", "76");
  await expect(page.getByLabel("实时指标")).toContainText("0.31");
  await expect(page.getByLabel("实时指标")).not.toContainText("0.42");
  await expect(page.getByLabel("实时指标")).not.toContainText("0.53");
  await expect(page.getByLabel("实时指标")).toContainText("11.5 GB / 24.0 GB");
  await expect(page.getByLabel("实时指标")).toContainText("72°C");
  await expect(page.getByLabel("日志读取状态")).toContainText("logs temporarily offline");
  await expect(page.locator(".monitor-log-panel")).not.toContainText("run-A old task log");

  recoveredLogs = true;
  await page.getByRole("button", { name: "重试日志" }).click();
  await expect(page.getByRole("log")).toContainText("run-B recovered task log");
  await expect(page.getByRole("log")).not.toContainText("run-A old task log");
  await expect(page.getByRole("group", { name: /Loss 趋势/ })).toBeVisible();
  await expect(page.getByText("最新 Loss: 0.5300")).toBeVisible();
  await expect(page.getByText("最新 Loss: 0.8300")).toHaveCount(0);
  expect(metricsTaskIds).toContain("run-A");
  expect(metricsTaskIds).toContain("run-B");
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("confirmed idle status does not present the previous run snapshot as current", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let idle = false;
  let confirmedIdle = false;
  const metricsTaskIds: Array<string | null> = [];
  const logsTaskIds: Array<string | null> = [];
  const postIdleRequests: string[] = [];

  await page.route((url) => url.pathname === "/api/training/status", (route) => route.fulfill({ json: idle ? {
    status: "idle",
    task_id: "old-run",
    job: "training",
    variant: "lora",
    preset: "default",
    log_count: 37,
    metric_count: 99,
    last_log_line: "old-run task log",
    latest_progress: { current: 99, total: 100, loss: 0.01 },
    latest_metric: { kind: "train", step: 99, loss: 0.02 },
    latest_system: { vram_used_gb: 19, vram_total_gb: 24, gpu_temp: 91, gpu_util: 100 },
  } : {
    status: "running", task_id: "old-run", job: "training",
    latest_progress: { current: 12, total: 100, loss: 0.81 },
    latest_metric: { kind: "train", step: 12, loss: 0.82 },
    latest_system: { vram_used_gb: 4, vram_total_gb: 24, gpu_temp: 55, gpu_util: 40 },
  } }));
  await page.route((url) => url.pathname === "/api/training/logs", (route) => {
    const taskId = new URL(route.request().url()).searchParams.get("task_id");
    logsTaskIds.push(taskId);
    if (confirmedIdle) postIdleRequests.push(`logs:${taskId ?? "none"}`);
    return route.fulfill({ json: { records: [{ id: 1, line: "old-run task log" }] } });
  });
  await page.route((url) => url.pathname === "/api/training/metrics", (route) => {
    const taskId = new URL(route.request().url()).searchParams.get("task_id");
    metricsTaskIds.push(taskId);
    if (confirmedIdle) postIdleRequests.push(`metrics:${taskId ?? "none"}`);
    return route.fulfill({ json: [{ kind: "train", step: 12, loss: 0.83 }] });
  });

  await page.goto("/next/monitor");
  await expect(page.getByRole("link", { name: "old-run", exact: true })).toBeVisible();
  await expect(page.getByRole("log")).toContainText("old-run task log");
  await expect(page.getByLabel("实时指标")).toContainText("4.0 GB / 24.0 GB");

  idle = true;
  await expect(page.getByRole("heading", { name: "暂无当前任务。" })).toBeVisible({ timeout: 7000 });
  await expect(page.getByLabel("任务进度")).toHaveCount(0);
  await expect(page.getByLabel("实时指标")).toHaveCount(0);
  await expect(page.getByLabel("训练趋势")).toHaveCount(0);
  await expect(page.getByLabel("实时日志")).toHaveCount(0);
  await expect(page.getByRole("log")).toHaveCount(0);
  await expect(page.getByText("old-run task log")).toHaveCount(0);
  await expect(page.getByText("91°C")).toHaveCount(0);
  await expect(page.getByText("19.0 GB / 24.0 GB")).toHaveCount(0);
  await expect(page.getByRole("link", { name: "old-run", exact: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "停止训练", exact: true })).toHaveCount(0);
  const idleNavigation = page.getByRole("navigation", { name: "空闲时操作" });
  await expect(idleNavigation.getByRole("link", { name: "训练配置" })).toBeVisible();
  await expect(idleNavigation.getByRole("link", { name: "训练队列" })).toBeVisible();
  confirmedIdle = true;
  await page.waitForTimeout(5500);
  expect(postIdleRequests).toEqual([]);
  expect(metricsTaskIds).toContain("old-run");
  expect(logsTaskIds).toContain("old-run");
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("switching tasks cancels stale metrics and logs without replacing the current task", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let currentTask: "run-A" | "run-B" = "run-A";
  let releaseA!: () => void;
  const aResponseGate = new Promise<void>((resolve) => { releaseA = resolve; });
  type ARequestKind = "metrics" | "logs";
  const aRequests: Record<ARequestKind, Set<Request>> = { metrics: new Set(), logs: new Set() };
  const terminalARequests = new Set<Request>();
  const failedARequests = new Set<Request>();
  const aResponseErrors: string[] = [];
  let notifyARequestChange!: () => void;
  let aRequestChange = new Promise<void>((resolve) => { notifyARequestChange = resolve; });
  const signalARequestChange = () => {
    notifyARequestChange();
    aRequestChange = new Promise<void>((resolve) => { notifyARequestChange = resolve; });
  };
  const markARequestTerminal = (request: Request, failed: boolean) => {
    if (![...aRequests.metrics, ...aRequests.logs].includes(request)) return;
    terminalARequests.add(request);
    if (failed) failedARequests.add(request);
    signalARequestChange();
  };
  page.on("requestfinished", (request) => markARequestTerminal(request, false));
  page.on("requestfailed", (request) => markARequestTerminal(request, true));
  const waitForARequests = async (predicate: () => boolean, description: string) => {
    const deadline = Date.now() + 7000;
    while (!predicate()) {
      const remaining = deadline - Date.now();
      if (remaining <= 0) throw new Error(`Timed out waiting for ${description}`);
      let timeout: ReturnType<typeof setTimeout> | undefined;
      try {
        await Promise.race([
          aRequestChange,
          new Promise<never>((_, reject) => {
            timeout = setTimeout(() => reject(new Error(`Timed out waiting for ${description}`)), remaining);
          }),
        ]);
      } finally {
        if (timeout) clearTimeout(timeout);
      }
    }
  };

  const fulfillAResponse = async (route: import("@playwright/test").Route, request: Request, json: unknown) => {
    try {
      await route.fulfill({ json });
    } catch (error: unknown) {
      const failure = await request.failure();
      if (!failedARequests.has(request) || !failure?.errorText.includes("ERR_ABORTED")) {
        aResponseErrors.push(String(error));
      }
    }
  };

  await page.route((url) => url.pathname === "/api/training/status", (route) => {
    const isB = currentTask === "run-B";
    return route.fulfill({ json: {
      status: "running", task_id: currentTask, job: "training",
      latest_progress: { current: isB ? 76 : 12, total: 100, loss: isB ? 0.31 : 0.81 },
      latest_metric: { step: isB ? 76 : 12, loss: isB ? 0.32 : 0.82 },
      latest_system: { vram_used_gb: isB ? 11.5 : 4, vram_total_gb: 24, gpu_temp: isB ? 72 : 55, gpu_util: isB ? 96 : 40 },
    } });
  });
  await page.route((url) => url.pathname === "/api/training/metrics", async (route) => {
    const taskId = new URL(route.request().url()).searchParams.get("task_id");
    if (taskId === "run-A") {
      const request = route.request();
      aRequests.metrics.add(request);
      signalARequestChange();
      await aResponseGate;
      await fulfillAResponse(route, request, [{ step: 12, loss: 0.91, lr: 0.0001 }]);
      return;
    }
    return route.fulfill({ json: [{ step: 76, loss: 0.27, lr: 0.00002 }] });
  });
  await page.route((url) => url.pathname === "/api/training/logs", async (route) => {
    const taskId = new URL(route.request().url()).searchParams.get("task_id");
    if (taskId === "run-A") {
      const request = route.request();
      aRequests.logs.add(request);
      signalARequestChange();
      await aResponseGate;
      await fulfillAResponse(route, request, { records: [{ id: 1, line: "run-A delayed log" }] });
      return;
    }
    return route.fulfill({ json: { records: [{ id: 2, line: "run-B current log" }] } });
  });

  try {
    await page.goto("/next/monitor");
    await waitForARequests(
      () => aRequests.metrics.size > 0 && aRequests.logs.size > 0,
      "run-A metrics and logs requests",
    );
    currentTask = "run-B";

    await expect(page.getByRole("link", { name: "run-B", exact: true })).toBeVisible({ timeout: 7000 });
    await expect(page.getByRole("progressbar", { name: "任务进度" })).toHaveAttribute("aria-valuenow", "76");
    await expect(page.getByLabel("实时指标")).toContainText("0.31");
    await expect(page.getByText("最新 Loss: 0.2700")).toBeVisible();
    await expect(page.getByRole("log")).toContainText("run-B current log");

    releaseA();
    await waitForARequests(
      () => [...aRequests.metrics, ...aRequests.logs].every((request) => terminalARequests.has(request)),
      "browser completion or cancellation of every run-A request",
    );
    await page.evaluate(() => new Promise<void>((resolve) => {
      requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
    }));
    expect([...aRequests.metrics].every((request) => failedARequests.has(request))).toBe(true);
    expect([...aRequests.logs].every((request) => failedARequests.has(request))).toBe(true);
    for (const request of [...aRequests.metrics, ...aRequests.logs]) {
      expect((await request.failure())?.errorText).toContain("ERR_ABORTED");
    }
    expect(aResponseErrors).toEqual([]);
    await expect(page.getByRole("link", { name: "run-B", exact: true })).toBeVisible();
    await expect(page.getByRole("progressbar", { name: "任务进度" })).toHaveAttribute("aria-valuenow", "76");
    await expect(page.getByLabel("实时指标")).toContainText("0.31");
    await expect(page.getByLabel("实时指标")).not.toContainText("0.81");
    await expect(page.getByText("最新 Loss: 0.2700")).toBeVisible();
    await expect(page.getByText("最新 Loss: 0.9100")).toHaveCount(0);
    await expect(page.getByRole("log")).toContainText("run-B current log");
    await expect(page.getByRole("log")).not.toContainText("run-A delayed log");
    expect(mocks.writes).toEqual([]);
    expect(mocks.unhandled).toEqual([]);
  } finally {
    releaseA();
    if (aRequests.metrics.size > 0 && aRequests.logs.size > 0) {
      await waitForARequests(
        () => [...aRequests.metrics, ...aRequests.logs].every((request) => terminalARequests.has(request)),
        "run-A request cleanup",
      ).catch(() => undefined);
    }
  }
});
