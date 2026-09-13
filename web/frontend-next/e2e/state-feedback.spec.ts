import { expect, test, type WebSocketRoute } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

test("monitor isolates failures, confirms WS hints and binds stop to the clicked task", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  const sockets: WebSocketRoute[] = [];
  await page.routeWebSocket("**/ws/training", (socket) => sockets.push(socket));
  let taskId = "run-A";
  let statusFails = false;
  let metricsFail = false;
  let logsFail = false;
  let stopTarget = "";
  await page.route((url) => url.pathname === "/api/training/status", (route) => route.fulfill({
    status: statusFails ? 503 : 200,
    json: statusFails ? { error: "status offline" } : {
      task_id: taskId, status: "running", job: "training", metric_count: 8000, log_count: 1500,
      latest_progress: { current: taskId === "run-A" ? 840 : 20, total: 1600, loss: 0.08, lr: 2e-7 },
    },
  }));
  await page.route((url) => url.pathname === "/api/training/metrics", (route) => route.fulfill({
    status: metricsFail ? 503 : 200,
    json: metricsFail ? { error: "metrics offline" } : [{ step: 1, loss: 0.08 }],
  }));
  await page.route((url) => url.pathname === "/api/training/logs", (route) => {
    const requested = new URL(route.request().url()).searchParams.get("task_id");
    return route.fulfill({ status: logsFail ? 503 : 200,
      json: logsFail ? { error: "logs offline" } : { records: [{ id: 1, line: `confirmed log ${requested}` }] } });
  });
  await page.route((url) => url.pathname === "/api/training/stop", (route) => {
    stopTarget = route.request().postDataJSON().task_id;
    taskId = "run-B";
    return route.fulfill({ status: 409, json: { error: "当前训练任务已发生变化" } });
  });
  await page.goto("/next/monitor");
  await expect(page.getByRole("log")).toContainText("confirmed log run-A");
  await expect(page.locator(".monitor-metrics")).toContainText("2.00e-7");
  await expect(page.getByRole("link", { name: "run-A", exact: true })).toHaveAttribute("href", "/next/history/run-A");
  await expect.poll(() => sockets.length).toBeGreaterThan(0);
  for (const event of [
    { type: "progress", current: 999999, total: 999999 },
    { type: "progress", task_id: "old-run", current: 999999 },
    { type: "log", task_id: "old-run", id: 99, line: "stale injected log" },
  ]) sockets.at(-1)!.send(JSON.stringify(event));
  await page.waitForTimeout(1000);
  await expect(page.locator(".monitor-progress-copy")).not.toContainText("999999");
  await expect(page.getByRole("log")).not.toContainText("stale injected log");

  statusFails = true;
  await expect(page.getByLabel("任务状态读取状态")).toContainText("status offline");
  await expect(page.getByRole("log")).toContainText("confirmed log run-A");
  await expect(page.getByRole("img", { name: /Loss 趋势/ })).toBeVisible();
  await expect(page.getByRole("button", { name: "停止训练", exact: true })).toBeDisabled();
  statusFails = false;
  await page.getByRole("button", { name: "重试任务状态", exact: true }).click();
  await expect(page.getByRole("button", { name: "停止训练", exact: true })).toBeEnabled();

  metricsFail = true;
  sockets.at(-1)!.send(JSON.stringify({ type: "metrics", task_id: taskId }));
  await expect(page.getByLabel("指标读取状态")).toContainText("metrics offline");
  await expect(page.getByRole("log")).toContainText("confirmed log run-A");
  await expect(page.getByLabel("指标读取状态")).toContainText("保留上次数据");
  metricsFail = false;
  await page.getByRole("button", { name: "重试指标", exact: true }).click();
  await expect(page.getByLabel("指标读取状态")).not.toContainText("offline");
  logsFail = true;
  await expect(page.getByLabel("日志读取状态")).toContainText("logs offline");
  await expect(page.getByRole("img", { name: /Loss 趋势/ })).toBeVisible();
  logsFail = false;
  await page.getByRole("button", { name: "重试日志", exact: true }).click();

  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "停止训练", exact: true }).click();
  await expect(page.getByRole("heading", { name: "停止训练失败" })).toBeVisible();
  expect(stopTarget).toBe("run-A");
  await expect(page.getByRole("link", { name: "run-B", exact: true })).toBeVisible();
  await expect(page.getByRole("log")).toContainText("confirmed log run-B");
  await expect(page.getByRole("log")).not.toContainText("run-A");
  sockets.at(-1)!.send(JSON.stringify({ type: "status", task_id: "run-A", state: "idle" }));
  await page.waitForTimeout(1000);
  await expect(page.getByRole("link", { name: "run-B", exact: true })).toBeVisible();
  await expect(page.locator(".monitor-state")).toHaveText("运行中");
  await sockets.at(-1)!.close();
  taskId = "run-C";
  await expect(page.getByRole("link", { name: "run-C", exact: true })).toBeVisible();
  await expect(page.getByRole("log")).toContainText("confirmed log run-C");
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("queue distinguishes loading, failure, cached snapshot and confirmed empty", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let release: () => void = () => {};
  const gate = new Promise<void>((resolve) => { release = resolve; });
  let phase: "loading" | "error" | "data" | "empty" = "loading";
  await page.route((url) => url.pathname === "/api/training/queue", async (route) => {
    if (phase === "loading") await gate;
    if (phase === "error") return route.fulfill({ status: 503, json: { error: "queue offline" } });
    return route.fulfill({ json: {
      paused: true, status: "idle", summary: { total: phase === "data" ? 1 : 0, queued: phase === "data" ? 1 : 0 },
      items: phase === "data" ? [{ id: "queued-a", state: "queued", variant: "Queue fixture" }] : [],
    } });
  });
  await page.goto("/next/queue");
  await expect(page.getByLabel("队列状态读取状态")).toContainText("正在读取");
  await expect(page.getByLabel("队列统计")).toHaveCount(0);
  await expect(page.getByText("当前筛选下没有队列任务。")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "暂停队列", exact: true })).toBeDisabled();
  phase = "error"; release();
  await expect(page.getByLabel("队列状态读取状态")).toContainText("queue offline");
  await expect(page.locator(".queue-badge")).toHaveText("状态待确认");
  phase = "data";
  await page.getByRole("button", { name: "重试队列状态" }).click();
  await expect(page.locator(".queue-card")).toHaveCount(1);
  await expect(page.getByRole("button", { name: "继续队列", exact: true })).toBeEnabled();
  await expect(page.getByRole("button", { name: "清理已完成", exact: true })).toBeDisabled();
  phase = "error";
  await page.getByRole("button", { name: "刷新", exact: true }).click();
  await expect(page.getByLabel("队列状态读取状态")).toContainText("保留上次数据");
  await expect(page.locator(".queue-card")).toHaveCount(1);
  await expect(page.getByRole("button", { name: "继续队列", exact: true })).toBeDisabled();
  phase = "empty";
  await page.getByRole("button", { name: "重试队列状态" }).click();
  await expect(page.getByText("当前筛选下没有队列任务。")).toBeVisible();
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("settings read failure is not synced or loading and can be retried", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let fail = true;
  await page.route((url) => url.pathname === "/api/settings/global", (route) => fail
    ? route.fulfill({ status: 503, json: { error: "settings offline" } }) : route.fallback());
  await page.goto("/next/settings");
  await expect(page.getByLabel("设置读取状态")).toContainText("settings offline");
  await expect(page.locator(".settings-page .state-label")).toHaveText("读取失败");
  await expect(page.getByText("正在读取设置", { exact: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "保存设置", exact: true })).toHaveCount(0);
  fail = false;
  await page.getByRole("button", { name: "重试设置" }).click();
  await expect(page.locator(".settings-page .state-label")).toHaveText("已同步");
  await expect(page.getByRole("button", { name: "保存设置", exact: true })).toBeDisabled();
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("model library read failure preserves its error state and retries", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let fail = true;
  await page.route((url) => url.pathname === "/api/settings/model-configs", (route) => fail
    ? route.fulfill({ status: 503, json: { error: "models offline" } }) : route.fallback());
  await page.goto("/next/models");
  await expect(page.getByLabel("模型配置读取状态")).toContainText("models offline");
  await expect(page.locator(".settings-page .state-label")).toHaveText("读取失败");
  await expect(page.getByText("正在读取模型库", { exact: true })).toHaveCount(0);
  fail = false;
  await page.getByRole("button", { name: "重试模型配置" }).click();
  await expect(page.getByRole("textbox", { name: "名称", exact: true })).toHaveValue("Krea-2 Studio");
  await expect(page.locator(".settings-page .state-label")).toHaveText("已同步");
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 720 }, { width: 768, height: 1024 }, { width: 390, height: 844 }]) {
  for (const theme of ["dark", "light"]) {
    test(`monitor and queue layout ${viewport.width} ${theme}`, async ({ page }, info) => {
      await page.setViewportSize(viewport);
      const mocks = await mockWorkspace(page);
      await page.addInitScript((theme) => localStorage.setItem("dragon-next-ui-v1-theme", theme), theme);
      for (const route of ["monitor", "queue"]) {
        await page.goto(`/next/${route}`);
        await expect(page.getByLabel(route === "monitor" ? "任务状态读取状态" : "队列状态读取状态")).toContainText("最近成功读取");
        if (route === "monitor") await expect(page.locator("canvas")).toHaveCount(1);
        else await expect(page.locator(".queue-card")).toHaveCount(3);
        await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
        await page.screenshot({ path: info.outputPath(`${route}.png`), fullPage: true });
      }
      expect(mocks.writes).toEqual([]);
      expect(mocks.unhandled).toEqual([]);
    });
  }
}
