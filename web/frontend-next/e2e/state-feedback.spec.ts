import { expect, test, type WebSocketRoute } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

test("monitor isolates failures, confirms WS hints and binds stop to the clicked task", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  const sockets: WebSocketRoute[] = [];
  await page.routeWebSocket("**/ws/training", (socket) => sockets.push(socket));
  let taskId = "run-A";
  let taskState = "running";
  let statusFails = false;
  let metricsFail = false;
  let logsFail = false;
  let stopTarget = "";
  await page.route((url) => url.pathname === "/api/training/status", (route) => route.fulfill({
    status: statusFails ? 503 : 200,
    json: statusFails ? { error: "status offline" } : {
      task_id: taskId, status: taskState, job: "training", metric_count: 8000, log_count: 1500,
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
  const taskStatus = page.getByRole("status", { name: "当前任务状态" });
  await expect(taskStatus).toHaveText("运行中");
  await expect(page.getByRole("status", { name: "实时连接状态" })).toHaveText("实时连接已建立");
  taskState = "idle";
  await expect(taskStatus).toHaveText("空闲");
  taskState = "running";
  await expect(taskStatus).toHaveText("运行中");
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
  await expect(page.locator(".monitor-stale")).toContainText("上次成功读取的任务快照");
  await expect(page.getByRole("log")).toContainText("confirmed log run-A");
  await expect(page.getByRole("img", { name: /Loss 趋势/ })).toBeVisible();
  await expect(page.getByRole("button", { name: "停止训练", exact: true })).toBeDisabled();
  statusFails = false;
  await page.getByRole("button", { name: "重试任务状态", exact: true }).click();
  await expect(page.locator(".monitor-stale")).toHaveCount(0);
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

  await page.getByRole("button", { name: "停止训练", exact: true }).click();
  await page.getByRole("dialog", { name: "停止训练" }).getByRole("button", { name: "停止训练" }).click();
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

test("empty model library conflict stays an error and exposes no destructive editor", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  const methods: string[] = [];
  await page.route((url) => url.pathname === "/api/settings/model-configs", (route) => {
    methods.push(route.request().method());
    return route.fulfill({ status: 409, json: { error: "模型配置库为空，拒绝覆盖" } });
  });

  await page.goto("/next/models");
  await expect(page.getByLabel("模型配置读取状态")).toContainText("模型配置库为空，拒绝覆盖");
  await expect(page.locator(".settings-page .state-label")).toHaveText("读取失败");
  await expect(page.getByRole("button", { name: "重试模型配置" })).toBeVisible();
  await expect(page.getByRole("button", { name: "新建模型配置" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "保存模型配置", exact: true })).toHaveCount(0);
  await expect(page.getByRole("textbox", { name: "名称", exact: true })).toHaveCount(0);
  expect(methods.length).toBeGreaterThan(0);
  expect(methods.every((method) => method === "GET")).toBe(true);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("model revision conflict keeps the draft until server reload is confirmed", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let conflicted = false;
  let reads = 0;
  let saveBody: unknown;
  const model = (name: string) => ({
    id: "model-a",
    name,
    model_family: "krea2_raw",
    pretrained_model_name_or_path: "models/diffusion_models/krea2_raw.safetensors",
    qwen3: "models/text_encoders/qwen3vl.safetensors",
    vae: "models/vae/qwen.safetensors",
    complete: true,
  });
  await page.route((url) => url.pathname === "/api/settings/model-configs", (route) => {
    if (route.request().method() === "GET") {
      reads += 1;
      return route.fulfill({ json: {
        items: [model(conflicted ? "Server revision" : "Krea-2 Studio")],
        groups: [{ id: "group-a", label: "Local models", item_ids: ["model-a"] }],
        revision: conflicted ? "revision-2" : "revision-1",
        default_id: "model-a",
      } });
    }
    if (route.request().method() === "PUT") {
      conflicted = true;
      saveBody = route.request().postDataJSON();
      return route.fulfill({ status: 409, json: { error: "model revision conflict" } });
    }
    return route.fallback();
  });

  await page.goto("/next/models");
  const name = page.getByRole("textbox", { name: "名称", exact: true });
  const save = page.getByRole("button", { name: "保存模型配置", exact: true });
  await expect(name).toHaveValue("Krea-2 Studio");
  await name.fill("Local draft");
  await save.click();

  await expect(page.getByRole("alert")).toContainText("model revision conflict");
  await expect(save).toBeDisabled();
  const reload = page.getByRole("button", { name: "重新载入服务器版本", exact: true });
  await expect(reload).toBeVisible();
  expect((saveBody as { revision: string }).revision).toBe("revision-1");
  expect((saveBody as { items: Array<{ name: string }> }).items[0].name).toBe("Local draft");
  const initialReads = reads;

  let dismissedMessage = "";
  page.once("dialog", async (dialog) => {
    dismissedMessage = dialog.message();
    await dialog.dismiss();
  });
  await reload.click();
  await expect(name).toHaveValue("Local draft");
  await expect(page.locator(".settings-page .state-label")).toHaveText("未保存");
  expect(dismissedMessage).toContain("会丢弃当前未保存修改");
  expect(reads).toBe(initialReads);

  page.once("dialog", (dialog) => dialog.accept());
  await reload.click();
  await expect(name).toHaveValue("Server revision");
  await expect(page.locator(".settings-page .state-label")).toHaveText("已同步");
  await expect(save).toBeDisabled();
  await expect(page.getByRole("alert")).toHaveCount(0);
  expect(reads).toBeGreaterThan(initialReads);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("model library protects its default and locks edits while saving a replacement", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  const models = [
    {
      id: "model-a",
      name: "Primary model",
      model_family: "anima",
      pretrained_model_name_or_path: "models/primary/dit.safetensors",
      qwen3: "models/primary/qwen.safetensors",
      vae: "models/primary/vae.safetensors",
    },
    {
      id: "model-b",
      name: "Alternate model",
      model_family: "krea2_raw",
      pretrained_model_name_or_path: "models/alternate/dit.safetensors",
      qwen3: "models/alternate/qwen.safetensors",
      vae: "models/alternate/vae.safetensors",
    },
  ];
  let submitted: Record<string, unknown> | undefined;
  let markSaveStarted!: () => void;
  const saveStarted = new Promise<void>((resolve) => {
    markSaveStarted = resolve;
  });
  let releaseSave!: () => void;
  const saveResponse = new Promise<void>((resolve) => {
    releaseSave = resolve;
  });
  await page.route((url) => url.pathname === "/api/settings/model-configs", async (route) => {
    if (route.request().method() === "GET")
      return route.fulfill({ json: {
        items: models,
        groups: [{ id: "group-a", label: "Local models", item_ids: ["model-a", "model-b"] }],
        revision: "revision-1",
        default_id: "model-a",
      } });
    if (route.request().method() === "PUT") {
      const body = route.request().postDataJSON() as Record<string, unknown>;
      submitted = body;
      markSaveStarted();
      await saveResponse;
      return route.fulfill({ json: { ...body, revision: "revision-2" } });
    }
    return route.fallback();
  });

  await page.goto("/next/models");
  const remove = page.getByRole("button", { name: "删除配置", exact: true });
  await expect(remove).toBeDisabled();
  await page.getByRole("button", { name: /Alternate model/ }).click();
  const defaultModel = page.getByRole("checkbox", { name: "默认模型组合" });
  await defaultModel.check();
  await expect(defaultModel).toBeChecked();
  await expect(defaultModel).toBeDisabled();

  await page.getByRole("button", { name: /Primary model/ }).click();
  await expect(remove).toBeEnabled();
  await page.getByRole("button", { name: /Alternate model/ }).click();
  const save = page.getByRole("button", { name: "保存模型配置", exact: true });
  await save.click();
  await saveStarted;
  await expect(save).toBeDisabled();
  await expect(page.getByRole("textbox", { name: "名称", exact: true })).toBeDisabled();
  expect(submitted).toMatchObject({ default_id: "model-b", revision: "revision-1" });

  releaseSave();
  await expect(page.getByRole("status")).toHaveText("模型配置已保存");
  await expect(page.locator(".settings-page .state-label")).toHaveText("已同步");
  await expect(page.getByRole("button", { name: /Alternate model/ })).toContainText("默认");
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("settings loading gate keeps the form unavailable until the server responds", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let release!: () => void;
  const responseGate = new Promise<void>((resolve) => { release = resolve; });
  await page.route((url) => url.pathname === "/api/settings/global", async (route) => {
    await responseGate;
    return route.fallback();
  });

  await page.goto("/next/settings");
  const state = page.getByLabel("设置读取状态");
  await expect(state.getByRole("status")).toContainText("正在读取设置");
  await expect(page.locator(".settings-page .state-label")).toHaveText("读取中");
  await expect(page.getByRole("button", { name: "保存设置", exact: true })).toHaveCount(0);

  release();
  await expect(page.locator(".settings-page .state-label")).toHaveText("已同步");
  await expect(page.getByRole("spinbutton", { name: "全局缩放 (%)", exact: true })).toHaveValue("100");
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("settings save locks the form while the request is pending", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let release!: () => void;
  const responseGate = new Promise<void>((resolve) => { release = resolve; });
  await page.route((url) => url.pathname === "/api/settings/global", async (route) => {
    if (route.request().method() !== "PUT") return route.fallback();
    await responseGate;
    return route.fulfill({ json: {
      output_root: "output/runs",
      ui_scale: 130,
      tagging_max_retained_jobs: 50,
      path_overrides: { configs_root: "", history_root: "", queue_root: "" },
      effective_paths: {
        configs_root: "/workspace/configs",
        history_root: "/workspace/history",
        queue_root: "/workspace/queue",
      },
      defaults: { output_root: "output/runs", ui_scale: 100 },
    } });
  });

  await page.goto("/next/settings");
  const scale = page.getByRole("spinbutton", { name: "全局缩放 (%)", exact: true });
  await scale.fill("125");
  await page.getByRole("button", { name: "保存设置", exact: true }).click();

  const pending = page.getByRole("button", { name: "保存中", exact: true });
  await expect(pending).toBeDisabled();
  await expect(scale).toBeDisabled();
  await expect(page.getByRole("button", { name: "还原修改", exact: true })).toBeDisabled();
  await expect(page.getByRole("button", { name: "恢复默认", exact: true })).toBeDisabled();

  release();
  await expect(page.getByRole("status")).toHaveText("全局设置已保存");
  await expect(page.locator(".settings-page .state-label")).toHaveText("已同步");
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("models loading gate keeps the editor unavailable until the server responds", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let release!: () => void;
  const responseGate = new Promise<void>((resolve) => { release = resolve; });
  await page.route((url) => url.pathname === "/api/settings/model-configs", async (route) => {
    await responseGate;
    return route.fallback();
  });

  await page.goto("/next/models");
  const state = page.getByLabel("模型配置读取状态");
  await expect(state.getByRole("status")).toContainText("正在读取模型配置");
  await expect(page.locator(".settings-page .state-label")).toHaveText("读取中");
  await expect(page.getByRole("textbox", { name: "名称", exact: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "新建模型配置", exact: true })).toHaveCount(0);

  release();
  await expect(page.getByRole("textbox", { name: "名称", exact: true })).toHaveValue("Krea-2 Studio");
  await expect(page.locator(".settings-page .state-label")).toHaveText("已同步");
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("models successful read exposes the synced default model", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  await page.goto("/next/models");
  await expect(page.locator(".settings-page .state-label")).toHaveText("已同步");
  await expect(page.getByRole("button", { name: /Krea-2 Studio/ })).toContainText("默认");
  await expect(page.getByRole("textbox", { name: "名称", exact: true })).toHaveValue("Krea-2 Studio");
  await expect(page.getByRole("button", { name: "保存模型配置", exact: true })).toBeDisabled();
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("model save with a lost response requires server reconciliation before retry", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let serverLibrary = {
    items: [{
      id: "model-a",
      name: "Primary model",
      model_family: "anima",
      pretrained_model_name_or_path: "models/primary/dit.safetensors",
      qwen3: "models/primary/qwen.safetensors",
      vae: "models/primary/vae.safetensors",
    }],
    groups: [{ id: "group-a", label: "Local models", item_ids: ["model-a"] }],
    revision: "revision-1",
    default_id: "model-a",
  };
  let reads = 0;
  let writes = 0;
  await page.route((url) => url.pathname === "/api/settings/model-configs", async (route) => {
    if (route.request().method() === "GET") {
      reads += 1;
      return route.fulfill({ json: serverLibrary });
    }
    if (route.request().method() === "PUT") {
      writes += 1;
      const body = route.request().postDataJSON() as typeof serverLibrary;
      serverLibrary = { ...body, revision: "revision-2" };
      return route.abort();
    }
    return route.fallback();
  });

  await page.goto("/next/models");
  const name = page.getByRole("textbox", { name: "名称", exact: true });
  const save = page.getByRole("button", { name: "保存模型配置", exact: true });
  await name.fill("Committed despite lost response");
  await save.click();

  await expect(page.getByRole("alert")).toContainText("操作结果尚未确认");
  await expect(name).toHaveValue("Committed despite lost response");
  await expect(page.locator(".settings-page .state-label")).toHaveText("未保存");
  await expect(save).toBeDisabled();
  const reconcile = page.getByRole("button", { name: "核对服务器版本", exact: true });
  await expect(reconcile).toBeVisible();
  const readsBeforeReconcile = reads;
  await page.waitForTimeout(1200);
  expect(writes).toBe(1);

  page.once("dialog", (dialog) => dialog.accept());
  await reconcile.click();
  await expect(name).toHaveValue("Committed despite lost response");
  await expect(page.locator(".settings-page .state-label")).toHaveText("已同步");
  await expect(save).toBeDisabled();
  await expect(page.getByRole("alert")).toHaveCount(0);
  await expect(page.getByRole("status")).toHaveText("已重新载入服务器模型库");
  expect(reads).toBeGreaterThan(readsBeforeReconcile);
  expect(writes).toBe(1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("global settings network loss reports unknown result and preserves the patch", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let putCount = 0;
  let putBody: unknown;
  await page.route((url) => url.pathname === "/api/settings/global", (route) => {
    if (route.request().method() !== "PUT") return route.fallback();
    putCount += 1;
    putBody = route.request().postDataJSON();
    return route.abort();
  });

  await page.goto("/next/settings");
  const scale = page.getByRole("spinbutton", { name: "全局缩放 (%)", exact: true });
  const save = page.getByRole("button", { name: "保存设置", exact: true });
  await scale.fill("125");
  await save.click();

  await expect(page.getByRole("alert")).toContainText("操作结果尚未确认");
  await expect(scale).toHaveValue("125");
  await expect(page.locator(".settings-page .state-label")).toHaveText("未保存");
  await expect(save).toBeEnabled();
  expect(putCount).toBe(1);
  expect(putBody).toEqual({ ui_scale: 125 });
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("global settings save accepts the server response and becomes synced", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let putCount = 0;
  let putBody: unknown;
  await page.route((url) => url.pathname === "/api/settings/global", (route) => {
    if (route.request().method() !== "PUT") return route.fallback();
    putCount += 1;
    putBody = route.request().postDataJSON();
    return route.fulfill({ json: {
      output_root: "output/runs",
      ui_scale: 130,
      tagging_max_retained_jobs: 50,
      path_overrides: { configs_root: "", history_root: "", queue_root: "" },
      effective_paths: {
        configs_root: "/workspace/configs",
        history_root: "/workspace/history",
        queue_root: "/workspace/queue",
      },
      defaults: { output_root: "output/runs", ui_scale: 100 },
    } });
  });

  await page.goto("/next/settings");
  const scale = page.getByRole("spinbutton", { name: "全局缩放 (%)", exact: true });
  const save = page.getByRole("button", { name: "保存设置", exact: true });
  await expect(scale).toHaveValue("100");
  await scale.fill("125");
  await save.click();

  await expect(page.getByRole("status")).toHaveText("全局设置已保存");
  await expect(scale).toHaveValue("130");
  await expect(page.locator(".settings-page .state-label")).toHaveText("已同步");
  await expect(save).toBeDisabled();
  expect(putCount).toBe(1);
  expect(putBody).toEqual({ ui_scale: 125 });
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("history list retries after an offline error and confirms the empty state", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let fail = true;
  let listReads = 0;
  await page.route((url) => url.pathname === "/api/training/history", (route) => {
    listReads += 1;
    return fail
      ? route.fulfill({ status: 503, json: { error: "history offline" } })
      : route.fulfill({ json: { ok: true, tasks: [], total: 0, next_cursor: null } });
  });

  await page.goto("/next/history");
  await expect(page.getByRole("alert")).toContainText("history offline");
  await expect(page.locator(".history-restore")).toContainText("正在恢复列表位置");
  await expect(page.getByText("没有匹配的历史记录。", { exact: true })).toHaveCount(0);

  fail = false;
  await page.getByRole("button", { name: "重试恢复位置", exact: true }).click();
  await expect(page.getByText("没有匹配的历史记录。", { exact: true })).toBeVisible();
  expect(listReads).toBeGreaterThan(1);
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
        if (route === "monitor") await expect(page.locator(".monitor-connection-row")).toContainText("最近成功读取");
        else await expect(page.getByLabel("队列状态读取状态")).toContainText("最近成功读取");
        if (route === "monitor") {
          await expect(page.locator("canvas")).toHaveCount(1);
          const chart = page.getByRole("img", { name: /Loss 趋势/ });
          await expect(chart).toBeVisible();
          const bounds = await chart.evaluate((node) => {
            const chartRect = node.getBoundingClientRect();
            const canvasRect = node.querySelector("canvas")?.getBoundingClientRect();
            const controlsRect = node.closest(".chart-section")?.querySelector(".chart-controls")?.getBoundingClientRect();
            return {
              chartLeft: chartRect.left,
              chartRight: chartRect.right,
              chartWidth: chartRect.width,
              canvasLeft: canvasRect?.left,
              canvasRight: canvasRect?.right,
              controlsLeft: controlsRect?.left,
              controlsRight: controlsRect?.right,
              viewportWidth: window.innerWidth,
            };
          });
          expect(bounds.chartWidth).toBeGreaterThan(0);
          expect(bounds.chartLeft).toBeGreaterThanOrEqual(-1);
          expect(bounds.chartRight).toBeLessThanOrEqual(bounds.viewportWidth + 1);
          expect(bounds.canvasLeft).toBeGreaterThanOrEqual(-1);
          expect(bounds.canvasRight).toBeLessThanOrEqual(bounds.viewportWidth + 1);
          expect(bounds.controlsLeft).toBeGreaterThanOrEqual(-1);
          expect(bounds.controlsRight).toBeLessThanOrEqual(bounds.viewportWidth + 1);
        } else await expect(page.locator(".queue-card")).toHaveCount(3);
        await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
        await page.screenshot({ path: info.outputPath(`${route}.png`), fullPage: true });
      }
      expect(mocks.writes).toEqual([]);
      expect(mocks.unhandled).toEqual([]);
    });
  }
}
