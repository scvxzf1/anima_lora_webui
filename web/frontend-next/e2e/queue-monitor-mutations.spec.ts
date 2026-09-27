import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

test("queue move locks controls, applies confirmed order and reconciles an unknown result", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let items = [
    { id: "queue-a", state: "queued", variant: "First fixture" },
    { id: "queue-b", state: "queued", variant: "Second fixture" },
  ];
  let releaseFirstMove: () => void = () => {};
  const firstMoveGate = new Promise<void>((resolve) => { releaseFirstMove = resolve; });
  const calls: { id: string; method: string; body: unknown }[] = [];
  const snapshot = () => ({
    ok: true,
    revision: `revision-${calls.length + 1}`,
    paused: true,
    status: "idle",
    summary: { total: items.length, queued: items.length, running: 0 },
    items: items.map((item) => ({ ...item })),
  });

  await page.route((url) => url.pathname === "/api/training/queue", (route) =>
    route.fulfill({ json: snapshot() }),
  );
  await page.route((url) => /\/api\/training\/queue\/queue-[ab]\/move$/.test(url.pathname), async (route) => {
    const id = new URL(route.request().url()).pathname.split("/").at(-2)!;
    calls.push({ id, method: route.request().method(), body: route.request().postDataJSON() });
    if (calls.length === 1) {
      await firstMoveGate;
      items = [items[1], items[0]];
      return route.fulfill({ json: snapshot() });
    }
    items = [items[1], items[0]];
    return route.abort();
  });

  await page.goto("/next/queue");
  await expect(page.locator(".queue-card")).toHaveCount(2);
  const moveDown = page.locator('.queue-card[data-item-id="queue-a"]').getByRole("button", { name: "下移", exact: true });
  await moveDown.click();
  await expect(moveDown).toBeDisabled();
  await expect.poll(() => calls.length).toBe(1);
  expect(calls[0]).toEqual({ id: "queue-a", method: "POST", body: { direction: "down" } });

  releaseFirstMove();
  await expect(page.locator(".queue-card").nth(0)).toHaveAttribute("data-item-id", "queue-b");
  await expect(page.getByRole("status")).toContainText("队列状态已更新");

  await page.locator('.queue-card[data-item-id="queue-b"]').getByRole("button", { name: "下移", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("操作结果尚未确认");
  expect(calls).toHaveLength(2);
  await expect(page.locator(".queue-card").nth(0)).toHaveAttribute("data-item-id", "queue-b");

  await page.getByRole("button", { name: "刷新", exact: true }).click();
  await expect(page.locator(".queue-card").nth(0)).toHaveAttribute("data-item-id", "queue-a");
  expect(calls).toHaveLength(2);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

for (const status of [409, 503] as const) {
  test(`queue move HTTP ${status} preserves the snapshot and unlocks without retry`, async ({ page }) => {
    const mocks = await mockWorkspace(page);
    const items = [
      { id: "queue-a", state: "queued", variant: "First fixture" },
      { id: "queue-b", state: "queued", variant: "Second fixture" },
    ];
    const calls: { method: string; id: string; body: unknown }[] = [];
    const snapshot = () => ({
      ok: true,
      revision: "revision-1",
      paused: true,
      status: "idle",
      summary: { total: items.length, queued: items.length, running: 0 },
      items,
    });

    await page.route((url) => url.pathname === "/api/training/queue", (route) =>
      route.fulfill({ json: snapshot() }),
    );
    await page.route(
      (url) => url.pathname === "/api/training/queue/queue-a/move",
      async (route) => {
        calls.push({
          method: route.request().method(),
          id: new URL(route.request().url()).pathname.split("/").at(-2)!,
          body: route.request().postDataJSON(),
        });
        return route.fulfill({ status, json: { ok: false, error: "队列版本冲突" } });
      },
    );

    await page.goto("/next/queue");
    await expect(page.locator(".queue-card")).toHaveCount(2);
    const moveDown = page.locator('.queue-card[data-item-id="queue-a"]')
      .getByRole("button", { name: "下移", exact: true });
    await moveDown.click();

    await expect(page.getByRole("alert")).toContainText("队列版本冲突");
    await expect(page.locator(".queue-card").nth(0)).toHaveAttribute("data-item-id", "queue-a");
    await expect(page.locator(".queue-card").nth(1)).toHaveAttribute("data-item-id", "queue-b");
    await expect(moveDown).toBeEnabled();
    expect(calls).toEqual([{ method: "POST", id: "queue-a", body: { direction: "down" } }]);

    await page.waitForTimeout(1200);
    expect(calls).toHaveLength(1);
    expect(mocks.writes).toEqual([]);
    expect(mocks.unhandled).toEqual([]);
  });
}

test("queue cancel reconciles a stale item after a 404 without retrying", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let serverHasItem = true;
  let releaseDelete: () => void = () => {};
  const deleteGate = new Promise<void>((resolve) => { releaseDelete = resolve; });
  const deleteCalls: { method: string; path: string; body: unknown }[] = [];
  const snapshot = () => ({
    ok: true,
    revision: serverHasItem ? "revision-1" : "revision-2",
    paused: true,
    status: "idle",
    summary: { total: serverHasItem ? 1 : 0, queued: serverHasItem ? 1 : 0 },
    items: serverHasItem ? [{ id: "queue-race", state: "queued", variant: "Stale fixture" }] : [],
  });
  await page.route((url) => url.pathname === "/api/training/queue", (route) =>
    route.fulfill({ json: snapshot() }),
  );
  await page.route((url) => url.pathname === "/api/training/queue/queue-race", async (route) => {
    if (route.request().method() !== "DELETE") return route.fallback();
    deleteCalls.push({
      method: route.request().method(),
      path: new URL(route.request().url()).pathname,
      body: route.request().postDataJSON(),
    });
    serverHasItem = false;
    await deleteGate;
    return route.fulfill({ status: 404, json: { error: "队列任务不存在" } });
  });

  await page.goto("/next/queue");
  const card = page.locator('.queue-card[data-item-id="queue-race"]');
  await expect(card).toBeVisible();
  const cancel = card.getByRole("button", { name: "取消", exact: true });
  let confirmation = "";
  page.once("dialog", async (dialog) => {
    confirmation = dialog.message();
    await dialog.accept();
  });
  await cancel.click();
  await expect(cancel).toBeDisabled();
  await expect.poll(() => deleteCalls.length).toBe(1);
  expect(deleteCalls[0]).toEqual({
    method: "DELETE",
    path: "/api/training/queue/queue-race",
    body: { delete_runtime: false },
  });
  expect(confirmation).toBe("确定取消这个等待任务吗？");
  await expect(card).toBeVisible();

  releaseDelete();
  await expect(page.getByRole("alert")).toContainText("队列任务不存在");
  await expect(card).toBeVisible();
  expect(deleteCalls).toHaveLength(1);

  await page.getByRole("button", { name: "刷新", exact: true }).click();
  await expect(page.getByText("当前筛选下没有队列任务。", { exact: true })).toBeVisible();
  await expect(card).toHaveCount(0);
  await expect(page.getByRole("alert")).toHaveCount(0);
  await expect(page.getByRole("status")).toContainText("当前快照已核对");
  expect(deleteCalls).toHaveLength(1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("queue retry confirms the frozen item, locks commands, and applies the returned snapshot", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let item = { id: "queue-retry", state: "error", variant: "Retry target", attempt: 2, max_attempts: 3 };
  let releaseRetry: () => void = () => {};
  const retryGate = new Promise<void>((resolve) => { releaseRetry = resolve; });
  const retryCalls: { method: string; path: string; body: string | null }[] = [];
  const snapshot = () => ({
    ok: true,
    revision: `revision-${retryCalls.length + 1}`,
    paused: true,
    status: "idle",
    summary: { total: 1, queued: item.state === "queued" ? 1 : 0, error: item.state === "error" ? 1 : 0 },
    items: [{ ...item }],
  });
  await page.route((url) => url.pathname === "/api/training/queue", (route) =>
    route.fulfill({ json: snapshot() }),
  );
  await page.route((url) => url.pathname === "/api/training/queue/queue-retry/retry", async (route) => {
    retryCalls.push({
      method: route.request().method(),
      path: new URL(route.request().url()).pathname,
      body: route.request().postData(),
    });
    await retryGate;
    item = { ...item, state: "queued", attempt: 3 };
    return route.fulfill({ json: snapshot() });
  });

  await page.goto("/next/queue");
  const card = page.locator('.queue-card[data-item-id="queue-retry"]');
  await expect(card).toBeVisible();
  const retry = card.getByRole("button", { name: "重试", exact: true });
  let confirmation = "";
  page.once("dialog", async (dialog) => {
    confirmation = dialog.message();
    await dialog.accept();
  });
  await retry.click();

  await expect(retry).toBeDisabled();
  await expect.poll(() => retryCalls.length).toBe(1);
  expect(confirmation).toBe("使用冻结配置创建重试任务？若队列未暂停，任务可能立即执行。");
  expect(retryCalls[0]).toEqual({
    method: "POST",
    path: "/api/training/queue/queue-retry/retry",
    body: null,
  });

  releaseRetry();
  await expect(card.getByRole("button", { name: "取消", exact: true })).toBeEnabled();
  await expect(card.getByRole("button", { name: "重试", exact: true })).toHaveCount(0);
  expect(retryCalls).toHaveLength(1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("queue retry reconciles a lost response from the refreshed server snapshot", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let item = { id: "queue-retry-race", state: "error", variant: "Retry race", attempt: 1, max_attempts: 3 };
  const retryCalls: { method: string; path: string }[] = [];
  const snapshot = () => ({
    ok: true,
    revision: item.state === "queued" ? "revision-2" : "revision-1",
    paused: true,
    status: "idle",
    summary: { total: 1, queued: item.state === "queued" ? 1 : 0, error: item.state === "error" ? 1 : 0 },
    items: [{ ...item }],
  });
  await page.route((url) => url.pathname === "/api/training/queue", (route) =>
    route.fulfill({ json: snapshot() }),
  );
  await page.route((url) => url.pathname === "/api/training/queue/queue-retry-race/retry", async (route) => {
    retryCalls.push({
      method: route.request().method(),
      path: new URL(route.request().url()).pathname,
    });
    item = { ...item, state: "queued", attempt: 2 };
    return route.abort();
  });

  await page.goto("/next/queue");
  const card = page.locator('.queue-card[data-item-id="queue-retry-race"]');
  await expect(card).toBeVisible();
  page.once("dialog", (dialog) => dialog.accept());
  await card.getByRole("button", { name: "重试", exact: true }).click();

  await expect(page.getByRole("alert")).toContainText("操作结果尚未确认");
  await expect(card).toHaveAttribute("data-state", "error");
  expect(retryCalls).toEqual([{
    method: "POST",
    path: "/api/training/queue/queue-retry-race/retry",
  }]);

  await page.getByRole("button", { name: "刷新", exact: true }).click();
  await expect(card).toHaveAttribute("data-state", "queued");
  await expect(card.getByRole("button", { name: "重试", exact: true })).toHaveCount(0);
  await expect(page.getByRole("alert")).toHaveCount(0);
  await expect(page.getByRole("status")).toContainText("当前快照已核对");
  expect(retryCalls).toHaveLength(1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("queue retry keeps the failed snapshot and unlocks after a 503", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  const item = { id: "queue-retry-503", state: "error", variant: "Unavailable fixture", attempt: 2, max_attempts: 3 };
  const retryCalls: { method: string; path: string; body: string | null }[] = [];
  let markStarted!: () => void;
  const started = new Promise<void>((resolve) => { markStarted = resolve; });
  let releaseResponse!: () => void;
  const responseGate = new Promise<void>((resolve) => { releaseResponse = resolve; });
  await page.route((url) => url.pathname === "/api/training/queue", (route) =>
    route.fulfill({ json: {
      ok: true,
      revision: "revision-1",
      paused: true,
      status: "idle",
      summary: { total: 1, queued: 0, error: 1 },
      items: [{ ...item }],
    } }),
  );
  await page.route((url) => url.pathname === "/api/training/queue/queue-retry-503/retry", async (route) => {
    retryCalls.push({
      method: route.request().method(),
      path: new URL(route.request().url()).pathname,
      body: route.request().postData(),
    });
    markStarted();
    await responseGate;
    return route.fulfill({ status: 503, json: { error: "队列服务暂不可用" } });
  });

  await page.goto("/next/queue");
  const card = page.locator('.queue-card[data-item-id="queue-retry-503"]');
  const retry = card.getByRole("button", { name: "重试", exact: true });
  await expect(card).toHaveAttribute("data-state", "error");
  page.once("dialog", (dialog) => dialog.accept());
  await retry.click();
  await started;
  await expect(retry).toBeDisabled();
  expect(retryCalls).toEqual([{
    method: "POST",
    path: "/api/training/queue/queue-retry-503/retry",
    body: null,
  }]);

  releaseResponse();
  await expect(page.getByRole("alert")).toContainText("队列服务暂不可用");
  await expect(card).toHaveAttribute("data-state", "error");
  await expect(retry).toBeEnabled();
  await page.waitForTimeout(1200);
  expect(retryCalls).toHaveLength(1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("queue stop is scoped to the running item and never requests runtime deletion", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let item = { id: "queue-stop", state: "running", variant: "Running target" };
  let releaseStop: () => void = () => {};
  const stopGate = new Promise<void>((resolve) => { releaseStop = resolve; });
  const stopCalls: { method: string; path: string; body: unknown }[] = [];
  const snapshot = () => ({
    ok: true,
    revision: `revision-${stopCalls.length + 1}`,
    paused: true,
    status: item.state === "running" ? "running" : "idle",
    summary: {
      total: 1,
      queued: 0,
      running: item.state === "running" ? 1 : 0,
      canceled: item.state === "canceled" ? 1 : 0,
    },
    items: [{ ...item }],
  });
  await page.route((url) => url.pathname === "/api/training/queue", (route) =>
    route.fulfill({ json: snapshot() }),
  );
  await page.route((url) => url.pathname === "/api/training/queue/queue-stop", async (route) => {
    if (route.request().method() !== "DELETE") return route.fallback();
    stopCalls.push({
      method: route.request().method(),
      path: new URL(route.request().url()).pathname,
      body: route.request().postDataJSON(),
    });
    await stopGate;
    item = { ...item, state: "canceled" };
    return route.fulfill({ json: snapshot() });
  });

  await page.goto("/next/queue");
  const card = page.locator('.queue-card[data-item-id="queue-stop"]');
  await expect(card).toBeVisible();
  const stop = card.getByRole("button", { name: "停止", exact: true });
  let confirmation = "";
  page.once("dialog", async (dialog) => {
    confirmation = dialog.message();
    await dialog.accept();
  });
  await stop.click();

  await expect(stop).toBeDisabled();
  await expect.poll(() => stopCalls.length).toBe(1);
  expect(confirmation).toBe("确定停止这个正在运行的任务吗？");
  expect(stopCalls[0]).toEqual({
    method: "DELETE",
    path: "/api/training/queue/queue-stop",
    body: { delete_runtime: false },
  });

  releaseStop();
  await expect(page.getByText("当前筛选下没有队列任务。", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "取消 1", exact: true }).click();
  await expect(card).toHaveAttribute("data-state", "canceled");
  await expect(card.getByRole("button", { name: "移出列表", exact: true })).toBeEnabled();
  expect(stopCalls).toHaveLength(1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("queue stop keeps the old snapshot after a lost response and reconciles on refresh", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let item = { id: "queue-stop-race", state: "running", variant: "Stop race target" };
  const stopCalls: { method: string; path: string; body: unknown }[] = [];
  const snapshot = () => ({
    ok: true,
    revision: item.state === "running" ? "revision-1" : "revision-2",
    paused: true,
    status: item.state === "running" ? "running" : "idle",
    summary: {
      total: 1,
      queued: 0,
      running: item.state === "running" ? 1 : 0,
      canceled: item.state === "canceled" ? 1 : 0,
    },
    items: [{ ...item }],
  });
  await page.route((url) => url.pathname === "/api/training/queue", (route) =>
    route.fulfill({ json: snapshot() }),
  );
  await page.route((url) => url.pathname === "/api/training/queue/queue-stop-race", async (route) => {
    if (route.request().method() !== "DELETE") return route.fallback();
    stopCalls.push({
      method: route.request().method(),
      path: new URL(route.request().url()).pathname,
      body: route.request().postDataJSON(),
    });
    item = { ...item, state: "canceled" };
    return route.abort();
  });

  await page.goto("/next/queue");
  const card = page.locator('.queue-card[data-item-id="queue-stop-race"]');
  await expect(card).toHaveAttribute("data-state", "running");
  page.once("dialog", (dialog) => dialog.accept());
  await card.getByRole("button", { name: "停止", exact: true }).click();

  await expect(page.getByRole("alert")).toContainText("操作结果尚未确认");
  await expect(card).toHaveAttribute("data-state", "running");
  expect(stopCalls).toEqual([{
    method: "DELETE",
    path: "/api/training/queue/queue-stop-race",
    body: { delete_runtime: false },
  }]);

  await page.getByRole("button", { name: "刷新", exact: true }).click();
  await expect(page.getByText("当前筛选下没有队列任务。", { exact: true })).toBeVisible();
  await expect(page.getByRole("alert")).toHaveCount(0);
  await expect(page.getByRole("status")).toContainText("当前快照已核对");
  await page.getByRole("button", { name: "取消 1", exact: true }).click();
  await expect(card).toHaveAttribute("data-state", "canceled");
  expect(stopCalls).toHaveLength(1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("queue remove deletes only a completed list entry after explicit confirmation", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let item: { id: string; state: string; variant: string } | null = {
    id: "queue-remove",
    state: "done",
    variant: "Completed target",
  };
  const removeCalls: { method: string; path: string; body: unknown }[] = [];
  const snapshot = () => ({
    ok: true,
    revision: `revision-${removeCalls.length + 1}`,
    paused: true,
    status: "idle",
    summary: { total: item ? 1 : 0, done: item ? 1 : 0 },
    items: item ? [{ ...item }] : [],
  });
  await page.route((url) => url.pathname === "/api/training/queue", (route) =>
    route.fulfill({ json: snapshot() }),
  );
  await page.route((url) => url.pathname === "/api/training/queue/queue-remove", async (route) => {
    if (route.request().method() !== "DELETE") return route.fallback();
    removeCalls.push({
      method: route.request().method(),
      path: new URL(route.request().url()).pathname,
      body: route.request().postDataJSON(),
    });
    item = null;
    return route.fulfill({ json: snapshot() });
  });

  await page.goto("/next/queue");
  await page.getByRole("button", { name: "完成 1", exact: true }).click();
  const card = page.locator('.queue-card[data-item-id="queue-remove"]');
  await expect(card).toBeVisible();
  let confirmation = "";
  page.once("dialog", async (dialog) => {
    confirmation = dialog.message();
    await dialog.accept();
  });
  await card.getByRole("button", { name: "移出列表", exact: true }).click();

  await expect(page.getByText("当前筛选下没有队列任务。", { exact: true })).toBeVisible();
  expect(confirmation).toBe("只从队列列表移除这条已结束记录吗？");
  expect(removeCalls).toEqual([{
    method: "DELETE",
    path: "/api/training/queue/queue-remove",
    body: { delete_runtime: false },
  }]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("queue remove keeps the old snapshot after a lost response and reconciles on refresh", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let item: { id: string; state: string; variant: string } | null = {
    id: "queue-remove-race",
    state: "done",
    variant: "Completed remove race",
  };
  const removeCalls: { method: string; path: string; body: unknown }[] = [];
  const snapshot = () => ({
    ok: true,
    revision: item ? "revision-1" : "revision-2",
    paused: true,
    status: "idle",
    summary: { total: item ? 1 : 0, done: item ? 1 : 0 },
    items: item ? [{ ...item }] : [],
  });
  await page.route((url) => url.pathname === "/api/training/queue", (route) =>
    route.fulfill({ json: snapshot() }),
  );
  await page.route((url) => url.pathname === "/api/training/queue/queue-remove-race", async (route) => {
    if (route.request().method() !== "DELETE") return route.fallback();
    removeCalls.push({
      method: route.request().method(),
      path: new URL(route.request().url()).pathname,
      body: route.request().postDataJSON(),
    });
    item = null;
    return route.abort();
  });

  await page.goto("/next/queue");
  await page.getByRole("button", { name: "完成 1", exact: true }).click();
  const card = page.locator('.queue-card[data-item-id="queue-remove-race"]');
  await expect(card).toBeVisible();
  page.once("dialog", (dialog) => dialog.accept());
  await card.getByRole("button", { name: "移出列表", exact: true }).click();

  await expect(page.getByRole("alert")).toContainText("操作结果尚未确认");
  await expect(card).toHaveAttribute("data-state", "done");
  expect(removeCalls).toEqual([{
    method: "DELETE",
    path: "/api/training/queue/queue-remove-race",
    body: { delete_runtime: false },
  }]);

  await page.getByRole("button", { name: "刷新", exact: true }).click();
  await expect(page.getByText("当前筛选下没有队列任务。", { exact: true })).toBeVisible();
  await expect(card).toHaveCount(0);
  await expect(page.getByRole("alert")).toHaveCount(0);
  await expect(page.getByRole("status")).toContainText("当前快照已核对");
  expect(removeCalls).toHaveLength(1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("queue retry reconciles a stale error item after a 404 without retrying", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let serverHasItem = true;
  const retryCalls: { method: string; path: string }[] = [];
  const snapshot = () => ({
    ok: true,
    revision: serverHasItem ? "revision-1" : "revision-2",
    paused: true,
    status: "idle",
    summary: { total: serverHasItem ? 1 : 0, error: serverHasItem ? 1 : 0 },
    items: serverHasItem
      ? [{ id: "queue-retry-404", state: "error", variant: "Stale retry target" }]
      : [],
  });
  await page.route((url) => url.pathname === "/api/training/queue", (route) =>
    route.fulfill({ json: snapshot() }),
  );
  await page.route((url) => url.pathname === "/api/training/queue/queue-retry-404/retry", (route) => {
    retryCalls.push({
      method: route.request().method(),
      path: new URL(route.request().url()).pathname,
    });
    serverHasItem = false;
    return route.fulfill({ status: 404, json: { error: "队列任务不存在" } });
  });

  await page.goto("/next/queue");
  const card = page.locator('.queue-card[data-item-id="queue-retry-404"]');
  await expect(card).toHaveAttribute("data-state", "error");
  page.once("dialog", (dialog) => dialog.accept());
  await card.getByRole("button", { name: "重试", exact: true }).click();

  await expect(page.getByRole("alert")).toContainText("队列任务不存在");
  await expect(card).toHaveAttribute("data-state", "error");
  await expect(card.getByRole("button", { name: "重试", exact: true })).toBeEnabled();
  await page.waitForTimeout(1200);
  expect(retryCalls).toEqual([{
    method: "POST",
    path: "/api/training/queue/queue-retry-404/retry",
  }]);

  await page.getByRole("button", { name: "刷新", exact: true }).click();
  await expect(card).toHaveCount(0);
  await expect(page.getByRole("alert")).toHaveCount(0);
  await expect(page.getByRole("status")).toContainText("当前快照已核对");
  expect(retryCalls).toHaveLength(1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

for (const status of [404, 503]) {
  test(`queue stop preserves the running snapshot after a ${status}`, async ({ page }) => {
    const mocks = await mockWorkspace(page);
    let serverHasItem = true;
    const stopCalls: { method: string; path: string; body: unknown }[] = [];
    const snapshot = () => ({
      ok: true,
      revision: serverHasItem ? "revision-1" : "revision-2",
      paused: true,
      status: serverHasItem ? "running" : "idle",
      summary: {
        total: serverHasItem ? 1 : 0,
        running: serverHasItem ? 1 : 0,
      },
      items: serverHasItem
        ? [{ id: `queue-stop-${status}`, state: "running", variant: "Stop error target" }]
        : [],
    });
    await page.route((url) => url.pathname === "/api/training/queue", (route) =>
      route.fulfill({ json: snapshot() }),
    );
    await page.route((url) => url.pathname === `/api/training/queue/queue-stop-${status}`, (route) => {
      stopCalls.push({
        method: route.request().method(),
        path: new URL(route.request().url()).pathname,
        body: route.request().postDataJSON(),
      });
      if (status === 404) serverHasItem = false;
      return route.fulfill({
        status,
        json: { error: status === 404 ? "队列任务不存在" : "队列服务暂不可用" },
      });
    });

    await page.goto("/next/queue");
    const card = page.locator(`.queue-card[data-item-id="queue-stop-${status}"]`);
    const stop = card.getByRole("button", { name: "停止", exact: true });
    await expect(card).toHaveAttribute("data-state", "running");
    page.once("dialog", (dialog) => dialog.accept());
    await stop.click();

    await expect(page.getByRole("alert")).toContainText(
      status === 404 ? "队列任务不存在" : "队列服务暂不可用",
    );
    await expect(card).toHaveAttribute("data-state", "running");
    await expect(stop).toBeEnabled();
    await page.waitForTimeout(1200);
    expect(stopCalls).toEqual([{
      method: "DELETE",
      path: `/api/training/queue/queue-stop-${status}`,
      body: { delete_runtime: false },
    }]);

    await page.getByRole("button", { name: "刷新", exact: true }).click();
    if (status === 404) await expect(card).toHaveCount(0);
    else await expect(card).toHaveAttribute("data-state", "running");
    await expect(page.getByRole("alert")).toHaveCount(0);
    expect(stopCalls).toHaveLength(1);
    expect(mocks.writes).toEqual([]);
    expect(mocks.unhandled).toEqual([]);
  });
}

test("monitor distinguishes no task and unknown status without enabling stop", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let status: Record<string, unknown> = { status: "idle" };
  let metricReads = 0;
  let logReads = 0;
  await page.route((url) => url.pathname === "/api/training/status", (route) =>
    route.fulfill({ json: status }),
  );
  await page.route((url) => url.pathname === "/api/training/metrics", (route) => {
    metricReads += 1;
    return route.fallback();
  });
  await page.route((url) => url.pathname === "/api/training/logs", (route) => {
    logReads += 1;
    return route.fallback();
  });
  await page.route((url) => url.pathname === "/api/training/gpus", (route) =>
    route.fulfill({ json: { gpus: [] } }),
  );

  await page.goto("/next/monitor");
  await expect(page.locator(".monitor-state")).toHaveText("空闲");
  await expect(page.getByText("暂无当前任务。", { exact: true })).toBeVisible();
  await expect(page.getByRole("navigation", { name: "空闲时操作" }).getByRole("link", { name: "训练配置" })).toHaveAttribute("href", "/next/training");
  await expect(page.locator(".monitor-log-panel")).toHaveCount(0);
  await expect(page.getByText("未检测到 GPU 信息", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "停止训练", exact: true })).toHaveCount(0);
  expect(metricReads).toBe(0);
  expect(logReads).toBe(0);

  status = { task_id: "task-unknown", status: "future-state" };
  await expect(page.getByRole("link", { name: "task-unknown", exact: true })).toBeVisible({ timeout: 5000 });
  await expect(page.locator(".monitor-state")).toHaveText("未知");
  await expect(page.getByRole("button", { name: "停止训练", exact: true })).toBeDisabled();
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("monitor recovers GPU details after a 503 without losing task status", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let unavailable = true;
  let gpuReads = 0;
  await page.route((url) => url.pathname === "/api/training/gpus", (route) => {
    gpuReads += 1;
    return unavailable
      ? route.fulfill({ status: 503, json: { error: "GPU probe unavailable" } })
      : route.fulfill({ json: { gpus: [{ index: 2, name: "Recovered fixture GPU", memory_total_gb: 16 }] } });
  });

  await page.goto("/next/monitor");
  await expect(page.getByLabel("GPU读取状态")).toContainText("GPU probe unavailable");
  await expect(page.getByRole("button", { name: "重试GPU", exact: true })).toBeEnabled();
  await expect(page.locator(".monitor-state")).toHaveText("运行中");
  await expect(page.getByRole("log")).toContainText("step 840");
  await expect(page.getByText("未检测到 GPU 信息", { exact: true })).toHaveCount(0);

  unavailable = false;
  await page.getByRole("button", { name: "重试GPU", exact: true }).click();
  await expect(page.locator(".monitor-gpu-card")).toContainText("Recovered fixture GPU");
  await expect(page.getByLabel("GPU读取状态")).not.toContainText("unavailable");
  expect(gpuReads).toBeGreaterThanOrEqual(2);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("monitor stop locks while pending and reflects the confirmed idle state", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let status: Record<string, unknown> = { task_id: "run-success", status: "running", job: "training" };
  let releaseStop: () => void = () => {};
  const stopGate = new Promise<void>((resolve) => { releaseStop = resolve; });
  const stopCalls: unknown[] = [];
  await page.route((url) => url.pathname === "/api/training/status", (route) =>
    route.fulfill({ json: status }),
  );
  await page.route((url) => url.pathname === "/api/training/stop", async (route) => {
    stopCalls.push(route.request().postDataJSON());
    await stopGate;
    status = { task_id: "run-success", status: "idle", job: "training" };
    return route.fulfill({ json: { ok: true, message: "stopped" } });
  });

  await page.goto("/next/monitor");
  const stop = page.locator(".monitor-stop");
  await expect(stop).toBeEnabled();
  await stop.click();
  await expect(page.getByRole("dialog", { name: "停止训练" })).toContainText("run-success");
  await page.getByRole("dialog", { name: "停止训练" }).getByRole("button", { name: "停止训练" }).click();
  await expect(stop).toContainText("正在停止");
  await expect(stop).toBeDisabled();
  await expect.poll(() => stopCalls.length).toBe(1);
  expect(stopCalls[0]).toEqual({ task_id: "run-success" });

  releaseStop();
  await expect(page.locator(".monitor-state")).toHaveText("空闲");
  await expect(stop).toBeDisabled();
  await expect(page.getByRole("alert")).toHaveCount(0);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("monitor reports an unknown stop result and reconciles status without retrying", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let status: Record<string, unknown> = { task_id: "run-unknown", status: "running", job: "training" };
  const stopCalls: unknown[] = [];
  await page.route((url) => url.pathname === "/api/training/status", (route) =>
    route.fulfill({ json: status }),
  );
  await page.route((url) => url.pathname === "/api/training/stop", async (route) => {
    stopCalls.push(route.request().postDataJSON());
    status = { task_id: "run-unknown", status: "idle", job: "training" };
    return route.abort();
  });

  await page.goto("/next/monitor");
  const stop = page.getByRole("button", { name: "停止训练", exact: true });
  await expect(stop).toBeEnabled();
  await stop.click();
  await page.getByRole("dialog", { name: "停止训练" }).getByRole("button", { name: "停止训练" }).click();

  await expect(page.getByRole("alert")).toContainText("操作结果尚未确认");
  await expect(page.locator(".monitor-state")).toHaveText("空闲");
  await expect(stop).toBeDisabled();
  expect(stopCalls).toEqual([{ task_id: "run-unknown" }]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});
