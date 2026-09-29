import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

test("bulk queue command fixes the confirmed revision and requires refresh after a race", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let revision = "snapshot-1";
  const items = [{ id: "queued-1", state: "queued", variant: "First" }];
  const snapshot = () => ({ revision, paused: true, status: "idle", items, summary: { total: items.length, queued: items.filter((item) => item.state === "queued").length } });
  const requests: unknown[] = [];
  await page.route((url) => url.pathname === "/api/training/queue", (route) => route.fulfill({ json: snapshot() }));
  await page.route((url) => url.pathname === "/api/training/queue/cancel-waiting", (route) => {
    const body = route.request().postDataJSON(); requests.push(body);
    if (body.expected_revision !== revision) return route.fulfill({ status: 409, json: { error: "队列已发生变化，本次操作未执行。请刷新队列并重新确认范围。" } });
    items.forEach((item) => { item.state = "canceled"; }); revision = "snapshot-3";
    return route.fulfill({ json: { ...snapshot(), message: "已取消 2 个等待任务" } });
  });
  await page.goto("/next/queue");
  await expect(page.locator(".queue-card")).toHaveCount(1);
  // A server-side enqueue between the visible snapshot and command acceptance.
  page.once("dialog", async (dialog) => {
    expect(dialog.message()).toContain("当前快照涉及 1 条队列记录");
    items.push({ id: "queued-2", state: "queued", variant: "New" }); revision = "snapshot-2";
    await dialog.accept();
  });
  await page.getByRole("button", { name: "取消全部等待", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("本次操作未执行");
  expect(requests).toEqual([{ expected_revision: "snapshot-1" }]);
  expect(items.every((item) => item.state === "queued")).toBe(true);
  await page.getByRole("button", { name: "刷新", exact: true }).click();
  await expect(page.locator(".queue-card")).toHaveCount(2);
  page.once("dialog", async (dialog) => {
    expect(dialog.message()).toContain("当前快照涉及 2 条队列记录");
    await dialog.accept();
  });
  await page.getByRole("button", { name: "取消全部等待", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("已取消 2 个等待任务");
  expect(requests).toEqual([{ expected_revision: "snapshot-1" }, { expected_revision: "snapshot-2" }]);
  expect(mocks.writes).toEqual([]); expect(mocks.unhandled).toEqual([]);
});

test("bulk operations do not fall back to unguarded requests when the revision is absent", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  await page.goto("/next/queue");
  await page.getByRole("button", { name: "取消全部等待", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("缺少队列快照版本");
  expect(mocks.writes).toEqual([]);
});

test("bulk stop and cleanup commands confirm their scope and send only mocked queue requests", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let revision = "scope-revision-1";
  const items = [
    { id: "run-1", state: "running", variant: "running" },
    { id: "queued-1", state: "queued", variant: "queued one" },
    { id: "queued-2", state: "queued", variant: "queued two" },
    { id: "done-1", state: "done", variant: "completed" },
    { id: "canceled-1", state: "canceled", variant: "canceled" },
  ];
  const snapshot = () => ({
    ok: true,
    revision,
    status: items.some((item) => item.state === "running") ? "running" : "idle",
    paused: true,
    current_task_id: items.some((item) => item.state === "running") ? "run-1" : null,
    summary: {
      total: items.length,
      queued: items.filter((item) => item.state === "queued").length,
      running: items.filter((item) => item.state === "running").length,
      done: items.filter((item) => item.state === "done").length,
      canceled: items.filter((item) => item.state === "canceled").length,
    },
    items: items.map((item) => ({ ...item })),
  });
  const requests: Array<{ path: string; body: unknown }> = [];
  await page.route((url) => url.pathname === "/api/training/queue", (route) => route.fulfill({ json: snapshot() }));
  await page.route((url) => url.pathname.startsWith("/api/training/queue/"), async (route) => {
    const path = new URL(route.request().url()).pathname;
    const body = route.request().postDataJSON();
    requests.push({ path, body });
    if (body.expected_revision !== revision) {
      await route.fulfill({ status: 409, json: { ok: false, error: "队列已发生变化，本次操作未执行。请刷新队列并重新确认范围。" } });
      return;
    }
    if (path.endsWith("/abort-after-current")) {
      items.forEach((item) => { if (item.state === "queued") item.state = "canceled"; });
    } else if (path.endsWith("/force-abort")) {
      items.forEach((item) => { if (item.state === "running" || item.state === "queued") item.state = "canceled"; });
    } else if (path.endsWith("/clear-completed")) {
      for (let index = items.length - 1; index >= 0; index -= 1) if (items[index].state === "done") items.splice(index, 1);
    } else if (path.endsWith("/clear-canceled")) {
      for (let index = items.length - 1; index >= 0; index -= 1) if (items[index].state === "canceled") items.splice(index, 1);
    }
    revision = `scope-revision-${Number(revision.split("-").at(-1)) + 1}`;
    const messages: Record<string, string> = {
      "/abort-after-current": "已中止后续队列，取消 2 个等待任务",
      "/force-abort": "已强制中止队列，处理 1 个任务",
      "/clear-completed": "已清理 1 条已完成记录",
      "/clear-canceled": "已清理 4 条已取消记录",
    };
    await route.fulfill({ json: { ...snapshot(), message: messages[path.slice(path.lastIndexOf("/"))] } });
  });
  await page.goto("/next/queue");

  const confirmAndAccept = (expectedText: string) => page.once("dialog", async (dialog) => {
    expect(dialog.message()).toBe(expectedText);
    await dialog.accept();
  });
  confirmAndAccept("当前任务完成后停止队列，并取消后续等待任务吗？\n当前快照涉及 2 条队列记录（包含筛选隐藏项）。");
  await page.getByRole("button", { name: "中止后续队列", exact: true }).click();
  await expect(page.getByRole("status")).toHaveText("已中止后续队列，取消 2 个等待任务");
  await expect(page.locator(".queue-card")).toHaveCount(1);
  await expect(page.locator(".queue-badge")).toHaveText("队列已暂停");
  await expect(page.locator(".queue-stat").filter({ hasText: "运行" })).toContainText("1");
  await expect(page.locator(".queue-stat").filter({ hasText: "取消" })).toContainText("3");

  confirmAndAccept("立即强制中止运行任务和等待任务吗？训练文件不会删除。\n当前快照涉及 1 条队列记录（包含筛选隐藏项）。当前任务：run-1。");
  await page.getByRole("button", { name: "强制中止", exact: true }).click();
  await expect(page.getByRole("status")).toHaveText("已强制中止队列，处理 1 个任务");
  await expect(page.locator(".queue-card")).toHaveCount(0);
  await expect(page.locator(".queue-badge")).toHaveText("队列已暂停");
  await expect(page.locator(".queue-stat").filter({ hasText: "运行" })).toContainText("0");
  await expect(page.locator(".queue-stat").filter({ hasText: "取消" })).toContainText("4");

  confirmAndAccept("清理已完成队列记录吗？训练历史、运行目录和权重不会删除。\n当前快照涉及 1 条队列记录（包含筛选隐藏项）。");
  await page.getByRole("button", { name: "清理已完成", exact: true }).click();
  await expect(page.getByRole("status")).toHaveText("已清理 1 条已完成记录");
  await expect(page.locator(".queue-card")).toHaveCount(0);
  await expect(page.locator(".queue-stat").filter({ hasText: "完成" })).toContainText("0");
  await expect(page.locator(".queue-stat").filter({ hasText: "取消" })).toContainText("4");

  confirmAndAccept("清理已取消队列记录吗？训练历史、运行目录和权重不会删除。\n当前快照涉及 4 条队列记录（包含筛选隐藏项）。");
  await page.getByRole("button", { name: "清理已取消", exact: true }).click();
  await expect(page.getByRole("status")).toHaveText("已清理 4 条已取消记录");
  await expect(page.locator(".queue-card")).toHaveCount(0);
  await expect(page.locator(".queue-stat").filter({ hasText: "全部" })).toContainText("0");

  expect(requests).toEqual([
    { path: "/api/training/queue/abort-after-current", body: { expected_revision: "scope-revision-1" } },
    { path: "/api/training/queue/force-abort", body: { expected_revision: "scope-revision-2" } },
    { path: "/api/training/queue/clear-completed", body: { expected_revision: "scope-revision-3" } },
    { path: "/api/training/queue/clear-canceled", body: { expected_revision: "scope-revision-4" } },
  ]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

const staleRevisionCases = [
  {
    action: "abort-after-current",
    button: "中止后续队列",
    expectedRevision: "conflict-revision-1",
    expectedItems: 2,
  },
  {
    action: "force-abort",
    button: "强制中止",
    expectedRevision: "conflict-revision-1",
    expectedItems: 2,
  },
  {
    action: "clear-completed",
    button: "清理已完成",
    expectedRevision: "conflict-revision-1",
    expectedItems: 2,
  },
  {
    action: "clear-canceled",
    button: "清理已取消",
    expectedRevision: "conflict-revision-1",
    expectedItems: 2,
  },
] as const;

for (const scenario of staleRevisionCases) {
  test(`${scenario.action} requires refresh when the confirmed revision becomes stale`, async ({ page }) => {
    const mocks = await mockWorkspace(page);
    let revision = scenario.expectedRevision;
    const items = [
      { id: "run-1", state: "running", variant: "running" },
      { id: "queued-1", state: "queued", variant: "queued" },
      { id: "done-1", state: "done", variant: "completed" },
      { id: "canceled-1", state: "canceled", variant: "canceled" },
    ];
    const snapshot = () => ({
      ok: true,
      revision,
      status: "running",
      paused: true,
      current_task_id: "run-1",
      summary: {
        total: items.length,
        queued: items.filter((item) => item.state === "queued").length,
        running: items.filter((item) => item.state === "running").length,
        done: items.filter((item) => item.state === "done").length,
        canceled: items.filter((item) => item.state === "canceled").length,
      },
      items: items.map((item) => ({ ...item })),
    });
    const requests: Array<{ path: string; body: { expected_revision: string } }> = [];
    let operationExecuted = false;
    await page.route((url) => url.pathname === "/api/training/queue", (route) => {
      return route.fulfill({ json: snapshot() });
    });
    await page.route((url) => url.pathname.startsWith("/api/training/queue/"), async (route) => {
      const path = new URL(route.request().url()).pathname;
      const body = route.request().postDataJSON() as { expected_revision: string };
      requests.push({ path, body });
      if (body.expected_revision !== revision) {
        await route.fulfill({ status: 409, json: { ok: false, error: "队列已发生变化，本次操作未执行。请刷新队列并重新确认范围。" } });
        return;
      }
      operationExecuted = true;
      await route.fulfill({ json: snapshot() });
    });

    await page.goto("/next/queue");
    await expect(page.locator(".queue-card")).toHaveCount(scenario.expectedItems);
    page.once("dialog", async (dialog) => {
      revision = "conflict-revision-2";
      await dialog.accept();
    });
    await page.getByRole("button", { name: scenario.button, exact: true }).click();

    await expect(page.getByRole("alert")).toContainText("本次操作未执行");
    await expect(page.getByRole("alert")).toContainText("请刷新队列");
    expect(requests).toEqual([{
      path: `/api/training/queue/${scenario.action}`,
      body: { expected_revision: scenario.expectedRevision },
    }]);
    expect(operationExecuted).toBe(false);
    await expect(page.locator(".queue-card")).toHaveCount(scenario.expectedItems);
    expect(mocks.writes).toEqual([]);
    expect(mocks.unhandled).toEqual([]);
  });
}
