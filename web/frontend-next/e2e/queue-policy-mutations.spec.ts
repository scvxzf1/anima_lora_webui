import { expect, test, type Page } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

type QueuePolicy = {
  failure_policy: string;
  auto_retry: boolean;
  max_attempts: number;
  retry_backoff_sec: number;
};

type QueueSnapshot = QueuePolicy & {
  revision: string;
  paused: boolean;
  status: string;
  summary: {
    total: number;
    queued: number;
    running: number;
    done: number;
    error: number;
    canceled: number;
  };
  items: { id: string; state: string; variant: string }[];
  message?: string;
};

type SettingsPayload = Partial<QueuePolicy> & { paused?: boolean };

function queueSnapshot(overrides: Partial<QueueSnapshot> = {}): QueueSnapshot {
  return {
    revision: "queue-rev-1",
    paused: true,
    failure_policy: "pause",
    auto_retry: false,
    max_attempts: 2,
    retry_backoff_sec: 5,
    status: "idle",
    summary: { total: 2, queued: 1, running: 0, done: 1, error: 0, canceled: 0 },
    items: [
      { id: "queue-a", state: "queued", variant: "Queued fixture" },
      { id: "queue-done", state: "done", variant: "Completed fixture" },
    ],
    ...overrides,
  };
}

async function setupQueue(page: Page) {
  const mocks = await mockWorkspace(page);
  const state = { snapshot: queueSnapshot() };
  await page.route(
    (url) => url.pathname === "/api/training/queue",
    (route) => route.fulfill({ json: state.snapshot }),
  );
  return { mocks, state };
}

async function openQueue(page: Page) {
  await page.goto("/next/queue");
  await expect(page.getByRole("heading", { name: "训练队列" })).toBeVisible();
  await expect(page.getByRole("button", { name: "保存策略" })).toBeEnabled();
}

async function editPolicy(page: Page) {
  await page.getByLabel("任务失败后").selectOption("continue");
  await page.getByRole("checkbox").check();
  await page.getByLabel("最大尝试次数").fill("3");
  await page.getByLabel("重试等待（秒）").fill("10");
}

const editedPolicy: QueuePolicy = {
  failure_policy: "continue",
  auto_retry: true,
  max_attempts: 3,
  retry_backoff_sec: 10,
};

test("queue policy save locks queue commands while pending and applies the confirmed snapshot", async ({ page }) => {
  const { mocks, state } = await setupQueue(page);
  const writes: SettingsPayload[] = [];
  let releaseSave: () => void = () => {};
  const saveGate = new Promise<void>((resolve) => { releaseSave = resolve; });
  await page.route(
    (url) => url.pathname === "/api/training/queue/settings",
    async (route) => {
      const payload = route.request().postDataJSON() as SettingsPayload;
      writes.push(payload);
      await saveGate;
      state.snapshot = queueSnapshot({
        ...editedPolicy,
        revision: "queue-rev-2",
        message: "队列策略已保存",
      });
      return route.fulfill({ json: state.snapshot });
    },
  );

  await openQueue(page);
  await editPolicy(page);
  const save = page.getByRole("button", { name: "保存策略" });
  await save.click();
  await expect(save).toBeDisabled();
  await expect(page.getByRole("button", { name: "继续队列" })).toBeDisabled();
  await expect(page.getByRole("button", { name: "取消全部等待" })).toBeDisabled();
  await expect.poll(() => writes.length).toBe(1);
  expect(writes).toEqual([editedPolicy]);

  releaseSave();
  await expect(page.getByRole("status")).toHaveText("队列策略已保存");
  await expect(save).toBeEnabled();
  await expect(page.getByLabel("任务失败后")).toHaveValue("continue");
  await expect(page.getByRole("checkbox")).toBeChecked();
  await expect(page.getByLabel("最大尝试次数")).toHaveValue("3");
  await expect(page.getByLabel("重试等待（秒）")).toHaveValue("10");
  expect(writes).toEqual([editedPolicy]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

for (const failure of ["server", "offline"] as const) {
  test(`queue policy ${failure} failure preserves edits and requires explicit retry`, async ({ page }) => {
    const { mocks, state } = await setupQueue(page);
    const writes: SettingsPayload[] = [];
    let failFirstRequest = true;
    await page.route(
      (url) => url.pathname === "/api/training/queue/settings",
      async (route) => {
        writes.push(route.request().postDataJSON() as SettingsPayload);
        if (failFirstRequest) {
          return failure === "offline"
            ? route.abort()
            : route.fulfill({ status: 500, json: { error: "queue policy unavailable" } });
        }
        state.snapshot = queueSnapshot({
          ...editedPolicy,
          revision: "queue-rev-2",
          message: "队列策略已保存",
        });
        return route.fulfill({ json: state.snapshot });
      },
    );

    await openQueue(page);
    await editPolicy(page);
    const save = page.getByRole("button", { name: "保存策略" });
    await save.click();
    await expect(page.getByRole("alert")).toContainText(
      failure === "offline" ? "操作结果尚未确认" : "queue policy unavailable",
    );
    await expect(save).toBeEnabled();
    await expect(page.getByLabel("任务失败后")).toHaveValue("continue");
    await expect(page.getByRole("checkbox")).toBeChecked();
    await expect(page.getByLabel("最大尝试次数")).toHaveValue("3");
    await expect(page.getByLabel("重试等待（秒）")).toHaveValue("10");
    expect(writes).toEqual([editedPolicy]);

    failFirstRequest = false;
    await save.click();
    await expect(page.getByRole("status")).toHaveText("队列策略已保存");
    expect(writes).toEqual([editedPolicy, editedPolicy]);
    expect(mocks.writes).toEqual([]);
    expect(mocks.unhandled).toEqual([]);
  });
}

test("canceling queue continuation confirmation sends no settings request", async ({ page }) => {
  const { mocks, state } = await setupQueue(page);
  const writes: SettingsPayload[] = [];
  await page.route(
    (url) => url.pathname === "/api/training/queue/settings",
    async (route) => {
      const payload = route.request().postDataJSON() as SettingsPayload;
      writes.push(payload);
      state.snapshot = queueSnapshot({
        ...state.snapshot,
        ...payload,
        paused: false,
        revision: "queue-rev-2",
        message: "队列已继续",
      });
      return route.fulfill({ json: state.snapshot });
    },
  );

  await openQueue(page);
  const resume = page.getByRole("button", { name: "继续队列" });
  page.once("dialog", (dialog) => dialog.dismiss());
  await resume.click();
  expect(writes).toEqual([]);
  await expect(resume).toBeEnabled();
  await expect(page.locator(".queue-badge")).toHaveText("队列已暂停");

  page.once("dialog", (dialog) => dialog.accept());
  await resume.click();
  await expect(page.getByRole("button", { name: "暂停队列" })).toBeEnabled();
  await expect(page.getByRole("status")).toHaveText("队列已继续");
  expect(writes).toEqual([{ paused: false }]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});
