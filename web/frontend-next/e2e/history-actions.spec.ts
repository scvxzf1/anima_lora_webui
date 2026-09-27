import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

const taskId = "s1-history-action";

function historyTask(archived = false) {
  return {
    id: taskId,
    name: "S1 history action fixture",
    job: "training",
    state: "idle",
    archived,
    history_source_config_file: "configs/imported/s1-fixture.toml",
    run_dir: "output/runs/s1-history-action",
    started_at_text: "2026-09-25 10:00",
    metric_count: 1,
    log_count: 1,
  };
}

test("history archive, unarchive and delete use explicit scope without deleting runtime files", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let tasks = [historyTask()];
  let releaseDelete: () => void = () => {};
  const deleteGate = new Promise<void>((resolve) => { releaseDelete = resolve; });
  const commands: Record<string, unknown>[] = [];

  await page.route((url) => url.pathname === "/api/training/history" && url.searchParams.has("include_archived"), (route) =>
    route.fulfill({ json: { ok: true, total: tasks.length, tasks: tasks.map((task) => ({ ...task })) } }),
  );
  await page.route((url) => url.pathname === "/api/training/history/batch", async (route) => {
    const command = route.request().postDataJSON() as Record<string, unknown>;
    commands.push(command);
    const action = String(command.action);
    if (action === "archive" || action === "unarchive") {
      const archived = action === "archive";
      tasks = tasks.map((task) => ({ ...task, archived }));
      return route.fulfill({ json: { ok: true, message: archived ? "历史任务已归档" : "历史任务已取消归档" } });
    }
    await deleteGate;
    tasks = [];
    return route.fulfill({ json: { ok: true, message: "历史记录已删除" } });
  });

  await page.goto("/next/history");
  await page.getByRole("button", { name: "平铺任务", exact: true }).click();
  const checkbox = page.getByRole("checkbox", { name: "选择 S1 history action fixture" });
  await expect(checkbox).toBeVisible();

  await checkbox.check();
  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "归档已选", exact: true }).click();
  await expect(page.locator(".history-card")).toHaveCount(0);
  expect(commands[0]).toEqual({ action: "archive", task_ids: [taskId] });

  await page.getByRole("combobox", { name: "归档", exact: true }).selectOption("all");
  await expect(page.locator(`[data-history-task="${taskId}"]`)).toBeVisible();
  await page.getByRole("checkbox", { name: "选择 S1 history action fixture" }).check();
  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "取消归档", exact: true }).click();
  await expect(page.getByText("历史任务已取消归档", { exact: true })).toBeVisible();
  expect(commands[1]).toEqual({ action: "unarchive", task_ids: [taskId] });

  await page.getByRole("checkbox", { name: "选择 S1 history action fixture" }).check();
  let confirmation = "";
  page.once("dialog", async (dialog) => {
    confirmation = dialog.message();
    await dialog.accept();
  });
  const remove = page.getByRole("button", { name: "彻底删除", exact: true });
  await remove.click();
  await expect(remove).toBeDisabled();
  expect(confirmation).toBe("确定彻底删除已选 1 条历史记录吗？该操作不会删除运行目录和权重。");
  expect(commands[2]).toEqual({ action: "delete", task_ids: [taskId] });
  releaseDelete();
  await expect(page.locator(".history-card")).toHaveCount(0);
  await expect(page.getByText("历史记录已删除", { exact: true })).toBeVisible();
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("history reconciles a lost delete response and drops the stale selection", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let tasks = [historyTask()];
  const commands: Record<string, unknown>[] = [];

  await page.route((url) => url.pathname === "/api/training/history" && url.searchParams.has("include_archived"), (route) =>
    route.fulfill({ json: { ok: true, total: tasks.length, tasks: tasks.map((task) => ({ ...task })) } }),
  );
  await page.route((url) => url.pathname === "/api/training/history/batch", async (route) => {
    commands.push(route.request().postDataJSON() as Record<string, unknown>);
    tasks = [];
    return route.abort();
  });

  await page.goto("/next/history");
  await page.getByRole("button", { name: "平铺任务", exact: true }).click();
  await page.getByRole("checkbox", { name: "选择 S1 history action fixture" }).check();
  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "彻底删除", exact: true }).click();

  const card = page.locator(`[data-history-task="${taskId}"]`);
  await expect(page.getByRole("alert")).toContainText("操作结果尚未确认");
  await expect(card).toBeVisible();
  expect(commands).toEqual([{ action: "delete", task_ids: [taskId] }]);

  await page.getByRole("button", { name: "刷新", exact: true }).click();
  await expect(card).toHaveCount(0);
  await expect(page.getByRole("alert")).toHaveCount(0);
  await expect(page.getByText("已刷新历史记录，当前列表已核对。", { exact: true })).toBeVisible();
  await expect(page.getByText("已选 1 项", { exact: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "彻底删除", exact: true })).toHaveCount(0);
  expect(commands).toHaveLength(1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

for (const status of [409, 503] as const) {
  test(`history archive HTTP ${status} preserves selection and recovers after refresh`, async ({ page }) => {
    const mocks = await mockWorkspace(page);
    const tasks = [historyTask()];
    let announceArchiveStarted = () => {};
    let releaseArchiveResponse = () => {};
    const archiveStarted = new Promise<void>((resolve) => { announceArchiveStarted = resolve; });
    const archiveResponseGate = new Promise<void>((resolve) => { releaseArchiveResponse = resolve; });
    const commands: Record<string, unknown>[] = [];

    await page.route((url) => url.pathname === "/api/training/history" && url.searchParams.has("include_archived"), (route) =>
      route.fulfill({ json: { ok: true, total: tasks.length, tasks: tasks.map((task) => ({ ...task })) } }),
    );
    await page.route((url) => url.pathname === "/api/training/history/batch", async (route) => {
      commands.push(route.request().postDataJSON() as Record<string, unknown>);
      announceArchiveStarted();
      await archiveResponseGate;
      return route.fulfill({ status, json: { ok: false, error: "归档写入失败" } });
    });

    await page.goto("/next/history");
    await page.getByRole("button", { name: "平铺任务", exact: true }).click();
    const card = page.locator(`[data-history-task="${taskId}"]`);
    const checkbox = page.getByRole("checkbox", { name: "选择 S1 history action fixture" });
    await expect(checkbox).toBeVisible();
    await checkbox.check();

    let confirmation = "";
    page.once("dialog", async (dialog) => {
      confirmation = dialog.message();
      await dialog.accept();
    });
    const archive = page.getByRole("button", { name: "归档已选", exact: true });
    await archive.click();
    await archiveStarted;
    await expect(archive).toBeDisabled();
    expect(commands).toEqual([{ action: "archive", task_ids: [taskId] }]);
    expect(confirmation).toBe("确定归档已选 1 条历史记录吗？");

    releaseArchiveResponse();
    await expect(page.getByRole("alert")).toContainText("归档写入失败");
    await expect(card).toBeVisible();
    await expect(page.getByText("已选 1 项", { exact: true })).toBeVisible();
    await expect(archive).toBeEnabled();
    await page.waitForTimeout(1200);
    expect(commands).toHaveLength(1);

    await page.getByRole("button", { name: "刷新", exact: true }).click();
    await expect(page.getByRole("alert")).toHaveCount(0);
    await expect(page.getByText("已刷新历史记录，当前列表已核对。", { exact: true })).toBeVisible();
    await expect(card).toBeVisible();
    await expect(page.getByText("已选 1 项", { exact: true })).toBeVisible();
    expect(commands).toHaveLength(1);
    expect(mocks.writes).toEqual([]);
    expect(mocks.unhandled).toEqual([]);
  });
}
