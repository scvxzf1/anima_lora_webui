import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

test("S3 queue state matrix preserves confirmed scope across refresh", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  const items = [
    { id: "s3-waiting", state: "queued", variant: "Waiting" },
    { id: "s3-running", state: "running", variant: "Running" },
    { id: "s3-error", state: "error", variant: "Error", message: "launcher failed" },
    { id: "s3-done", state: "done", variant: "Done" },
    { id: "s3-canceled", state: "canceled", variant: "Canceled" },
  ];
  let revision = "s3-revision-1";
  const requests: unknown[] = [];
  const snapshot = () => {
    const summary = { total: items.length, queued: 0, running: 0, error: 0, done: 0, canceled: 0 };
    for (const item of items) summary[item.state as keyof typeof summary] += 1;
    return { ok: true, revision, paused: true, status: "idle", summary, items: items.map((item) => ({ ...item })) };
  };

  await page.route(
    (url) => url.pathname === "/api/training/queue" && url.search === "",
    (route) => route.fulfill({ json: snapshot() }),
  );
  await page.route((url) => url.pathname === "/api/training/queue/cancel-waiting", (route) => {
    const body = route.request().postDataJSON();
    requests.push(body);
    if (body.expected_revision !== revision) {
      return route.fulfill({ status: 409, json: { error: "队列已发生变化，本次操作未执行。" } });
    }
    for (const item of items) {
      if (item.state === "queued") item.state = "canceled";
    }
    revision = "s3-revision-2";
    return route.fulfill({ json: { ...snapshot(), message: "已取消 1 个等待任务" } });
  });

  await page.goto("/next/queue");
  await expect(page.locator(".queue-card")).toHaveCount(3);
  await page.locator(".queue-stat").filter({ hasText: "全部" }).click();
  await expect(page.locator(".queue-card")).toHaveCount(5);
  for (const state of ["queued", "running", "error", "done", "canceled"]) {
    await expect(page.locator(`.queue-card[data-state="${state}"]`)).toHaveCount(1);
  }

  page.once("dialog", (dialog) => {
    expect(dialog.message()).toContain("当前快照涉及 1 条队列记录");
    dialog.accept();
  });
  await page.getByRole("button", { name: "取消全部等待", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("已取消 1 个等待任务");
  expect(requests).toEqual([{ expected_revision: "s3-revision-1" }]);
  await expect(page.locator('[data-item-id="s3-waiting"]')).toHaveAttribute("data-state", "canceled");

  await page.getByRole("button", { name: "刷新", exact: true }).click();
  await expect(page.locator('[data-item-id="s3-waiting"]')).toHaveAttribute("data-state", "canceled");
  await expect(page.locator(".queue-card")).toHaveCount(5);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("S3 history reports artifact states and converts an absolute resume target", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  const taskId = "s3-history-task";
  const checkpoint = "output/runs/s3-history-task/checkpoints/step-40";
  const submissions: unknown[] = [];
  await page.route((url) => url.pathname === `/api/training/history/${taskId}`, (route) => route.fulfill({
    json: {
      ok: true,
      task: { id: taskId, name: "S3 history fixture", job: "training", state: "error", last_step: 40, final_loss: 0.12, message: "launcher failed" },
      metrics: [], logs: [], system: [], config_toml: "max_train_steps = 100",
    },
  }));
  await page.route((url) => url.pathname === `/api/training/history/${taskId}/artifacts`, (route) => route.fulfill({
    json: {
      ok: true,
      task_id: taskId,
      artifacts: [
        { key: "config-snapshot", state: "available", name: "config.snapshot.toml", size_bytes: 42 },
        { key: "runtime-config", state: "missing", message: "未保存或文件已不存在" },
        { key: "dataset-config", state: "blocked", message: "文件路径不在允许范围内" },
        { key: "logs", state: "unreadable", message: "文件暂不可读" },
      ],
    },
  }));
  await page.route((url) => url.pathname === `/api/training/history/${taskId}/resume-options`, (route) => route.fulfill({
    json: {
      checkpoints: [{
        path: checkpoint,
        name: "step-40",
        step: 40,
        target_total_steps: 100,
        remaining_steps: 60,
        resume_available: true,
        state_complete: true,
        state_integrity: { ok: true, train_state: true, model: true, optimizer: true, scheduler: true, random_state: true },
      }],
      default_checkpoint: checkpoint,
      message: "",
    },
  }));
  await page.route((url) => url.pathname === "/api/training/queue/resume", (route) => {
    submissions.push(route.request().postDataJSON());
    return route.fulfill({ json: { ok: true, message: "续训已排队" } });
  });

  await page.goto(`/next/history/${taskId}`);
  await expect(page.getByText("未保存或文件已不存在", { exact: true })).toBeVisible();
  await expect(page.getByText("文件路径不在允许范围内", { exact: true })).toBeVisible();
  await expect(page.getByText("文件暂不可读", { exact: true })).toBeVisible();
  const configArtifact = page.getByRole("link", { name: "config.snapshot.toml", exact: true });
  await expect(configArtifact).toBeVisible();
  await expect(configArtifact).toHaveAttribute(
    "href",
    `/api/training/history/${taskId}/artifacts/config-snapshot?download=1`,
  );
  await expect(configArtifact).toHaveAttribute("download", "");
  for (const key of ["runtime-config", "dataset-config", "logs"]) {
    await expect(page.locator(`a[href*="/artifacts/${key}"]`)).toHaveCount(0);
  }

  await page.getByRole("button", { name: "检查点续训", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "从历史检查点续训" });
  await expect(dialog.getByText("检查点 40 步 · 历史目标 100 步 · 剩余 60 步", { exact: true })).toBeVisible();
  await dialog.getByRole("spinbutton", { name: "训练目标总步数", exact: true }).fill("120");
  await dialog.getByRole("combobox", { name: "执行方式", exact: true }).selectOption("queue");
  await dialog.getByRole("checkbox").check();
  await dialog.getByRole("button", { name: "确认续训", exact: true }).click();

  await expect(dialog.getByRole("status")).toContainText("续训已排队");
  expect(submissions).toEqual([{
    task_id: taskId,
    checkpoint,
    duration_overrides: { max_train_steps: 80 },
  }]);
  await expect(dialog.getByRole("link", { name: "查看队列", exact: true })).toHaveAttribute("href", "/next/queue");
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("S3 history blocks resume from a checkpoint with incomplete optimizer state", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  const taskId = "s3-damaged-checkpoint";
  const checkpoint = "output/runs/s3-damaged-checkpoint/checkpoints/step-40";
  const submissions: unknown[] = [];
  await page.route((url) => url.pathname === `/api/training/history/${taskId}`, (route) => route.fulfill({
    json: {
      ok: true,
      task: { id: taskId, name: "Damaged checkpoint fixture", job: "training", state: "error", last_step: 40 },
      metrics: [], logs: [], system: [], config_toml: "max_train_steps = 100",
    },
  }));
  await page.route((url) => url.pathname === `/api/training/history/${taskId}/artifacts`, (route) => route.fulfill({
    json: { ok: true, task_id: taskId, artifacts: [] },
  }));
  await page.route((url) => url.pathname === `/api/training/history/${taskId}/resume-options`, (route) => route.fulfill({
    json: {
      checkpoints: [{
        path: checkpoint,
        name: "step-40",
        step: 40,
        target_total_steps: 100,
        remaining_steps: 60,
        resume_available: false,
        state_complete: false,
        state_integrity: { ok: false, train_state: true, model: true, optimizer: false, scheduler: true, random_state: true, missing: ["optimizer.bin"] },
        unavailable_reason: "缺少 optimizer.bin",
      }],
      default_checkpoint: checkpoint,
      message: "",
    },
  }));
  await page.route((url) => url.pathname === "/api/training/queue/resume", (route) => {
    submissions.push(route.request().postDataJSON());
    return route.fulfill({ json: { ok: true, message: "续训已排队" } });
  });

  await page.goto(`/next/history/${taskId}`);
  await page.getByRole("button", { name: "检查点续训", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "从历史检查点续训" });
  await expect(dialog.getByRole("status")).toContainText("缺少 optimizer.bin");
  const submit = dialog.getByRole("button", { name: "确认续训", exact: true });
  await expect(submit).toBeDisabled();
  await submit.click({ force: true });
  expect(submissions).toEqual([]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("history resume binds confirmation to the selected checkpoint in a mixed list", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  const taskId = "s3-mixed-checkpoints";
  const brokenCheckpoint = "output/runs/s3-mixed-checkpoints/checkpoints/step-20";
  const validCheckpoint = "output/runs/s3-mixed-checkpoints/checkpoints/step-40";
  const submissions: unknown[] = [];
  await page.route((url) => url.pathname === `/api/training/history/${taskId}`, (route) => route.fulfill({
    json: {
      ok: true,
      task: { id: taskId, name: "Mixed checkpoint fixture", job: "training", state: "error", last_step: 40 },
      metrics: [], logs: [], system: [], config_toml: "max_train_steps = 100",
    },
  }));
  await page.route((url) => url.pathname === `/api/training/history/${taskId}/artifacts`, (route) => route.fulfill({
    json: { ok: true, task_id: taskId, artifacts: [] },
  }));
  await page.route((url) => url.pathname === `/api/training/history/${taskId}/resume-options`, (route) => route.fulfill({
    json: {
      checkpoints: [
        {
          path: brokenCheckpoint,
          name: "step-20",
          step: 20,
          target_total_steps: 100,
          remaining_steps: 80,
          resume_available: false,
          state_complete: false,
          state_integrity: { ok: false, train_state: true, model: true, optimizer: false, scheduler: true, missing: ["optimizer.bin"] },
          unavailable_reason: "缺少 optimizer.bin",
        },
        {
          path: validCheckpoint,
          name: "step-40",
          step: 40,
          target_total_steps: 100,
          remaining_steps: 60,
          resume_available: true,
          state_complete: true,
          state_integrity: { ok: true, train_state: true, model: true, optimizer: true, scheduler: true, random_state: true },
        },
      ],
      default_checkpoint: validCheckpoint,
      message: "",
    },
  }));
  await page.route((url) => url.pathname === "/api/training/resume", (route) => {
    submissions.push(route.request().postDataJSON());
    return route.fulfill({ json: { ok: true, message: "续训已提交" } });
  });

  await page.goto(`/next/history/${taskId}`);
  await page.getByRole("button", { name: "检查点续训", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "从历史检查点续训" });
  const checkpoint = dialog.getByRole("combobox", { name: "检查点", exact: true });
  const confirmation = dialog.getByRole("checkbox");
  const submit = dialog.getByRole("button", { name: "确认续训", exact: true });
  await expect(checkpoint).toHaveValue(validCheckpoint);
  await confirmation.check();
  await expect(submit).toBeEnabled();

  await checkpoint.selectOption(brokenCheckpoint);
  await expect(dialog.getByRole("status")).toContainText("缺少 optimizer.bin");
  await expect(confirmation).toBeDisabled();
  await expect(submit).toBeDisabled();

  await checkpoint.selectOption(validCheckpoint);
  await expect(confirmation).toBeEnabled();
  await expect(confirmation).not.toBeChecked();
  await expect(submit).toBeDisabled();
  await confirmation.check();
  await submit.click();
  await expect(dialog.getByRole("status")).toContainText("续训已提交");
  expect(submissions).toEqual([{ task_id: taskId, checkpoint: validCheckpoint }]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("history resume options error blocks submission and recovers on explicit retry", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  const taskId = "s1-resume-options-error";
  const checkpoint = "output/runs/s1-resume-options-error/checkpoints/step-40";
  let reads = 0;
  let allowRead = false;
  const submissions: unknown[] = [];
  await page.route((url) => url.pathname === `/api/training/history/${taskId}`, (route) => route.fulfill({
    json: {
      ok: true,
      task: { id: taskId, name: "Resume options error fixture", job: "training", state: "error", last_step: 40 },
      metrics: [], logs: [], system: [], config_toml: "max_train_steps = 100",
    },
  }));
  await page.route((url) => url.pathname === `/api/training/history/${taskId}/artifacts`, (route) => route.fulfill({
    json: { ok: true, task_id: taskId, artifacts: [] },
  }));
  await page.route((url) => url.pathname === `/api/training/history/${taskId}/resume-options`, (route) => {
    reads += 1;
    if (!allowRead) return route.fulfill({ status: 503, json: { error: "检查点目录暂不可读" } });
    return route.fulfill({ json: {
      checkpoints: [{
        path: checkpoint,
        name: "step-40",
        step: 40,
        target_total_steps: 100,
        remaining_steps: 60,
        resume_available: true,
        state_complete: true,
        state_integrity: { ok: true, train_state: true, model: true, optimizer: true, scheduler: true, random_state: true },
      }],
      default_checkpoint: checkpoint,
      message: "",
    } });
  });
  await page.route((url) => url.pathname === "/api/training/queue/resume", (route) => {
    submissions.push(route.request().postDataJSON());
    return route.fulfill({ json: { ok: true, message: "续训已排队" } });
  });

  await page.goto(`/next/history/${taskId}`);
  await page.getByRole("button", { name: "检查点续训", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "从历史检查点续训" });
  const error = dialog.getByRole("alert");
  const confirmation = dialog.getByRole("checkbox");
  const submit = dialog.getByRole("button", { name: "确认续训", exact: true });
  await expect(error).toContainText("检查点目录暂不可读");
  await expect(confirmation).toBeDisabled();
  await expect(submit).toBeDisabled();
  const failedReads = reads;
  await page.waitForTimeout(1200);
  expect(reads).toBe(failedReads);
  expect(submissions).toEqual([]);

  allowRead = true;
  await error.getByRole("button", { name: "重试读取", exact: true }).click();
  await expect(dialog.getByRole("alert")).toHaveCount(0);
  await expect(confirmation).toBeEnabled();
  await confirmation.check();
  await expect(submit).toBeEnabled();
  expect(reads).toBe(failedReads + 1);
  expect(submissions).toEqual([]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("history resume reconciles an accepted queue submission after the response is lost", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  const taskId = "s1-resume-unknown-task";
  const checkpoint = "output/runs/s1-resume-unknown-task/checkpoints/step-40";
  const resumedItemId = "s1-resumed-queue-item";
  const submissions: unknown[] = [];
  let accepted = false;
  let releaseResume: () => void = () => {};
  const resumeGate = new Promise<void>((resolve) => { releaseResume = resolve; });
  const queueSnapshot = () => {
    const items = accepted
      ? [{ id: resumedItemId, state: "queued", variant: "Queued", name: "Resumed from history", task_id: taskId }]
      : [];
    return {
      ok: true,
      revision: accepted ? "resume-revision-2" : "resume-revision-1",
      paused: false,
      status: "idle",
      summary: { total: items.length, queued: items.length, running: 0, error: 0, done: 0, canceled: 0 },
      items,
    };
  };

  await page.route((url) => url.pathname === `/api/training/history/${taskId}`, (route) => route.fulfill({
    json: {
      ok: true,
      task: { id: taskId, name: "Resume unknown fixture", job: "training", state: "error", last_step: 40, message: "launcher failed" },
      metrics: [], logs: [], system: [], config_toml: "max_train_steps = 100",
    },
  }));
  await page.route((url) => url.pathname === `/api/training/history/${taskId}/artifacts`, (route) => route.fulfill({
    json: { ok: true, task_id: taskId, artifacts: [] },
  }));
  await page.route((url) => url.pathname === `/api/training/history/${taskId}/resume-options`, (route) => route.fulfill({
    json: {
      checkpoints: [{
        path: checkpoint,
        name: "step-40",
        step: 40,
        target_total_steps: 100,
        remaining_steps: 60,
        resume_available: true,
        state_complete: true,
        state_integrity: { ok: true, train_state: true, model: true, optimizer: true, scheduler: true, random_state: true },
      }],
      default_checkpoint: checkpoint,
      message: "",
    },
  }));
  await page.route((url) => url.pathname === "/api/training/queue" && url.search === "", (route) => route.fulfill({
    json: queueSnapshot(),
  }));
  await page.route((url) => url.pathname === "/api/training/queue/resume", async (route) => {
    submissions.push(route.request().postDataJSON());
    await resumeGate;
    accepted = true;
    return route.abort();
  });

  await page.goto(`/next/history/${taskId}`);
  await page.getByRole("button", { name: "检查点续训", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "从历史检查点续训" });
  await dialog.getByRole("combobox", { name: "执行方式", exact: true }).selectOption("queue");
  await dialog.getByRole("checkbox").check();
  await dialog.getByRole("button", { name: "确认续训", exact: true }).click();

  await expect.poll(() => submissions.length).toBe(1);
  await expect(dialog.getByRole("button", { name: "取消", exact: true })).toBeDisabled();
  await expect(dialog.getByRole("button", { name: "确认续训", exact: true })).toBeDisabled();
  releaseResume();
  await expect(dialog.getByRole("alert")).toContainText("操作结果尚未确认");
  await expect(dialog.getByRole("link", { name: "核对历史任务", exact: true })).toHaveAttribute("href", "/next/history");
  await expect(dialog.getByRole("link", { name: "核对队列", exact: true })).toHaveAttribute("href", "/next/queue");
  await expect(dialog.getByRole("button", { name: "确认续训", exact: true })).toBeDisabled();
  expect(submissions).toHaveLength(1);
  expect(submissions[0]).toMatchObject({ task_id: taskId, checkpoint });

  await dialog.getByRole("link", { name: "核对队列", exact: true }).click();
  await expect(page).toHaveURL(/\/next\/queue$/);
  await expect(page.locator(`[data-item-id="${resumedItemId}"]`)).toHaveAttribute("data-state", "queued");
  expect(submissions).toHaveLength(1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});
