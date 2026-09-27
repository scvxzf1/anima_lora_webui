import { expect, test, type Page } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

async function captionFixture(page: Page) {
  const mocks = await mockWorkspace(page);
  const writes: { method: string; path: string; body: unknown }[] = [];
  const job = {
    id: "caption-1", state: "done", profile_name: "Studio captions", profile_id: "provider-a", total: 55, completed: 55, failed: 0,
    items: Array.from({ length: 55 }, (_, i) => ({
      id: `image-${i}`, name: `studio-${i}.png`, file: `studio-${i}.png`, state: "ready",
      caption: "Original caption", proposed_caption: `Candidate ${i}`,
      url: `/api/config/dataset-presets/image?image=${i}`,
    })),
  };
  await page.route((url) => url.pathname.startsWith("/api/captioning/jobs/caption-1"), (route) => {
    const path = new URL(route.request().url()).pathname;
    const method = route.request().method();
    if (method !== "GET") {
      const body = route.request().postDataJSON();
      writes.push({ method, path, body });
      if (method === "PATCH") {
        const id = path.split("/").at(-1);
        const item = job.items.find((item) => item.id === id)!;
        item.proposed_caption = body.proposed_caption;
      } else return route.fulfill({ status: 409, json: { error: "TXT changed externally" } });
    }
    return route.fulfill({ json: { ok: true, job } });
  });
  return { ...mocks, commandWrites: writes, captionJob: job };
}

test("caption drafts retain object identity across pages and block silent navigation loss", async ({ page }) => {
  const mocks = await captionFixture(page);
  await page.goto("/next/captioning?job=caption-1");
  const editor = page.getByRole("textbox", { name: "候选标注", exact: true });
  await editor.fill("Draft zero");
  await page.getByRole("checkbox", { name: "选择 studio-0.png", exact: true }).check();
  await page.getByRole("button", { name: "下一页候选" }).click();
  await expect(page.locator(".caption-current-object")).toContainText("studio-0.png · 第 1 页");
  await expect(editor).toHaveValue("Draft zero");
  await expect(page.locator(".caption-review-context")).toContainText("其他页 1 项");
  await page.getByRole("button", { name: "studio-50.png ready", exact: true }).click();
  await editor.fill("Draft fifty");
  await page.getByRole("checkbox", { name: "选择 studio-50.png", exact: true }).check();
  await expect(page.locator(".caption-review-context")).toContainText("未保存草稿 2 项");
  await expect(page.getByRole("button", { name: "写回选中 (2)", exact: true })).toBeDisabled();
  await page.locator(".caption-draft-list summary").click();
  await page.locator(".caption-draft-list").getByRole("button", { name: "studio-0.png · 第 1 页" }).click();
  await expect(editor).toHaveValue("Draft zero");
  await expect(page.getByRole("button", { name: "上一页候选" })).toBeDisabled();

  let prompt = "";
  page.once("dialog", async (dialog) => { prompt = dialog.message(); await dialog.dismiss(); });
  await page.getByRole("link", { name: "接入预设", exact: true }).click();
  await expect(editor).toHaveValue("Draft zero");
  await expect.poll(() => prompt).toContain("离开会丢失");
  await page.getByRole("button", { name: "保存候选", exact: true }).click();
  await expect(page.locator(".caption-review-context")).toContainText("未保存草稿 1 项");
  await page.locator(".caption-draft-list").getByRole("button", { name: "studio-50.png · 第 2 页" }).click();
  await expect(editor).toHaveValue("Draft fifty");
  await page.getByRole("button", { name: "保存候选", exact: true }).click();
  await expect(page.locator(".caption-review-context")).toContainText("未保存草稿 0 项");
  page.once("dialog", async (dialog) => { prompt = dialog.message(); await dialog.accept(); });
  await page.getByRole("button", { name: "写回选中 (2)", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("TXT changed externally");
  expect(prompt).toContain("2 项已保存候选");
  expect(prompt).toContain("其他页");
  expect(mocks.commandWrites.map((write) => write.method)).toEqual(["PATCH", "PATCH", "POST"]);
  expect(mocks.commandWrites.at(-1)?.body).toEqual({ item_ids: ["image-0", "image-50"] });
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("captioning success commits the selected candidate and keeps the task summary", async ({ page }) => {
  const mocks = await captionFixture(page);
  const commits: string[][] = [];
  await page.route(
    url => url.pathname === "/api/captioning/jobs/caption-1/commit",
    async route => {
      const body = route.request().postDataJSON() as { item_ids: string[] };
      commits.push(body.item_ids);
      return route.fulfill({ json: {
        written: body.item_ids.length,
        conflicts: 0,
        skipped: 0,
        errors: [],
        job: mocks.captionJob,
      } });
    },
  );

  await page.goto("/next/captioning?job=caption-1");
  await expect(page.getByRole("heading", { name: "Studio captions", exact: true })).toBeVisible();
  await expect(page.locator(".review-heading")).toContainText("done · 55/55");
  await page.getByRole("checkbox", { name: "选择 studio-0.png", exact: true }).check();
  page.once("dialog", dialog => dialog.accept());
  await page.getByRole("button", { name: "写回选中 (1)", exact: true }).click();
  await expect(page.getByRole("status").filter({ hasText: "写入 1" })).toBeVisible();
  await expect(page.getByRole("status").filter({ hasText: "冲突 0" })).toBeVisible();
  expect(commits).toEqual([["image-0"]]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("caption rerun locks controls until the selected items enter the active job", async ({ page }) => {
  const mocks = await captionFixture(page);
  const job = mocks.captionJob;
  job.state = "failed";
  job.failed = 1;
  job.items[0].state = "failed";
  const attempts: { profile_id: string; item_ids: string[] }[] = [];
  let markStarted!: () => void;
  const started = new Promise<void>((resolve) => {
    markStarted = resolve;
  });
  let releaseResponse!: () => void;
  const responseGate = new Promise<void>((resolve) => {
    releaseResponse = resolve;
  });
  await page.route(
    (url) => url.pathname === "/api/captioning/jobs/caption-1/rerun",
    async (route) => {
      attempts.push(route.request().postDataJSON() as { profile_id: string; item_ids: string[] });
      markStarted();
      await responseGate;
      return route.fulfill({
        status: 202,
        json: {
          ok: true,
          job: {
            ...job,
            state: "queued",
            completed: 0,
            failed: 0,
            items: job.items.map((item) => item.id === "image-0" ? { ...item, state: "queued" } : item),
          },
        },
      });
    },
  );

  await page.goto("/next/captioning?job=caption-1");
  const selected = page.getByRole("checkbox", { name: "选择 studio-0.png", exact: true });
  await selected.check();
  const rerun = page.getByRole("button", { name: "重新打标", exact: true });
  page.once("dialog", (dialog) => dialog.accept());
  await rerun.click();
  await started;
  await expect(rerun).toBeDisabled();
  await expect(selected).toBeDisabled();
  await expect(page.getByRole("button", { name: "取消任务", exact: true })).toBeDisabled();
  expect(attempts).toEqual([{ profile_id: "provider-a", item_ids: ["image-0"] }]);

  releaseResponse();
  await expect(page.getByText("queued · 0/55 · 失败 0", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "取消任务", exact: true })).toBeEnabled();
  await expect(rerun).toBeDisabled();
  expect(mocks.commandWrites).toEqual([]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("caption rerun conflict refreshes a stale terminal job before exposing controls", async ({ page }) => {
  const mocks = await captionFixture(page);
  const job = mocks.captionJob;
  job.state = "failed";
  job.failed = 1;
  job.items[0].state = "failed";
  let reads = 0;
  let reruns = 0;
  await page.route((url) => url.pathname === "/api/captioning/jobs/caption-1", (route) => {
    if (route.request().method() !== "GET") return route.fallback();
    reads += 1;
    return route.fulfill({ json: { ok: true, job } });
  });
  await page.route((url) => url.pathname === "/api/captioning/jobs/caption-1/rerun", (route) => {
    reruns += 1;
    job.state = "running";
    job.completed = 0;
    job.failed = 0;
    job.items[0].state = "running";
    return route.fulfill({ status: 409, json: { error: "任务仍在运行，结束后才能重新打标" } });
  });

  await page.goto("/next/captioning?job=caption-1");
  await page.getByRole("checkbox", { name: "选择 studio-0.png", exact: true }).check();
  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "重新打标", exact: true }).click();

  await expect(page.getByRole("alert")).toContainText("任务仍在运行，结束后才能重新打标");
  await expect(page.getByText("running · 0/55 · 失败 0", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "重新打标", exact: true })).toBeDisabled();
  await expect(page.getByRole("button", { name: "取消任务", exact: true })).toBeEnabled();
  expect(reads).toBeGreaterThan(1);
  await page.waitForTimeout(1200);
  expect(reruns).toBe(1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("caption rerun preserves the failed snapshot and unlocks after a server error", async ({ page }) => {
  const mocks = await captionFixture(page);
  const job = mocks.captionJob;
  job.state = "failed";
  job.failed = 1;
  job.items[0].state = "failed";
  let reruns = 0;
  await page.route((url) => url.pathname === "/api/captioning/jobs/caption-1/rerun", (route) => {
    reruns += 1;
    return route.fulfill({ status: 500, json: { error: "打标任务重试失败" } });
  });

  await page.goto("/next/captioning?job=caption-1");
  const rerun = page.getByRole("button", { name: "重新打标", exact: true });
  await expect(rerun).toBeEnabled();
  page.once("dialog", (dialog) => dialog.accept());
  await rerun.click();

  await expect(page.getByRole("alert")).toContainText("打标任务重试失败");
  await expect(page.getByText("failed · 55/55 · 失败 1", { exact: true })).toBeVisible();
  await expect(rerun).toBeEnabled();
  await page.waitForTimeout(1200);
  expect(reruns).toBe(1);
  expect(mocks.commandWrites).toEqual([]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("caption rerun locks after an unknown response until the job is reconciled", async ({ page }) => {
  const mocks = await captionFixture(page);
  const initial = mocks.captionJob;
  initial.state = "failed";
  initial.failed = 1;
  initial.items[0].state = "failed";
  let authoritativeJob = { ...initial, items: initial.items.map((item) => ({ ...item })) };
  let reruns = 0;
  let reads = 0;
  await page.route((url) => url.pathname === "/api/captioning/jobs/caption-1", (route) => {
    if (route.request().method() !== "GET") return route.fallback();
    reads += 1;
    return route.fulfill({ json: { ok: true, job: authoritativeJob } });
  });
  await page.route((url) => url.pathname === "/api/captioning/jobs/caption-1/rerun", (route) => {
    reruns += 1;
    authoritativeJob = {
      ...authoritativeJob,
      state: "queued",
      completed: 0,
      failed: 0,
      items: authoritativeJob.items.map((item) =>
        item.id === "image-0" ? { ...item, state: "queued" } : item,
      ),
    };
    return route.abort();
  });

  await page.goto("/next/captioning?job=caption-1");
  await page.getByRole("checkbox", { name: "选择 studio-0.png", exact: true }).check();
  const selected = page.getByRole("checkbox", { name: "选择 studio-0.png", exact: true });
  const rerun = page.getByRole("button", { name: "重新打标", exact: true });
  page.once("dialog", (dialog) => dialog.accept());
  await rerun.click();

  await expect(page.getByText("操作结果尚未确认。核对服务器任务状态后再继续。", { exact: true })).toBeVisible();
  await expect(rerun).toBeDisabled();
  await expect(selected).toBeDisabled();
  await expect(page.getByRole("button", { name: "核对任务状态", exact: true })).toBeEnabled();
  expect(reruns).toBe(1);
  const readsBeforeCheck = reads;
  expect(readsBeforeCheck).toBeGreaterThan(0);

  await page.getByRole("button", { name: "核对任务状态", exact: true }).click();
  await expect(page.getByText("queued · 0/55 · 失败 0", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "核对任务状态", exact: true })).toHaveCount(0);
  await expect(page.getByRole("alert")).toHaveCount(0);
  await expect(rerun).toBeDisabled();
  expect(reads).toBeGreaterThan(readsBeforeCheck);
  expect(reruns).toBe(1);
  expect(mocks.commandWrites).toEqual([]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("running caption cancellation confirms once, locks controls, and settles to canceled", async ({ page }) => {
  const mocks = await captionFixture(page);
  const job = mocks.captionJob;
  job.state = "running";
  job.completed = 0;
  job.failed = 0;
  const attempts: { method: string; path: string; body: string | null }[] = [];
  let markStarted!: () => void;
  const started = new Promise<void>((resolve) => { markStarted = resolve; });
  let releaseResponse!: () => void;
  const responseGate = new Promise<void>((resolve) => { releaseResponse = resolve; });
  await page.route(
    (url) => url.pathname === "/api/captioning/jobs/caption-1/cancel",
    async (route) => {
      attempts.push({
        method: route.request().method(),
        path: new URL(route.request().url()).pathname,
        body: route.request().postData(),
      });
      markStarted();
      await responseGate;
      return route.fulfill({ json: { ok: true, job: { ...job, state: "canceled" } } });
    },
  );

  await page.goto("/next/captioning?job=caption-1");
  const cancel = page.getByRole("button", { name: "取消任务", exact: true });
  const rerun = page.getByRole("button", { name: "重新打标", exact: true });
  const selected = page.getByRole("checkbox", { name: "选择 studio-0.png", exact: true });
  await expect(cancel).toBeEnabled();
  let confirmation = "";
  page.once("dialog", async (dialog) => {
    confirmation = dialog.message();
    await dialog.accept();
  });
  await cancel.click();
  await started;
  await expect(cancel).toBeDisabled();
  await expect(rerun).toBeDisabled();
  await expect(selected).toBeDisabled();
  expect(attempts).toEqual([{
    method: "POST",
    path: "/api/captioning/jobs/caption-1/cancel",
    body: null,
  }]);

  releaseResponse();
  await expect(page.getByText("canceled · 0/55 · 失败 0", { exact: true })).toBeVisible();
  await expect(cancel).toBeDisabled();
  await expect(rerun).toBeEnabled();
  await expect(selected).toBeEnabled();
  expect(confirmation).toBe("取消此打标任务？已生成候选会保留。");
  await page.waitForTimeout(1200);
  expect(attempts).toHaveLength(1);
  expect(mocks.commandWrites).toEqual([]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("caption cancel conflict refreshes a stale running job before exposing terminal controls", async ({ page }) => {
  const mocks = await captionFixture(page);
  const job = mocks.captionJob;
  job.state = "running";
  job.completed = 4;
  let serverJob = { ...job };
  let reads = 0;
  const cancelCalls: { method: string; path: string; body: string | null }[] = [];
  await page.route((url) => url.pathname === "/api/captioning/jobs/caption-1", (route) => {
    if (route.request().method() !== "GET") return route.fallback();
    reads += 1;
    return route.fulfill({ json: { ok: true, job: serverJob } });
  });
  await page.route((url) => url.pathname === "/api/captioning/jobs/caption-1/cancel", (route) => {
    cancelCalls.push({
      method: route.request().method(),
      path: new URL(route.request().url()).pathname,
      body: route.request().postData(),
    });
    serverJob = { ...serverJob, state: "done", completed: 55 };
    return route.fulfill({ status: 409, json: { error: "当前任务已经结束" } });
  });

  await page.goto("/next/captioning?job=caption-1");
  const cancel = page.getByRole("button", { name: "取消任务", exact: true });
  const rerun = page.getByRole("button", { name: "重新打标", exact: true });
  await expect(cancel).toBeEnabled();
  page.once("dialog", (dialog) => dialog.accept());
  await cancel.click();

  await expect(page.getByRole("alert")).toContainText("当前任务已经结束");
  await expect(page.getByText("done · 55/55 · 失败 0", { exact: true })).toBeVisible();
  await expect(cancel).toBeDisabled();
  await expect(rerun).toBeEnabled();
  expect(reads).toBeGreaterThan(1);
  expect(cancelCalls).toEqual([{
    method: "POST",
    path: "/api/captioning/jobs/caption-1/cancel",
    body: null,
  }]);
  await page.waitForTimeout(1200);
  expect(cancelCalls).toHaveLength(1);
  expect(mocks.commandWrites).toEqual([]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("caption cancel HTTP 500 preserves the running snapshot and requires explicit retry", async ({ page }) => {
  const mocks = await captionFixture(page);
  const job = mocks.captionJob;
  job.state = "running";
  job.completed = 4;
  const cancelCalls: { method: string; path: string }[] = [];
  await page.route((url) => url.pathname === "/api/captioning/jobs/caption-1/cancel", async (route) => {
    cancelCalls.push({
      method: route.request().method(),
      path: new URL(route.request().url()).pathname,
    });
    if (cancelCalls.length === 1) {
      return route.fulfill({ status: 500, json: { error: "取消任务失败" } });
    }
    job.state = "canceled";
    return route.fulfill({ status: 202, json: { ok: true, job } });
  });

  await page.goto("/next/captioning?job=caption-1");
  const cancel = page.getByRole("button", { name: "取消任务", exact: true });
  const rerun = page.getByRole("button", { name: "重新打标", exact: true });
  page.on("dialog", (dialog) => { void dialog.accept(); });
  await cancel.click();

  await expect(page.getByRole("alert")).toContainText("取消任务失败");
  await expect(page.getByText("running · 4/55 · 失败 0", { exact: true })).toBeVisible();
  await expect(cancel).toBeEnabled();
  await expect(rerun).toBeDisabled();
  await page.waitForTimeout(1200);
  expect(cancelCalls).toEqual([{
    method: "POST",
    path: "/api/captioning/jobs/caption-1/cancel",
  }]);

  await cancel.click();
  await expect(page.getByText("canceled · 4/55 · 失败 0", { exact: true })).toBeVisible();
  expect(cancelCalls).toHaveLength(2);
  expect(mocks.commandWrites).toEqual([]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("caption cancel locks after an unknown response until the job is reconciled", async ({ page }) => {
  const mocks = await captionFixture(page);
  const initial = mocks.captionJob;
  initial.state = "running";
  initial.completed = 4;
  let authoritativeJob = { ...initial, items: initial.items.map((item) => ({ ...item })) };
  const cancelCalls: string[] = [];
  let reads = 0;
  await page.route((url) => url.pathname === "/api/captioning/jobs/caption-1", (route) => {
    if (route.request().method() !== "GET") return route.fallback();
    reads += 1;
    return route.fulfill({ json: { ok: true, job: authoritativeJob } });
  });
  await page.route((url) => url.pathname === "/api/captioning/jobs/caption-1/cancel", (route) => {
    cancelCalls.push(new URL(route.request().url()).pathname);
    authoritativeJob = { ...authoritativeJob, state: "canceled" };
    return route.abort();
  });

  await page.goto("/next/captioning?job=caption-1");
  const cancel = page.getByRole("button", { name: "取消任务", exact: true });
  const rerun = page.getByRole("button", { name: "重新打标", exact: true });
  const selected = page.getByRole("checkbox", { name: "选择 studio-0.png", exact: true });
  page.once("dialog", (dialog) => dialog.accept());
  await cancel.click();

  await expect(page.getByText("操作结果尚未确认。核对服务器任务状态后再继续。", { exact: true })).toBeVisible();
  await expect(cancel).toBeDisabled();
  await expect(rerun).toBeDisabled();
  await expect(selected).toBeDisabled();
  expect(cancelCalls).toEqual(["/api/captioning/jobs/caption-1/cancel"]);

  await page.getByRole("button", { name: "核对任务状态", exact: true }).click();
  await expect(page.getByText("canceled · 4/55 · 失败 0", { exact: true })).toBeVisible();
  await expect(cancel).toBeDisabled();
  await expect(rerun).toBeEnabled();
  await expect(page.getByRole("alert")).toHaveCount(0);
  expect(reads).toBeGreaterThan(1);
  expect(cancelCalls).toHaveLength(1);
  expect(mocks.commandWrites).toEqual([]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("caption candidate save keeps its draft after a lost response and can be retried", async ({ page }) => {
  const mocks = await captionFixture(page);
  const attempts: { method: string; path: string; body: unknown }[] = [];
  await page.route((url) => url.pathname === "/api/captioning/jobs/caption-1/items/image-0", async (route) => {
    attempts.push({ method: route.request().method(), path: new URL(route.request().url()).pathname, body: route.request().postDataJSON() });
    if (attempts.length === 1) return route.abort();
    return route.fallback();
  });

  await page.goto("/next/captioning?job=caption-1");
  const editor = page.getByRole("textbox", { name: "候选标注", exact: true });
  const save = page.getByRole("button", { name: "保存候选", exact: true });
  await editor.fill("Offline candidate draft");
  await save.click();

  await expect(page.getByRole("alert")).toContainText("操作结果尚未确认");
  await expect(editor).toHaveValue("Offline candidate draft");
  await expect(page.locator(".caption-review-context")).toContainText("未保存草稿 1 项");
  await expect(save).toBeEnabled();
  expect(attempts).toHaveLength(1);

  await save.click();
  await expect(page.locator(".caption-review-context")).toContainText("未保存草稿 0 项");
  await expect(editor).toHaveValue("Offline candidate draft");
  await expect(page.getByRole("alert")).toHaveCount(0);
  expect(attempts).toHaveLength(2);
  expect(attempts).toEqual(Array.from({ length: 2 }, () => ({
    method: "PATCH",
    path: "/api/captioning/jobs/caption-1/items/image-0",
    body: { proposed_caption: "Offline candidate draft" },
  })));
  expect(mocks.commandWrites).toEqual([{ method: "PATCH", path: "/api/captioning/jobs/caption-1/items/image-0", body: { proposed_caption: "Offline candidate draft" } }]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("caption TXT commit stays pending without auto-retry and supports explicit retry", async ({ page }) => {
  const mocks = await captionFixture(page);
  const attempts: { method: string; path: string; body: unknown }[] = [];
  let announceCommitStarted = () => {};
  let releaseCommitResponse = () => {};
  const commitStarted = new Promise<void>((resolve) => { announceCommitStarted = resolve; });
  const commitResponseGate = new Promise<void>((resolve) => { releaseCommitResponse = resolve; });
  const committedJob = {
    ...mocks.captionJob,
    items: mocks.captionJob.items.map((item, index) => index === 0 ? { ...item, state: "committed" } : item),
  };
  await page.route((url) => url.pathname === "/api/captioning/jobs/caption-1/commit", async (route) => {
    attempts.push({
      method: route.request().method(),
      path: new URL(route.request().url()).pathname,
      body: route.request().postDataJSON(),
    });
    if (attempts.length === 1) {
      announceCommitStarted();
      await commitResponseGate;
      return route.abort();
    }
    return route.fulfill({ json: {
      ok: true,
      written: 1,
      conflicts: 0,
      skipped: 0,
      errors: [],
      job: committedJob,
    } });
  });

  await page.goto("/next/captioning?job=caption-1");
  const selected = page.getByRole("checkbox", { name: "选择 studio-0.png", exact: true });
  const commit = page.getByRole("button", { name: "写回选中 (1)", exact: true });
  await selected.check();
  page.once("dialog", (dialog) => dialog.accept());
  await commit.click();
  await commitStarted;
  await expect(commit).toBeDisabled();
  await expect(selected).toBeDisabled();
  expect(attempts).toHaveLength(1);

  releaseCommitResponse();
  await expect(page.getByRole("alert")).toContainText("操作结果尚未确认");
  await expect(commit).toBeEnabled();
  await expect(selected).toBeEnabled();
  await expect(selected).toBeChecked();
  expect(attempts).toHaveLength(1);

  page.once("dialog", (dialog) => dialog.accept());
  await commit.click();
  await expect(page.getByText("写入 1 · 冲突 0 · 跳过 0", { exact: true })).toBeVisible();
  expect(attempts).toEqual([
    {
      method: "POST",
      path: "/api/captioning/jobs/caption-1/commit",
      body: { item_ids: ["image-0"] },
    },
    {
      method: "POST",
      path: "/api/captioning/jobs/caption-1/commit",
      body: { item_ids: ["image-0"] },
    },
  ]);
  expect(mocks.commandWrites).toEqual([]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 720 }, { width: 768, height: 1024 }, { width: 390, height: 844 }]) {
  for (const theme of ["dark", "light"]) {
    test(`caption focused workspace ${viewport.width} ${theme}`, async ({ page }, info) => {
      await page.setViewportSize(viewport);
      const mocks = await captionFixture(page);
      await page.addInitScript((theme) => localStorage.setItem("dragon-next-ui-v1-theme", theme), theme);
      await page.goto("/next/captioning?job=caption-1");
      await expect(page.getByRole("textbox", { name: "候选标注", exact: true })).toHaveValue("Candidate 0");
      await expect.poll(() => page.locator(".caption-original").evaluate((node) => (node as HTMLImageElement).naturalWidth)).toBe(480);
      await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      if (viewport.width < 650) {
        await expect(page.locator(".review-items")).not.toBeVisible();
        const imageTop = await page.locator(".caption-original").evaluate((node) => node.getBoundingClientRect().top);
        expect(imageTop).toBeLessThan(viewport.height);
        await page.getByRole("button", { name: "候选列表 (55)", exact: true }).click();
        await expect(page.locator(".review-items")).toBeVisible();
        await page.getByRole("button", { name: "下一页候选" }).click();
        await page.getByRole("button", { name: "studio-50.png ready", exact: true }).click();
        await expect(page.getByRole("textbox", { name: "候选标注", exact: true })).toHaveValue("Candidate 50");
      }
      await page.screenshot({ path: info.outputPath("caption-editor.png"), fullPage: true });
      expect(mocks.commandWrites).toEqual([]);
      expect(mocks.unhandled).toEqual([]);
    });
  }
}

test("missing image remains distinct from editable text and retry recovers the preview", async ({ page }) => {
  const mocks = await captionFixture(page);
  let missing = true;
  await page.route((url) => url.pathname === "/api/config/dataset-presets/image", (route) => missing
    ? route.fulfill({ status: 404, body: "missing image" }) : route.fallback());
  await page.goto("/next/captioning?job=caption-1");
  await expect(page.locator(".caption-image-error")).toContainText("无法读取图片：studio-0.png");
  await expect(page.getByRole("textbox", { name: "候选标注", exact: true })).toHaveValue("Candidate 0");
  missing = false;
  await page.getByRole("button", { name: "重试图片" }).click();
  await expect.poll(() => page.locator(".caption-original").evaluate((node) => (node as HTMLImageElement).naturalWidth)).toBe(480);
  expect(mocks.commandWrites).toEqual([]); expect(mocks.writes).toEqual([]);
});

test("switching caption jobs requires explicit draft discard and isolates item state", async ({ page }) => {
  const mocks = await captionFixture(page);
  const next = { id: "caption-2", profile_name: "Second job", state: "done", total: 1, completed: 1, failed: 0,
    items: [{ id: "image-0", name: "second.png", state: "ready", caption: "", proposed_caption: "Second candidate", url: "" }] };
  await page.route((url) => url.pathname === "/api/captioning/jobs", (route) => route.fulfill({ json: { jobs: [next] } }));
  await page.route((url) => url.pathname === "/api/captioning/jobs/caption-2", (route) => route.fulfill({ json: { job: next } }));
  await page.goto("/next/captioning?job=caption-1");
  await page.getByRole("textbox", { name: "候选标注", exact: true }).fill("Unsaved first job");
  await Promise.all([
    page.waitForEvent("dialog").then((dialog) => dialog.dismiss()),
    page.getByRole("button", { name: /Second job/ }).click(),
  ]);
  await expect(page).toHaveURL(/job=caption-1/);
  await expect(page.getByRole("textbox", { name: "候选标注", exact: true })).toHaveValue("Unsaved first job");
  await Promise.all([
    page.waitForEvent("dialog").then((dialog) => dialog.accept()),
    page.getByRole("button", { name: /Second job/ }).click(),
  ]);
  await expect(page).toHaveURL(/job=caption-2/);
  await expect(page.getByRole("textbox", { name: "候选标注", exact: true })).toHaveValue("Second candidate");
  await expect(page.locator(".caption-review-context")).toContainText("未保存草稿 0 项");
  await expect(page.locator(".caption-image-error")).toContainText("未保存可用图片地址");
  expect(mocks.commandWrites).toEqual([]); expect(mocks.writes).toEqual([]);
});
