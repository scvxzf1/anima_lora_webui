import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

test("history root switch reloads log metadata and the selected page", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  const taskId = "history-log-root-task";
  let activeRoot = "history";
  const settingsWrites: unknown[] = [];
  const listRoots: string[] = [];
  const detailRoots: string[] = [];
  const logReads: Array<{ root: string; offset: string | null; limit: number }> = [];
  const searchReads: Array<{ root: string; query: string | null; cursor: string | null; direction: string | null }> = [];
  let announceExternalRead!: () => void;
  let releaseExternalRead!: () => void;
  const externalReadStarted = new Promise<void>((resolve) => {
    announceExternalRead = resolve;
  });
  const externalReadGate = new Promise<void>((resolve) => {
    releaseExternalRead = resolve;
  });
  let announceExternalSearchRead!: () => void;
  let releaseExternalSearchRead!: () => void;
  const externalSearchReadStarted = new Promise<void>((resolve) => {
    announceExternalSearchRead = resolve;
  });
  const externalSearchReadGate = new Promise<void>((resolve) => {
    releaseExternalSearchRead = resolve;
  });

  const settingsPayload = () => ({
    revision: activeRoot === "history" ? "settings-r1" : "settings-r2",
    output_root: "output/runs",
    ui_scale: 100,
    tagging_max_retained_jobs: 40,
    path_overrides: {
      configs_root: "",
      history_root: activeRoot === "history" ? "" : activeRoot,
      queue_root: "",
    },
    effective_paths: {
      configs_root: "/workspace/configs",
      history_root: `/workspace/${activeRoot}`,
      queue_root: "/workspace/queue",
    },
    defaults: { output_root: "output/runs", ui_scale: 100 },
  });

  await page.addInitScript(() => { window.confirm = () => true; });
  await page.route(
    (url) => url.pathname === "/api/settings/global" ||
      url.pathname === "/api/training/history" ||
      url.pathname === `/api/training/history/${taskId}` ||
      url.pathname === `/api/training/history/${taskId}/artifacts` ||
      url.pathname.startsWith(`/api/training/history/${taskId}/logs`),
    async (route) => {
      const url = new URL(route.request().url());
      const method = route.request().method();
      if (url.pathname === "/api/settings/global") {
        if (method === "GET") return route.fulfill({ json: settingsPayload() });
        if (method === "PUT") {
          const body = route.request().postDataJSON() as { history_root?: string };
          settingsWrites.push(body);
          activeRoot = body.history_root || "history";
          return route.fulfill({ json: {
            ...settingsPayload(),
            ok: true,
            requires_reload: true,
          } });
        }
        return route.fallback();
      }
      if (url.pathname === "/api/training/history" && method === "GET") {
        listRoots.push(activeRoot);
        return route.fulfill({ json: {
          ok: true,
          tasks: [{
            id: taskId,
            name: `${activeRoot} log task`,
            job: "training",
            state: "error",
            group: "Root audit",
            run_dir: `/workspace/${activeRoot}/run`,
            metric_count: 0,
            log_count: activeRoot === "history" ? 800 : 500,
          }],
        } });
      }
      if (url.pathname === `/api/training/history/${taskId}` && method === "GET") {
        detailRoots.push(activeRoot);
        return route.fulfill({ json: {
          task: {
            id: taskId,
            name: `${activeRoot} log task`,
            job: "training",
            state: "error",
            group: "Root audit",
            run_dir: `/workspace/${activeRoot}/run`,
            metric_count: 0,
            log_count: activeRoot === "history" ? 800 : 500,
          },
          metrics: [],
          logs: [],
          system: [],
          config_toml: "network_dim = 32",
        } });
      }
      if (url.pathname === `/api/training/history/${taskId}/artifacts` && method === "GET")
        return route.fulfill({ json: { artifacts: [] } });
      if (url.pathname === `/api/training/history/${taskId}/logs/search` && method === "GET") {
        searchReads.push({
          root: activeRoot,
          query: url.searchParams.get("query"),
          cursor: url.searchParams.get("cursor"),
          direction: url.searchParams.get("direction"),
        });
        if (activeRoot === "external-history") {
          announceExternalSearchRead();
          await externalSearchReadGate;
        }
        return route.fulfill({ json: {
          match_index: activeRoot === "history" ? 799 : 499,
          match_ordinal: activeRoot === "history" ? 1 : 2,
          matches_total: 2,
          total: activeRoot === "history" ? 800 : 500,
        } });
      }
      if (url.pathname === `/api/training/history/${taskId}/logs` && method === "GET") {
        const params = url.searchParams;
        const limit = Number(params.get("limit"));
        const offsetParam = params.get("offset");
        const total = activeRoot === "history" ? 800 : 500;
        const offset = offsetParam === null ? total - limit : Number(offsetParam);
        logReads.push({ root: activeRoot, offset: offsetParam, limit });
        if (activeRoot === "external-history" && offsetParam === null) {
          announceExternalRead();
          await externalReadGate;
        }
        return route.fulfill({ json: {
          total,
          offset,
          logs: Array.from(
            { length: Math.min(limit, Math.max(0, total - offset)) },
            (_, index) => ({ line: `${activeRoot} paged log row ${offset + index + 1}` }),
          ),
        } });
      }
      return route.fallback();
    },
  );

  await page.goto(`/next/history/${taskId}?view=logs`);
  await expect(page.getByRole("log")).toContainText("history paged log row 800");
  await expect(page.locator(".history-log-footer")).toContainText("共 800 行");
  await page.getByLabel("搜索全部日志", { exact: true }).fill("root needle");
  await page.getByLabel("执行全局搜索").click();
  await expect(page.getByText("1 / 2 匹配", { exact: true })).toBeVisible();
  await expect(page.locator('[data-match="true"]')).toContainText("history paged log row 800");
  await page.getByRole("link", { name: "全局设置", exact: true }).click();
  await page.getByRole("textbox", { name: /历史目录/ }).fill("external-history");
  await page.getByRole("button", { name: "保存设置", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("现有文件未迁移");

  await page.getByRole("link", { name: "历史任务", exact: true }).click();
  await page.getByRole("button", { name: /external-history log task 1 条/ }).click();
  await page.locator("a.history-card-link").filter({ hasText: "external-history log task" }).click();
  await expect(page.getByRole("heading", { name: "external-history log task" })).toBeVisible();
  await page.getByRole("link", { name: "日志", exact: true }).click();
  try {
    await externalReadStarted;
    await expect(page.getByRole("log")).not.toContainText("history paged log row 800");
    await expect(page.locator(".history-log-footer")).not.toContainText("共 800 行");
  } finally {
    releaseExternalRead();
  }
  await expect(page.getByRole("log")).toContainText("external-history paged log row 500");
  await expect(page.locator(".history-log-footer")).toContainText("共 500 行");
  await page.getByLabel("搜索全部日志", { exact: true }).fill("root needle");
  await page.getByLabel("执行全局搜索").click();
  try {
    await externalSearchReadStarted;
    await expect(page.getByText("搜索中…", { exact: true })).toBeVisible();
    await expect(page.locator('[data-match="true"]')).toHaveCount(0);
  } finally {
    releaseExternalSearchRead();
  }
  await expect(page.getByText("2 / 2 匹配", { exact: true })).toBeVisible();
  await expect(page.locator('[data-match="true"]')).toContainText("external-history paged log row 500");

  for (const root of ["history", "external-history"]) {
    expect(logReads).toContainEqual({ root, offset: null, limit: 1 });
    expect(logReads).toContainEqual({ root, offset: "400", limit: 400 });
  }
  expect(searchReads).toEqual([
    { root: "history", query: "root needle", cursor: "0", direction: "forward" },
    { root: "external-history", query: "root needle", cursor: "0", direction: "forward" },
  ]);
  expect(listRoots).toContain("external-history");
  expect(detailRoots).toContain("external-history");
  expect(settingsWrites).toHaveLength(1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});
