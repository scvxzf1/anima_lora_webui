import { mkdir } from "node:fs/promises";
import { expect, test, type Page } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

const screenshotDir = "/tmp/legacy-retirement-p2";

async function inspectViewport(page: Page, name: string) {
  await expect.poll(() => page.evaluate(() => document.readyState)).toBe("complete");
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await mkdir(screenshotDir, { recursive: true });
  await page.screenshot({ path: `${screenshotDir}/${name}.png`, fullPage: true });
}

const task = {
  id: "p2-run-1",
  name: "P2 fixture run",
  job: "training",
  state: "idle",
  history_group_key: "p2-group",
  history_group_label: "P2 group",
  history_source_config_file: "configs/p2.toml",
  methods_subdir: "imported",
  variant: "lora",
  preset: "default",
  run_dir: "output/runs/p2-run-1",
  metric_count: 1,
  log_count: 1,
};

test("preview source and scope switch, responsive layout, and API parameters", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const mocks = await mockWorkspace(page);
  const imageRequests: URL[] = [];
  const weightRequests: URL[] = [];
  await page.route((url) => url.pathname === "/api/preview/settings", (route) =>
    route.fulfill({ json: { training_dir: "output/samples", inference_dir: "output/inference", custom_dir: "output/custom", effective_training_dir: "output/samples" } }),
  );
  await page.route((url) => url.pathname === "/api/training/history" && url.searchParams.get("limit") === "100", (route) =>
    route.fulfill({ json: { tasks: [task] } }),
  );
  await page.route((url) => url.pathname === "/api/preview/images", (route) => {
    imageRequests.push(new URL(route.request().url()));
    return route.fulfill({ json: { images: [], count: 0, total: 0, directory: "output/samples" } });
  });
  await page.route((url) => url.pathname === "/api/preview/weights", (route) => {
    weightRequests.push(new URL(route.request().url()));
    return route.fulfill({ json: { weights: [] } });
  });

  await page.goto("/next/preview");
  await expect(page.getByRole("heading", { name: "预览工作区" })).toBeVisible();
  await expect.poll(() => imageRequests.length).toBeGreaterThan(0);
  expect(imageRequests.at(-1)?.searchParams.get("source")).toBe("training");
  expect(imageRequests.at(-1)?.searchParams.has("task_id")).toBe(false);

  await page.getByLabel("训练范围").selectOption("task");
  await page.locator(".preview-controls label").filter({ hasText: "任务" }).last().locator("select").selectOption("p2-run-1");
  await expect.poll(() => imageRequests.some((url) => url.searchParams.get("task_id") === "p2-run-1")).toBe(true);
  expect(weightRequests.some((url) => url.searchParams.get("task_id") === "p2-run-1")).toBe(true);

  await page.getByLabel("训练范围").selectOption("group");
  await page.locator(".preview-controls label").filter({ hasText: "分组" }).last().locator("select").selectOption("p2-group");
  await expect.poll(() => imageRequests.some((url) => url.searchParams.get("mode") === "config_group")).toBe(true);
  await expect(page.getByRole("button", { name: /删除所选/ })).toBeDisabled();
  await expect(page.getByText(/只能浏览/)).toBeVisible();

  await page.getByRole("button", { name: "推理输出" }).click();
  await expect.poll(() => imageRequests.some((url) => url.searchParams.get("source") === "inference")).toBe(true);
  await expect(page.getByLabel("训练范围")).toBeDisabled();
  await page.setViewportSize({ width: 1440, height: 1000 });
  await inspectViewport(page, "preview-desktop");
  await page.setViewportSize({ width: 390, height: 844 });
  await inspectViewport(page, "preview-mobile");
  expect(errors).toEqual([]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("aggregate history deep link renders the requested tasks on desktop and mobile", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const mocks = await mockWorkspace(page);
  let timelineRequest: URL | undefined;
  await page.route((url) => url.pathname === "/api/training/history/config-group/timeline", (route) => {
    timelineRequest = new URL(route.request().url());
    return route.fulfill({ json: {
      ok: true,
      tasks: [{ ...task, label: "P2 first" }, { ...task, id: "p2-run-2", label: "P2 second" }],
      segments: [
        { task: { ...task, label: "P2 first" }, metric_count: 2, log_count: 1, start_display_step: 1, end_display_step: 2 },
        { task: { ...task, id: "p2-run-2", label: "P2 second" }, metric_count: 1, log_count: 1, start_display_step: 3, end_display_step: 3 },
      ],
      metrics: [
        { loss: 0.5, source_task_id: "p2-run-1" },
        { loss: 0.3, source_task_id: "p2-run-1" },
        { loss: 0.2, source_task_id: "p2-run-2", stage_break_before: true },
      ],
      logs: [{ line: "fixture log", source_task_id: "p2-run-1" }],
    } });
  });

  await page.goto("/next/history/aggregate?task=p2-run-1&task=p2-run-2&from=collection%3DP2");
  await expect(page.getByRole("heading", { name: "合并查看" })).toBeVisible();
  await expect(page.getByText("1. P2 first", { exact: true })).toBeVisible();
  await expect(page.getByText("2. P2 second", { exact: true })).toBeVisible();
  await expect(page.getByRole("img", { name: "合并 Loss 曲线，共 3 个点" })).toBeVisible();
  await expect(page.locator(".history-timeline-logs")).toContainText("[任务1] fixture log");
  expect(timelineRequest?.searchParams.getAll("task_id")).toEqual(["p2-run-1", "p2-run-2"]);

  await page.setViewportSize({ width: 1440, height: 1000 });
  await inspectViewport(page, "aggregate-desktop");
  await page.setViewportSize({ width: 390, height: 844 });
  await inspectViewport(page, "aggregate-mobile");
  expect(errors).toEqual([]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("runtime deletion dry-run displays blockers and cannot submit", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const mocks = await mockWorkspace(page);
  const commands: Record<string, unknown>[] = [];
  await page.route((url) => url.pathname === "/api/training/history" && url.searchParams.has("include_archived"), (route) =>
    route.fulfill({ json: { ok: true, total: 1, tasks: [task] } }),
  );
  await page.route((url) => url.pathname === "/api/training/history/batch", (route) => {
    const body = route.request().postDataJSON() as Record<string, unknown>;
    commands.push(body);
    if (body.dry_run) return route.fulfill({ json: {
      ok: true,
      dry_run: true,
      tasks: [{ id: "p2-run-1", name: task.name, state: "idle" }],
      runtime_dirs: [{ path: "/outside/p2-run-1", status: "ready" }],
      blocked: [{ id: "p2-run-1", path: "/outside/p2-run-1", reason: "运行目录不在 WebUI 输出根目录内" }],
      task_count: 1,
      runtime_dir_count: 1,
    } });
    return route.fulfill({ status: 500, json: { ok: false, error: "blocked submission must not happen" } });
  });

  await page.goto("/next/history");
  await page.getByRole("button", { name: "平铺任务", exact: true }).click();
  await page.getByRole("checkbox", { name: "选择 P2 fixture run" }).check();
  await page.getByRole("button", { name: "删除记录及运行目录", exact: true }).click();
  await expect(page.getByRole("heading", { name: "彻底删除历史与运行目录" })).toBeVisible();
  await expect(page.getByRole("region", { name: "阻止项" })).toContainText("运行目录不在 WebUI 输出根目录内");
  await expect(page.getByLabel("我已核对上述列表，并确认永久删除")).toBeDisabled();
  await expect(page.getByRole("button", { name: "确认彻底删除" })).toBeDisabled();
  expect(commands.length).toBeGreaterThan(0);
  expect(commands.every((command) => command.dry_run === true && command.confirmed !== true)).toBe(true);
  expect(commands.every((command) => JSON.stringify(command.task_ids) === '["p2-run-1"]')).toBe(true);
  await page.setViewportSize({ width: 390, height: 844 });
  await inspectViewport(page, "runtime-delete-blocked-mobile");
  expect(errors).toEqual([]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});
