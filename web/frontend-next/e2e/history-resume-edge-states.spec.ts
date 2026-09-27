import { expect, test, type Page } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

async function installHistoryFixture(page: Page, taskId: string) {
  await page.route(
    (url) => url.pathname === `/api/training/history/${taskId}`,
    (route) => route.fulfill({
      json: {
        ok: true,
        task: {
          id: taskId,
          name: "Resume edge fixture",
          job: "training",
          state: "error",
          last_step: 50,
          target_total_steps: 100,
        },
        metrics: [],
        logs: [],
        system: [],
        config_toml: "max_train_steps = 100",
      },
    }),
  );
  await page.route(
    (url) => url.pathname === `/api/training/history/${taskId}/artifacts`,
    (route) => route.fulfill({ json: { ok: true, task_id: taskId, artifacts: [] } }),
  );
}

const checkpoint = {
  path: "output/runs/resume-edge/checkpoints/step-50",
  name: "step-50",
  step: 50,
  target_total_steps: 100,
  remaining_steps: 50,
  resume_available: true,
  state_integrity: { ok: true },
};

test("empty resume options keep confirmation disabled and never submit", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  const taskId = "resume-empty";
  const submissions: unknown[] = [];
  await installHistoryFixture(page, taskId);
  await page.route(
    (url) => url.pathname === `/api/training/history/${taskId}/resume-options`,
    (route) => route.fulfill({ json: { checkpoints: [], default_checkpoint: "", message: "无可恢复检查点" } }),
  );
  await page.route("/api/training/resume", (route) => {
    submissions.push(route.request().postDataJSON());
    return route.fulfill({ json: { ok: true, message: "unexpected" } });
  });

  await page.goto(`/next/history/${taskId}`);
  await page.getByRole("button", { name: "检查点续训", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "从历史检查点续训" });
  await expect(dialog.getByRole("status")).toContainText("无可恢复检查点");
  await expect(dialog.getByRole("checkbox")).toBeDisabled();
  await expect(dialog.getByRole("button", { name: "确认续训", exact: true })).toBeDisabled();
  expect(submissions).toEqual([]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("resume response loss shows reconciliation links and blocks blind retry", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  const taskId = "resume-unknown";
  const submissions: unknown[] = [];
  await installHistoryFixture(page, taskId);
  await page.route(
    (url) => url.pathname === `/api/training/history/${taskId}/resume-options`,
    (route) => route.fulfill({
      json: { checkpoints: [checkpoint], default_checkpoint: checkpoint.path, message: "" },
    }),
  );
  await page.route("/api/training/resume", async (route) => {
    submissions.push(route.request().postDataJSON());
    return route.abort("failed");
  });

  await page.goto(`/next/history/${taskId}`);
  await page.getByRole("button", { name: "检查点续训", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "从历史检查点续训" });
  const consent = dialog.getByRole("checkbox");
  await expect(consent).toBeEnabled();
  await consent.check();
  await dialog.getByRole("button", { name: "确认续训", exact: true }).click();

  await expect(dialog.getByRole("alert")).toContainText("连接中断，操作结果尚未确认");
  await expect(dialog.getByRole("link", { name: "核对历史任务", exact: true })).toHaveAttribute("href", "/next/history");
  await expect(dialog.getByRole("link", { name: "核对队列", exact: true })).toHaveAttribute("href", "/next/queue");
  await expect(dialog.getByRole("button", { name: "确认续训", exact: true })).toBeDisabled();
  expect(submissions).toEqual([{ task_id: taskId, checkpoint: checkpoint.path }]);
  await page.waitForTimeout(600);
  expect(submissions).toHaveLength(1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});
