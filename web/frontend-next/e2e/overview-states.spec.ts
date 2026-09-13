import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

const scenarios = [
  { id: "success", job: "training", state: "idle", message: "训练目标已完成", step: 100, loss: 0, config: "max_train_steps = 100" },
  { id: "error", job: "training", state: "error", message: "CUDA checkpoint save failed", step: 64, loss: 0.08, config: "max_train_steps = 100" },
  { id: "interrupted", job: "training", state: "interrupted", message: "用户中断，最近保存检查点为 40 步", step: 64, loss: 0.08, config: "max_train_steps = 100" },
  { id: "running", job: "training", state: "running", message: "", step: 64, loss: 0.08, config: "max_train_steps = 100" },
  { id: "preprocess", job: "preprocess", state: "idle", message: "缓存预处理已结束", step: 7, loss: 0.1, config: "" },
  { id: "preprocess-error", job: "preprocess", state: "error", message: "Text cache failed", step: 7, loss: 0.1, config: "" },
  { id: "epoch", job: "training", state: "idle", message: "", step: 64, loss: 0.08, config: "max_train_steps = 100\nmax_train_epochs = 3" },
  { id: "legacy", job: "", state: "", message: "", step: undefined, loss: undefined, config: "" },
];

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 720 }, { width: 768, height: 1024 }, { width: 390, height: 844 }]) {
  for (const theme of ["dark", "light"]) {
    test(`overview state and task-type matrix ${viewport.width} ${theme}`, async ({ page }, info) => {
      await page.setViewportSize(viewport);
      const mocks = await mockWorkspace(page);
      const resumeReads: string[] = [];
      await page.addInitScript((value) => localStorage.setItem("dragon-next-ui-v1-theme", value), theme);
      await page.route(/\/api\/training\/history\/matrix-[^/]+$/, (route) => {
        const id = new URL(route.request().url()).pathname.split("matrix-")[1];
        const scenario = scenarios.find((entry) => entry.id === id)!;
        return route.fulfill({ json: {
          task: { id: `matrix-${id}`, name: `${id} / 模型训练工作流与长名称检查`, job: scenario.job, state: scenario.state, message: scenario.message,
            last_step: scenario.step, final_loss: scenario.loss, metric_count: 73, started_at: 100, finished_at: id === "running" || id === "legacy" ? undefined : 3700,
            source_image_dir: "images/studio", dataset_cache_dir: "output/runs/studio/cache", source_task_id: id === "interrupted" ? "source-run" : undefined,
          }, metrics: [], logs: [], config_toml: scenario.config,
        } });
      });
      await page.route(/\/matrix-[^/]+\/artifacts$/, (route) => route.fulfill({ json: { artifacts: [] } }));
      await page.route(/\/matrix-[^/]+\/resume-options$/, (route) => { resumeReads.push(route.request().url()); return route.fulfill({ json: { checkpoints: [], default_checkpoint: "", message: "无可恢复检查点" } }); });
      for (const scenario of scenarios) {
        await page.goto(`/next/history/matrix-${scenario.id}`);
        await expect(page.getByRole("heading", { name: `${scenario.id} / 模型训练工作流与长名称检查`, exact: true })).toBeVisible();
        if (scenario.message) await expect(page.getByLabel("执行结果")).toContainText(scenario.message);
        if (scenario.job === "training") {
          await expect(page.getByLabel("训练摘要")).toContainText(String(scenario.step));
          if (scenario.id === "success") await expect(page.getByLabel("训练摘要")).toContainText("0.0000");
          if (scenario.id === "epoch") { await expect(page.getByLabel("训练摘要")).not.toContainText(" / 100"); await expect(page.getByText("3 轮（总步数未估算）")).toBeVisible(); }
          else await expect(page.getByLabel("训练摘要")).toContainText(" / 100");
          await expect(page.getByRole("button", { name: "检查点续训", exact: true })).toBeVisible();
        } else {
          await expect(page.getByLabel("训练摘要")).toHaveCount(0);
          await expect(page.getByRole("button", { name: "检查点续训", exact: true })).toHaveCount(0);
          if (scenario.job === "preprocess") await expect(page.getByRole("heading", { name: "预处理结果", exact: true })).toBeVisible();
        }
        if (scenario.id === "running") await expect(page.getByRole("link", { name: "查看当前监控" })).toHaveAttribute("href", "/next/monitor?from_task=matrix-running");
        if (scenario.id === "legacy") await expect(page.getByText("此记录未保存可识别的任务类型。")).toBeVisible();
        await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
        await page.screenshot({ path: info.outputPath(`${scenario.id}.png`), fullPage: true });
      }
      expect(resumeReads).toEqual([]);
      expect(mocks.writes).toEqual([]); expect(mocks.unhandled).toEqual([]);
    });
  }
}
