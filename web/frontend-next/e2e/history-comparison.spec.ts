import { expect, test, type Page } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

async function fixture(page: Page) {
  const mocks = await mockWorkspace(page);
  const tasks = [0, 1, 2, 3].map((index) => ({ id: `compare-${index}`, name: `Studio ${index} / configuration with a long readable name`, job: "training", state: "idle" }));
  let failed = false;
  await page.route((url) => url.pathname === "/api/training/history", (route) => route.fulfill({ json: { tasks, total: 4 } }));
  await page.route(/\/api\/training\/history\/compare-\d$/, (route) => {
    const index = Number(new URL(route.request().url()).pathname.at(-1));
    return route.fulfill({ status: failed && index === 3 ? 503 : 200, json: failed && index === 3 ? { error: "snapshot offline" } : {
      task: tasks[index],
      metrics: Array.from({ length: 20 }, (_, point) => ({ step: point * 10 + index * 40, loss: 0.3 / (point + 1) + index * 0.02 })),
      limits: { metrics_total: 100 },
      config_toml: `network_dim = ${16 + index * 8}\nmodel_family = "anima"\nlearning_rate = 0.0000002\n` + Array.from({ length: 35 }, (_, field) => `custom_${field} = ${index}`).join("\n"),
    } });
  });
  const open = async () => {
    await page.goto("/next/history?layout=list");
    for (const task of tasks) await page.getByRole("checkbox", { name: `选择 ${task.name}`, exact: true }).check();
    await page.getByRole("button", { name: "对比记录 (2-4)", exact: true }).click();
  };
  return { ...mocks, open, fail: () => { failed = true; } };
}

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 720 }, { width: 768, height: 1024 }, { width: 390, height: 844 }]) {
  for (const theme of ["dark", "light"]) {
    test(`common comparison window and snapshot differences ${viewport.width} ${theme}`, async ({ page }, info) => {
      await page.setViewportSize(viewport);
      const mocks = await fixture(page);
      await page.addInitScript((value) => localStorage.setItem("dragon-next-ui-v1-theme", value), theme);
      await mocks.open();
      const chart = page.getByLabel("任务 Loss 对比", { exact: true });
      await expect(chart).toContainText("步数 0 - 310");
      await expect(chart.locator("canvas")).toHaveCount(1);
      await expect.poll(() => chart.locator("canvas").evaluate((canvas: HTMLCanvasElement) => {
        const data = canvas.getContext("2d")!.getImageData(0, 0, canvas.width, canvas.height).data;
        let painted = 0;
        for (let i = 3; i < data.length; i += 4) if (data[i] > 0) painted++;
        return painted;
      })).toBeGreaterThan(1000);
      await page.getByLabel("对比步数窗口").selectOption("overlap");
      await expect(chart).toContainText("步数 120 - 190");
      await expect(chart.locator(".comparison-legend li")).toHaveCount(4);
      await expect(chart.locator(".comparison-legend li").first()).toContainText("当前范围 8 个有效点 / 已读取 20 点 / 共 100 点");
      const parameters = page.getByLabel("历史快照参数对比");
      await expect(parameters.locator("tbody tr")).toHaveCount(30);
      await page.getByRole("button", { name: "下一页参数" }).click();
      await expect(parameters).toContainText("network_dim");
      await expect(parameters.locator("tbody")).not.toContainText("model_family");
      await page.getByRole("checkbox", { name: "仅显示差异" }).uncheck();
      await page.getByRole("button", { name: "下一页参数" }).click();
      await expect(parameters.locator("tbody")).toContainText("model_family");
      const dialog = page.getByRole("dialog", { name: "历史任务对比" });
      await expect.poll(() => dialog.evaluate((node) => node.scrollWidth <= node.clientWidth + 1)).toBe(true);
      await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await chart.scrollIntoViewIfNeeded();
      await page.screenshot({ path: info.outputPath("comparison.png") });
      await page.keyboard.press("Escape");
      await expect(dialog).toHaveCount(0);
      await expect(page.getByRole("button", { name: "对比记录 (2-4)", exact: true })).toBeFocused();
      expect(mocks.writes).toEqual([]); expect(mocks.unhandled).toEqual([]);
    });
  }
}

test("comparison isolates one failed snapshot and keeps other records readable", async ({ page }) => {
  const mocks = await fixture(page); mocks.fail();
  await mocks.open();
  await expect(page.getByRole("alert")).toContainText("snapshot offline");
  await expect(page.getByLabel("任务 Loss 对比", { exact: true })).toContainText("3 条有效曲线");
  await expect(page.getByLabel("历史快照参数对比")).toContainText("未记录");
  expect(mocks.writes).toEqual([]); expect(mocks.unhandled).toEqual([]);
});
