import { expect, test, type Page } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

async function deviceFixture(page: Page) {
  const mocks = await mockWorkspace(page);
  let gpus = [{ index: 0, name: "GPU Alpha", memory_total_gb: 24 }, { index: 1, name: "GPU Beta", memory_total_gb: 32 }];
  const commands: { path: string; ids: string[] }[] = [];
  await page.route("**/api/training/gpus", (route) => route.fulfill({ json: { gpus } }));
  await page.route((url) => ["/api/training/preflight", "/api/training/start", "/api/training/queue"].includes(url.pathname), (route) => {
    if (route.request().method() === "GET") return route.fallback();
    const path = new URL(route.request().url()).pathname;
    commands.push({ path, ids: route.request().postDataJSON().gpu_whitelist });
    return route.fulfill({ json: path.endsWith("preflight")
      ? { ok: true, summary: { errors: 0, warnings: 0, checks: 1 }, checks: [{ level: "ok", key: "gpu", message: "设备检查通过" }] }
      : { ok: true, message: "模拟请求已完成" } });
  });
  return { ...mocks, commands, removeSecond: () => { gpus = gpus.slice(0, 1); } };
}

for (const width of [1440, 390]) {
  test(`shared GPU selection and launch contracts ${width}`, async ({ page }, info) => {
    const mocks = await deviceFixture(page);
    await page.setViewportSize({ width, height: 900 });
    await page.goto("/next/training");
    const quick = page.getByRole("region", { name: "训练设备快捷选择" });
    await expect(quick.getByRole("radio", { name: /GPU 0/ })).toBeChecked();
    await quick.getByRole("radio", { name: /GPU 1/ }).check();
    await page.reload();
    await expect(quick.getByRole("radio", { name: /GPU 1/ })).toBeChecked();
    await page.getByRole("button", { name: "运行预检测", exact: true }).click();
    await expect(page.getByRole("dialog")).toContainText("设备检查通过");
    expect(mocks.commands.at(-1)?.ids).toEqual(["1"]);
    await page.keyboard.press("Escape");
    await page.getByRole("button", { name: "立即启动", exact: true }).click();
    const dialog = page.getByRole("dialog");
    await expect(dialog).toContainText("单卡 · GPU 1 · GPU Beta");
    expect(await dialog.getByRole("radio").count()).toBe(0);
    await dialog.getByRole("checkbox").check();
    await dialog.getByRole("button", { name: "确认启动", exact: true }).click();
    await expect(dialog).toContainText("模拟请求已完成");
    expect(mocks.commands.at(-1)).toEqual({ path: "/api/training/start", ids: ["1"] });
    await page.keyboard.press("Escape");
    await quick.getByRole("button", { name: "多卡 · 数据并行" }).click();
    await expect(page.getByRole("button", { name: "立即启动", exact: true })).toBeDisabled();
    await quick.getByRole("checkbox", { name: /GPU 0/ }).check();
    await page.getByRole("tab", { name: "设备与性能", exact: true }).click();
    const details = page.getByRole("region", { name: "设备与并行设置" });
    await expect(details).toContainText("数据并行 · GPU 1 · GPU Beta / GPU 0 · GPU Alpha");
    await expect(details).toContainText("暂不可用");
    await page.screenshot({ path: info.outputPath("devices-performance.png") });
    if (width < 800) {
      await page.getByRole("button", { name: /精度与计算后端/ }).scrollIntoViewIfNeeded();
      await page.screenshot({ path: info.outputPath("devices-performance-scrolled.png") });
    }
    await page.getByRole("button", { name: "加入队列", exact: true }).click();
    await expect(dialog).toContainText("数据并行");
    await dialog.getByRole("checkbox").check();
    await dialog.getByRole("button", { name: "确认入队", exact: true }).click();
    await expect(dialog).toContainText("模拟请求已完成");
    expect(mocks.commands.slice(-2)).toEqual([
      { path: "/api/training/preflight", ids: ["1", "0"] },
      { path: "/api/training/queue", ids: ["1", "0"] },
    ]);
    await page.keyboard.press("Escape");
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    expect(mocks.writes).toEqual([]);
    expect(mocks.unhandled).toEqual([]);
  });
}

test("missing saved GPU blocks launch instead of falling back", async ({ page }) => {
  const mocks = await deviceFixture(page);
  await page.goto("/next/training");
  const quick = page.getByRole("region", { name: "训练设备快捷选择" });
  await quick.getByRole("radio", { name: /GPU 1/ }).check();
  mocks.removeSecond();
  await quick.getByRole("button", { name: "刷新 GPU 列表" }).click();
  await expect(quick).toContainText("已选设备不可用");
  await expect(page.getByRole("button", { name: "立即启动", exact: true })).toBeDisabled();
  await quick.getByRole("radio", { name: /GPU 0/ }).check();
  await expect(page.getByRole("button", { name: "立即启动", exact: true })).toBeEnabled();
  expect(mocks.commands).toEqual([]);
});

test("GPU inventory errors and empty inventory block execution but allow editing", async ({ page }) => {
  const mocks = await deviceFixture(page);
  await page.route("**/api/training/gpus", (route) => route.fulfill({ status: 503, json: { error: "GPU inventory offline" } }));
  await page.goto("/next/training");
  const quick = page.getByRole("region", { name: "训练设备快捷选择" });
  await expect(quick).toContainText("无法读取 GPU");
  await expect(page.getByRole("button", { name: "运行预检测", exact: true })).toBeDisabled();
  await expect(page.getByRole("button", { name: "立即启动", exact: true })).toBeDisabled();
  await page.getByRole("tab", { name: "训练计划", exact: true }).click();
  await expect(page.getByLabel("输出名称", { exact: true })).toBeEnabled();
  await page.route("**/api/training/gpus", (route) => route.fulfill({ json: { gpus: [] } }));
  await quick.getByRole("button", { name: "刷新 GPU 列表" }).click();
  await expect(quick).toContainText("未检测到可用 GPU");
  expect(mocks.commands).toEqual([]);
});

test("imported unavailable pipeline cannot launch and can explicitly return to single GPU", async ({ page }) => {
  const mocks = await deviceFixture(page);
  await page.route("**/api/config/merged?**", (route) => route.fulfill({ json: { model_family: "krea2_raw", pipeline_parallel: true } }));
  await page.goto("/next/training");
  const quick = page.getByRole("region", { name: "训练设备快捷选择" });
  await expect(quick).toContainText("当前配置启用了");
  await expect(page.getByRole("button", { name: "立即启动", exact: true })).toBeDisabled();
  await expect(page.getByRole("button", { name: "加入队列", exact: true })).toBeDisabled();
  await quick.getByRole("button", { name: "单卡", exact: true }).click();
  await expect(quick).not.toContainText("当前配置启用了");
  await expect(page.getByRole("button", { name: "保存并启动", exact: true })).toBeEnabled();
  expect(mocks.commands).toEqual([]);
});
