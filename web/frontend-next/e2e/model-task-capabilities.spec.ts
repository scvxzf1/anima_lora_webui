import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

const capabilities = { items: [
  { name: "anima", display_name: "Anima", aliases: ["anima"], supported_tasks: ["t2i"], supported_attention_modes: ["torch", "flash"] },
  { name: "krea2_raw", display_name: "Krea-2", aliases: ["krea2"], supported_tasks: ["t2i"], supported_attention_modes: ["torch", "flash"] },
  { name: "qwen_image_2_1", display_name: "Qwen Image 2.1", aliases: ["qwen21"], supported_tasks: ["edit", "t2i"], plain_lora_only: true, supported_attention_modes: ["torch", "flash"] },
] };

for (const width of [1440, 390]) {
  test(`model task tags and preflight summary ${width}`, async ({ page }, info) => {
    await page.setViewportSize({ width, height: 950 });
    const mocks = await mockWorkspace(page);
    await page.route("**/api/config/model-families", (route) => route.fulfill({ json: capabilities }));
    await page.goto("/next/models");
    const tags = page.getByLabel("模型训练能力");
    await expect(tags).toContainText("普通文生图");
    await expect(tags).not.toContainText("编辑训练");
    await page.getByRole("combobox", { name: "模型族", exact: true }).selectOption("qwen_image_2_1");
    await expect(tags).toContainText("编辑训练");
    await page.getByLabel("筛选训练能力").selectOption("edit");
    await expect(page.locator(".object-row")).toHaveCount(1);
    await tags.scrollIntoViewIfNeeded();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);
    await page.screenshot({ path: info.outputPath("model-tags.png"), fullPage: true });
    await page.getByRole("button", { name: "还原修改", exact: true }).click();
    await page.route("**/api/training/preflight", (route) => route.fulfill({ json: {
      ok: false, summary: { errors: 1, warnings: 0, checks: 1 },
      training_task: { model_family: "krea2_raw", model_name: "Krea-2", supported_tasks: ["t2i"], configured_task: "t2i", dataset_task: "edit" },
      checks: [{ level: "error", key: "model_family", message: "当前数据集需要编辑训练；当前模型 Krea-2 不支持编辑数据集训练" }],
    } }));
    await page.goto("/next/training");
    await page.getByRole("button", { name: "运行预检测", exact: true }).click();
    const summary = page.getByLabel("模型与数据集能力检查");
    await expect(summary).toContainText("Krea-2");
    await expect(summary).toContainText("编辑训练");
    await expect(page.getByText("当前数据集需要编辑训练；当前模型 Krea-2 不支持编辑数据集训练")).toBeVisible();
    await page.screenshot({ path: info.outputPath("preflight-task.png"), fullPage: true });
    expect(mocks.writes).toEqual([]);
    expect(mocks.unhandled).toEqual([]);
  });
}
