import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

for (const viewport of [{ width: 1172, height: 900 }, { width: 390, height: 844 }]) {
  test(`Qwen edit sample keeps ordered references at ${viewport.width}px`, async ({ page }) => {
    await page.setViewportSize(viewport);
    const mocks = await mockWorkspace(page);
    await page.route((url) => url.pathname === "/api/config/merged", (route) => route.fulfill({ json: {
      model_family: "qwen_image_2_1", output_name: "edit-preview", qwen_image_2_1_task: "edit",
      sample_prompts: "configs/sample-prompts/edit-preview.txt",
    } }));
    await page.route((url) => url.pathname === "/api/config/model-families", (route) => route.fulfill({ json: { items: [{
      name: "qwen_image_2_1", display_name: "Qwen Image 2.1", supported_preview_tasks: ["t2i", "edit"],
      max_preview_references: 4,
    }] } }));
    await page.route((url) => url.pathname === "/api/config/sample-prompts", (route) => route.fulfill({ json: {
      ok: true, exists: true, file: "configs/sample-prompts/edit-preview.txt",
      content: `${JSON.stringify({ prompt: "colorize", sample_task: "edit", reference_images: ["/tmp/before-a.png", "/tmp/before-b.png"], width: 512, height: 512 })}\n`,
      prompts: ["colorize"], revision: "fixture",
    } }));

    await page.goto("/next/training");
    await expect(page.getByText("已同步", { exact: true })).toBeVisible();
    await page.getByRole("button", { name: "采样样张" }).click();
    await page.getByRole("button", { name: /样张 1.*编辑.*2 图/ }).click();
    const dialog = page.getByRole("dialog", { name: "采样样张" });
    await expect(dialog.getByText("参考图 (2/4)")).toBeVisible();
    await expect(dialog.getByText("缩放后参考图总像素上限 4 Mi。")).toBeVisible();
    await expect(dialog.getByRole("listitem").nth(0)).toContainText("before-a.png");
    await dialog.getByRole("button", { name: "下移参考图 1" }).click();
    await expect(dialog.getByRole("listitem").nth(0)).toContainText("before-b.png");
    await expect(dialog.getByRole("listitem").nth(1)).toContainText("before-a.png");
    const overflow = await dialog.evaluate((element) => element.scrollWidth > element.clientWidth + 1);
    expect(overflow).toBe(false);
    await page.screenshot({ path: `test-results/multi-reference-${viewport.width}.png` });
    expect(mocks.writes).toEqual([]);
    expect(mocks.unhandled).toEqual([]);
  });
}
