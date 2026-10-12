import { expect, test, type Locator } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

async function expectContainedThumbnails(scope: Locator) {
  const buttons = scope.locator(".sample-reference-thumbnail > button");
  await expect(buttons).toHaveCount(3);
  for (const button of await buttons.all()) {
    await expect.poll(() => button.locator("img").evaluate((image: HTMLImageElement) => image.complete && image.naturalWidth > 0)).toBe(true);
    const bounds = await button.evaluate((element) => {
      const frame = element.getBoundingClientRect();
      const image = element.querySelector("img")!.getBoundingClientRect();
      return { width: frame.width, height: frame.height, contained: image.left >= frame.left && image.top >= frame.top && image.right <= frame.right && image.bottom <= frame.bottom };
    });
    expect(bounds.width).toBeCloseTo(64, 1);
    expect(bounds.height).toBeCloseTo(64, 1);
    expect(bounds.contained).toBe(true);
  }
}

for (const width of [1172, 390, 320]) {
  test(`extreme reference aspect ratios stay contained at ${width}px`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height: 844 });
    const mocks = await mockWorkspace(page);
    const references = ["/tmp/tall.png", "/tmp/wide.png", `/tmp/${"long-path-".repeat(30)}square.png`];
    await page.route((url) => url.pathname === "/api/config/merged", (route) => route.fulfill({ json: {
      model_family: "qwen_image_2_1", output_name: "edit-preview", qwen_image_2_1_task: "edit", sample_prompts: "configs/sample-prompts/edit-preview.txt",
    } }));
    await page.route((url) => url.pathname === "/api/config/model-families", (route) => route.fulfill({ json: { items: [{
      name: "qwen_image_2_1", display_name: "Qwen Image 2.1", supported_preview_tasks: ["t2i", "edit"], max_preview_references: 4,
    }] } }));
    await page.route((url) => url.pathname === "/api/config/sample-prompts", (route) => route.fulfill({ json: {
      ok: true, exists: true, file: "configs/sample-prompts/edit-preview.txt",
      content: `${JSON.stringify({ prompt: "colorize", sample_task: "edit", reference_images: references, width: 512, height: 512 })}\n`, prompts: ["colorize"], revision: "fixture",
    } }));
    await page.route((url) => url.pathname === "/api/config/sample-references/preview", (route) => {
      const path = new URL(route.request().url()).searchParams.get("path")!;
      const [w, h] = path.includes("tall") ? [32, 4096] : path.includes("wide") ? [4096, 32] : [1024, 1024];
      return route.fulfill({ contentType: "image/svg+xml", body: `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}"><rect width="100%" height="100%" fill="teal"/></svg>` });
    });
    await page.goto("/next/training");
    await expect(page.getByText("已同步", { exact: true })).toBeVisible();
    await page.getByRole("button", { name: "采样样张" }).click();
    const editor = page.getByRole("dialog", { name: "采样样张" });
    await expectContainedThumbnails(editor);
    await editor.getByRole("button", { name: /样张 1.*编辑.*3 图/ }).click();
    await expectContainedThumbnails(editor);
    const layout = await editor.evaluate((element) => {
      const list = element.querySelector(".sample-reference-list")!.getBoundingClientRect();
      const actions = element.querySelector(".sample-reference-actions")!.getBoundingClientRect();
      return { overlap: list.bottom > actions.top, overflow: element.scrollWidth > element.clientWidth + 1 };
    });
    expect(layout).toEqual({ overlap: false, overflow: false });
    await page.screenshot({ path: testInfo.outputPath("reference-editor.png") });
    for (let i = 1; i <= 3; i++) {
      await editor.getByRole("button", { name: `展开参考图 ${i}`, exact: true }).click();
      const preview = page.getByRole("dialog", { name: `参考图 ${i}`, exact: true });
      await expect(preview.getByRole("img")).toBeVisible();
      const bounds = await preview.evaluate((element) => {
        const frame = element.getBoundingClientRect();
        const image = element.querySelector("img")!.getBoundingClientRect();
        return { overflow: element.scrollWidth > element.clientWidth + 1, inViewport: frame.left >= 0 && frame.right <= innerWidth && frame.top >= 0 && frame.bottom <= innerHeight, imageContained: image.left >= frame.left && image.right <= frame.right && image.top >= frame.top && image.bottom <= frame.bottom };
      });
      expect(bounds).toEqual({ overflow: false, inViewport: true, imageContained: true });
      await page.screenshot({ path: testInfo.outputPath(`reference-preview-${i}.png`) });
      await page.getByRole("button", { name: "关闭参考图预览" }).click();
    }
    await editor.getByRole("button", { name: "下移参考图 1" }).click();
    await expect(editor.getByRole("listitem").first()).toContainText("wide.png");
    await expectContainedThumbnails(editor);
    expect(mocks.writes).toEqual([]);
    expect(mocks.unhandled).toEqual([]);
  });
}
