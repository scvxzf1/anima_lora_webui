import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

const routes = [
  ["training", "训练配置"],
  ["datasets", "数据集蓝图"],
  ["queue", "训练队列"],
  ["monitor", "当前监控"],
  ["history", "历史任务"],
  ["models", "模型配置"],
  ["captioning", "打标工作台"],
  ["settings", "全局设置"],
] as const;

for (const viewport of [
  { width: 1920, height: 1080 },
  { width: 1440, height: 900 },
  { width: 1280, height: 720 },
  { width: 768, height: 1024 },
  { width: 390, height: 844 },
]) {
  for (const theme of ["dark", "light"]) {
    test(`eight workspaces ${viewport.width} ${theme}`, async ({
      page,
    }, testInfo) => {
      await page.setViewportSize(viewport);
      const mocks = await mockWorkspace(page);
      const errors: string[] = [];
      page.on("pageerror", (error) => errors.push(error.message));
      await page.addInitScript(
        (value) => localStorage.setItem("dragon-next-ui-v1-theme", value),
        theme,
      );
      for (const [path, title] of routes) {
        await page.goto(`/next/${path}`);
        await expect(
          page.getByRole("heading", { name: title, exact: true }),
        ).toBeVisible();
        if (path === "training") {
          await page.getByRole("tab", { name: "训练计划", exact: true }).click();
          await expect(
            page.getByLabel("输出名称", { exact: true }),
          ).toHaveValue("studio-portrait");
        }
        if (path === "monitor")
          await expect(page.locator("canvas").first()).toBeVisible();
        await expect
          .poll(() =>
            page.evaluate(
              () => document.documentElement.scrollWidth <= window.innerWidth,
            ),
          )
          .toBe(true);
        await page.screenshot({
          path: testInfo.outputPath(`${path}.png`),
          fullPage: false,
        });
      }
      expect(errors).toEqual([]);
      expect(mocks.writes).toEqual([]);
      expect(mocks.unhandled).toEqual([]);
    });
  }
}

test("failed save blocks training and protects unsaved navigation", async ({
  page,
}) => {
  const mocks = await mockWorkspace(page);
  await page.goto("/next/training");
  await page.getByRole("tab", { name: "训练计划", exact: true }).click();
  await expect(page.getByLabel("输出名称", { exact: true })).toHaveValue(
    "studio-portrait",
  );
  await page.getByLabel("输出名称", { exact: true }).fill("changed");
  await page.getByRole("button", { name: "保存并启动", exact: true }).click();
  await expect(page.getByText("Fixture: readonly disk")).toBeVisible();
  expect(mocks.writes.map((entry) => entry.path)).toEqual(["/api/config/raw"]);
  page.once("dialog", (dialog) => dialog.dismiss());
  await page
    .getByRole("navigation", { name: "主导航" })
    .getByRole("link", { name: "训练队列" })
    .click();
  await expect(page).toHaveURL(/\/training$/);
  await expect(page.getByLabel("输出名称", { exact: true })).toHaveValue(
    "changed",
  );
});

test("caption image review and chart pixels are rendered", async ({
  page,
}, testInfo) => {
  await mockWorkspace(page);
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/next/captioning?job=caption-1");
  await expect(page.getByLabel("候选标注")).toBeVisible();
  await expect
    .poll(() =>
      page
        .locator(".caption-original")
        .evaluate((image) => (image as HTMLImageElement).naturalWidth),
    )
    .toBe(480);
  await page.screenshot({
    path: testInfo.outputPath("caption-review-desktop.png"),
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect
    .poll(() =>
      page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
    )
    .toBe(true);
  await page.screenshot({
    path: testInfo.outputPath("caption-review-mobile.png"),
  });
  await page.goto("/next/monitor");
  const canvas = page.locator("canvas").first();
  await expect(canvas).toBeVisible();
  await expect
    .poll(() =>
      canvas.evaluate((node) => {
        const canvas = node as HTMLCanvasElement;
        const pixels = canvas
          .getContext("2d")!
          .getImageData(0, 0, canvas.width, canvas.height).data;
        let nonblank = 0;
        for (let i = 3; i < pixels.length; i += 4)
          if (pixels[i] > 0) nonblank++;
        return nonblank;
      }),
    )
    .toBeGreaterThan(100);
});
