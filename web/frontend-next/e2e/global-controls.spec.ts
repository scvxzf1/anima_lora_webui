import { expect, test, type Locator, type Page } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

async function switchGeometry(control: Locator) {
  return control.evaluate((node) => {
    const track = node.getBoundingClientRect();
    const thumb = node.querySelector(".setting-switch-thumb")!.getBoundingClientRect();
    return {
      width: track.width,
      height: track.height,
      thumbWidth: thumb.width,
      thumbHeight: thumb.height,
      left: thumb.left - track.left,
      top: thumb.top - track.top,
      right: track.right - thumb.right,
      bottom: track.bottom - thumb.bottom,
      transform: getComputedStyle(node).transform,
    };
  });
}

async function expectSwitchGeometry(control: Locator) {
  const geometry = await switchGeometry(control);
  expect(geometry.width).toBe(34);
  expect(geometry.height).toBe(20);
  expect(geometry.thumbWidth).toBe(16);
  expect(geometry.thumbHeight).toBe(16);
  expect(geometry.transform).toBe("none");
  for (const edge of [geometry.left, geometry.top, geometry.right, geometry.bottom]) {
    expect(edge).toBeGreaterThanOrEqual(1);
  }
}

async function navigate(page: Page, label: string) {
  const link = page.getByRole("navigation", { name: "主导航" })
    .getByRole("link", { name: label, exact: true });
  if (!await link.isVisible()) await page.getByRole("button", { name: "切换导航" }).click();
  await link.click();
  await expect(page.getByRole("heading", { name: label, exact: true })).toBeVisible();
}

async function expectUndecoratedInput(input: Locator) {
  await expect(input).toBeVisible();
  const style = await input.evaluate((node) => {
    const css = getComputedStyle(node);
    return {
      padding: [css.paddingTop, css.paddingRight, css.paddingBottom, css.paddingLeft],
      shadow: css.boxShadow,
    };
  });
  expect(style.padding).toEqual(["0px", "0px", "0px", "0px"]);
  expect(style.shadow).toBe("none");
}

for (const width of [1920, 390]) {
  for (const theme of ["dark", "light"]) {
    test(`global controls preserve component geometry ${width} ${theme}`, async ({ page }, testInfo) => {
      await page.setViewportSize({ width, height: width === 390 ? 844 : 1080 });
      const mocks = await mockWorkspace(page);
      const errors: string[] = [];
      page.on("pageerror", (error) => errors.push(error.message));
      await page.addInitScript((value) => localStorage.setItem("dragon-next-ui-v1-theme", value), theme);
      await page.goto("/next/training");
      await expect(page.getByRole("heading", { name: "训练配置", exact: true })).toBeVisible({ timeout: 15000 });
      if (width === 390) await page.getByRole("button", { name: "展开配置库", exact: true }).click();
      const management = page.getByRole("switch", { name: "详细管理", exact: true });
      await expect(management).not.toBeChecked();
      await expectSwitchGeometry(management);
      await management.hover();
      await expectSwitchGeometry(management);
      await management.click();
      await expect(management).toBeChecked();
      // Sample the real animation, including its end, to catch escaped thumbs.
      const frames = await management.evaluate(async (node) => {
        const samples = [];
        const start = performance.now();
        do {
          await new Promise(requestAnimationFrame);
          const track = node.getBoundingClientRect();
          const thumb = node.querySelector(".setting-switch-thumb")!.getBoundingClientRect();
          samples.push(thumb.left >= track.left + 1 && thumb.right <= track.right - 1
            && thumb.top >= track.top + 1 && thumb.bottom <= track.bottom - 1);
        } while (performance.now() - start < 200);
        return samples;
      });
      expect(frames.length).toBeGreaterThan(1);
      expect(frames.every(Boolean)).toBe(true);
      await expectSwitchGeometry(management);
      await management.hover();
      await expectSwitchGeometry(management);
      await management.click();
      await expect(management).not.toBeChecked();
      await expect.poll(async () => (await switchGeometry(management)).left).toBe(2);
      const before = await switchGeometry(management);
      await expect(page.getByText("有未保存修改", { exact: true })).toHaveCount(0);
      if (width === 1920) {
        const compact = page.locator(".training-library-layout-switch > .icon-button").first();
        await expect(compact).toBeVisible();
        const rect = await compact.boundingBox();
        expect(rect?.width).toBe(25);
        expect(rect?.height).toBe(26);
      }
      const help = page.locator(".training-field-help-trigger").first();
      await help.scrollIntoViewIfNeeded();
      const helpRect = await help.boundingBox();
      expect(helpRect?.width).toBe(28);
      expect(helpRect?.height).toBe(28);
      await page.screenshot({ path: testInfo.outputPath("training-controls.png") });

      // A marker proves these visits keep the same document and CSS cascade.
      await page.evaluate(() => { document.documentElement.dataset.controlsRegression = "same-document"; });
      await navigate(page, "全局设置");
      await expect(page.getByRole("button", { name: "恢复默认", exact: true })).toBeVisible();
      for (const name of ["保存设置", "还原修改", "恢复默认"]) {
        const button = page.getByRole("button", { name, exact: true });
        const rect = await button.boundingBox();
        expect(rect?.height).toBeGreaterThanOrEqual(width === 390 ? 44 : 36);
      }
      await expectSwitchGeometry(page.getByRole("switch").first());
      await navigate(page, "数据集蓝图");
      await expectSwitchGeometry(page.getByRole("switch", { name: "详细管理", exact: true }));
      await navigate(page, "当前监控");
      await expect(page.locator("canvas")).toHaveCount(1);
      await expectUndecoratedInput(page.locator('input[type="checkbox"]').first());
      await expectUndecoratedInput(page.locator('input[type="range"]').first());
      await page.screenshot({ path: testInfo.outputPath("monitor-controls.png") });
      await navigate(page, "训练配置");
      await expect(management).not.toBeChecked();
      await expectSwitchGeometry(management);
      expect(await switchGeometry(management)).toEqual(before);
      await expect(page.getByText("有未保存修改", { exact: true })).toHaveCount(0);
      expect(await page.evaluate(() => document.documentElement.dataset.controlsRegression)).toBe("same-document");
      expect(errors).toEqual([]);
      expect(mocks.writes).toEqual([]);
      expect(mocks.unhandled).toEqual([]);
    });
  }
}
