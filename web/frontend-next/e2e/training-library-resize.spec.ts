import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

test("training library keeps bounded side and top sizes independently", async ({ page }, info) => {
  await page.setViewportSize({ width: 1600, height: 900 });
  const mocks = await mockWorkspace(page);
  await page.goto("/next/training");
  await expect(page.getByText("已同步", { exact: true })).toBeVisible();

  const layout = page.locator(".training-workspace-layout");
  const library = page.getByRole("complementary", { name: "训练配置库" });
  const side = page.getByRole("separator", { name: "调整配置库宽度" });
  const top = page.getByRole("separator", { name: "调整配置库高度" });
  const editGroups = page.locator(".training-edit-groups");
  for (const [width, height, minEditorHeight] of [
    [1440, 900, 420],
    [1280, 720, 300],
    [768, 1024, 0],
  ]) {
    await page.setViewportSize({ width, height });
    await page.screenshot({ path: info.outputPath(`training-layout-${width}x${height}.png`) });
    const box = await editGroups.boundingBox();
    expect(box, `Editor is visible at ${width}x${height}`).not.toBeNull();
    expect(box!.height, `Editor height at ${width}x${height}`).toBeGreaterThanOrEqual(minEditorHeight);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1),
      `No horizontal overflow at ${width}x${height}`).toBe(true);
  }
  await page.setViewportSize({ width: 1600, height: 900 });
  await expect(side).toBeVisible();
  await expect(top).toBeHidden();

  const sideHandle = (await side.boundingBox())!;
  const sideLayout = (await layout.boundingBox())!;
  await page.mouse.move(sideHandle.x + sideHandle.width / 2, sideHandle.y + 50);
  await page.mouse.down();
  await page.mouse.move(sideLayout.x + sideLayout.width * 0.95, sideHandle.y + 50, { steps: 5 });
  await page.mouse.up();
  await expect(side).toHaveAttribute("aria-valuenow", "42");
  expect((await library.boundingBox())!.width / (await layout.boundingBox())!.width).toBeCloseTo(0.42, 2);

  await side.focus();
  await page.keyboard.press("Home");
  await page.keyboard.press("ArrowRight");
  await expect(side).toHaveAttribute("aria-valuenow", "11");
  expect(await page.evaluate(() => localStorage.getItem("dragon-next.training-library-side-percent"))).toBe("11");

  await page.setViewportSize({ width: 1200, height: 900 });
  await expect(side).toBeHidden();
  await expect(top).toBeVisible();
  const topHandle = (await top.boundingBox())!;
  const topLayout = (await layout.boundingBox())!;
  await page.mouse.move(topHandle.x + 50, topHandle.y + topHandle.height / 2);
  await page.mouse.down();
  await page.mouse.move(topHandle.x + 50, topLayout.y + topLayout.height * 0.95, { steps: 5 });
  await page.mouse.up();
  await expect(top).toHaveAttribute("aria-valuenow", "80");
  await top.focus();
  await page.keyboard.press("Home");
  await page.keyboard.press("Shift+ArrowDown");
  await expect(top).toHaveAttribute("aria-valuenow", "15");
  expect(await page.evaluate(() => localStorage.getItem("dragon-next.training-library-top-percent"))).toBe("15");
  expect(await page.evaluate(() => localStorage.getItem("dragon-next.training-library-side-percent"))).toBe("11");

  await page.reload();
  await expect(top).toBeVisible();
  expect((await library.boundingBox())!.height / (await layout.boundingBox())!.height).toBeCloseTo(0.15, 2);
  await page.setViewportSize({ width: 1600, height: 900 });
  await expect(side).toBeVisible();
  expect(await layout.evaluate((node) => getComputedStyle(node).getPropertyValue("--training-library-side-size").trim())).toBe("11%");
  expect((await library.boundingBox())!.width).toBeGreaterThanOrEqual(260);
  expect((await library.boundingBox())!.width / (await layout.boundingBox())!.width).toBeLessThanOrEqual(0.42);
  await side.dblclick();
  expect(await page.evaluate(() => localStorage.getItem("dragon-next.training-library-side-percent"))).toBeNull();
  expect(await page.evaluate(() => localStorage.getItem("dragon-next.training-library-top-percent"))).toBe("15");

  await page.setViewportSize({ width: 390, height: 844 });
  await expect(side).toBeHidden();
  await expect(top).toBeHidden();
  expect((await library.boundingBox())!.height).toBeLessThanOrEqual(181);
  expect((await page.locator(".training-editor-column").boundingBox())!.y)
    .toBeGreaterThan((await library.boundingBox())!.y + (await library.boundingBox())!.height);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});
