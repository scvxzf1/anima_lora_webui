import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

test("sidebar resizer survives switching to top navigation and back", async ({ page }) => {
  await page.setViewportSize({ width: 1600, height: 900 });
  await mockWorkspace(page);
  await page.goto("/next/training");

  const layout = page.locator(".next-layout");
  const navigation = page.locator(".app-navigation");
  const resizer = page.getByRole("separator", { name: "调整侧边栏宽度" });
  await expect(resizer).toBeVisible();
  await page.getByRole("button", { name: "切换为顶部栏" }).click();
  await expect(layout).toHaveAttribute("data-navigation-layout", "top");
  await expect(resizer).toBeHidden();
  await page.getByRole("button", { name: "切换为侧边栏" }).click();
  await expect(layout).toHaveAttribute("data-navigation-layout", "sidebar");
  await expect(resizer).toBeVisible();

  const navBox = (await navigation.boundingBox())!;
  const handleBox = (await resizer.boundingBox())!;
  expect(Math.abs(handleBox.x + handleBox.width - navBox.x - navBox.width)).toBeLessThan(2);
  expect(handleBox.height).toBeGreaterThan(800);

  const handleX = handleBox.x + handleBox.width / 2;
  const handleY = handleBox.y + handleBox.height / 2;
  await page.mouse.move(handleX, handleY);
  await page.mouse.down();
  await page.mouse.move(handleX + 80, handleY);
  await page.mouse.up();
  const draggedWidth = (await navigation.boundingBox())!.width;
  expect(draggedWidth).toBeGreaterThan(navBox.width + 40);
  expect(Number(await page.evaluate(() => localStorage.getItem("dragon-next-ui-v1-sidebar-width")))).toBeGreaterThan(navBox.width + 40);

  await resizer.focus();
  await page.keyboard.press("Home");
  await page.keyboard.press("ArrowRight");
  await expect(resizer).toHaveAttribute("aria-valuenow", "11");
  expect(Number(await page.evaluate(() => localStorage.getItem("dragon-next-ui-v1-sidebar-width")))).toBe(170);
  await resizer.dblclick();
  expect(await page.evaluate(() => localStorage.getItem("dragon-next-ui-v1-sidebar-width"))).toBeNull();
});
