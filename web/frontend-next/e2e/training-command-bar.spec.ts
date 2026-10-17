import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

test("training command menus preserve core actions and dismiss accessibly", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await mockWorkspace(page);
  await page.goto("/next/training");

  const commandBar = page.getByRole("group", { name: "训练配置操作" });
  await expect(commandBar.getByRole("button", { name: "加入队列" })).toBeVisible();
  await expect(commandBar.getByRole("button", { name: "立即启动" })).toBeVisible();
  await expect(commandBar.getByRole("button", { name: "运行预检测" })).toBeVisible();
  await expect(commandBar.getByRole("button", { name: "保存配置" })).toBeVisible();

  const saveMenu = commandBar.getByRole("button", { name: "保存菜单" });
  await saveMenu.click();
  await expect(page.getByRole("menu", { name: "保存菜单" })).toBeVisible();
  await expect(page.getByRole("menuitem", { name: "导出 TOML" })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("menu", { name: "保存菜单" })).toHaveCount(0);
  await expect(saveMenu).toBeFocused();

  const moreMenu = commandBar.getByRole("button", { name: "更多" });
  await moreMenu.click();
  await expect(page.getByRole("menu", { name: "更多" })).toBeVisible();
  await expect(page.getByRole("menuitem", { name: "采样样张" })).toBeVisible();
  await page.getByRole("heading", { name: "训练配置" }).click();
  await expect(page.getByRole("menu", { name: "更多" })).toHaveCount(0);
});

for (const width of [800, 520, 390]) {
  test(`training command bar fits ${width}px and keeps touch targets`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await mockWorkspace(page);
    await page.goto("/next/training");
    await expect(page.getByRole("heading", { name: "训练配置" })).toBeVisible();

    const layout = await page.evaluate(() => {
      const button = document.querySelector<HTMLElement>(
        ".training-command-bar .training-display-menu > button",
      );
      const bar = document.querySelector<HTMLElement>(".training-command-bar");
      return {
        viewportWidth: window.innerWidth,
        documentWidth: document.documentElement.scrollWidth,
        displayButton: button?.getBoundingClientRect().toJSON(),
        commandBarHeight: bar?.getBoundingClientRect().height,
      };
    });
    expect(layout.documentWidth).toBeLessThanOrEqual(layout.viewportWidth);
    expect(layout.displayButton?.width).toBeGreaterThanOrEqual(43.5);
    expect(layout.displayButton?.height).toBeGreaterThanOrEqual(43.5);
    expect(layout.commandBarHeight).toBeLessThanOrEqual(62);
    await expect(page.getByRole("button", { name: "加入队列" })).toBeVisible();
    await expect(page.getByRole("button", { name: "立即启动" })).toBeVisible();
  });
}
