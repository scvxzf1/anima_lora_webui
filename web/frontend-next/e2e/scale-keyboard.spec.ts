import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

for (const theme of ["dark", "light"]) {
  test(`scaled workspaces and keyboard overview navigation ${theme}`, async ({ page }, info) => {
    await page.setViewportSize({ width: 1280, height: 720 });
    const mocks = await mockWorkspace(page);
    await page.addInitScript((value) => localStorage.setItem("dragon-next-ui-v1-theme", value), theme);
    await page.route((url) => url.pathname === "/api/settings/global", (route) => route.fulfill({ json: { ui_scale: 150, ui_scale_config: 200, ui_scale_history_overview: 200, dragon_motion_enabled: false } }));
    for (const route of ["training", "datasets", "queue", "monitor", "history", "models", "captioning?job=caption-1", "settings", "history/fixture-run"]) {
      await page.goto(`/next/${route}`);
      await expect(page.locator("main h1")).toBeVisible();
      await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth ? "" : JSON.stringify({
        width: innerWidth, scroll: document.documentElement.scrollWidth,
        offenders: [...document.querySelectorAll("main *")].filter((node) => node.getBoundingClientRect().right > innerWidth + 1).slice(0, 12).map((node) => ({ tag: node.tagName, class: node.className, right: node.getBoundingClientRect().right })),
      })), { message: `No horizontal overflow on ${route}` }).toBe("");
      await page.screenshot({ path: info.outputPath(`${route.replaceAll(/[/?=]/g, "-")}.png`) });
    }
    const nav = page.getByRole("navigation", { name: "历史详情视图" });
    await nav.getByRole("link", { name: "概览", exact: true }).focus();
    await page.keyboard.press("Tab");
    await expect(nav.getByRole("link", { name: "指标", exact: true })).toBeFocused();
    await page.keyboard.press("Enter");
    await expect(nav.getByRole("link", { name: "指标", exact: true })).toHaveAttribute("aria-current", "page");
    await nav.getByRole("link", { name: "概览", exact: true }).click();
    const trigger = page.getByRole("button", { name: "检查点续训", exact: true });
    await page.route((url) => url.pathname.endsWith("/resume-options"), (route) => route.fulfill({ json: { checkpoints: [], default_checkpoint: "", message: "无检查点" } }));
    await trigger.focus(); await page.keyboard.press("Enter");
    const dialog = page.getByRole("dialog", { name: "从历史检查点续训" });
    await expect(dialog).toBeVisible();
    for (let index = 0; index < 10; index++) {
      await page.keyboard.press(index % 2 ? "Tab" : "Shift+Tab");
      expect(await dialog.evaluate((node) => node.contains(document.activeElement))).toBe(true);
    }
    await page.keyboard.press("Escape");
    await expect(trigger).toBeFocused();
    expect(mocks.writes).toEqual([]); expect(mocks.unhandled).toEqual([]);
  });
}
