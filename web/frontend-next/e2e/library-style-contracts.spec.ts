import { expect, test } from "@playwright/test";
import { configFile, mockWorkspace } from "./fixtures";

test("group-name searches include their files and absent matches are explicit", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  await page.route((url) => url.pathname === "/api/config/file-groups", (route) => route.fulfill({ json: [
    { id: "imported", label: "Archive Alpha", files: [configFile] },
  ] }));
  await page.route((url) => url.pathname === "/api/config/dataset-presets", (route) => {
    const preset = { path: "configs/datasets/studio.toml", filename: "studio.toml", label: "Portrait blueprint", locked: false };
    return route.fulfill({ json: { ok: true, presets: [preset], groups: [{ id: "studio", label: "Archive Alpha", files: [preset] }] } });
  });
  await page.goto("/next/training");
  await page.getByRole("textbox", { name: "搜索配置", exact: true }).fill("archive alpha");
  await expect(page.locator(".training-library-item")).toHaveCount(1);
  await expect(page.locator(".training-library-item")).toContainText("Studio portrait");
  await page.getByRole("textbox", { name: "搜索配置", exact: true }).fill("absent-query");
  await expect(page.getByText("没有匹配的配置", { exact: true })).toBeVisible();
  await page.goto("/next/datasets");
  await page.getByRole("searchbox", { name: "搜索预设", exact: true }).fill("archive alpha");
  await expect(page.locator(".dataset-group-list")).toContainText("Portrait blueprint");
  await page.getByRole("searchbox", { name: "搜索预设", exact: true }).fill("absent-query");
  await expect(page.getByText("没有匹配的数据集预设", { exact: true })).toBeVisible();
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

for (const theme of ["dark", "light"]) {
  test(`shared chart and checkbox styles survive same-document routes ${theme}`, async ({ page }) => {
    const mocks = await mockWorkspace(page);
    await page.addInitScript((theme) => localStorage.setItem("dragon-next-ui-v1-theme", theme), theme);
    await page.goto("/next/monitor");
    await expect(page.locator("canvas")).toHaveCount(1);
    const measure = () => page.locator(".chart-heading, .chart-controls .checkbox-row").evaluateAll((nodes) => nodes.map((node) => {
      const rect = node.getBoundingClientRect();
      const style = getComputedStyle(node);
      return { x: rect.x, y: rect.y, width: rect.width, height: rect.height, display: style.display, gap: style.gap };
    }));
    const before = await measure();
    await page.evaluate(() => { (window as unknown as { routeMarker: number }).routeMarker = 1; });
    const nav = page.getByRole("navigation", { name: "主导航" });
    for (const destination of ["打标工作台", "全局设置"]) {
      await nav.getByRole("link", { name: destination, exact: true }).click();
      await expect(page.getByRole("heading", { name: destination, exact: true })).toBeVisible();
      await nav.getByRole("link", { name: "当前监控", exact: true }).click();
      await expect(page.locator("canvas")).toHaveCount(1);
      expect(await measure()).toEqual(before);
      expect(await page.evaluate(() => (window as unknown as { routeMarker: number }).routeMarker)).toBe(1);
    }
    expect(mocks.writes).toEqual([]);
    expect(mocks.unhandled).toEqual([]);
  });
}
