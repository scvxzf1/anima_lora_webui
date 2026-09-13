import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

for (const width of [1440, 390]) {
  test(`preflight locates fields across categories ${width}`, async ({ page }, info) => {
    await page.setViewportSize({ width, height: 900 });
    const mocks = await mockWorkspace(page);
    await page.route("**/api/training/preflight", (route) => route.fulfill({
      json: {
        ok: true,
        summary: { errors: 0, warnings: 2, checks: 2 },
        checks: ["compile_dynamic_seq", "base_compute"].map((key) => ({
          level: "warning", key, message: `Check ${key}`,
        })),
      },
    }));
    await page.goto("/next/training");
    await page.getByRole("button", { name: "运行预检测", exact: true }).click();
    for (const key of ["compile_dynamic_seq", "base_compute"]) {
      await page.getByRole("button", { name: `定位 ${key}`, exact: true }).click();
      await expect(page.getByRole("dialog")).toHaveCount(0);
      await expect(page.getByRole("tab", { name: /设备与性能/ })).toHaveAttribute("aria-selected", "true");
      await expect(page.getByLabel("搜索参数")).toHaveValue(key);
      await expect(page.getByLabel("参数视图")).toHaveValue("all");
      const field = page.locator(`#training-field-${key}`);
      await expect(field).toBeVisible();
      await expect(field.locator("input, select, textarea")).toBeFocused();
      await expect(page.getByText("当前分类没有符合筛选条件的配置项。")).toHaveCount(0);
      await page.screenshot({ path: info.outputPath(`${key}.png`) });
      await page.getByRole("tab", { name: /输入准备/ }).click();
      await page.getByLabel("参数视图").selectOption("changed");
      await page.getByRole("button", { name: "预览与预检", exact: true }).click();
    }
    expect(mocks.writes).toEqual([]);
    expect(mocks.unhandled).toEqual([]);
  });
}
