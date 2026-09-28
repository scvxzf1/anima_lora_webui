import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

test("AUTO swap children render only while AUTO is enabled", async ({ page }, testInfo) => {
  const mocks = await mockWorkspace(page);
  await page.route((url) => url.pathname === "/api/config/merged", (route) =>
    route.fulfill({ json: {
      model_family: "krea2_raw",
      output_name: "auto-swap-visibility",
      auto_block_swap: false,
      auto_block_swap_mode: "startup",
      auto_block_swap_interval: 8,
      auto_block_swap_max_trials: 6,
      auto_block_swap_timeout: 120,
      auto_block_swap_swap_io_limit_mb: 512,
      auto_block_swap_vram_reserve_percent: 10,
      auto_block_swap_preference: "balanced",
      blocks_to_swap: 20,
    } }),
  );

  for (const [name, width, height] of [["desktop", 1280, 800], ["mobile", 390, 844]] as const) {
    await page.setViewportSize({ width, height });
    await page.goto("/next/training");
    await expect(page.getByText("已同步", { exact: true })).toBeVisible();
    await page.getByRole("tab", { name: "设备与性能" }).click();
    await page.getByLabel("参数视图").selectOption("all");
    await page.getByRole("button", { name: /块交换与内存卸载/ }).click();

    const swap = page.locator("#resource-fields-residency");
    const toggle = page.getByLabel("AUTO 块交换（实验）");
    await expect(toggle).not.toBeChecked();
    await expect(swap.locator('[id^="training-field-auto_block_swap_"]')).toHaveCount(0);
    await expect(page.getByLabel("Block swap 数量")).toBeEnabled();
    await swap.screenshot({ path: testInfo.outputPath(`auto-swap-off-${name}.png`) });

    await toggle.check();
    await expect(page.getByLabel("AUTO 调整模式")).toBeVisible();
    await expect(swap.locator('[id^="training-field-auto_block_swap_"]')).toHaveCount(7);
    await swap.screenshot({ path: testInfo.outputPath(`auto-swap-on-${name}.png`) });
  }

  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});
