import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

test("AUTO swap controls follow the enabled mode without losing manual values", async ({ page }, testInfo) => {
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
    const toggle = page.getByRole("checkbox", { name: "AUTO 块交换（实验）" });
    await expect(toggle).not.toBeChecked();
    await expect(swap.locator('[id^="training-field-auto_block_swap_"]')).toHaveCount(0);
    const manualSwap = page.getByRole("spinbutton", { name: "Block swap 数量" });
    await expect(manualSwap).toBeEnabled();
    await expect(manualSwap).toHaveValue("20");

    await toggle.check();
    const mode = page.getByRole("combobox", { name: "AUTO 调整模式" });
    await expect(mode).toBeVisible();
    await expect(swap.locator('[id^="training-field-auto_block_swap_"]')).toHaveCount(7);
    await expect(manualSwap).toBeDisabled();
    await expect(manualSwap).toHaveValue("20");
    const reservePercent = page.getByRole("spinbutton", { name: "保留显存（总容量 %）" });
    await expect(reservePercent).toHaveAttribute("min", "0");
    await expect(reservePercent).toHaveAttribute("max", "90");
    await expect(reservePercent).toHaveAttribute("step", "0.1");
    const preference = page.getByRole("combobox", { name: "显存 / 内存倾向" });
    await expect(preference.getByRole("option", { name: "均衡" })).toBeAttached();
    await expect(preference.getByRole("option", { name: "优先节省显存" })).toBeAttached();
    await expect(preference.getByRole("option", { name: "优先节省内存" })).toBeEnabled();
    await reservePercent.scrollIntoViewIfNeeded();
    await expect(reservePercent).toBeInViewport();
    await page.locator("#training-field-auto_block_swap_vram_reserve_percent").screenshot({ path: testInfo.outputPath(`auto-swap-reserve-${name}.png`) });
    await preference.scrollIntoViewIfNeeded();
    await expect(preference).toBeInViewport();
    await page.locator("#training-field-auto_block_swap_preference").screenshot({ path: testInfo.outputPath(`auto-swap-preference-${name}.png`) });

    await mode.selectOption("dynamic");
    await expect(page.getByRole("spinbutton", { name: "动态评估窗口（更新数）" })).toBeEnabled();
    await expect(page.getByRole("spinbutton", { name: "AUTO 最大候选数" })).toBeDisabled();

    await mode.selectOption("startup");
    await expect(page.getByRole("spinbutton", { name: "动态评估窗口（更新数）" })).toBeDisabled();
    await expect(page.getByRole("spinbutton", { name: "AUTO 最大候选数" })).toBeEnabled();

    await toggle.uncheck();
    await expect(toggle).not.toBeChecked();
    await expect(swap.locator('[id^="training-field-auto_block_swap_"]')).toHaveCount(0);
    await expect(manualSwap).toBeEnabled();
    await expect(manualSwap).toHaveValue("20");
  }

  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});
