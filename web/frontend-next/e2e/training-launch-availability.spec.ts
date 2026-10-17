import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

test("unavailable saved GPU exposes a recovery entry beside disabled launch actions", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  await page.addInitScript(() => localStorage.setItem("dragon-next.training-devices.v1", JSON.stringify({
    mode: "single", devices: [{ id: "9", name: "Old GPU" }],
  })));
  await page.route("**/api/training/gpus", (route) => route.fulfill({
    json: { gpus: [{ index: 1, name: "Available GPU", memory_total_gb: 24 }] },
  }));
  await page.goto("/next/training");
  await expect(page.getByText("已同步", { exact: true })).toBeVisible();
  const warning = page.locator(".training-launch-availability");
  await expect(warning).toContainText("已选设备不可用");
  await expect(page.getByRole("button", { name: "立即启动", exact: true })).toBeDisabled();
  await expect(page.getByRole("button", { name: "加入队列", exact: true })).toBeDisabled();
  await warning.getByRole("button", { name: "检查训练设备" }).click();
  const dialog = page.getByRole("dialog", { name: "检查训练设备" });
  await dialog.getByRole("radio", { name: "GPU 1 · Available GPU" }).check();
  await expect(warning).toHaveCount(0);
  await expect(dialog).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(dialog).toHaveCount(0);
  await expect(page.getByRole("button", { name: "立即启动", exact: true })).toBeEnabled();
  await expect(page.getByRole("button", { name: "加入队列", exact: true })).toBeEnabled();
  expect(mocks.writes).toEqual([]);
});
