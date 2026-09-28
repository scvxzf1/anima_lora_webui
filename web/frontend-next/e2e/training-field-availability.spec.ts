import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

test("boolean drafts and audit-only fields keep their editing boundaries", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  await page.route((url) => url.pathname === "/api/config/merged", (route) => route.fulfill({ json: {
    model_family: "anima",
    output_name: "field-audit",
    dim_from_weights: false,
    network_weights: "adapter.safetensors",
    network_dim: 16,
    __future_runtime_field__: 7,
  } }));
  await page.goto("/next/training");
  await expect(page.getByText("已同步", { exact: true })).toBeVisible();

  await page.getByRole("tab", { name: "方法配置" }).click();
  await page.getByLabel("搜索参数").fill("dim_from_weights");
  const dimensionSwitch = page.getByLabel("从权重读取秩");
  await expect(dimensionSwitch).toHaveAttribute("type", "checkbox");
  await expect(dimensionSwitch).not.toBeChecked();

  await page.getByLabel("搜索参数").fill("network_dim");
  await expect(page.getByLabel("LoRA rank")).toBeEnabled();

  await page.getByRole("tab", { name: "训练计划" }).click();
  await page.getByLabel("搜索参数").fill("__future_runtime_field__");
  await expect(page.getByLabel("__future_runtime_field__")).toHaveCount(0);
  await page.getByLabel("参数视图").selectOption("all");
  await expect(page.getByLabel("__future_runtime_field__")).toBeDisabled();
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});
