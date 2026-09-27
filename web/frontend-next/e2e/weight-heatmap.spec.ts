import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

test("weight heatmap renders and scrolls within the analysis page", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 850 });
  const mocks = await mockWorkspace(page);
  const components = ["attention_to_q", "attention_to_k", "attention_to_v", "attention_to_out_0", "feed_forward_net_0"];
  const blocks = [0, 1, 2, 3];
  const matrix = blocks.map((block) => components.map((_, index) => block === 0 && index === 0 ? 0 : (block + 1) * (index + 1)));
  const cells = blocks.flatMap((block, row) => components.flatMap((component, column) => matrix[row][column] ? [{ block, component, fro_norm: matrix[row][column], layer_count: 1, top_layer: `layers_${block}_${component}`, intensity: matrix[row][column] / 20 }] : []));
  await page.route("**/api/analysis/weights", (route) => route.fulfill({ json: { weights: [], count: 0 } }));
  await page.route("**/api/analysis/inspect", (route) => route.fulfill({ json: {
    ok: true,
    file: { name: "fixture.safetensors", path: "output/fixture.safetensors" },
    adapter_type: "LoRA",
    summary: { layer_count: cells.length + 1, block_count: blocks.length, total_energy: 200 },
    layers: [...cells.map((cell) => ({ block: cell.block })), { block: null }],
    component_summary: [{ label: "attention", layer_count: 4, fro_norm: 10 }],
    block_summary: [], style_top20: [], character_top20: [],
    heatmap: { blocks, components, matrix, max_value: 20, cells },
  } }));

  await page.goto("/next/weight-analysis");
  await page.getByRole("textbox", { name: "权重 A路径" }).fill("output/fixture.safetensors");
  await page.getByRole("button", { name: "开始分析" }).click();
  const heatmap = page.getByRole("region", { name: "区块 × 组件热力图" });
  await expect(heatmap.getByRole("columnheader", { name: "attention_to_out_0" })).toBeVisible();
  await expect(heatmap.getByRole("cell", { name: /Block 3.*attention_to_out_0.*范数: 16/ })).toBeVisible();
  await expect(page.getByText(/纳入 19\/20 层/)).toBeVisible();

  await page.setViewportSize({ width: 390, height: 844 });
  expect(await heatmap.evaluate((element) => element.scrollWidth > element.clientWidth)).toBe(true);
  await heatmap.evaluate((element) => { element.scrollLeft = element.scrollWidth; });
  await expect(heatmap.getByRole("columnheader", { name: "feed_forward_net_0" })).toBeVisible();
  await page.getByRole("tab", { name: "组件" }).click();
  await expect(page.getByRole("cell", { name: "attention" })).toBeVisible();
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});
