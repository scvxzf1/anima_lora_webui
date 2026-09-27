import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

test("model library keeps long paths intact and persists the visible item order", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  const longPath = `/mnt/model-archive/${"very-long-directory/".repeat(12)}qwen3vl.safetensors`;
  let serverData = {
    ok: true,
    revision: "models-revision-1",
    default_id: "model-a",
    items: [
      {
        id: "model-a",
        name: "Alpha",
        model_family: "krea2_raw",
        pretrained_model_name_or_path: "models/alpha-dit.safetensors",
        qwen3: "models/alpha-qwen.safetensors",
        vae: "models/alpha-vae.safetensors",
      },
      {
        id: "model-b",
        name: "Beta",
        model_family: "anima",
        pretrained_model_name_or_path: "models/beta-dit.safetensors",
        qwen3: "models/beta-qwen.safetensors",
        vae: "models/beta-vae.safetensors",
      },
      {
        id: "model-c",
        name: "Gamma",
        model_family: "z_image",
        pretrained_model_name_or_path: longPath,
        qwen3: "models/gamma-qwen.safetensors",
        vae: "models/gamma-vae.safetensors",
      },
    ],
    groups: [{ id: "group-main", label: "Main models", item_ids: ["model-a", "model-b", "model-c"] }],
  };
  const saves: Record<string, unknown>[] = [];
  await page.route((url) => url.pathname === "/api/settings/model-configs", async (route) => {
    if (route.request().method() === "GET") return route.fulfill({ json: serverData });
    const body = route.request().postDataJSON() as Record<string, unknown>;
    saves.push(body);
    serverData = { ...body, ok: true, revision: "models-revision-2" } as typeof serverData;
    return route.fulfill({ json: serverData });
  });

  await page.goto("/next/models");
  await expect(page.locator(".object-row")).toHaveCount(3);
  await page.locator(".object-row").nth(2).click();
  await expect(page.getByLabel("名称")).toHaveValue("Gamma");
  await expect(page.getByLabel("DiT 模型")).toHaveValue(longPath);

  const order = async () => page.locator(".object-row").evaluateAll((rows) =>
    rows.map((row) => row.querySelector("span")?.textContent || ""),
  );
  const moveUp = page.getByRole("button", { name: "上移模型配置", exact: true });
  await moveUp.click();
  await moveUp.click();
  await expect.poll(order).toEqual(["Gamma", "Alpha", "Beta"]);
  await moveUp.click();
  await expect.poll(order).toEqual(["Gamma", "Alpha", "Beta"]);

  await page.locator(".object-row").nth(2).click();
  const moveDown = page.getByRole("button", { name: "下移模型配置", exact: true });
  await moveDown.click();
  await expect.poll(order).toEqual(["Gamma", "Alpha", "Beta"]);

  await page.getByRole("button", { name: "保存模型配置", exact: true }).click();
  await expect(page.getByText("模型配置已保存", { exact: true })).toBeVisible();
  expect(saves).toHaveLength(1);
  expect(saves[0]).toMatchObject({
    revision: "models-revision-1",
    default_id: "model-a",
    groups: [{ id: "group-main", label: "Main models", item_ids: ["model-c", "model-a", "model-b"] }],
  });
  expect((saves[0].items as Array<Record<string, unknown>>).find((item) => item.id === "model-c")?.pretrained_model_name_or_path)
    .toBe(longPath);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});
