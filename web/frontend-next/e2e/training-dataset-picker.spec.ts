import { test, expect, type Page } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

async function fixture(page: Page) {
  const mocks = await mockWorkspace(page);
  const datasets = [
    {
      source_dir: "images/studio",
      image_dir: "cache/studio",
      num_repeats: 2,
      settings: { keep_tokens: 1 },
    },
    {
      source_dir: "images/reference",
      image_dir: "cache/reference",
      num_repeats: 3,
      is_reg: true,
      settings: {},
    },
  ];
  await page.route("**/api/config/dataset-presets/read?*", (route) =>
    route.fulfill({
      json: {
        ok: true,
        file: "configs/datasets/studio.toml",
        name: "studio",
        content: "",
        datasets,
        defaults: { resolution: 1024, batch_size: 1 },
        readonly: false,
        summary: { dataset_count: 2 },
      },
    }),
  );
  return mocks;
}

for (const width of [1440, 390]) {
  test(`dataset modal preview and multi-subset drafts ${width}`, async ({
    page,
  }, info) => {
    await page.setViewportSize({ width, height: 900 });
    const mocks = await fixture(page);
    await page.goto("/next/training");
    const trigger = page.getByRole("button", {
      name: "选择与配置数据集",
      exact: true,
    });
    await trigger.click();
    const dialog = page.getByRole("dialog", { name: "选择与配置数据集" });
    await expect(dialog).toBeVisible();
    await expect(page.getByLabel("子集 2 重复次数")).toHaveValue("3");
    await expect(
      dialog.locator(".training-dataset-thumbnails img").first(),
    ).toBeVisible();
    expect(
      await dialog
        .locator(".training-dataset-thumbnails img")
        .first()
        .evaluate((node) => (node as HTMLImageElement).naturalWidth),
    ).toBeGreaterThan(0);
    await expect(dialog.getByText("3 张", { exact: true })).toHaveCount(2);
    await page.screenshot({
      path: info.outputPath(`dataset-dialog-${width}.png`),
    });
    expect(
      await dialog.evaluate((node) => node.scrollWidth <= node.clientWidth),
    ).toBe(true);
    await dialog.locator(".training-dataset-thumbnails button").first().click();
    await expect(dialog.getByLabel("图像详情")).toBeVisible();
    await dialog.getByRole("button", { name: "收起图像" }).click();
    await page.getByLabel("子集 2 重复次数").fill("6");
    await expect(
      dialog.getByRole("button", { name: "使用此数据集" }),
    ).toBeDisabled();
    page.once("dialog", (prompt) => prompt.dismiss());
    await dialog.getByRole("button", { name: "取消", exact: true }).click();
    await expect(dialog).toBeVisible();
    await expect(page.getByLabel("子集 2 重复次数")).toHaveValue("6");
    page.once("dialog", (prompt) => prompt.accept());
    await dialog.getByRole("button", { name: "取消", exact: true }).click();
    await expect(dialog).not.toBeVisible();
    await expect(trigger).toBeFocused();
    await expect(page.getByLabel("数据集配置", { exact: true })).toHaveValue(
      "configs/datasets/studio.toml",
    );
    expect(mocks.writes).toEqual([]);
  });
}

test("saves blueprint parameters separately from applying the reference", async ({
  page,
}) => {
  await fixture(page);
  const writes: Record<string, unknown>[] = [];
  await page.route(
    (url) => url.pathname === "/api/config/dataset-presets",
    async (route) => {
      if (route.request().method() !== "PUT") return route.fallback();
      const body = route.request().postDataJSON();
      writes.push(body);
      await route.fulfill({
        json: { ok: true, ...body, content: "", summary: { dataset_count: 2 } },
      });
    },
  );
  await page.goto("/next/training");
  await page.getByRole("button", { name: "选择与配置数据集" }).click();
  await page.getByLabel("子集 1 重复次数").fill("5");
  page.once("dialog", (prompt) => prompt.accept());
  await page.getByRole("button", { name: "保存蓝图参数" }).click();
  await expect(page.getByText("蓝图参数已保存")).toBeVisible();
  expect(writes).toHaveLength(1);
  expect(writes[0]).toMatchObject({
    datasets: [
      { num_repeats: 5, settings: { keep_tokens: 1 } },
      { num_repeats: 3, is_reg: true },
    ],
  });
  await page.getByRole("button", { name: "使用此数据集" }).click();
  await expect(page.getByRole("dialog")).not.toBeVisible();
});

test("dataset management validates the initial deep link against the library", async ({
  page,
}) => {
  const mocks = await mockWorkspace(page);
  const reads: string[] = [];
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.pathname === "/api/config/dataset-presets/read")
      reads.push(url.searchParams.get("file") || "");
  });
  await page.goto("/next/datasets?dataset=missing.toml");
  await expect(
    page.getByRole("heading", { name: "studio.toml", exact: true }),
  ).toBeVisible();
  expect(reads).not.toContain("missing.toml");
  expect(mocks.writes).toEqual([]);
});
