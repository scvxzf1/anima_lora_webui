import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

const buckets = [[1024, 1024, 82], [832, 1280, 1], [896, 1152, 1], [1024, 960, 1], [1152, 960, 1], [1280, 896, 1], [1408, 704, 1]]
  .map(([width, height, count]) => ({ width, height, count }));
const datasets = ["train", "reg"].map((name, index) => ({
  index: index + 1, is_reg: index === 1, source_dir: `image_dataset/${name}/data`, image_dir: `post_image_dataset/${name}/data`,
  train_image_count: 88, num_repeats: 1, sample_ratio: 1, sampled_image_count: 88, sampled_weighted_image_count: 88,
  trigger_clone_sampled_weighted_image_count: 0, uses_preprocessed_images: true,
  bucket_distribution: { basis: "image_dimensions", status: "ready", image_count: 88, unreadable_count: 0, buckets },
}));

for (const [width, theme] of [[1440, "light"], [1440, "dark"], [390, "light"]] as const) {
  test(`estimate distribution ${width} ${theme}`, async ({ page }, info) => {
    await page.setViewportSize({ width, height: 1000 });
    const mocks = await mockWorkspace(page);
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.route("**/api/config/steps?**", (route) => route.fulfill({ json: {
      total_steps: 3520, train_image_count: 176, effective_batch_size: 1, steps_per_epoch: 176,
      repeated_image_count: 176, train_batch_size: 1, gradient_accumulation_steps: 1,
      duration_mode: "epochs", max_train_epochs: 20, datasets,
    } }));
    await page.goto("/next/training");
    await page.evaluate((theme) => { document.documentElement.dataset.theme = theme; }, theme);
    await page.getByRole("button", { name: "训练量估算", exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "训练量估算" });
    await expect(dialog.getByRole("table")).toBeVisible();
    await expect(dialog.getByText("共 176 张")).toBeVisible();
    expect(await dialog.evaluate((element) => element.scrollWidth > element.clientWidth)).toBe(false);
    expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false);
    await page.screenshot({ path: info.outputPath("estimate.png") });
    await dialog.getByLabel("分桶数据集").selectOption("2");
    await expect(dialog.getByText("共 88 张")).toBeVisible();
    await dialog.getByLabel("分桶排序").selectOption("aspect");
    await expect(dialog.getByRole("row").nth(1)).toContainText("832 × 1280");
    await dialog.getByRole("row").last().scrollIntoViewIfNeeded();
    await page.screenshot({ path: info.outputPath("estimate-filtered.png") });
    await page.keyboard.press("Escape");
    await expect(dialog).toHaveCount(0);
    await expect(page.getByRole("button", { name: "训练量估算", exact: true })).toBeFocused();
    expect(errors).toEqual([]);
    expect(mocks.writes).toEqual([]);
  });
}

test("estimate errors keep the dialog empty until an explicit retry", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  const mocks = await mockWorkspace(page);
  let attempts = 0;
  let recovered = false;
  await page.route("**/api/config/steps?**", (route) => {
    attempts += 1;
    if (!recovered)
      return route.fulfill({
        status: 503,
        contentType: "application/json",
        body: JSON.stringify({ error: "训练量估算暂不可用" }),
      });
    return route.fulfill({ json: {
      total_steps: 3520, train_image_count: 176, effective_batch_size: 1, steps_per_epoch: 176,
      repeated_image_count: 176, train_batch_size: 1, gradient_accumulation_steps: 1,
      duration_mode: "epochs", max_train_epochs: 20, datasets,
    } });
  });

  await page.goto("/next/training");
  await page.getByRole("button", { name: "训练量估算", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "训练量估算" });
  await expect(dialog.getByRole("alert")).toContainText("训练量估算暂不可用");
  await expect(dialog.getByRole("table")).toHaveCount(0);
  const failedAttempts = attempts;
  await page.waitForTimeout(1200);
  expect(attempts).toBe(failedAttempts);

  recovered = true;
  await dialog.getByRole("button", { name: "重新估算", exact: true }).click();
  await expect(dialog.getByRole("table", { name: "分桶尺寸明细" })).toBeVisible();
  expect(attempts).toBe(failedAttempts + 1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

for (const width of [1440, 390]) {
  test(`many source buckets scroll without moving background ${width}`, async ({ page }, info) => {
    await page.setViewportSize({ width, height: 800 });
    const mocks = await mockWorkspace(page);
    const many = Array.from({ length: 24 }, (_, index) => ({ width: 512 + index * 64, height: 1024, count: 1 }));
    await page.route("**/api/config/steps?**", (route) => route.fulfill({ json: {
      total_steps: 240, train_image_count: 24, effective_batch_size: 1, steps_per_epoch: 24,
      repeated_image_count: 24, train_batch_size: 1, gradient_accumulation_steps: 1,
      duration_mode: "epochs", max_train_epochs: 10,
      datasets: Array.from({ length: 8 }, (_, index) => ({ ...datasets[0], index: index + 1,
        uses_preprocessed_images: false,
        bucket_distribution: { basis: "source_projection", status: "ready", image_count: 24, unreadable_count: 0, buckets: many },
      })),
    } }));
    await page.goto("/next/training");
    await page.getByRole("button", { name: "训练量估算", exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "训练量估算" });
    const table = dialog.getByRole("table");
    await expect(table).toBeAttached();
    const background = () => page.evaluate(() => [window.scrollY, ...Array.from(document.querySelectorAll<HTMLElement>("body *"))
      .filter((e) => !e.closest(".command-backdrop") && e.scrollHeight > e.clientHeight).map((e) => e.scrollTop)]);
    const before = await background();
    const bounds = await dialog.boundingBox();
    await page.mouse.move(bounds!.x + bounds!.width / 2, bounds!.y + bounds!.height / 2);
    await page.mouse.wheel(0, 600);
    await expect.poll(() => dialog.evaluate((e) => e.scrollTop)).toBeGreaterThan(0);
    await dialog.evaluate((e) => { e.scrollTop = e.scrollHeight; });
    await table.hover();
    await page.mouse.wheel(0, 400);
    await expect.poll(() => table.evaluate((e) => e.scrollTop)).toBeGreaterThan(0);
    await table.evaluate((e) => { e.scrollTop = e.scrollHeight; });
    await page.mouse.wheel(0, 1500);
    await page.mouse.move(4, 400);
    await page.mouse.wheel(0, 1500);
    await page.waitForTimeout(200);
    expect(await background()).toEqual(before);
    expect(await page.evaluate(() => getComputedStyle(document.body).overflow)).toBe("hidden");
    expect(await dialog.evaluate((e) => e.scrollWidth > e.clientWidth)).toBe(false);
    await page.screenshot({ path: info.outputPath("many-buckets-bottom.png") });
    await dialog.getByRole("button", { name: "关闭", exact: true }).click();
    await expect(dialog).toHaveCount(0);
    await expect(page.getByRole("button", { name: "训练量估算", exact: true })).toBeFocused();
    expect(mocks.writes).toEqual([]);
  });
}
