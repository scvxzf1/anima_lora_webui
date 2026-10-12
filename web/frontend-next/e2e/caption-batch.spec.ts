import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";
for (const width of [390, 1280]) test(`cross-page batch preview and candidate-only save at ${width}`, async ({ page }) => {
  await page.setViewportSize({ width, height: 844 });
  await mockWorkspace(page);
  const writes: string[] = [];
  const job = { id: "caption-1", state: "completed", profile_name: "Batch", profile_id: "p1", settings: { provider: "wd14" }, total: 65, completed: 65, failed: 0,
    items: Array.from({ length: 65 }, (_, i) => ({ id: String(i), name: `${i}.png`, file: `${i}.png`, url: "", state: "ready", caption: "", proposed_caption: "red, red" })) };
  await page.route((url) => url.pathname.startsWith("/api/captioning/jobs/caption-1"), async (route) => {
    if (route.request().method() !== "GET") {
      writes.push(`${route.request().method()} ${new URL(route.request().url()).pathname}`);
      for (const entry of route.request().postDataJSON().items) job.items.find((i) => i.id === entry.id)!.proposed_caption = entry.proposed_caption;
    }
    await route.fulfill({ json: { ok: true, job } });
  });
  await page.goto("/next/captioning?job=caption-1");
  await expect(page.getByRole("button", { name: "red (65)", exact: true })).toBeVisible();
  await page.getByLabel("操作范围").selectOption("all");
  await page.getByLabel("批量操作").selectOption("dedupe");
  await page.getByRole("button", { name: "预览差异 (65)", exact: true }).click();
  expect(writes).toEqual([]);
  await page.getByRole("button", { name: "应用到草稿", exact: true }).click();
  expect(writes).toEqual([]);
  await page.getByRole("button", { name: "保存全部草稿 (65)", exact: true }).click();
  await expect(page.getByRole("button", { name: "保存全部草稿 (0)", exact: true })).toBeDisabled();
  expect(writes).toEqual(["PATCH /api/captioning/jobs/caption-1/items"]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});
