import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

for (const width of [1285, 390]) {
  test(`full log navigation and bounded rendering ${width}`, async ({ page }, info) => {
    await page.setViewportSize({ width, height: 1054 });
    const mocks = await mockWorkspace(page);
    const offsets: number[] = [];
    const total = 2000000;
    await page.route((url) => url.pathname.endsWith("/fixture-run/logs"), (route) => {
      const params = new URL(route.request().url()).searchParams;
      const limit = Number(params.get("limit"));
      const offset = params.has("offset") ? Number(params.get("offset")) : total - limit;
      offsets.push(offset);
      return route.fulfill({ json: { total, offset, logs: Array.from({ length: Math.min(limit, total - offset) }, (_, i) => ({ line: `training row ${offset + i + 1} | loss=0.123 | ${"model/path/".repeat(20)}` })) } });
    });
    await page.route((url) => url.pathname.endsWith("/logs/search"), (route) => route.fulfill({ json: {
      match_index: 1234566, match_ordinal: 1, matches_total: 1, total,
    } }));
    await page.goto("/next/history/fixture-run?view=logs");
    await expect(page.getByRole("log")).toContainText("training row 2000000");
    await expect(page.getByLabel("下一页", { exact: true })).toBeDisabled();
    await expect(page.locator(".history-log-row")).not.toHaveCount(0);
    expect(await page.locator(".history-log-row").count()).toBeLessThan(100);
    await page.getByLabel("日志开头", { exact: true }).click();
    await expect(page.getByRole("log")).toContainText("training row 1 |");
    await page.getByRole("log").evaluate((element) => { element.scrollTop = 9950 * 24; });
    await expect(page.getByRole("log")).toContainText("training row 9951 |");
    await expect(page.getByLabel("跟随末尾")).not.toBeChecked();
    await page.getByLabel("跳转位置").fill("900001");
    await page.getByLabel("跳转", { exact: true }).click();
    await expect(page.getByRole("log")).toContainText("training row 900001 |");
    await page.getByLabel("跳转单位").selectOption("page");
    await page.getByLabel("跳转位置").fill("2000");
    await page.getByLabel("跳转", { exact: true }).click();
    await expect(page.getByRole("log")).toContainText("training row 799601 |");
    await page.getByLabel("搜索全部日志", { exact: true }).fill("global needle");
    await page.getByLabel("执行全局搜索").click();
    await expect(page.locator('[data-match="true"]')).toContainText("training row 1234567 |");
    await expect(page.getByText("1 / 1 匹配", { exact: true })).toBeVisible();
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    const box = await page.getByRole("log").boundingBox();
    expect(box!.height).toBeGreaterThan(width === 390 ? 400 : 650);
    await page.screenshot({ path: info.outputPath("logs.png"), fullPage: true });
    expect(offsets.length).toBeLessThan(20);
    expect(mocks.writes).toEqual([]);
    expect(mocks.unhandled).toEqual([]);
  });
}

test("log page failure can be retried and empty search is explicit", async ({ page }) => {
  await mockWorkspace(page);
  let fail = true;
  await page.route((url) => url.pathname.endsWith("/fixture-run/logs"), (route) => {
    const params = new URL(route.request().url()).searchParams;
    if (params.has("offset") && fail) return route.fulfill({ status: 500, json: { error: "日志读取失败测试" } });
    return route.fulfill({ json: { total: 1, offset: 0, logs: [{ line: "recovered log" }] } });
  });
  await page.route((url) => url.pathname.endsWith("/logs/search"), (route) => route.fulfill({ json: { match_index: null, match_ordinal: 0, matches_total: 0, total: 1 } }));
  await page.goto("/next/history/fixture-run?view=logs");
  await expect(page.getByRole("alert")).toContainText("日志读取失败测试");
  fail = false;
  await page.getByRole("button", { name: "重试", exact: true }).click();
  await expect(page.getByRole("log")).toContainText("recovered log");
  await page.getByLabel("搜索全部日志", { exact: true }).fill("missing");
  await page.getByLabel("执行全局搜索").click();
  await expect(page.getByText("0 / 0 匹配", { exact: true })).toBeVisible();
});
