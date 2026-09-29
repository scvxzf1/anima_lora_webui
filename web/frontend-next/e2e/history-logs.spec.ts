import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

async function expectBoundedRows(page: import("@playwright/test").Page) {
  await expect
    .poll(() => page.locator(".history-log-row").count())
    .toBeGreaterThan(0);
  await expect
    .poll(() => page.locator(".history-log-row").count())
    .toBeLessThan(100);
}

for (const width of [1285, 390]) {
  test(`full log navigation and bounded rendering ${width}`, async ({
    page,
  }, info) => {
    await page.setViewportSize({ width, height: 1054 });
    const mocks = await mockWorkspace(page);
    const offsets: number[] = [];
    const searches: URL[] = [];
    const total = 2000000;
    const matches = new Map<number, string>([
      [799607, "global needle"],
      [799611, "second needle"],
      [799613, "second needle"],
      [1200000, "global needle"],
    ]);
    const lineAt = (index: number) =>
      `training row ${index + 1} | loss=0.123 | ${"model/path/".repeat(20)} ${matches.get(index) ?? ""}`;
    await page.route(
      (url) => url.pathname.endsWith("/fixture-run/logs"),
      (route) => {
        const params = new URL(route.request().url()).searchParams;
        const limit = Number(params.get("limit"));
        const offset = params.has("offset")
          ? Number(params.get("offset"))
          : total - limit;
        offsets.push(offset);
        return route.fulfill({
          json: {
            total,
            offset,
            logs: Array.from(
              { length: Math.min(limit, total - offset) },
              (_, i) => ({ line: lineAt(offset + i) }),
            ),
          },
        });
      },
    );
    await page.route(
      (url) => url.pathname.endsWith("/logs/search"),
      (route) => {
        const requestUrl = new URL(route.request().url());
        searches.push(requestUrl);
        const query =
          requestUrl.searchParams.get("query")?.trim().toLocaleLowerCase() ??
          "";
        const cursor = Number(requestUrl.searchParams.get("cursor"));
        const direction = requestUrl.searchParams.get("direction");
        const matching = [...matches.entries()]
          .filter(([, text]) => text.toLocaleLowerCase().includes(query))
          .map(([index]) => index);
        const ordinal =
          direction === "backward"
            ? [...matching].reverse().findIndex((index) => index <= cursor)
            : matching.findIndex((index) => index >= cursor);
        const resolvedOrdinal =
          ordinal < 0
            ? direction === "backward"
              ? matching.length - 1
              : 0
            : direction === "backward"
              ? matching.length - 1 - ordinal
              : ordinal;
        const match_index = matching[resolvedOrdinal] ?? null;
        return route.fulfill({
          json: {
            match_index,
            match_ordinal: match_index === null ? 0 : resolvedOrdinal + 1,
            matches_total: matching.length,
            total,
          },
        });
      },
    );
    await page.goto("/next/history/fixture-run?view=logs");
    await expect(page.getByRole("log")).toContainText("training row 2000000");
    await expect(page.getByLabel("下一页", { exact: true })).toBeDisabled();
    await expectBoundedRows(page);
    await page.getByLabel("日志开头", { exact: true }).click();
    await expect(page.getByRole("log")).toContainText("training row 1 |");
    await expectBoundedRows(page);
    await page.getByRole("log").evaluate((element) => {
      element.scrollTop = 9950 * 24;
    });
    await expect(page.getByRole("log")).toContainText("training row 9951 |");
    await expectBoundedRows(page);
    await expect(page.getByLabel("跟随末尾")).not.toBeChecked();
    await page.getByLabel("跳转位置").fill("900001");
    await page.getByLabel("跳转", { exact: true }).click();
    await expect(page.getByRole("log")).toContainText("training row 900001 |");
    await expectBoundedRows(page);
    await page.getByLabel("跳转单位").selectOption("page");
    await page.getByLabel("跳转位置").fill("2000");
    await page.getByLabel("跳转", { exact: true }).click();
    await expect(page.getByRole("log")).toContainText("training row 799601 |");
    await expectBoundedRows(page);
    expect(offsets).toContain(799600);
    await page
      .getByLabel("搜索全部日志", { exact: true })
      .fill("global needle");
    await page.getByLabel("执行全局搜索").click();
    await expect(page.locator('[data-match="true"]')).toContainText(
      "training row 799608 |",
    );
    await expectBoundedRows(page);
    await expect(page.getByText("1 / 2 匹配", { exact: true })).toBeVisible();
    expect(searches).toHaveLength(1);
    expect(searches[0].searchParams.get("query")).toBe("global needle");
    expect(searches[0].searchParams.get("cursor")).toBe("799600");
    expect(searches[0].searchParams.get("direction")).toBe("forward");
    expect(offsets).toContain(799600);
    await page.getByLabel("搜索全部日志", { exact: true }).fill("global");
    await page.getByLabel("执行全局搜索").click();
    await expect(page.locator('[data-match="true"]')).toContainText(
      "training row 799608 |",
    );
    await expect(page.getByText("1 / 2 匹配", { exact: true })).toBeVisible();
    await expectBoundedRows(page);
    await expect.poll(() => searches.length).toBe(2);
    expect(searches[1].searchParams.get("cursor")).toBe("799607");
    expect(searches[1].searchParams.get("direction")).toBe("forward");
    expect(searches[1].searchParams.get("query")).toBe("global");
    await page
      .getByLabel("搜索全部日志", { exact: true })
      .fill("second needle");
    await page.getByLabel("执行全局搜索").click();
    await expect(page.locator('[data-match="true"]')).toContainText(
      "training row 799612 |",
    );
    await expect.poll(() => searches.length).toBe(3);
    expect(searches[2].searchParams.get("cursor")).toBe("799607");
    await page.getByLabel("上一匹配", { exact: true }).click();
    await expect(page.locator('[data-match="true"]')).toContainText(
      "training row 799614 |",
    );
    await expect.poll(() => searches.length).toBe(4);
    expect(searches[3].searchParams.get("cursor")).toBe("799610");
    expect(searches[3].searchParams.get("direction")).toBe("backward");
    await page.getByLabel("下一匹配", { exact: true }).click();
    await expect(page.locator('[data-match="true"]')).toContainText(
      "training row 799612 |",
    );
    await page.getByLabel("下一匹配", { exact: true }).click();
    await expect(page.locator('[data-match="true"]')).toContainText(
      "training row 799614 |",
    );
    await page
      .getByLabel("搜索全部日志", { exact: true })
      .fill("global needle");
    await page.getByLabel("执行全局搜索").click();
    await expect(page.locator('[data-match="true"]')).toContainText(
      "training row 1200001 |",
    );
    await page.getByLabel("下一匹配", { exact: true }).click();
    await expect(page.locator('[data-match="true"]')).toContainText(
      "training row 799608 |",
    );
    await page.getByLabel("下一匹配", { exact: true }).click();
    await expect(page.locator('[data-match="true"]')).toContainText(
      "training row 1200001 |",
    );
    await page.getByLabel("搜索全部日志", { exact: true }).fill("");
    await expect(page.locator('[data-match="true"]')).toHaveCount(0);
    await expect(page.getByText(/匹配$/, { exact: false })).toHaveCount(0);
    await expect
      .poll(() =>
        page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
      )
      .toBe(true);
    const box = await page.getByRole("log").boundingBox();
    expect(box!.height).toBeGreaterThan(width === 390 ? 400 : 650);
    await page.screenshot({
      path: info.outputPath("logs.png"),
      fullPage: true,
    });
    expect(offsets.length).toBeLessThan(20);
    expect(mocks.writes).toEqual([]);
    expect(mocks.unhandled).toEqual([]);
  });
}

test("log page failure can be retried and empty search is explicit", async ({
  page,
}) => {
  await mockWorkspace(page);
  let fail = true;
  await page.route(
    (url) => url.pathname.endsWith("/fixture-run/logs"),
    (route) => {
      const params = new URL(route.request().url()).searchParams;
      if (params.has("offset") && fail)
        return route.fulfill({
          status: 500,
          json: { error: "日志读取失败测试" },
        });
      return route.fulfill({
        json: { total: 1, offset: 0, logs: [{ line: "recovered log" }] },
      });
    },
  );
  await page.route(
    (url) => url.pathname.endsWith("/logs/search"),
    (route) =>
      route.fulfill({
        json: {
          match_index: null,
          match_ordinal: 0,
          matches_total: 0,
          total: 1,
        },
      }),
  );
  await page.goto("/next/history/fixture-run?view=logs");
  await expect(page.getByRole("alert")).toContainText("日志读取失败测试");
  fail = false;
  await page.getByRole("button", { name: "重试", exact: true }).click();
  await expect(page.getByRole("log")).toContainText("recovered log");
  await page.getByLabel("搜索全部日志", { exact: true }).fill("missing");
  await page.getByLabel("执行全局搜索").click();
  await expect(page.getByText("0 / 0 匹配", { exact: true })).toBeVisible();
});

test("empty logs report zero search matches", async ({ page }) => {
  await mockWorkspace(page);
  await page.route(
    (url) => url.pathname.endsWith("/fixture-run/logs"),
    (route) => route.fulfill({ json: { total: 0, offset: 0, logs: [] } }),
  );
  await page.route(
    (url) => url.pathname.endsWith("/logs/search"),
    (route) =>
      route.fulfill({
        json: {
          match_index: null,
          match_ordinal: 0,
          matches_total: 0,
          total: 0,
        },
      }),
  );
  await page.goto("/next/history/fixture-run?view=logs");
  await expect(page.getByRole("log")).toContainText("暂无日志");
  await page.getByLabel("搜索全部日志", { exact: true }).fill("missing");
  await page.getByLabel("执行全局搜索").click();
  await expect(page.getByText("0 / 0 匹配", { exact: true })).toBeVisible();
});
