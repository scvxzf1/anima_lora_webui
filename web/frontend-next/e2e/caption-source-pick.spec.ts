import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

const PAGE_SIZE = 60;
const TOTAL_IMAGES = 600;

function pageImages(offset: number) {
  return Array.from(
    { length: Math.min(PAGE_SIZE, TOTAL_IMAGES - offset) },
    (_, index) => {
      const number = offset + index;
      return {
        id: `source-${number}`,
        name: `source-${number}.png`,
        file: `source-${number}.png`,
        url: `/api/config/dataset-presets/image?image=${number}`,
        state: "ready",
        caption: "",
        proposed_caption: "",
      };
    },
  );
}

test("caption source selection persists across pages and submits at most 500 images", async ({
  page,
}) => {
  const mocks = await mockWorkspace(page);
  const scanRequests: string[] = [];
  let submitted: Record<string, unknown> | undefined;
  await page.route(
    (url) => url.pathname === "/api/config/dataset-presets/images",
    async (route) => {
      const url = new URL(route.request().url());
      const offset = Number(url.searchParams.get("offset") || 0);
      scanRequests.push(`${url.searchParams.get("source")}:${offset}`);
      return route.fulfill({
        json: {
          ok: true,
          file: "configs/datasets/studio.toml",
          dataset_index: 0,
          source: url.searchParams.get("source"),
          total: TOTAL_IMAGES,
          limit: PAGE_SIZE,
          images: pageImages(offset),
        },
      });
    },
  );
  await page.route(
    (url) => url.pathname === "/api/captioning/jobs",
    async (route) => {
      if (route.request().method() !== "POST") return route.fallback();
      submitted = route.request().postDataJSON() as Record<string, unknown>;
      return route.fulfill({
        json: {
          ok: true,
          job: {
            id: "caption-source-pick",
            state: "queued",
            profile_name: "Studio captions",
            profile_id: "provider-a",
            dataset_file: "configs/datasets/studio.toml",
            dataset_index: 0,
            total: 500,
            completed: 0,
            failed: 0,
            items: [],
          },
        },
      });
    },
  );

  await page.goto("/next/captioning");
  await page
    .getByRole("combobox", { name: "数据集预设", exact: true })
    .selectOption("configs/datasets/studio.toml");
  await page.getByRole("button", { name: "扫描图片", exact: true }).click();
  await expect(page.locator(".caption-image-grid img")).toHaveCount(PAGE_SIZE);

  for (let pageIndex = 0; pageIndex < 8; pageIndex += 1) {
    await page.getByLabel("全选本页").check();
    await page.getByRole("button", { name: "下一页图片" }).click();
    await expect(
      page.locator(".caption-image-grid img").first(),
    ).toHaveAttribute("alt", `source-${(pageIndex + 1) * PAGE_SIZE}.png`);
  }
  for (let index = 0; index < 21; index += 1) {
    await page
      .locator(".caption-image-grid input[type=checkbox]")
      .nth(index)
      .check();
  }
  await expect(
    page.getByText("600 张 · 已选 501", { exact: true }),
  ).toBeVisible();
  await expect(page.getByRole("alert")).toContainText("最多处理 500 张图片");
  await expect(
    page.getByRole("button", { name: "开始打标 (501)", exact: true }),
  ).toBeDisabled();
  await page
    .locator(".caption-image-grid input[type=checkbox]")
    .nth(20)
    .uncheck();
  await expect(
    page.getByRole("button", { name: "开始打标 (500)", exact: true }),
  ).toBeEnabled();

  page.on("dialog", (dialog) => void dialog.accept());
  await page
    .getByRole("button", { name: "开始打标 (500)", exact: true })
    .click();
  await expect(page).toHaveURL(/job=caption-source-pick/);
  expect(scanRequests).toEqual([
    "source:0",
    "source:60",
    "source:120",
    "source:180",
    "source:240",
    "source:300",
    "source:360",
    "source:420",
    "source:480",
  ]);
  expect(submitted?.items).toHaveLength(500);
  expect((submitted?.items as { file: string }[])[0].file).toBe("source-0.png");
  expect((submitted?.items as { file: string }[])[499].file).toBe(
    "source-499.png",
  );
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});
