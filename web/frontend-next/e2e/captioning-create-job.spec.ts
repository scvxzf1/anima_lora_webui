import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

const createdJob = {
  id: "caption-created",
  state: "queued",
  profile_name: "Studio captions",
  profile_id: "provider-a",
  dataset_file: "configs/datasets/studio.toml",
  dataset_index: 0,
  total: 1,
  completed: 0,
  failed: 0,
  items: [{
    id: "image-0",
    name: "studio-1.png",
    file: "studio-1.png",
    url: "/api/config/dataset-presets/image?image=0",
    state: "ready",
    caption: "fixture caption",
    proposed_caption: "Fixture candidate",
  }],
};

async function prepareSource(page: import("@playwright/test").Page) {
  await page.goto("/next/captioning");
  await page
    .getByRole("combobox", { name: "数据集预设", exact: true })
    .selectOption("configs/datasets/studio.toml");
  await page.getByRole("button", { name: "扫描图片", exact: true }).click();
  await expect(page.locator(".caption-image-grid img")).toHaveCount(3);
  await page.locator(".caption-image-grid input[type=checkbox]").first().check();
  await expect(page.getByRole("button", { name: "开始打标 (1)", exact: true })).toBeEnabled();
}

test("caption job creation keeps the selection after failure and succeeds only on explicit retry", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  const submissions: Record<string, unknown>[] = [];
  let recovered = false;
  await page.route(
    (url) => url.pathname === "/api/captioning/jobs" || url.pathname === "/api/captioning/jobs/caption-created",
    async (route) => {
      const request = route.request();
      if (request.method() === "GET") {
        if (new URL(request.url()).pathname.endsWith("caption-created"))
          return route.fulfill({ json: { ok: true, job: createdJob } });
        return route.fallback();
      }
      submissions.push(request.postDataJSON() as Record<string, unknown>);
      if (!recovered)
        return route.fulfill({ status: 503, json: { error: "创建打标任务暂不可用" } });
      return route.fulfill({ json: { ok: true, job: createdJob } });
    },
  );

  await prepareSource(page);
  page.on("dialog", (dialog) => void dialog.accept());
  await page.getByRole("button", { name: "开始打标 (1)", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("创建打标任务暂不可用");
  await expect(page.getByRole("button", { name: "开始打标 (1)", exact: true })).toBeEnabled();
  await page.waitForTimeout(600);
  expect(submissions).toHaveLength(1);
  expect(submissions[0]).toMatchObject({
    dataset_file: "configs/datasets/studio.toml",
    dataset_index: 0,
    source: "source",
    profile_id: "provider-a",
    items: [{ file: "studio-1.png" }],
  });

  recovered = true;
  await page.getByRole("button", { name: "开始打标 (1)", exact: true }).click();
  await expect(page).toHaveURL(/\/next\/captioning\?.*job=caption-created/);
  await expect(page.getByRole("heading", { name: "Studio captions", exact: true })).toBeVisible();
  expect(submissions).toHaveLength(2);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("caption job creation rejects a successful response without a task id", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  await page.route(
    (url) => url.pathname === "/api/captioning/jobs",
    (route) => route.request().method() === "POST"
      ? route.fulfill({ status: 202, json: { ok: true } })
      : route.fallback(),
  );

  await prepareSource(page);
  page.on("dialog", (dialog) => void dialog.accept());
  await page.getByRole("button", { name: "开始打标 (1)", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("打标任务响应缺少任务 ID");
  await expect(page).toHaveURL(/\/next\/captioning\?dataset=.*&subset=0$/);
  await expect(page.getByRole("button", { name: "开始打标 (1)", exact: true })).toBeEnabled();
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});
