import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

test("captioning assets and dictionary errors recover only after an explicit refresh", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let assetAttempts = 0;
  let dictionaryAttempts = 0;
  let recovered = false;
  const reply = async (route: import("@playwright/test").Route, kind: "assets" | "dictionary") => {
    const attempts = kind === "assets" ? ++assetAttempts : ++dictionaryAttempts;
    if (!recovered && attempts <= 3) {
      return route.fulfill({
        status: 503,
        contentType: "application/json",
        body: JSON.stringify({ error: kind === "assets" ? "模型资源暂不可读" : "标签词典暂不可读" }),
      });
    }
    if (kind === "assets") {
      return route.fulfill({ json: {
        assets: [{
          id: "fixture-asset",
          label: "Local caption model",
          repo_id: "fixture/model",
          license: "Apache 2.0",
          state: "installed",
          installed: true,
          total_size: 400000000,
          requires_auth: false,
          auth_configured: true,
          auth_hint: "",
        }],
        downloads: [],
      } });
    }
    return route.fulfill({ json: {
      installed: false,
      state: "missing",
      source_name: "Tag dictionary",
      entry_count: 0,
      download_size: 40000000,
    } });
  };
  await page.route("**/api/captioning/model-assets**", (route) => {
    return reply(route, "assets");
  });
  await page.route("**/api/captioning/tag-dictionary**", (route) => {
    return reply(route, "dictionary");
  });

  await page.goto("/next/captioning/assets");
  await expect(page.getByRole("heading", { name: "本地资源", exact: true })).toBeVisible();
  await expect(page.getByRole("alert")).toContainText("模型资源暂不可读");
  await expect(page.getByText("Local caption model", { exact: true })).toHaveCount(0);
  await expect.poll(() => assetAttempts).toBe(3);
  await expect.poll(() => dictionaryAttempts).toBe(3);
  await page.waitForTimeout(1200);
  expect(assetAttempts).toBe(3);
  expect(dictionaryAttempts).toBe(3);

  recovered = true;
  await page.getByRole("button", { name: "刷新状态", exact: true }).click();
  await expect(page.getByText("Local caption model", { exact: true })).toBeVisible();
  await expect(page.getByText("Tag dictionary", { exact: true })).toBeVisible();
  expect(assetAttempts).toBe(4);
  expect(dictionaryAttempts).toBe(4);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});
