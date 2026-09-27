import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

function profilesResponse(name = "Studio captions", config: Record<string, unknown> = {}) {
  return {
    active_profile_id: "provider-a",
    profiles: [{
      id: "provider-a",
      name,
      provider: "openai_compatible",
      kind: "external",
      status: "可用",
      available: true,
      api_key_configured: true,
      api_key_hint: "已配置 API Key",
      config: { base_url: "https://example.invalid/v1", model: "fixture-model", ...config },
    }],
    provider_types: [{ id: "openai_compatible", label: "OpenAI Compatible", kind: "external" }],
  };
}

test("caption profile editing never renders or resubmits a server-provided secret", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let submitted: Record<string, unknown> | undefined;
  const serverSecret = "server-only-caption-secret";
  await page.route(
    (url) => url.pathname === "/api/captioning/profiles" || url.pathname === "/api/captioning/profiles/provider-a",
    async (route) => {
      const request = route.request();
      if (request.method() === "GET" && new URL(request.url()).pathname === "/api/captioning/profiles") {
        return route.fulfill({ json: profilesResponse("Studio captions", { api_key: serverSecret }) });
      }
      if (request.method() === "PUT") {
        submitted = request.postDataJSON() as Record<string, unknown>;
        return route.fulfill({ json: profilesResponse() });
      }
      return route.fallback();
    },
  );

  await page.goto("/next/captioning/providers");
  await page.getByRole("button", { name: "编辑", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "编辑接入预设" });
  await expect(dialog).toBeVisible();
  await expect(dialog.locator("pre.config-json")).not.toContainText(serverSecret);
  await expect(dialog.locator('input[type="password"]')).toHaveValue("");
  await dialog.getByRole("button", { name: "保存接入", exact: true }).click();
  await expect(dialog).toHaveCount(0);

  expect(submitted).toBeDefined();
  expect(JSON.stringify(submitted)).not.toContain(serverSecret);
  expect(submitted).not.toHaveProperty("api_key");
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("caption profile save keeps the draft after a server error and retries explicitly", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  const submissions: Record<string, unknown>[] = [];
  await page.route(
    (url) => url.pathname === "/api/captioning/profiles/provider-a",
    async (route) => {
      submissions.push(route.request().postDataJSON() as Record<string, unknown>);
      if (submissions.length === 1) {
        return route.fulfill({ status: 503, json: { error: "接入预设保存暂不可用" } });
      }
      return route.fulfill({ json: profilesResponse("Studio captions revised") });
    },
  );

  await page.goto("/next/captioning/providers");
  await page.getByRole("button", { name: "编辑", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "编辑接入预设" });
  await dialog.getByLabel("名称", { exact: true }).fill("Studio captions revised");
  await dialog.locator('input[type="password"]').fill("new-caption-secret");
  await dialog.getByRole("button", { name: "保存接入", exact: true }).click();

  await expect(dialog.getByRole("alert")).toContainText("接入预设保存暂不可用");
  await expect(dialog.getByLabel("名称", { exact: true })).toHaveValue("Studio captions revised");
  await expect(dialog.locator('input[type="password"]')).toHaveValue("new-caption-secret");
  await page.waitForTimeout(1200);
  expect(submissions).toHaveLength(1);

  await dialog.getByRole("button", { name: "保存接入", exact: true }).click();
  await expect(dialog).toHaveCount(0);
  await expect(page.getByRole("heading", { name: /Studio captions revised · 当前/ })).toBeVisible();
  expect(submissions).toHaveLength(2);
  expect(submissions.every((body) => body.api_key === "new-caption-secret")).toBe(true);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});
