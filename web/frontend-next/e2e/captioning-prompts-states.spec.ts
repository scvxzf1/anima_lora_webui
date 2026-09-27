import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

async function installPromptFailure(
  page: import("@playwright/test").Page,
  error: string,
) {
  let attempts = 0;
  let recovered = false;
  await page.route("**/api/captioning/prompt-presets", (route) => {
    attempts += 1;
    if (!recovered)
      return route.fulfill({
        status: 503,
        contentType: "application/json",
        body: JSON.stringify({ error }),
      });
    return route.fulfill({
      json: {
        presets: [
          {
            id: "prompt",
            name: "Studio captions",
            system_prompt: "",
            user_prompt: "Describe the image.",
            builtin: true,
          },
        ],
      },
    });
  });
  return {
    attempts: () => attempts,
    recover: () => {
      recovered = true;
    },
  };
}

test("captioning empty job library keeps the new-image workflow available", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let jobReads = 0;
  await page.route("**/api/captioning/jobs", (route) => {
    jobReads += 1;
    return route.fulfill({ json: { jobs: [] } });
  });

  await page.goto("/next/captioning");
  await expect(page.getByRole("heading", { name: "打标工作台", exact: true })).toBeVisible();
  await expect(page.getByText("暂无打标任务", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "图片来源", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "新任务", exact: true })).toBeVisible();
  await expect(page.getByRole("alert")).toHaveCount(0);
  expect(jobReads).toBeGreaterThan(0);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("captioning loading gate keeps provider and commit actions unavailable", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  const gates = [
    "/api/captioning/profiles",
    "/api/captioning/jobs",
    "/api/captioning/prompt-presets",
  ].map((path) => {
    let announceStarted = () => {};
    let releaseResponse = () => {};
    const started = new Promise<void>(resolve => { announceStarted = resolve; });
    const responseGate = new Promise<void>(resolve => { releaseResponse = resolve; });
    return { path, announceStarted, releaseResponse, started, responseGate };
  });
  for (const gate of gates) {
    await page.route(url => url.pathname === gate.path, async route => {
      if (route.request().method() !== "GET") return route.fallback();
      gate.announceStarted();
      await gate.responseGate;
      return route.fallback();
    });
  }

  await page.goto("/next/captioning");
  await Promise.all(gates.map(gate => gate.started));
  await expect(page.getByRole("heading", { name: "打标工作台", exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "打标任务", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "开始打标 (0)", exact: true })).toBeDisabled();
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);

  for (const gate of gates) gate.releaseResponse();
  await expect(page.locator("label").filter({ hasText: "接入预设" })).toContainText("Studio captions");
  await expect(page.getByRole("button", { name: "开始打标 (0)", exact: true })).toBeDisabled();
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("caption provider library read errors stop polling and recover explicitly", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let attempts = 0;
  let recovered = false;
  await page.route("**/api/captioning/profiles", (route) => {
    attempts += 1;
    if (!recovered) {
      return route.fulfill({ status: 503, json: { error: "接入预设暂不可读" } });
    }
    return route.fulfill({ json: {
      active_profile_id: "provider-a",
      profiles: [{
        id: "provider-a",
        name: "Studio captions",
        provider: "openai_compatible",
        kind: "external",
        status: "可用",
        available: true,
        api_key_configured: true,
        api_key_hint: "****",
        config: { base_url: "https://example.invalid/v1" },
      }],
      provider_types: [{ id: "openai_compatible", label: "OpenAI Compatible", kind: "external" }],
    } });
  });

  await page.goto("/next/captioning/providers");
  await expect(page.getByRole("heading", { name: "接入预设", exact: true })).toBeVisible();
  const feedback = page.locator('[aria-label="接入预设读取状态"]');
  await expect(feedback.getByRole("alert")).toContainText("接入预设暂不可读");
  await expect(feedback.getByRole("button", { name: "重试接入预设", exact: true })).toBeVisible();
  const failedAttempts = attempts;
  await page.waitForTimeout(1200);
  expect(attempts).toBe(failedAttempts);

  recovered = true;
  await feedback.getByRole("button", { name: "重试接入预设", exact: true }).click();
  await expect(page.getByRole("heading", { name: /Studio captions/ })).toBeVisible();
  expect(attempts).toBe(failedAttempts + 1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("caption job library read errors stop polling and recover explicitly", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let attempts = 0;
  let recovered = false;
  await page.route("**/api/captioning/jobs", (route) => {
    attempts += 1;
    return recovered
      ? route.fulfill({ json: { jobs: [] } })
      : route.fulfill({ status: 503, json: { error: "打标任务暂不可读" } });
  });

  await page.goto("/next/captioning");
  await expect(page.getByRole("heading", { name: "图片来源", exact: true })).toBeVisible();
  const feedback = page.locator('[aria-label="打标任务读取状态"]');
  await expect(feedback.getByRole("alert")).toContainText("打标任务暂不可读");
  await expect(feedback.getByRole("button", { name: "重试打标任务", exact: true })).toBeVisible();
  const failedAttempts = attempts;
  await page.waitForTimeout(1200);
  expect(attempts).toBe(failedAttempts);

  recovered = true;
  await feedback.getByRole("button", { name: "重试打标任务", exact: true }).click();
  await expect(page.getByText("暂无打标任务", { exact: true })).toBeVisible();
  expect(attempts).toBe(failedAttempts + 1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("caption job detail read errors recover only after an explicit retry", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let attempts = 0;
  let recovered = false;
  await page.route("**/api/captioning/jobs/caption-1", (route) => {
    attempts += 1;
    if (!recovered) {
      return route.fulfill({ status: 503, json: { error: "打标任务详情暂不可读" } });
    }
    return route.fulfill({ json: {
      ok: true,
      job: {
        id: "caption-1",
        state: "done",
        profile_name: "Studio captions",
        profile_id: "provider-a",
        dataset_file: "configs/datasets/studio.toml",
        dataset_index: 0,
        total: 0,
        completed: 0,
        failed: 0,
        items: [],
      },
    } });
  });

  await page.goto("/next/captioning?job=caption-1");
  const feedback = page.locator('[aria-label="打标任务读取状态"]');
  await expect(feedback.getByRole("alert")).toContainText("打标任务详情暂不可读");
  await expect(feedback.getByRole("button", { name: "重试打标任务", exact: true })).toBeVisible();
  const failedAttempts = attempts;
  await page.waitForTimeout(1200);
  expect(attempts).toBe(failedAttempts);

  recovered = true;
  await feedback.getByRole("button", { name: "重试打标任务", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Studio captions", exact: true })).toBeVisible();
  await expect(page.getByRole("alert")).toHaveCount(0);
  expect(attempts).toBe(failedAttempts + 1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("prompt preset errors stop retrying and recover explicitly on the presets page", async ({
  page,
}) => {
  const mocks = await mockWorkspace(page);
  const prompt = await installPromptFailure(page, "提示词预设暂不可读");
  await page.goto("/next/captioning/prompts");

  await expect(
    page.getByRole("heading", { name: "提示词预设", exact: true }),
  ).toBeVisible();
  await expect(page.getByRole("alert")).toContainText("提示词预设暂不可读");
  await expect(
    page.getByRole("button", { name: "重试提示词", exact: true }),
  ).toBeVisible();
  const failedAttempts = prompt.attempts();
  await page.waitForTimeout(1200);
  expect(prompt.attempts()).toBe(failedAttempts);

  prompt.recover();
  await page.getByRole("button", { name: "重试提示词", exact: true }).click();
  await expect(
    page.getByText("Studio captions", { exact: true }),
  ).toBeVisible();
  expect(prompt.attempts()).toBe(failedAttempts + 1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("prompt preset errors expose the same explicit recovery on the source page", async ({
  page,
}) => {
  const mocks = await mockWorkspace(page);
  const prompt = await installPromptFailure(page, "提示词预设暂不可读");
  await page.goto("/next/captioning");

  await expect(
    page.getByRole("heading", { name: "图片来源", exact: true }),
  ).toBeVisible();
  await expect(page.getByRole("alert")).toContainText("提示词预设暂不可读");
  const failedAttempts = prompt.attempts();
  await page.waitForTimeout(1200);
  expect(prompt.attempts()).toBe(failedAttempts);

  prompt.recover();
  await page.getByRole("button", { name: "重试提示词", exact: true }).click();
  await expect(
    page
      .getByLabel("提示词预设")
      .getByRole("option", { name: "Studio captions", exact: true }),
  ).toHaveCount(1);
  expect(prompt.attempts()).toBe(failedAttempts + 1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});
