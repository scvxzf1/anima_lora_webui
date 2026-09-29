import { expect, test } from "@playwright/test";
import { configFile, mockWorkspace } from "./fixtures";

test("training group queues its ordered files only after explicit confirmation", async ({
  page,
}, info) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const mocks = await mockWorkspace(page);
  const requests: Record<string, unknown>[] = [];
  await page.route("**/api/training/queue/batch/start", (route) => {
    requests.push(route.request().postDataJSON());
    return route.fulfill({
      json: { ok: true, message: "已将 1 个配置加入训练队列", queued_count: 1 },
    });
  });
  await page.goto("/next/training");
  await page.getByRole("button", { name: "展开配置库" }).click();
  await page
    .getByRole("button", { name: "整组加入队列 Studio presets" })
    .click();
  const dialog = page.getByRole("dialog", { name: "批量加入训练队列" });
  await expect(dialog).toContainText(configFile.path);
  await expect(dialog).toContainText("队列保持暂停");
  expect(
    await dialog.evaluate((node) => node.scrollWidth <= node.clientWidth),
  ).toBe(true);
  await page.screenshot({
    path: info.outputPath("training-group-queue-mobile.png"),
  });
  const submit = dialog.getByRole("button", { name: "确认加入队列" });
  await expect(submit).toBeDisabled();
  expect(requests).toEqual([]);
  await dialog.getByRole("checkbox", { name: /确认批量预检/ }).check();
  await submit.click();
  await expect(dialog.getByRole("status")).toContainText(
    "已将 1 个配置加入训练队列",
  );
  expect(requests).toHaveLength(1);
  expect(requests[0]).toMatchObject({
    preset: "default",
    gpu_whitelist: ["0"],
    start_paused: true,
    items: [
      {
        variant: "lora",
        methods_subdir: "imported",
        config_file: configFile.path,
        confirm_preprocess: true,
      },
    ],
  });
  await expect(
    dialog.getByRole("link", { name: "查看训练队列" }),
  ).toBeVisible();
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("training group preserves file order and the selected hardware preset", async ({
  page,
}) => {
  const mocks = await mockWorkspace(page);
  await page.route("**/api/training/gpus", (route) =>
    route.fulfill({
      json: {
        gpus: [
          { index: 0, name: "Fixture GPU 0", memory_total_gb: 24 },
          { index: 1, name: "Fixture GPU 1", memory_total_gb: 24 },
        ],
      },
    }),
  );
  const second = {
    ...configFile,
    path: "configs/imported/second.toml",
    filename: "second.toml",
    label: "Second",
    method: "loha",
  };
  await page.route("**/api/config/file-groups?kind=training", (route) =>
    route.fulfill({
      json: [
        {
          id: "imported",
          label: "Studio presets",
          files: [configFile, second],
        },
      ],
    }),
  );
  const requests: Record<string, unknown>[] = [];
  await page.route("**/api/training/queue/batch/start", (route) => {
    requests.push(route.request().postDataJSON());
    return route.fulfill({
      json: { ok: true, message: "已将 2 个配置加入训练队列", queued_count: 2 },
    });
  });

  await page.goto("/next/training");
  await page.getByLabel("当前硬件预设").selectOption("low_vram");
  await page.getByRole("radio", { name: "GPU 1 · Fixture GPU 1" }).check();
  await page
    .getByRole("button", { name: "整组加入队列 Studio presets" })
    .click();
  const dialog = page.getByRole("dialog", { name: "批量加入训练队列" });
  await dialog.getByRole("checkbox", { name: /确认批量预检/ }).check();
  await dialog.getByRole("button", { name: "确认加入队列" }).click();

  await expect(dialog.getByRole("status")).toContainText("已将 2 个配置");
  expect(requests).toHaveLength(1);
  expect(requests[0]).toMatchObject({
    preset: "low_vram",
    gpu_whitelist: ["1"],
    start_paused: true,
    items: [
      {
        variant: "lora",
        preset: "low_vram",
        methods_subdir: "imported",
        config_file: configFile.path,
        confirm_preprocess: true,
      },
      {
        variant: "loha",
        preset: "low_vram",
        methods_subdir: "imported",
        config_file: second.path,
        confirm_preprocess: true,
      },
    ],
  });
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("training group cannot submit when the saved GPU is unavailable", async ({
  page,
}) => {
  const mocks = await mockWorkspace(page);
  await page.addInitScript(() => {
    localStorage.setItem(
      "dragon-next.training-devices.v1",
      JSON.stringify({
        mode: "single",
        devices: [{ id: "9", name: "Missing GPU" }],
      }),
    );
  });
  const requests: Record<string, unknown>[] = [];
  await page.route("**/api/training/queue/batch/start", (route) => {
    requests.push(route.request().postDataJSON());
    return route.fulfill({ json: { ok: true } });
  });

  await page.goto("/next/training");
  await page
    .getByRole("button", { name: "整组加入队列 Studio presets" })
    .click();
  const dialog = page.getByRole("dialog", { name: "批量加入训练队列" });
  await expect(dialog.getByRole("alert")).toContainText("已选设备不可用");
  await dialog.getByRole("checkbox", { name: /确认批量预检/ }).check();
  await expect(
    dialog.getByRole("button", { name: "确认加入队列" }),
  ).toBeDisabled();
  expect(requests).toEqual([]);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("training group reports a partial failure without resubmitting", async ({
  page,
}) => {
  const mocks = await mockWorkspace(page);
  const second = {
    ...configFile,
    path: "configs/imported/second.toml",
    filename: "second.toml",
    label: "Second",
  };
  await page.route("**/api/config/file-groups?kind=training", (route) =>
    route.fulfill({
      json: [
        {
          id: "imported",
          label: "Studio presets",
          files: [configFile, second],
        },
      ],
    }),
  );
  let calls = 0;
  await page.route("**/api/training/queue/batch/start", (route) => {
    calls += 1;
    return route.fulfill({
      json: {
        ok: false,
        message: "已加入 1 个配置，批量加入在第 2 项停止",
        queued_count: 1,
        failures: [
          { index: 1, config_file: second.path, error: "配置无法冻结" },
        ],
      },
    });
  });
  await page.goto("/next/training");
  await page
    .getByRole("button", { name: "整组加入队列 Studio presets" })
    .click();
  const dialog = page.getByRole("dialog", { name: "批量加入训练队列" });
  await dialog.getByRole("checkbox", { name: /确认批量预检/ }).check();
  await dialog.getByRole("button", { name: "确认加入队列" }).click();
  await expect(dialog.getByRole("alert")).toContainText("已加入 1 个配置");
  await expect(dialog.getByRole("alert")).toContainText(second.path);
  await expect(dialog.getByRole("alert")).toContainText("配置无法冻结");
  await expect(
    dialog.getByRole("button", { name: "确认加入队列" }),
  ).toBeDisabled();
  expect(calls).toBe(1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("training group treats a lost batch response as unknown and never retries", async ({
  page,
}) => {
  const mocks = await mockWorkspace(page);
  let calls = 0;
  await page.route("**/api/training/queue/batch/start", (route) => {
    calls += 1;
    return route.abort();
  });
  await page.goto("/next/training");
  await page
    .getByRole("button", { name: "整组加入队列 Studio presets" })
    .click();
  const dialog = page.getByRole("dialog", { name: "批量加入训练队列" });
  await dialog.getByRole("checkbox", { name: /确认批量预检/ }).check();
  await dialog.getByRole("button", { name: "确认加入队列" }).click();
  await expect(dialog.getByRole("alert")).toContainText("结果可能未知");
  await expect(dialog.getByRole("alert")).toContainText("先核对训练队列");
  await expect(
    dialog.getByRole("button", { name: "确认加入队列" }),
  ).toBeDisabled();
  expect(calls).toBe(1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});
