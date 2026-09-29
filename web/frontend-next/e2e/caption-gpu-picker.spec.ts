import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

for (const width of [1280, 390]) {
  test(`local caption profile GPU picker preserves selection contracts at ${width}px`, async ({
    page,
  }, info) => {
    await page.setViewportSize({ width, height: 900 });
    const mocks = await mockWorkspace(page);
    let saved: Record<string, unknown> | undefined;
    let gpuRequests = 0;
    let config: Record<string, unknown> = {
      asset_id: "wd14-eva02-large-v3",
      device: "cuda",
      gpu_index: 8,
      batch_size: 8,
    };
    const profile = () => ({
      id: "local-profile",
      name: "Local captions",
      provider: "wd14",
      kind: "local",
      available: true,
      status: "ready",
      config,
      api_key_hint: "",
      api_key_configured: false,
    });
    await page.route(
      (url) => url.pathname.startsWith("/api/captioning/profiles"),
      async (route) => {
        if (route.request().method() === "PUT") {
          saved = route.request().postDataJSON() as Record<string, unknown>;
          config = (saved.config as Record<string, unknown>) || {};
        }
        const current = profile();
        return route.fulfill({
          json: {
            ok: true,
            active_profile_id: "local-profile",
            profiles: [current],
            profile: current,
            provider_types: [
              { id: "wd14", label: "WD14", kind: "local" },
              {
                id: "openai_compatible",
                label: "OpenAI Compatible",
                kind: "external",
              },
            ],
          },
        });
      },
    );
    await page.route(
      (url) => url.pathname === "/api/training/gpus",
      async (route) => {
        gpuRequests += 1;
        if (gpuRequests >= 3)
          await new Promise((resolve) => setTimeout(resolve, 250));
        return route.fulfill({
          json: {
            gpus:
              gpuRequests >= 3
                ? [{ index: 0, name: "GPU Alpha", memory_total_gb: 10 }]
                : [
                    { index: 0, name: "GPU Alpha", memory_total_gb: 10 },
                    {
                      index: 2,
                      label: "GPU 2 · 24 GB",
                      name: "GPU Beta",
                      memory_total_gb: 24,
                    },
                  ],
          },
        });
      },
    );

    await page.goto("/next/captioning/providers");
    await page.getByRole("button", { name: "编辑", exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "编辑接入预设" });
    const gpu = dialog.getByRole("combobox", { name: "GPU" });
    await expect.poll(() => gpuRequests).toBe(1);
    await expect(gpu).toContainText("GPU 8（当前不可用）");
    await expect(
      dialog.getByRole("button", { name: "保存接入" }),
    ).toBeDisabled();
    await gpu.selectOption("2");
    await expect(
      dialog.getByRole("button", { name: "保存接入" }),
    ).toBeEnabled();
    const bounds = await page.evaluate(() => ({
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth,
    }));
    expect(bounds.scrollWidth).toBeLessThanOrEqual(bounds.clientWidth);
    await page.screenshot({
      path: info.outputPath("gpu-picker.png"),
      fullPage: true,
    });
    await dialog.getByRole("button", { name: "保存接入" }).click();
    await expect(dialog).toHaveCount(0);
    expect((saved?.config as Record<string, unknown>).gpu_index).toBe(2);
    await expect.poll(() => gpuRequests).toBe(2);

    await page.getByRole("button", { name: "编辑", exact: true }).click();
    const nextDialog = page.getByRole("dialog", { name: "编辑接入预设" });
    await expect.poll(() => gpuRequests).toBe(3);
    await expect(
      nextDialog.getByRole("button", { name: "保存接入" }),
    ).toBeDisabled();
    await expect(
      nextDialog.getByRole("combobox", { name: "GPU" }),
    ).toContainText("GPU 2（当前不可用）");
    await nextDialog.getByLabel("执行设备").selectOption("cpu");
    await nextDialog.getByRole("button", { name: "保存接入" }).click();
    await expect(nextDialog).toHaveCount(0);
    const savedConfig = saved?.config as Record<string, unknown>;
    expect(savedConfig.device).toBe("cpu");
    expect(savedConfig).not.toHaveProperty("gpu_index");
    expect(gpuRequests).toBe(3);
    expect(mocks.writes).toEqual([]);
    expect(mocks.unhandled).toEqual([]);
  });
}

test("local caption CUDA profile cannot be saved without an available GPU", async ({
  page,
}) => {
  const mocks = await mockWorkspace(page);
  const localProfile = {
    id: "local-profile",
    name: "Local captions",
    provider: "wd14",
    kind: "local",
    available: true,
    status: "ready",
    config: { asset_id: "wd14-eva02-large-v3", device: "cuda" },
    api_key_hint: "",
    api_key_configured: false,
  };
  await page.route(
    (url) => url.pathname.startsWith("/api/captioning/profiles"),
    (route) =>
      route.fulfill({
        json: {
          ok: true,
          active_profile_id: "local-profile",
          profiles: [localProfile],
          provider_types: [{ id: "wd14", label: "WD14", kind: "local" }],
        },
      }),
  );
  await page.route(
    (url) => url.pathname === "/api/training/gpus",
    (route) => route.fulfill({ json: { gpus: [] } }),
  );

  await page.goto("/next/captioning/providers");
  await page.getByRole("button", { name: "编辑", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "编辑接入预设" });
  await expect(dialog.getByRole("status")).toContainText("未检测到可用 GPU");
  await expect(dialog.getByRole("button", { name: "保存接入" })).toBeDisabled();
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("local caption GPU inventory can be retried after a failed request", async ({
  page,
}) => {
  await mockWorkspace(page);
  let gpuRequests = 0;
  await page.route(
    (url) => url.pathname.startsWith("/api/captioning/profiles"),
    (route) =>
      route.fulfill({
        json: {
          ok: true,
          active_profile_id: "local-profile",
          profiles: [
            {
              id: "local-profile",
              name: "Local captions",
              provider: "wd14",
              kind: "local",
              available: true,
              status: "ready",
              config: { asset_id: "wd14-eva02-large-v3", device: "cuda" },
              api_key_hint: "",
              api_key_configured: false,
            },
          ],
          provider_types: [{ id: "wd14", label: "WD14", kind: "local" }],
        },
      }),
  );
  await page.route(
    (url) => url.pathname === "/api/training/gpus",
    (route) => {
      gpuRequests += 1;
      return gpuRequests === 1
        ? route.fulfill({
            status: 503,
            json: { error: "temporarily unavailable" },
          })
        : route.fulfill({
            json: {
              gpus: [{ index: 1, name: "GPU One", memory_total_gb: 10 }],
            },
          });
    },
  );

  await page.goto("/next/captioning/providers");
  await page.getByRole("button", { name: "编辑", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "编辑接入预设" });
  await expect(dialog.getByRole("status")).toContainText("无法读取 GPU 列表");
  await dialog.getByRole("button", { name: "重试" }).click();
  await expect(dialog.getByRole("combobox", { name: "GPU" })).toContainText(
    "GPU One",
  );
  expect(gpuRequests).toBe(2);
});

test("advanced local profile config cannot bypass CUDA GPU validation", async ({
  page,
}) => {
  await mockWorkspace(page);
  let writes = 0;
  const profile = {
    id: "local-profile",
    name: "Local captions",
    provider: "wd14",
    kind: "local",
    available: true,
    status: "ready",
    config: { asset_id: "wd14-eva02-large-v3", device: "cuda", gpu_index: 0 },
    api_key_hint: "",
    api_key_configured: false,
  };
  await page.route(
    (url) => url.pathname.startsWith("/api/captioning/profiles"),
    (route) => {
      if (route.request().method() === "PUT") writes += 1;
      return route.fulfill({
        json: {
          ok: true,
          active_profile_id: "local-profile",
          profiles: [profile],
          profile,
          provider_types: [{ id: "wd14", label: "WD14", kind: "local" }],
        },
      });
    },
  );
  let gpuRequests = 0;
  await page.route(
    (url) => url.pathname === "/api/training/gpus",
    (route) => {
      gpuRequests += 1;
      return route.fulfill({
        json: {
          gpus:
            gpuRequests === 1
              ? [
                  { index: 0, name: "GPU Zero", memory_total_gb: 10 },
                  { index: 1, name: "GPU One", memory_total_gb: 10 },
                ]
              : [{ index: 0, name: "GPU Zero", memory_total_gb: 10 }],
        },
      });
    },
  );

  await page.goto("/next/captioning/providers");
  await page.getByRole("button", { name: "编辑", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "编辑接入预设" });
  await dialog.getByText("高级配置").click();
  await dialog
    .getByLabel("配置覆盖 (JSON)")
    .fill('{"device":"cuda","gpu_index":1}');
  await dialog.getByRole("button", { name: "保存接入" }).click();
  await expect(dialog.getByRole("alert")).toContainText("当前可用的 GPU");
  expect(gpuRequests).toBe(2);
  expect(writes).toBe(0);
});

test("advanced local profile preview controls CUDA validation and reports invalid JSON", async ({
  page,
}) => {
  await mockWorkspace(page);
  let saved: Record<string, unknown> | undefined;
  const profile = {
    id: "local-profile",
    name: "Local captions",
    provider: "wd14",
    kind: "local",
    available: true,
    status: "ready",
    config: { asset_id: "wd14-eva02-large-v3", device: "cuda", gpu_index: 99 },
    api_key_hint: "",
    api_key_configured: false,
  };
  await page.route(
    (url) => url.pathname.startsWith("/api/captioning/profiles"),
    (route) => {
      if (route.request().method() === "PUT") {
        saved = route.request().postDataJSON() as Record<string, unknown>;
      }
      return route.fulfill({
        json: {
          ok: true,
          active_profile_id: "local-profile",
          profiles: [profile],
          profile,
          provider_types: [{ id: "wd14", label: "WD14", kind: "local" }],
        },
      });
    },
  );
  await page.route(
    (url) => url.pathname === "/api/training/gpus",
    (route) =>
      route.fulfill({
        json: { gpus: [{ index: 2, name: "GPU Two", memory_total_gb: 10 }] },
      }),
  );

  await page.goto("/next/captioning/providers");
  await page.getByRole("button", { name: "编辑", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "编辑接入预设" });
  await expect
    .poll(async () =>
      dialog.getByRole("button", { name: "保存接入" }).isEnabled(),
    )
    .toBe(false);
  await dialog.getByText("高级配置").click();
  const save = dialog.getByRole("button", { name: "保存接入" });
  const override = dialog.getByLabel("配置覆盖 (JSON)");

  await override.fill('{"device":"cuda","gpu_index":2}');
  await expect(save).toBeEnabled();
  await override.fill('{"device":"cpu"}');
  await expect(save).toBeEnabled();
  await save.click();
  await expect(dialog).toHaveCount(0);
  const savedConfig = saved?.config as Record<string, unknown>;
  expect(savedConfig.device).toBe("cpu");
  expect(savedConfig).not.toHaveProperty("gpu_index");
});

test("visual local profile fields take precedence over advanced JSON overrides", async ({
  page,
}) => {
  await mockWorkspace(page);
  let config: Record<string, unknown> = {
    asset_id: "wd14-eva02-large-v3",
    device: "cuda",
    gpu_index: 0,
  };
  let saved: Record<string, unknown> | undefined;
  const profile = () => ({
    id: "local-profile",
    name: "Local captions",
    provider: "wd14",
    kind: "local",
    available: true,
    status: "ready",
    config,
    api_key_hint: "",
    api_key_configured: false,
  });
  await page.route(
    (url) => url.pathname.startsWith("/api/captioning/profiles"),
    (route) => {
      if (route.request().method() === "PUT") {
        saved = route.request().postDataJSON() as Record<string, unknown>;
        config = (saved.config as Record<string, unknown>) || {};
      }
      const current = profile();
      return route.fulfill({
        json: {
          ok: true,
          active_profile_id: "local-profile",
          profiles: [current],
          profile: current,
          provider_types: [{ id: "wd14", label: "WD14", kind: "local" }],
        },
      });
    },
  );
  await page.route(
    (url) => url.pathname === "/api/training/gpus",
    (route) =>
      route.fulfill({
        json: {
          gpus: [
            { index: 0, name: "GPU Zero", memory_total_gb: 10 },
            { index: 1, name: "GPU One", memory_total_gb: 10 },
          ],
        },
      }),
  );

  await page.goto("/next/captioning/providers");
  const openEditor = async () => {
    await page.getByRole("button", { name: "编辑", exact: true }).click();
    return page.getByRole("dialog", { name: "编辑接入预设" });
  };
  let dialog = await openEditor();
  await dialog.getByText("高级配置").click();
  await dialog
    .getByLabel("配置覆盖 (JSON)")
    .fill('{"device":"cuda","gpu_index":1,"batch_size":4}');
  await dialog.getByRole("combobox", { name: "GPU" }).selectOption("0");
  await dialog.getByRole("button", { name: "保存接入" }).click();
  await expect(dialog).toHaveCount(0);
  expect((saved?.config as Record<string, unknown>).gpu_index).toBe(0);
  expect((saved?.config as Record<string, unknown>).batch_size).toBe(4);

  dialog = await openEditor();
  await dialog.getByText("高级配置").click();
  await dialog.getByLabel("配置覆盖 (JSON)").fill('{"device":"cpu"}');
  await dialog.getByLabel("执行设备").selectOption("cuda");
  await dialog.getByRole("combobox", { name: "GPU" }).selectOption("1");
  await dialog.getByRole("button", { name: "保存接入" }).click();
  await expect(dialog).toHaveCount(0);
  const finalConfig = saved?.config as Record<string, unknown>;
  expect(finalConfig.device).toBe("cuda");
  expect(finalConfig.gpu_index).toBe(1);
});

test("invalid advanced local profile JSON can be submitted to show its error", async ({
  page,
}) => {
  await mockWorkspace(page);
  await page.route(
    (url) => url.pathname.startsWith("/api/captioning/profiles"),
    (route) =>
      route.fulfill({
        json: {
          ok: true,
          profiles: [
            {
              id: "local-profile",
              name: "Local captions",
              provider: "wd14",
              kind: "local",
              available: true,
              status: "ready",
              config: { device: "cuda", gpu_index: 99 },
              api_key_hint: "",
              api_key_configured: false,
            },
          ],
          provider_types: [{ id: "wd14", label: "WD14", kind: "local" }],
        },
      }),
  );
  await page.route(
    (url) => url.pathname === "/api/training/gpus",
    (route) => route.fulfill({ json: { gpus: [] } }),
  );
  await page.goto("/next/captioning/providers");
  await page.getByRole("button", { name: "编辑", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "编辑接入预设" });
  await dialog.getByText("高级配置").click();
  await dialog.getByLabel("配置覆盖 (JSON)").fill("{");
  const save = dialog.getByRole("button", { name: "保存接入" });
  await expect(save).toBeEnabled();
  await save.click();
  await expect(dialog.getByRole("alert")).toContainText("JSON 格式无效");
});
