import { expect, test, type Page, type Route } from "@playwright/test";
import { configFile, mockWorkspace } from "./fixtures";

type RootMaskFixture = {
  activeRoot: string;
  settingsWrites: unknown[];
  presetRoots: string[];
  maskPageRoots: string[];
  maskPageReads: Array<{ root: string; offset: number }>;
  maskImageReads: Array<{ root: string; image: string | null }>;
  externalPageReadStarted: Promise<void>;
  announceExternalPageRead: () => void;
  externalPageReadGate: Promise<void>;
  releaseExternalPageRead: () => void;
  externalImageReadStarted: Promise<void>;
  announceExternalImageRead: () => void;
  externalImageReadGate: Promise<void>;
  releaseExternalImageRead: () => void;
};

type RootMaskAssets = { imageUrl: string; maskUrl: string };

function rootSettingsPayload(activeRoot: string) {
  return {
    output_root: "output/runs",
    ui_scale: 100,
    tagging_max_retained_jobs: 40,
    path_overrides: {
      configs_root: activeRoot === "configs" ? "" : activeRoot,
      history_root: "",
      queue_root: "",
    },
    effective_paths: { configs_root: `/workspace/${activeRoot}` },
    defaults: { output_root: "output/runs", ui_scale: 100 },
  };
}

function rootDatasetLibrary(activeRoot: string, datasetFile: string) {
  const item = {
    path: datasetFile,
    filename: "studio.toml",
    label: `${activeRoot} dataset`,
    summary: { dataset_count: 1, repeat_total: 1 },
    locked: false,
  };
  return {
    ok: true,
    presets: [item],
    groups: [{ id: "studio", label: "Studio", files: [item] }],
  };
}

function rootPresetPayload(activeRoot: string, datasetFile: string) {
  const sourceDir = "images/studio";
  return {
    ok: true,
    file: datasetFile,
    name: `${activeRoot} studio`,
    content: "",
    datasets: [{
      source_dir: sourceDir,
      image_dir: `cache/${activeRoot}`,
      num_repeats: 1,
      settings: {},
    }],
    defaults: { resolution: 1024, batch_size: 1 },
    readonly: false,
    summary: {},
  };
}

function rootMaskPagePayload(
  activeRoot: string,
  datasetFile: string,
  imageUrl: string,
  offset = 0,
) {
  const sourceDir = "images/studio";
  const maskDir = `masks/${activeRoot}`;
  const total = 60;
  const returned = offset === 0 ? 48 : 12;
  const images = Array.from({ length: returned }, (_, position) => {
    const name = offset === 0
      ? position === 0 ? "photo.png" : `image-${position}.png`
      : `${activeRoot}-later-${position + 1}.png`;
    const file = `${sourceDir}/${name}`;
    return {
      file,
      name,
      url: imageUrl,
      thumbnail_url: imageUrl,
      has_mask: false,
      caption: {
        ok: true,
        file: file.replace(/\.png$/, ".txt"),
        extension: ".txt",
        source_mode: "txt",
        source_label: "TXT",
        detected_mode: "txt",
        format_label: "TXT",
        caption_count: 0,
        text: "",
        truncated: false,
        length: 0,
      },
    };
  });
  return {
    ok: true,
    file: datasetFile,
    dataset_index: 0,
    dataset_label: `${activeRoot} studio`,
    source: "source",
    source_label: "源图",
    directory: sourceDir,
    directory_exists: true,
    caption_extension: ".txt",
    prefer_json_caption: false,
    caption_source_mode: "txt",
    caption_source_label: "TXT",
    caption_summary: "",
    count: images.length,
    total,
    limit: 48,
    images,
    row: { source_dir: sourceDir, mask_mode: "external", mask_dir: maskDir },
    settings: {},
    message: "",
    mask_dir: maskDir,
    mask_mode: "external",
    config_revision: `config-rev-${activeRoot}`,
    readonly: false,
    offset,
    next_offset: offset + returned,
    has_more_before: offset > 0,
    has_more_after: offset + returned < total,
  };
}

function rootDatasetPreviewPayload(activeRoot: string, datasetFile: string, imageUrl: string) {
  const directory = `images/${activeRoot}/studio`;
  const caption = `${activeRoot} caption`;
  return {
    ok: true,
    file: datasetFile,
    dataset_index: 0,
    dataset_label: `${activeRoot} studio`,
    source: "source",
    source_label: "源图",
    directory,
    directory_exists: true,
    caption_extension: ".txt",
    prefer_json_caption: false,
    caption_source_mode: "txt",
    caption_source_label: "TXT",
    caption_summary: `${activeRoot} caption summary`,
    count: 1,
    total: 1,
    limit: 120,
    images: [{
      file: `${directory}/photo.png`,
      name: "photo.png",
      url: imageUrl,
      caption: {
        ok: true,
        file: `${directory}/photo.txt`,
        extension: ".txt",
        source_mode: "txt",
        source_label: "TXT",
        detected_mode: "txt",
        format_label: "TXT",
        caption_count: 1,
        text: caption,
        truncated: false,
        length: caption.length,
      },
    }],
    row: {
      source_dir: directory,
      image_dir: `cache/${activeRoot}`,
      num_repeats: 1,
      settings: {},
    },
    settings: { resolution: 1024, enable_bucket: true },
    message: "",
  };
}

function rootMaskImagePayload(activeRoot: string, assets: RootMaskAssets) {
  return {
    ok: true,
    revision: `mask-rev-${activeRoot}`,
    width: 64,
    height: 64,
    image_url: assets.imageUrl,
    mask_url: assets.maskUrl,
    has_mask: false,
    readonly: false,
    basis: "training",
    mask_dir: `masks/${activeRoot}`,
    mask_file: `masks/${activeRoot}/photo.png`,
  };
}

async function rootMaskRoute(
  route: Route,
  state: RootMaskFixture,
  datasetFile: string,
  assets: RootMaskAssets,
) {
  const url = new URL(route.request().url());
  const method = route.request().method();
  if (url.pathname === "/api/settings/global") {
    if (method === "GET")
      return route.fulfill({ json: rootSettingsPayload(state.activeRoot) });
    if (method === "PUT") {
      const body = route.request().postDataJSON() as { configs_root?: string };
      state.settingsWrites.push(body);
      state.activeRoot = body.configs_root || "configs";
      return route.fulfill({ json: {
        ...rootSettingsPayload(state.activeRoot),
        ok: true,
        requires_reload: true,
      } });
    }
    return route.fallback();
  }
  if (method !== "GET") return route.fallback();
  if (url.pathname === "/api/config/dataset-presets")
    return route.fulfill({ json: rootDatasetLibrary(state.activeRoot, datasetFile) });
  if (url.pathname === "/api/config/dataset-presets/read") {
    state.presetRoots.push(state.activeRoot);
    return route.fulfill({ json: rootPresetPayload(state.activeRoot, datasetFile) });
  }
  if (url.pathname === "/api/config/dataset-masks") {
    const offset = Number(url.searchParams.get("offset") || 0);
    state.maskPageRoots.push(state.activeRoot);
    state.maskPageReads.push({ root: state.activeRoot, offset });
    if (state.activeRoot === "external-configs") {
      state.announceExternalPageRead();
      await state.externalPageReadGate;
    }
    return route.fulfill({
      json: rootMaskPagePayload(state.activeRoot, datasetFile, assets.imageUrl, offset),
    });
  }
  if (url.pathname === "/api/config/dataset-masks/image") {
    const image = url.searchParams.get("image");
    state.maskImageReads.push({ root: state.activeRoot, image });
    if (state.activeRoot === "external-configs") {
      state.announceExternalImageRead();
      await state.externalImageReadGate;
    }
    return route.fulfill({ json: rootMaskImagePayload(state.activeRoot, assets) });
  }
  return route.fallback();
}

async function setupRootMaskFixture(page: Page, datasetFile: string) {
  let announceExternalPageRead = () => {};
  let releaseExternalPageRead = () => {};
  let announceExternalImageRead = () => {};
  let releaseExternalImageRead = () => {};
  const externalPageReadStarted = new Promise<void>((resolve) => {
    announceExternalPageRead = resolve;
  });
  const externalPageReadGate = new Promise<void>((resolve) => {
    releaseExternalPageRead = resolve;
  });
  const externalImageReadStarted = new Promise<void>((resolve) => {
    announceExternalImageRead = resolve;
  });
  const externalImageReadGate = new Promise<void>((resolve) => {
    releaseExternalImageRead = resolve;
  });
  const state: RootMaskFixture = {
    activeRoot: "configs",
    settingsWrites: [],
    presetRoots: [],
    maskPageRoots: [],
    maskPageReads: [],
    maskImageReads: [],
    externalPageReadStarted,
    announceExternalPageRead,
    externalPageReadGate,
    releaseExternalPageRead,
    externalImageReadStarted,
    announceExternalImageRead,
    externalImageReadGate,
    releaseExternalImageRead,
  };
  const assets = await page.evaluate(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 64;
    canvas.height = 64;
    const context = canvas.getContext("2d")!;
    context.fillStyle = "#587d72";
    context.fillRect(0, 0, 64, 64);
    const imageUrl = canvas.toDataURL("image/png");
    context.fillStyle = "#fff";
    context.fillRect(0, 0, 64, 64);
    return { imageUrl, maskUrl: canvas.toDataURL("image/png") };
  });
  await page.route(
    (url) => [
      "/api/settings/global",
      "/api/config/dataset-presets",
      "/api/config/dataset-presets/read",
      "/api/config/dataset-masks",
      "/api/config/dataset-masks/image",
    ].includes(url.pathname),
    (route) => rootMaskRoute(route, state, datasetFile, assets),
  );
  return state;
}

test("settings root switch locks duplicate saves and invalidates cached workspaces", async ({
  page,
}) => {
  const mocks = await mockWorkspace(page);
  let activeRoot = "configs";
  let activeHistoryRoot = "history";
  let activeQueueRoot = "queue";
  let finishSettingsWrite: (() => void) | undefined;
  const settingsWriteGate = new Promise<void>((resolve) => {
    finishSettingsWrite = resolve;
  });
  const settingsWrites: unknown[] = [];
  const datasetRoots: string[] = [];
  const modelRoots: string[] = [];
  const trainingRoots: string[] = [];
  const historyRoots: string[] = [];
  const queueRoots: string[] = [];

  await page.addInitScript(() => {
    const target = window as unknown as { __s2ConfirmMessages: string[] };
    target.__s2ConfirmMessages = [];
    window.confirm = (message) => {
      target.__s2ConfirmMessages.push(String(message));
      return true;
    };
  });
  await page.route(
    (url) =>
      url.pathname === "/api/settings/global" ||
      url.pathname === "/api/settings/model-configs" ||
      url.pathname === "/api/config/dataset-presets" ||
      (url.pathname === "/api/config/file-groups" &&
        url.searchParams.get("kind") === "training") ||
      url.pathname === "/api/presets" ||
      url.pathname === "/api/config/merged" ||
      url.pathname === "/api/training/history" ||
      url.pathname === "/api/training/queue",
    async (route) => {
      const { pathname } = new URL(route.request().url());
      const method = route.request().method();
      if (pathname === "/api/settings/global" && method === "GET") {
        return route.fulfill({
          json: {
            output_root: "output/runs",
            ui_scale: 100,
            tagging_max_retained_jobs: 40,
            path_overrides: {
              configs_root: activeRoot === "configs" ? "" : activeRoot,
              history_root:
                activeHistoryRoot === "history" ? "" : activeHistoryRoot,
              queue_root:
                activeQueueRoot === "queue" ? "" : activeQueueRoot,
            },
            effective_paths: {
              configs_root: "/workspace/" + activeRoot,
              history_root: "/workspace/" + activeHistoryRoot,
              queue_root: "/workspace/" + activeQueueRoot,
            },
            defaults: { output_root: "output/runs", ui_scale: 100 },
          },
        });
      }
      if (pathname === "/api/settings/global" && method === "PUT") {
        settingsWrites.push(route.request().postDataJSON());
        await settingsWriteGate;
        activeRoot = "external-configs";
        activeHistoryRoot = "external-history";
        activeQueueRoot = "external-queue";
        return route.fulfill({
          json: {
            ok: true,
            requires_reload: true,
            configs_root: activeRoot,
            output_root: "output/runs",
            ui_scale: 100,
            tagging_max_retained_jobs: 40,
            path_overrides: {
              configs_root: activeRoot,
              history_root: activeHistoryRoot,
              queue_root: activeQueueRoot,
            },
            effective_paths: {
              configs_root: "/workspace/" + activeRoot,
              history_root: "/workspace/" + activeHistoryRoot,
              queue_root: "/workspace/" + activeQueueRoot,
            },
            defaults: { output_root: "output/runs", ui_scale: 100 },
          },
        });
      }
      if (pathname === "/api/settings/model-configs" && method === "GET") {
        modelRoots.push(activeRoot);
        return route.fulfill({
          json: {
            items: [
              {
                id: "model-a",
                name: activeRoot + " model",
                model_family: "krea2_raw",
                pretrained_model_name_or_path: "models/krea2/dit.safetensors",
                qwen3: "models/krea2/qwen3vl.safetensors",
                vae: "models/krea2/vae.safetensors",
                complete: true,
              },
            ],
            groups: [
              { id: "group-a", label: "Models", item_ids: ["model-a"] },
            ],
            revision: "revision-" + activeRoot,
            default_id: "model-a",
          },
        });
      }
      if (pathname === "/api/config/dataset-presets" && method === "GET") {
        datasetRoots.push(activeRoot);
        const item = {
          path: "configs/datasets/studio.toml",
          filename: "studio.toml",
          label: activeRoot + " dataset",
          summary: { dataset_count: 1, repeat_total: 48 },
          locked: false,
        };
        return route.fulfill({
          json: {
            ok: true,
            presets: [item],
            groups: [{ id: "studio", label: "Studio", files: [item] }],
          },
        });
      }
      if (pathname === "/api/config/file-groups" && method === "GET") {
        trainingRoots.push(activeRoot);
        const item = {
          path: `${activeRoot}/training.toml`,
          filename: "training.toml",
          label: `${activeRoot} training`,
          method: "lora",
          methods_subdir: "imported",
          trainable: true,
          locked: false,
        };
        return route.fulfill({
          json: [{ id: activeRoot, label: activeRoot, files: [item] }],
        });
      }
      if (pathname === "/api/presets" && method === "GET") {
        trainingRoots.push(activeRoot);
        return route.fulfill({ json: ["default", activeRoot] });
      }
      if (pathname === "/api/config/merged" && method === "GET") {
        trainingRoots.push(activeRoot);
        return route.fulfill({ json: { output_name: activeRoot, max_train_steps: 100 } });
      }
      if (pathname === "/api/training/history" && method === "GET") {
        historyRoots.push(activeHistoryRoot);
        return route.fulfill({
          json: {
            ok: true,
            tasks: [
              {
                id: "history-task",
                name: `${activeHistoryRoot} history`,
                job: "training",
                state: "idle",
                group: "Root audit",
                history_source_config_file: `${activeHistoryRoot}/task.toml`,
                run_dir: `${activeHistoryRoot}/run`,
                metric_count: 1,
                log_count: 1,
              },
            ],
          },
        });
      }
      if (pathname === "/api/training/queue" && method === "GET") {
        queueRoots.push(activeQueueRoot);
        return route.fulfill({
          json: {
            ok: true,
            paused: false,
            status: "idle",
            summary: { total: 1, queued: 1 },
            items: [
              {
                id: "root-queue-item",
                state: "queued",
                variant: "lora",
                preset: "default",
                source_config_file: `${activeQueueRoot}/task.toml`,
                runtime_config_file: `${activeQueueRoot}/runtime.toml`,
              },
            ],
          },
        });
      }
      return route.fallback();
    },
  );

  await page.goto("/next/datasets");
  await expect.poll(() => datasetRoots.length).toBeGreaterThan(0);
  const initialDatasetReads = datasetRoots.length;
  await page.getByRole("link", { name: "训练配置", exact: true }).click();
  await expect.poll(() => trainingRoots.length).toBeGreaterThan(0);
  const initialTrainingReads = trainingRoots.length;
  await page.getByRole("link", { name: "训练队列", exact: true }).click();
  await expect(page.getByText("queue/task.toml", { exact: true })).toBeVisible();
  const initialQueueReads = queueRoots.length;
  await page.getByRole("link", { name: "历史任务", exact: true }).click();
  await expect.poll(() => historyRoots.length).toBeGreaterThan(0);
  const initialHistoryReads = historyRoots.length;
  await page.getByRole("link", { name: "模型配置", exact: true }).click();
  await expect(page.getByRole("textbox", { name: "名称", exact: true })).toHaveValue(
    "configs model",
  );
  await expect.poll(() => modelRoots.length).toBeGreaterThan(0);
  const initialModelReads = modelRoots.length;

  await page.getByRole("link", { name: "全局设置", exact: true }).click();
  const root = page.getByRole("textbox", { name: /配置根目录/ });
  await root.fill("external-configs");
  await page
    .getByRole("textbox", { name: /历史目录/ })
    .fill("external-history");
  await page
    .getByRole("textbox", { name: /队列目录/ })
    .fill("external-queue");
  const save = page.locator(".settings-actions button[type=submit]");
  await expect(save).toBeEnabled();
  await save.click();
  const confirmation = await page.evaluate(
    () => (window as unknown as { __s2ConfirmMessages: string[] }).__s2ConfirmMessages,
  );
  expect(confirmation.join("\n")).toContain("更改目录");
  await expect.poll(() => settingsWrites.length).toBe(1);
  await expect(save).toBeDisabled();
  await save.evaluate((button) => (button as HTMLButtonElement).click());
  expect(settingsWrites).toHaveLength(1);
  finishSettingsWrite?.();
  await expect(page.getByRole("status")).toContainText("现有文件未迁移");

  await page.getByRole("link", { name: "训练配置", exact: true }).click();
  await expect.poll(() => trainingRoots.length).toBeGreaterThan(initialTrainingReads);
  await page.getByRole("link", { name: "训练队列", exact: true }).click();
  await expect(
    page.getByText("external-queue/task.toml", { exact: true }),
  ).toBeVisible();
  await expect.poll(() => queueRoots.length).toBeGreaterThan(initialQueueReads);
  await page.getByRole("link", { name: "历史任务", exact: true }).click();
  await expect.poll(() => historyRoots.length).toBeGreaterThan(initialHistoryReads);
  await page.getByRole("link", { name: "数据集蓝图", exact: true }).click();
  await expect.poll(() => datasetRoots.length).toBeGreaterThan(initialDatasetReads);
  await page.getByRole("link", { name: "模型配置", exact: true }).click();
  await expect(page.getByRole("textbox", { name: "名称", exact: true })).toHaveValue(
    "external-configs model",
  );
  await expect.poll(() => modelRoots.length).toBeGreaterThan(initialModelReads);

  expect(datasetRoots.at(-1)).toBe("external-configs");
  expect(modelRoots.at(-1)).toBe("external-configs");
  expect(trainingRoots.at(-1)).toBe("external-configs");
  expect(queueRoots.at(-1)).toBe("external-queue");
  expect(historyRoots.at(-1)).toBe("external-history");
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("settings root switch removes cached mask data before loading the new root", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  const datasetFile = "configs/datasets/studio.toml";
  await page.addInitScript(() => { window.confirm = () => true; });
  const state = await setupRootMaskFixture(page, datasetFile);

  await page.goto("/next/datasets");
  await expect(page.getByText("configs dataset", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "打开子集 1 图片工作台", exact: true }).click();
  await expect(page.getByRole("heading", { name: "图片工作台" })).toBeVisible();
  await page.getByRole("tab", { name: "编辑蒙版", exact: true }).click();
  await expect(page.getByText("masks/configs", { exact: true }).first()).toBeVisible();
  await expect(page.locator('canvas[aria-label="蒙版编辑画布"]')).toBeVisible();
  expect(state.presetRoots.at(-1)).toBe("configs");
  expect(state.maskPageRoots.at(-1)).toBe("configs");
  expect(state.maskImageReads.at(-1)).toEqual({
    root: "configs",
    image: "images/studio/photo.png",
  });
  const imageReadsBeforeRootSwitch = state.maskImageReads.length;

  await page.getByRole("link", { name: "全局设置", exact: true }).click();
  await page.getByRole("textbox", { name: /配置根目录/ }).fill("external-configs");
  await page.getByRole("button", { name: "保存设置", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("现有文件未迁移");
  expect(state.settingsWrites).toEqual([{ configs_root: "external-configs" }]);

  await page.getByRole("link", { name: "数据集蓝图", exact: true }).click();
  await expect(page.getByText("external-configs dataset", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "打开子集 1 图片工作台", exact: true }).click();
  await expect(page.getByRole("heading", { name: "图片工作台" })).toBeVisible();
  await page.getByRole("tab", { name: "编辑蒙版", exact: true }).click();
  await state.externalPageReadStarted;
  await expect(page.getByText("masks/configs", { exact: true })).toHaveCount(0);
  await expect(page.getByText("正在加载图片", { exact: true })).toBeVisible();

  state.releaseExternalPageRead();
  await state.externalImageReadStarted;
  await expect(page.locator('section[aria-label="编辑区域"]')).toHaveAttribute("aria-busy", "true");
  await expect(page.locator('canvas[aria-label="蒙版编辑画布"]')).toHaveCount(0);
  await expect(page.getByText("masks/external-configs", { exact: true }).first()).toBeVisible();
  state.releaseExternalImageRead();
  await expect(page.locator('canvas[aria-label="蒙版编辑画布"]')).toBeVisible();
  expect(state.presetRoots.at(-1)).toBe("external-configs");
  expect(state.maskPageRoots.at(-1)).toBe("external-configs");
  expect(state.maskImageReads.slice(imageReadsBeforeRootSwitch)).toContainEqual({
    root: "external-configs",
    image: "images/studio/photo.png",
  });
  expect(state.settingsWrites).toHaveLength(1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("settings root switch refetches a cached nonzero mask page", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  const datasetFile = "configs/datasets/studio.toml";
  await page.addInitScript(() => { window.confirm = () => true; });
  const state = await setupRootMaskFixture(page, datasetFile);

  await page.goto("/next/datasets");
  await page.getByRole("button", { name: "打开子集 1 图片工作台", exact: true }).click();
  await page.getByRole("tab", { name: "编辑蒙版", exact: true }).click();
  await page.getByRole("button", { name: "下一页图片" }).click();
  await expect(page.getByRole("button", { name: "configs-later-1.png" })).toBeVisible();
  expect(state.maskPageReads.at(-1)).toEqual({ root: "configs", offset: 48 });
  const readsBeforeRootSwitch = state.maskPageReads.length;

  await page.getByRole("link", { name: "全局设置", exact: true }).click();
  await page.getByRole("textbox", { name: /配置根目录/ }).fill("external-configs");
  await page.getByRole("button", { name: "保存设置", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("现有文件未迁移");
  expect(state.settingsWrites).toEqual([{ configs_root: "external-configs" }]);

  await page.getByRole("link", { name: "数据集蓝图", exact: true }).click();
  await page.getByRole("button", { name: "打开子集 1 图片工作台", exact: true }).click();
  await page.getByRole("tab", { name: "编辑蒙版", exact: true }).click();
  await state.externalPageReadStarted;
  await expect(page.getByRole("button", { name: "configs-later-1.png" })).toHaveCount(0);
  state.releaseExternalPageRead();
  await state.externalImageReadStarted;
  state.releaseExternalImageRead();
  await expect(page.locator('canvas[aria-label="蒙版编辑画布"]')).toBeVisible();
  await page.getByRole("button", { name: "下一页图片" }).click();
  await expect(
    page.getByRole("button", { name: "external-configs-later-1.png" }),
  ).toBeVisible();
  const refreshedImageNames = await page
    .locator(".mask-images .mask-image-item span")
    .allTextContents();
  expect(refreshedImageNames.length).toBeGreaterThan(0);
  expect(refreshedImageNames.every((name) => name.startsWith("external-configs-"))).toBe(true);

  expect(state.maskPageReads.slice(readsBeforeRootSwitch)).toContainEqual({
    root: "external-configs",
    offset: 48,
  });
  expect(state.maskPageReads.at(-1)).toEqual({ root: "external-configs", offset: 48 });
  expect(state.settingsWrites).toHaveLength(1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("settings root switch removes cached dataset cover and preview data", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  const datasetFile = "configs/datasets/studio.toml";
  const state = await setupRootMaskFixture(page, datasetFile);
  const coverImages = await page.evaluate(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 48;
    canvas.height = 48;
    const context = canvas.getContext("2d")!;
    const render = (color: string) => {
      context.fillStyle = color;
      context.fillRect(0, 0, 48, 48);
      return canvas.toDataURL("image/png");
    };
    return { configs: render("#587d72"), external: render("#c4513e") };
  });
  const coverReads: Array<{ root: string; file: string | null }> = [];
  const previewReads: Array<{
    root: string;
    file: string | null;
    datasetIndex: string | null;
    source: string | null;
    limit: string | null;
  }> = [];
  let announceExternalCoverRead = () => {};
  let releaseExternalCoverRead = () => {};
  const externalCoverReadStarted = new Promise<void>((resolve) => {
    announceExternalCoverRead = resolve;
  });
  const externalCoverReadGate = new Promise<void>((resolve) => {
    releaseExternalCoverRead = resolve;
  });
  let announceExternalPreviewRead = () => {};
  let releaseExternalPreviewRead = () => {};
  const externalPreviewReadStarted = new Promise<void>((resolve) => {
    announceExternalPreviewRead = resolve;
  });
  const externalPreviewReadGate = new Promise<void>((resolve) => {
    releaseExternalPreviewRead = resolve;
  });

  await page.addInitScript(() => { window.confirm = () => true; });
  await page.route(
    (url) => [
      "/api/config/dataset-presets/cover",
      "/api/config/dataset-presets/images",
    ].includes(url.pathname),
    async (route) => {
      const url = new URL(route.request().url());
      const root = state.activeRoot;
      if (url.pathname === "/api/config/dataset-presets/cover") {
        coverReads.push({ root, file: url.searchParams.get("file") });
        if (root === "external-configs") {
          announceExternalCoverRead();
          await externalCoverReadGate;
        }
        return route.fulfill({ json: {
          ok: true,
          image: root === "configs" ? coverImages.configs : coverImages.external,
          reason: "",
        } });
      }
      previewReads.push({
        root,
        file: url.searchParams.get("file"),
        datasetIndex: url.searchParams.get("dataset_index"),
        source: url.searchParams.get("source"),
        limit: url.searchParams.get("limit"),
      });
      if (root === "external-configs") {
        announceExternalPreviewRead();
        await externalPreviewReadGate;
      }
      return route.fulfill({
        json: rootDatasetPreviewPayload(
          root,
          datasetFile,
          root === "configs" ? coverImages.configs : coverImages.external,
        ),
      });
    },
  );

  await page.goto("/next/datasets");
  await expect(page.getByText("configs dataset", { exact: true })).toBeVisible();
  const initialCover = page.locator(".dataset-cover img").first();
  await expect(initialCover).toBeVisible();
  await expect(initialCover).toHaveAttribute("src", coverImages.configs);
  expect(coverReads.at(-1)).toEqual({ root: "configs", file: datasetFile });

  await page.getByRole("button", { name: "打开子集 1 图片工作台", exact: true }).click();
  await expect(page.getByRole("heading", { name: "图片工作台" })).toBeVisible();
  await expect(page.getByText("images/configs/studio", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("configs caption summary", { exact: true })).toBeVisible();
  const initialPreviewRead = previewReads.at(-1);
  expect(initialPreviewRead).toEqual({
    root: "configs",
    file: datasetFile,
    datasetIndex: "0",
    source: "source",
    limit: "120",
  });

  await page.getByRole("link", { name: "全局设置", exact: true }).click();
  await page.getByRole("textbox", { name: /配置根目录/ }).fill("external-configs");
  await page.getByRole("button", { name: "保存设置", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("现有文件未迁移");
  expect(state.settingsWrites).toEqual([{ configs_root: "external-configs" }]);

  await page.getByRole("link", { name: "数据集蓝图", exact: true }).click();
  await expect(page.getByText("external-configs dataset", { exact: true })).toBeVisible();
  await externalCoverReadStarted;
  await expect(page.locator(".dataset-cover img")).toHaveCount(0);
  expect(coverReads.at(-1)).toEqual({ root: "external-configs", file: datasetFile });
  releaseExternalCoverRead();
  const refreshedCover = page.locator(".dataset-cover img").first();
  await expect(refreshedCover).toBeVisible();
  await expect(refreshedCover).toHaveAttribute("src", coverImages.external);

  await page.getByRole("button", { name: "打开子集 1 图片工作台", exact: true }).click();
  await expect(page.getByRole("heading", { name: "图片工作台" })).toBeVisible();
  await externalPreviewReadStarted;
  await expect(page.getByText("正在读取图片与标注", { exact: true })).toBeVisible();
  await expect(page.getByText("images/configs/studio", { exact: true })).toHaveCount(0);
  await expect(page.getByText("configs caption summary", { exact: true })).toHaveCount(0);
  await expect(page.locator(".dataset-preview-body")).toHaveAttribute("aria-busy", "true");
  expect(previewReads.slice(1)).toContainEqual({
    root: "external-configs",
    file: datasetFile,
    datasetIndex: "0",
    source: "source",
    limit: "120",
  });
  releaseExternalPreviewRead();
  await expect(page.getByText("images/external-configs/studio", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("external-configs caption summary", { exact: true })).toBeVisible();
  expect(state.presetRoots.at(-1)).toBe("external-configs");
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("settings root switch refetches the same raw training config key", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let activeRoot = "configs";
  const rawReads: Array<{ root: string; file: string | null }> = [];
  const settingsWrites: unknown[] = [];
  const rawContent = (root: string) =>
    `output_name = "${root}-raw"\nraw_marker = "${root}-only"\n`;

  await page.addInitScript(() => { window.confirm = () => true; });
  await page.route(
    (url) =>
      url.pathname === "/api/settings/global" ||
      (url.pathname === "/api/config/file-groups" &&
        url.searchParams.get("kind") === "training") ||
      url.pathname === "/api/presets" ||
      url.pathname === "/api/config/merged" ||
      url.pathname === "/api/config/raw",
    async (route) => {
      const url = new URL(route.request().url());
      const method = route.request().method();
      if (url.pathname === "/api/settings/global") {
        if (method === "GET")
          return route.fulfill({ json: rootSettingsPayload(activeRoot) });
        if (method === "PUT") {
          const body = route.request().postDataJSON() as { configs_root?: string };
          settingsWrites.push(body);
          activeRoot = body.configs_root || "configs";
          return route.fulfill({ json: {
            ...rootSettingsPayload(activeRoot),
            ok: true,
            requires_reload: true,
          } });
        }
        return route.fallback();
      }
      if (method !== "GET") return route.fallback();
      if (url.pathname === "/api/config/file-groups")
        return route.fulfill({ json: [{
          id: "imported",
          label: `${activeRoot} configs`,
          files: [{ ...configFile }],
        }] });
      if (url.pathname === "/api/presets")
        return route.fulfill({ json: ["default"] });
      if (url.pathname === "/api/config/merged")
        return route.fulfill({ json: { output_name: `${activeRoot}-merged`, max_train_steps: 100 } });
      const file = url.searchParams.get("file");
      rawReads.push({ root: activeRoot, file });
      return route.fulfill({ json: {
        file,
        content: rawContent(activeRoot),
        revision: `${activeRoot}-revision`,
      } });
    },
  );

  await page.goto("/next/training");
  const configRow = page.locator(".training-library-item", {
    hasText: configFile.label,
  });
  await expect(configRow).toBeVisible();
  await configRow.click();
  const rawButton = page.getByRole("button", { name: "TOML", exact: true });
  await expect(rawButton).toBeEnabled();
  await rawButton.click();
  const rawEditor = page.locator(".raw-config-text");
  await expect(rawEditor).toHaveValue(rawContent("configs"));
  await page.getByRole("dialog", { name: "TOML 配置" }).getByRole("button", { name: "关闭" }).click();

  const previousReadCount = rawReads.length;
  await page.getByRole("link", { name: "全局设置", exact: true }).click();
  await page.getByRole("textbox", { name: /配置根目录/ }).fill("external-configs");
  await page.getByRole("button", { name: "保存设置", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("现有文件未迁移");
  expect(settingsWrites).toEqual([{ configs_root: "external-configs" }]);

  await page.getByRole("link", { name: "训练配置", exact: true }).click();
  const refreshedRow = page.locator(".training-library-item", {
    hasText: configFile.label,
  });
  await expect(refreshedRow).toBeVisible();
  await refreshedRow.click();
  await expect.poll(() => rawReads.length).toBeGreaterThan(previousReadCount);
  await expect(rawButton).toBeEnabled();
  await rawButton.click();
  await expect(rawEditor).toHaveValue(rawContent("external-configs"));
  expect(rawReads[0]).toEqual({ root: "configs", file: configFile.path });
  expect(rawReads.at(-1)).toEqual({ root: "external-configs", file: configFile.path });
  expect(rawReads.every((read) => read.file === configFile.path)).toBe(true);
  expect(settingsWrites).toHaveLength(1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("settings history root switch reloads cached detail and result summaries", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  const taskId = "root-history-task";
  let activeHistoryRoot = "history";
  const settingsWrites: unknown[] = [];
  const listRoots: string[] = [];
  const detailRoots: string[] = [];
  const artifactRoots: string[] = [];
  const collectionRoots: string[] = [];
  const resumeRoots: string[] = [];
  const imageRoots: string[] = [];
  const weightRoots: string[] = [];
  let announceExternalResumeRead!: () => void;
  let releaseExternalResumeRead!: () => void;
  const externalResumeReadStarted = new Promise<void>((resolve) => {
    announceExternalResumeRead = resolve;
  });
  const externalResumeReadGate = new Promise<void>((resolve) => {
    releaseExternalResumeRead = resolve;
  });
  const historySettings = () => {
    const base = rootSettingsPayload("configs");
    return {
      ...base,
      path_overrides: {
        ...base.path_overrides,
        history_root: activeHistoryRoot === "history" ? "" : activeHistoryRoot,
      },
      effective_paths: {
        ...base.effective_paths,
        history_root: `/workspace/${activeHistoryRoot}`,
      },
    };
  };

  await page.addInitScript(() => { window.confirm = () => true; });
  await page.route(
    (url) =>
      url.pathname === "/api/settings/global" ||
      url.pathname === "/api/training/history" ||
      url.pathname === "/api/training/history/collections/settings" ||
      url.pathname === `/api/training/history/${taskId}` ||
      url.pathname === `/api/training/history/${taskId}/artifacts` ||
      url.pathname === `/api/training/history/${taskId}/resume-options` ||
      url.pathname === "/api/preview/images" ||
      url.pathname === "/api/preview/weights",
    async (route) => {
      const url = new URL(route.request().url());
      const method = route.request().method();
      if (url.pathname === "/api/settings/global") {
        if (method === "GET") return route.fulfill({ json: historySettings() });
        if (method === "PUT") {
          const body = route.request().postDataJSON() as { history_root?: string };
          settingsWrites.push(body);
          activeHistoryRoot = body.history_root || "history";
          return route.fulfill({ json: {
            ...historySettings(),
            ok: true,
            requires_reload: true,
          } });
        }
        return route.fallback();
      }
      if (url.pathname === "/api/training/history" && method === "GET") {
        listRoots.push(activeHistoryRoot);
        return route.fulfill({ json: {
          ok: true,
          tasks: [{
            id: taskId,
            name: `${activeHistoryRoot} task`,
            job: "training",
            state: "error",
            group: "Root audit",
            run_dir: `/workspace/${activeHistoryRoot}/run`,
            metric_count: 0,
            log_count: 0,
          }],
        } });
      }
      if (url.pathname === "/api/training/history/collections/settings" && method === "GET") {
        collectionRoots.push(activeHistoryRoot);
        return route.fulfill({ json: {
          collection_order: [activeHistoryRoot === "history" ? "history-only" : "external-only", "Root audit"],
          config_group_order: {},
        } });
      }
      if (url.pathname === `/api/training/history/${taskId}` && method === "GET") {
        detailRoots.push(activeHistoryRoot);
        return route.fulfill({ json: {
          task: {
            id: taskId,
            name: `${activeHistoryRoot} task detail`,
            job: "training",
            state: "error",
            group: "Root audit",
            run_dir: `/workspace/${activeHistoryRoot}/run`,
            metric_count: 0,
            log_count: 0,
          },
          metrics: [],
          logs: [],
          system: [],
          config_toml: "network_dim = 32",
        } });
      }
      if (url.pathname === `/api/training/history/${taskId}/artifacts` && method === "GET") {
        artifactRoots.push(activeHistoryRoot);
        return route.fulfill({ json: { artifacts: [{
          key: "config-snapshot",
          state: "available",
          name: `${activeHistoryRoot}-config.toml`,
          size_bytes: 32,
        }] } });
      }
      if (url.pathname === `/api/training/history/${taskId}/resume-options` && method === "GET") {
        const root = activeHistoryRoot;
        resumeRoots.push(root);
        if (root === "external-history") {
          announceExternalResumeRead();
          await externalResumeReadGate;
        }
        const checkpoint = {
          path: `/workspace/${root}/checkpoints/state`,
          name: `${root}-checkpoint`,
          step: 42,
          target_total_steps: 100,
          remaining_steps: 58,
          state_complete: true,
          state_integrity: { ok: true },
          resume_available: true,
        };
        return route.fulfill({ json: {
          checkpoints: [checkpoint],
          default_checkpoint: checkpoint.path,
          message: "",
        } });
      }
      if (url.pathname === "/api/preview/images" && method === "GET") {
        imageRoots.push(activeHistoryRoot);
        return route.fulfill({ json: {
          images: [{ file: `${activeHistoryRoot}-sample.png`, name: `${activeHistoryRoot}-sample.png` }],
          total: 1,
          directory_exists: true,
        } });
      }
      if (url.pathname === "/api/preview/weights" && method === "GET") {
        weightRoots.push(activeHistoryRoot);
        return route.fulfill({ json: {
          weights: [{ file: `${activeHistoryRoot}.safetensors`, name: `${activeHistoryRoot}.safetensors`, size_bytes: 128, scope_label: "LoRA" }],
          total: 1,
          directory_exists: true,
        } });
      }
      return route.fallback();
    },
  );

  await page.goto("/next/history");
  await expect(page.getByRole("button", { name: /history-only 0 条已加载/ })).toBeVisible();
  await expect(page.getByRole("button", { name: /external-only 0 条已加载/ })).toHaveCount(0);
  await page.getByRole("button", { name: /history task 1 条/ }).click();
  await page.locator("a.history-card-link").filter({ hasText: "history task" }).click();
  await expect(page.getByRole("heading", { name: "history task detail" })).toBeVisible();
  await expect(page.getByRole("link", { name: "history-config.toml" })).toBeVisible();
  await expect(page.getByLabel("训练产物摘要")).toContainText("最近：history-sample.png");
  await expect(page.getByLabel("训练产物摘要")).toContainText("history.safetensors");
  await page.getByRole("button", { name: "检查点续训" }).click();
  const resumeCheckpointSelect = page.getByRole("dialog", { name: "从历史检查点续训" }).getByRole("combobox").first();
  await expect(resumeCheckpointSelect).toHaveValue("/workspace/history/checkpoints/state");
  await page.getByRole("button", { name: "取消", exact: true }).click();

  await page.getByRole("link", { name: "全局设置", exact: true }).click();
  await page.getByRole("textbox", { name: /历史目录/ }).fill("external-history");
  await page.getByRole("button", { name: "保存设置", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("现有文件未迁移");
  expect(settingsWrites).toEqual([{ history_root: "external-history" }]);

  await page.getByRole("link", { name: "历史任务", exact: true }).click();
  await expect(page.getByRole("button", { name: /external-only 0 条已加载/ })).toBeVisible();
  await expect(page.getByRole("button", { name: /history-only 0 条已加载/ })).toHaveCount(0);
  await page.getByRole("button", { name: /external-history task 1 条/ }).click();
  await page.locator("a.history-card-link").filter({ hasText: "external-history task" }).click();
  await expect(page.getByRole("heading", { name: "external-history task detail" })).toBeVisible();
  await expect(page.getByRole("link", { name: "external-history-config.toml" })).toBeVisible();
  await expect(page.getByLabel("训练产物摘要")).toContainText("最近：external-history-sample.png");
  await expect(page.getByLabel("训练产物摘要")).toContainText("external-history.safetensors");
  await page.getByRole("button", { name: "检查点续训" }).click();
  try {
    await externalResumeReadStarted;
    await expect(resumeCheckpointSelect).not.toHaveValue("/workspace/history/checkpoints/state");
  } finally {
    releaseExternalResumeRead();
  }
  await expect(resumeCheckpointSelect).toHaveValue("/workspace/external-history/checkpoints/state");

  expect(listRoots).toContain("history");
  expect(listRoots.at(-1)).toBe("external-history");
  for (const roots of [collectionRoots, detailRoots, artifactRoots, resumeRoots, imageRoots, weightRoots]) {
    expect(roots).toContain("history");
    expect(roots.at(-1)).toBe("external-history");
  }
  expect(settingsWrites).toHaveLength(1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("settings root switch exposes and recovers from a failed new-root read", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let activeRoot = "configs";
  let allowExternalRead = false;
  const settingsWrites: unknown[] = [];
  const datasetRoots: string[] = [];
  await page.addInitScript(() => { window.confirm = () => true; });
  await page.route(
    (url) => ["/api/settings/global", "/api/config/dataset-presets"].includes(url.pathname),
    async (route) => {
      const { pathname } = new URL(route.request().url());
      const method = route.request().method();
      if (pathname === "/api/settings/global" && method === "GET") {
        return route.fulfill({ json: {
          revision: "settings-revision-1",
          output_root: "output/runs",
          ui_scale: 100,
          tagging_max_retained_jobs: 40,
          path_overrides: { configs_root: "", history_root: "", queue_root: "" },
          effective_paths: { configs_root: "/workspace/configs" },
          defaults: { output_root: "output/runs", ui_scale: 100 },
        } });
      }
      if (pathname === "/api/settings/global" && method === "PUT") {
        settingsWrites.push(route.request().postDataJSON());
        activeRoot = "external-configs";
        return route.fulfill({ json: {
          ok: true,
          requires_reload: true,
          revision: "settings-revision-2",
          output_root: "output/runs",
          ui_scale: 100,
          tagging_max_retained_jobs: 40,
          path_overrides: { configs_root: activeRoot },
          effective_paths: { configs_root: "/workspace/" + activeRoot },
          defaults: { output_root: "output/runs", ui_scale: 100 },
        } });
      }
      if (pathname === "/api/config/dataset-presets" && method === "GET") {
        datasetRoots.push(activeRoot);
        if (activeRoot === "external-configs" && !allowExternalRead) {
          return route.fulfill({ status: 503, json: { error: "新配置根的数据集预设暂不可读" } });
        }
        const item = {
          path: "configs/datasets/studio.toml",
          filename: "studio.toml",
          label: activeRoot + " dataset",
          summary: { dataset_count: 1, repeat_total: 48 },
          locked: false,
        };
        return route.fulfill({ json: {
          ok: true,
          presets: [item],
          groups: [{ id: "studio", label: "Studio", files: [item] }],
        } });
      }
      return route.fallback();
    },
  );

  await page.goto("/next/datasets");
  await expect(page.getByText("configs dataset", { exact: true })).toBeVisible();
  await page.getByRole("link", { name: "全局设置", exact: true }).click();
  await page.getByRole("textbox", { name: /配置根目录/ }).fill("external-configs");
  await page.getByRole("button", { name: "保存设置", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("现有文件未迁移");
  expect(settingsWrites).toHaveLength(1);

  await page.getByRole("link", { name: "数据集蓝图", exact: true }).click();
  const error = page.getByRole("alert");
  await expect(error).toContainText("新配置根的数据集预设暂不可读");
  await expect(page.getByText("configs dataset", { exact: true })).toHaveCount(0);
  const failedReads = datasetRoots.filter((root) => root === "external-configs").length;
  expect(failedReads).toBeGreaterThan(0);

  allowExternalRead = true;
  await error.getByRole("button", { name: "重试", exact: true }).click();
  await expect(page.getByText("external-configs dataset", { exact: true })).toBeVisible();
  expect(datasetRoots.at(-1)).toBe("external-configs");
  expect(datasetRoots.filter((root) => root === "external-configs")).toHaveLength(failedReads + 1);
  expect(settingsWrites).toHaveLength(1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("settings preserves an unknown post-commit result and does not retry it", async ({
  page,
}) => {
  const mocks = await mockWorkspace(page);
  let serverScale = 100;
  let writes = 0;
  await page.route(
    (url) => url.pathname === "/api/settings/global",
    async (route) => {
      if (route.request().method() === "GET") {
        return route.fulfill({
          json: {
            output_root: "output/runs",
            ui_scale: serverScale,
            tagging_max_retained_jobs: 40,
            path_overrides: {
              configs_root: "",
              history_root: "",
              queue_root: "",
            },
            effective_paths: { configs_root: "/workspace/configs" },
            defaults: { output_root: "output/runs", ui_scale: 100 },
          },
        });
      }
      if (route.request().method() === "PUT") {
        writes += 1;
        const body = route.request().postDataJSON() as { ui_scale?: number };
        serverScale = body.ui_scale ?? serverScale;
        return route.abort();
      }
      return route.fallback();
    },
  );

  await page.goto("/next/settings");
  const scale = page.getByRole("spinbutton", {
    name: "全局缩放 (%)",
    exact: true,
  });
  await scale.fill("125");
  await page.getByRole("button", { name: "保存设置", exact: true }).click();

  await expect(page.getByRole("alert")).toContainText("操作结果尚未确认");
  await expect(scale).toHaveValue("125");
  await expect(page.locator(".settings-page .state-label")).toHaveText("未保存");
  await expect.poll(() => writes).toBe(1);
  expect(serverScale).toBe(125);
  expect(mocks.writes).toEqual([]);

  await page.reload();
  await expect(scale).toHaveValue("125");
  await expect(page.locator(".settings-page .state-label")).toHaveText("已同步");
  expect(writes).toBe(1);
  expect(mocks.unhandled).toEqual([]);
});

test("settings keeps a stale draft and requires explicit reload after 409", async ({ page }) => {
  await mockWorkspace(page);
  let writes = 0;
  await page.route((url) => url.pathname === "/api/settings/global", async (route) => {
    if (route.request().method() === "GET")
      return route.fulfill({ json: {
        ok: true, revision: "read-revision", output_root: "output/runs", ui_scale: 100,
        tagging_max_retained_jobs: 40,
        path_overrides: { configs_root: "", history_root: "", queue_root: "" },
        effective_paths: { configs_root: "/workspace/configs" },
        defaults: { output_root: "output/runs", ui_scale: 100 },
      } });
    writes += 1;
    expect(route.request().postDataJSON().revision).toBe("read-revision");
    return route.fulfill({ status: 409, json: { ok: false, error: "全局设置已在其他位置修改" } });
  });
  await page.goto("/next/settings");
  const scale = page.getByRole("spinbutton", { name: "全局缩放 (%)", exact: true });
  await scale.fill("125");
  await page.getByRole("button", { name: "保存设置", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("全局设置已在其他位置修改");
  await expect(scale).toHaveValue("125");
  await expect(page.getByRole("button", { name: "保存设置", exact: true })).toBeDisabled();
  await expect(page.getByRole("button", { name: "重新读取设置" })).toBeVisible();
  expect(writes).toBe(1);
});
