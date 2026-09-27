import { expect, test, type Page, type Route } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

const datasetFile = "configs/datasets/multi.toml";

type MediaState = {
  activeRoot: string;
  settingsWrites: unknown[];
  presetReads: string[];
  previewReads: Array<{
    root: string;
    datasetIndex: string | null;
    source: string | null;
    limit: string | null;
    offset: string | null;
  }>;
  maskPageReads: Array<{
    root: string;
    datasetIndex: string | null;
    offset: string | null;
  }>;
  maskImageReads: Array<{
    root: string;
    datasetIndex: string | null;
    image: string | null;
  }>;
  externalPresetReadStarted: Promise<void>;
  announceExternalPresetRead: () => void;
  releaseExternalPresetRead: () => void;
  externalPreviewReadStarted: Promise<void>;
  announceExternalPreviewRead: () => void;
  releaseExternalPreviewRead: () => void;
  externalMaskPageReadStarted: Promise<void>;
  announceExternalMaskPageRead: () => void;
  releaseExternalMaskPageRead: () => void;
  externalMaskImageReadStarted: Promise<void>;
  announceExternalMaskImageRead: () => void;
  releaseExternalMaskImageRead: () => void;
};

type MediaAssets = { imageUrl: string; maskUrl: string };

function settingsPayload(activeRoot: string) {
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

function presetPayload(activeRoot: string) {
  return {
    ok: true,
    file: datasetFile,
    name: `${activeRoot} multi-subset`,
    content: "",
    datasets: [
      {
        source_dir: `images/${activeRoot}/subset-0`,
        image_dir: `cache/${activeRoot}/subset-0`,
        num_repeats: 1,
        settings: {},
      },
      {
        source_dir: `images/${activeRoot}/subset-1`,
        image_dir: `cache/${activeRoot}/subset-1`,
        num_repeats: 2,
        settings: { resolution: 768 },
      },
    ],
    defaults: { resolution: 1024, batch_size: 1 },
    readonly: false,
    summary: { dataset_count: 2, repeat_total: 3 },
  };
}

function imageMeta(activeRoot: string) {
  const directory = `images/${activeRoot}/subset-1`;
  const file = `${directory}/photo.png`;
  return {
    file,
    name: "photo.png",
    url: "",
    thumbnail_url: "",
    has_mask: false,
    caption: {
      ok: true,
      file: file.replace(/\.png$/, ".txt"),
      extension: ".txt",
      source_mode: "txt",
      source_label: "TXT",
      detected_mode: "txt",
      format_label: "TXT",
      caption_count: 1,
      text: `${activeRoot} subset one caption`,
      truncated: false,
      length: `${activeRoot} subset one caption`.length,
    },
  };
}

function previewPayload(activeRoot: string, assets: MediaAssets) {
  const image = imageMeta(activeRoot);
  image.url = assets.imageUrl;
  image.thumbnail_url = assets.imageUrl;
  return {
    ok: true,
    file: datasetFile,
    dataset_index: 1,
    dataset_label: `${activeRoot} subset one`,
    source: "source",
    source_label: "原始图目录",
    directory: `images/${activeRoot}/subset-1`,
    directory_exists: true,
    caption_extension: ".txt",
    prefer_json_caption: false,
    caption_source_mode: "txt",
    caption_source_label: "TXT",
    caption_summary: `${activeRoot} subset one caption`,
    count: 1,
    total: 1,
    returned: 1,
    offset: 0,
    limit: 120,
    next_offset: null,
    has_more_before: false,
    has_more_after: false,
    images: [image],
    row: {
      source_dir: `images/${activeRoot}/subset-1`,
      image_dir: `cache/${activeRoot}/subset-1`,
      num_repeats: 2,
      settings: { resolution: 768 },
    },
    settings: { resolution: 768, enable_bucket: true },
    message: "",
  };
}

function maskPagePayload(activeRoot: string, assets: MediaAssets) {
  const image = imageMeta(activeRoot);
  image.url = assets.imageUrl;
  image.thumbnail_url = assets.imageUrl;
  return {
    ...previewPayload(activeRoot, assets),
    images: [image],
    mask_dir: `masks/${activeRoot}/subset-1`,
    mask_mode: "external",
    config_revision: `mask-rev-${activeRoot}`,
    readonly: false,
    offset: 0,
    next_offset: 1,
    has_more_before: false,
    has_more_after: false,
  };
}

function maskImagePayload(activeRoot: string, assets: MediaAssets) {
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
    mask_dir: `masks/${activeRoot}/subset-1`,
    mask_file: `masks/${activeRoot}/subset-1/photo.png`,
  };
}

async function setupMediaFixture(page: Page): Promise<MediaState> {
  let releaseExternalPresetRead = () => {};
  let releaseExternalPreviewRead = () => {};
  let releaseExternalMaskPageRead = () => {};
  let releaseExternalMaskImageRead = () => {};
  let announceExternalPresetRead = () => {};
  let announceExternalPreviewRead = () => {};
  let announceExternalMaskPageRead = () => {};
  let announceExternalMaskImageRead = () => {};
  const externalPresetReadStarted = new Promise<void>((resolve) => {
    announceExternalPresetRead = resolve;
  });
  const externalPreviewReadStarted = new Promise<void>((resolve) => {
    announceExternalPreviewRead = resolve;
  });
  const externalMaskPageReadStarted = new Promise<void>((resolve) => {
    announceExternalMaskPageRead = resolve;
  });
  const externalMaskImageReadStarted = new Promise<void>((resolve) => {
    announceExternalMaskImageRead = resolve;
  });
  const externalPresetReadGate = new Promise<void>((resolve) => {
    releaseExternalPresetRead = resolve;
  });
  const externalPreviewReadGate = new Promise<void>((resolve) => {
    releaseExternalPreviewRead = resolve;
  });
  const externalMaskPageReadGate = new Promise<void>((resolve) => {
    releaseExternalMaskPageRead = resolve;
  });
  const externalMaskImageReadGate = new Promise<void>((resolve) => {
    releaseExternalMaskImageRead = resolve;
  });
  const assets = await page.evaluate(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 64;
    canvas.height = 64;
    const context = canvas.getContext("2d")!;
    context.fillStyle = "#587d72";
    context.fillRect(0, 0, 64, 64);
    const imageUrl = canvas.toDataURL("image/png");
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, 64, 64);
    return { imageUrl, maskUrl: canvas.toDataURL("image/png") };
  });
  const state: MediaState = {
    activeRoot: "configs",
    settingsWrites: [],
    presetReads: [],
    previewReads: [],
    maskPageReads: [],
    maskImageReads: [],
    externalPresetReadStarted,
    announceExternalPresetRead,
    releaseExternalPresetRead,
    externalPreviewReadStarted,
    announceExternalPreviewRead,
    releaseExternalPreviewRead,
    externalMaskPageReadStarted,
    announceExternalMaskPageRead,
    releaseExternalMaskPageRead,
    externalMaskImageReadStarted,
    announceExternalMaskImageRead,
    releaseExternalMaskImageRead,
  };
  await page.route(
    (url) => [
      "/api/settings/global",
      "/api/config/dataset-presets",
      "/api/config/dataset-presets/read",
      "/api/config/dataset-presets/images",
      "/api/config/dataset-masks",
      "/api/config/dataset-masks/image",
    ].includes(url.pathname),
    (route) => handleMediaRoute(
      route,
      state,
      assets,
      externalPresetReadGate,
      externalPreviewReadGate,
      externalMaskPageReadGate,
      externalMaskImageReadGate,
    ),
  );
  return state;
}

async function handleMediaRoute(
  route: Route,
  state: MediaState,
  assets: MediaAssets,
  externalPresetReadGate: Promise<void>,
  externalPreviewReadGate: Promise<void>,
  externalMaskPageReadGate: Promise<void>,
  externalMaskImageReadGate: Promise<void>,
) {
  const url = new URL(route.request().url());
  const method = route.request().method();
  if (url.pathname === "/api/settings/global") {
    if (method === "GET") return route.fulfill({ json: settingsPayload(state.activeRoot) });
    if (method === "PUT") {
      const body = route.request().postDataJSON() as { configs_root?: string };
      state.settingsWrites.push(body);
      state.activeRoot = body.configs_root || "configs";
      return route.fulfill({ json: { ...settingsPayload(state.activeRoot), ok: true, requires_reload: true } });
    }
    return route.fallback();
  }
  if (method !== "GET") return route.fallback();
  if (url.pathname === "/api/config/dataset-presets") {
    const item = { path: datasetFile, filename: "multi.toml", label: `${state.activeRoot} dataset`, summary: { dataset_count: 2, repeat_total: 3 }, locked: false };
    return route.fulfill({ json: { ok: true, presets: [item], groups: [{ id: "multi", label: "Multi", files: [item] }] } });
  }
  if (url.pathname === "/api/config/dataset-presets/read") {
    state.presetReads.push(state.activeRoot);
    if (state.activeRoot === "external-configs") {
      state.announceExternalPresetRead();
      await externalPresetReadGate;
    }
    return route.fulfill({ json: presetPayload(state.activeRoot) });
  }
  if (url.pathname === "/api/config/dataset-presets/images") {
    state.previewReads.push({
      root: state.activeRoot,
      datasetIndex: url.searchParams.get("dataset_index"),
      source: url.searchParams.get("source"),
      limit: url.searchParams.get("limit"),
      offset: url.searchParams.get("offset"),
    });
    if (state.activeRoot === "external-configs") {
      state.announceExternalPreviewRead();
      await externalPreviewReadGate;
    }
    return route.fulfill({ json: previewPayload(state.activeRoot, assets) });
  }
  if (url.pathname === "/api/config/dataset-masks") {
    state.maskPageReads.push({
      root: state.activeRoot,
      datasetIndex: url.searchParams.get("dataset_index"),
      offset: url.searchParams.get("offset"),
    });
    if (state.activeRoot === "external-configs") {
      state.announceExternalMaskPageRead();
      await externalMaskPageReadGate;
    }
    return route.fulfill({ json: maskPagePayload(state.activeRoot, assets) });
  }
  if (url.pathname === "/api/config/dataset-masks/image") {
    state.maskImageReads.push({
      root: state.activeRoot,
      datasetIndex: url.searchParams.get("dataset_index"),
      image: url.searchParams.get("image"),
    });
    if (state.activeRoot === "external-configs") {
      state.announceExternalMaskImageRead();
      await externalMaskImageReadGate;
    }
    return route.fulfill({ json: maskImagePayload(state.activeRoot, assets) });
  }
  return route.fallback();
}

async function switchConfigRoot(page: Page) {
  await page.getByRole("link", { name: "全局设置", exact: true }).click();
  await page.getByRole("textbox", { name: /配置根目录/ }).fill("external-configs");
  await page.getByRole("button", { name: "保存设置", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("现有文件未迁移");
}

test("preview deep link keeps the second subset query contract across a root switch", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  const state = await setupMediaFixture(page);
  const query = new URLSearchParams({ dataset: datasetFile, subset: "1" }).toString();
  await page.addInitScript(() => { window.confirm = () => true; });

  await page.goto(`/next/datasets/workspace/preview?${query}`);
  await expect(page.getByRole("heading", { name: "图片工作台" })).toBeVisible();
  await expect(page.getByText("images/configs/subset-1", { exact: true }).first()).toBeVisible();
  await expect(page.getByRole("region", { name: "预览图片" }).getByText("configs subset one caption", { exact: true })).toBeVisible();
  expect(state.presetReads.at(-1)).toBe("configs");
  expect(state.previewReads.at(-1)).toEqual({
    root: "configs",
    datasetIndex: "1",
    source: "source",
    limit: "120",
    offset: null,
  });

  await switchConfigRoot(page);
  await page.goto(`/next/datasets/workspace/preview?${query}`);
  await state.externalPresetReadStarted;
  await expect(page.getByText("images/configs/subset-1", { exact: true })).toHaveCount(0);
  state.releaseExternalPresetRead();
  await state.externalPreviewReadStarted;
  await expect(page.getByText("images/configs/subset-1", { exact: true })).toHaveCount(0);
  await expect(page.getByText("正在读取图片与标注", { exact: true })).toBeVisible();
  state.releaseExternalPreviewRead();
  await expect(page.getByText("images/external-configs/subset-1", { exact: true }).first()).toBeVisible();
  await expect(page.getByRole("region", { name: "预览图片" }).getByText("external-configs subset one caption", { exact: true })).toBeVisible();
  expect(state.previewReads.at(-1)).toEqual({
    root: "external-configs",
    datasetIndex: "1",
    source: "source",
    limit: "120",
    offset: null,
  });
  expect(state.settingsWrites).toHaveLength(1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("mask deep link keeps the second subset image identity across a root switch", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  const state = await setupMediaFixture(page);
  const query = new URLSearchParams({ dataset: datasetFile, subset: "1" }).toString();
  await page.addInitScript(() => { window.confirm = () => true; });

  await page.goto(`/next/datasets/workspace/masks?${query}`);
  await expect(page.getByRole("heading", { name: "蒙版编辑" })).toBeVisible();
  await expect(page.getByText("masks/configs/subset-1", { exact: true }).first()).toBeVisible();
  await expect(page.locator('canvas[aria-label="蒙版编辑画布"]')).toBeVisible();
  expect(state.maskPageReads.at(-1)).toEqual({ root: "configs", datasetIndex: "1", offset: "0" });
  expect(state.maskImageReads.at(-1)).toEqual({
    root: "configs",
    datasetIndex: "1",
    image: "images/configs/subset-1/photo.png",
  });

  await switchConfigRoot(page);
  await page.goto(`/next/datasets/workspace/masks?${query}`);
  await state.externalMaskPageReadStarted;
  await expect(page.getByText("masks/configs/subset-1", { exact: true })).toHaveCount(0);
  state.releaseExternalMaskPageRead();
  await state.externalMaskImageReadStarted;
  await expect(page.locator('canvas[aria-label="蒙版编辑画布"]')).toHaveCount(0);
  await expect(page.getByText("masks/external-configs/subset-1", { exact: true }).first()).toBeVisible();
  state.releaseExternalMaskImageRead();
  await expect(page.locator('canvas[aria-label="蒙版编辑画布"]')).toBeVisible();
  expect(state.maskImageReads.at(-1)).toEqual({
    root: "external-configs",
    datasetIndex: "1",
    image: "images/external-configs/subset-1/photo.png",
  });
  expect(state.settingsWrites).toHaveLength(1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});
