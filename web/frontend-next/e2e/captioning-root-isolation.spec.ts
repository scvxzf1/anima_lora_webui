import { expect, test } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

const datasetFile = "configs/datasets/studio.toml";

function settingsPayload(root: string) {
  return {
    revision: `settings-${root}`,
    output_root: "output/runs",
    ui_scale: 100,
    tagging_max_retained_jobs: 40,
    path_overrides: {
      configs_root: root === "configs" ? "" : root,
      history_root: "",
      queue_root: "",
    },
    effective_paths: {
      configs_root: `/workspace/${root}`,
      history_root: "/workspace/history",
      queue_root: "/workspace/queue",
    },
    defaults: { output_root: "output/runs", ui_scale: 100 },
  };
}

function datasetPayload(root: string) {
  const dataset = {
    path: datasetFile,
    filename: "studio.toml",
    label: `${root} dataset`,
    summary: { dataset_count: 1, repeat_total: 1 },
    locked: false,
  };
  return {
    ok: true,
    presets: [dataset],
    groups: [{ id: "studio", label: `${root} datasets`, files: [dataset] }],
  };
}

function presetPayload(root: string) {
  return {
    ok: true,
    file: datasetFile,
    name: `${root} studio`,
    content: "",
    datasets: [{
      source_dir: `images/${root}/studio`,
      image_dir: `cache/${root}/studio`,
      num_repeats: 1,
      settings: {},
    }],
    defaults: { resolution: 1024, batch_size: 1 },
    readonly: false,
    summary: { dataset_count: 1, repeat_total: 1 },
  };
}

function imagesPayload(root: string, offset = 0) {
  const total = 61;
  const images = Array.from({ length: Math.min(60, total - offset) }, (_, position) => {
    const imageIndex = offset + position + 1;
    const name = imageIndex === 1 ? `${root}-image.png` : `${root}-image-${imageIndex}.png`;
    const file = `images/${root}/studio/${name}`;
    return {
      file,
      name,
      url: `/api/config/dataset-presets/image?file=${encodeURIComponent(file)}`,
      thumbnail_url: `/api/config/dataset-presets/image?file=${encodeURIComponent(file)}`,
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
    dataset_label: `${root} studio`,
    source: "source",
    source_label: "原始图目录",
    directory: `images/${root}/studio`,
    directory_exists: true,
    caption_extension: ".txt",
    prefer_json_caption: false,
    caption_source_mode: "txt",
    caption_source_label: "TXT",
    caption_summary: "",
    count: images.length,
    total,
    limit: 60,
    offset,
    images,
    row: {
      source_dir: `images/${root}/studio`,
      image_dir: `cache/${root}/studio`,
      num_repeats: 1,
      settings: {},
    },
    settings: { resolution: 1024, enable_bucket: true },
    message: "",
  };
}

test("config root switch refreshes captioning dataset preset and image queries", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let activeRoot = "configs";
  const settingsWrites: unknown[] = [];
  const libraryRoots: string[] = [];
  const presetReads: Array<{ root: string; file: string | null }> = [];
  const imageReads: Array<{ root: string; params: string }> = [];

  let announcePresetRead = () => {};
  const externalPresetReadStarted = new Promise<void>((resolve) => {
    announcePresetRead = resolve;
  });
  let releasePresetRead = () => {};
  const externalPresetReadGate = new Promise<void>((resolve) => {
    releasePresetRead = resolve;
  });
  let announceImageRead = () => {};
  const externalImageReadStarted = new Promise<void>((resolve) => {
    announceImageRead = resolve;
  });
  let releaseImageRead = () => {};
  const externalImageReadGate = new Promise<void>((resolve) => {
    releaseImageRead = resolve;
  });
  let announceNextImageRead = () => {};
  const externalNextImageReadStarted = new Promise<void>((resolve) => {
    announceNextImageRead = resolve;
  });
  let releaseNextImageRead = () => {};
  const externalNextImageReadGate = new Promise<void>((resolve) => {
    releaseNextImageRead = resolve;
  });

  await page.addInitScript(() => {
    window.confirm = () => true;
  });
  await page.route(
    (url) =>
      url.pathname === "/api/settings/global" ||
      url.pathname === "/api/config/dataset-presets" ||
      url.pathname === "/api/config/dataset-presets/read" ||
      url.pathname === "/api/config/dataset-presets/images",
    async (route) => {
      const url = new URL(route.request().url());
      const method = route.request().method();
      const reply = (body: unknown) => route.fulfill({ json: body });

      if (url.pathname === "/api/settings/global") {
        if (method === "GET") return reply(settingsPayload(activeRoot));
        if (method === "PUT") {
          const body = route.request().postDataJSON() as { configs_root?: string };
          settingsWrites.push(body);
          activeRoot = body.configs_root || "configs";
          return reply({
            ...settingsPayload(activeRoot),
            ok: true,
            requires_reload: true,
          });
        }
        return route.fallback();
      }

      if (method !== "GET") return route.fallback();
      if (url.pathname === "/api/config/dataset-presets") {
        libraryRoots.push(activeRoot);
        return reply(datasetPayload(activeRoot));
      }
      if (url.pathname === "/api/config/dataset-presets/read") {
        presetReads.push({ root: activeRoot, file: url.searchParams.get("file") });
        if (activeRoot === "external-configs") {
          announcePresetRead();
          await externalPresetReadGate;
        }
        return reply(presetPayload(activeRoot));
      }
      if (url.pathname === "/api/config/dataset-presets/images") {
        imageReads.push({ root: activeRoot, params: url.searchParams.toString() });
        const offset = Number(url.searchParams.get("offset") || 0);
        if (activeRoot === "external-configs" && offset === 0) {
          announceImageRead();
          await externalImageReadGate;
        }
        if (activeRoot === "external-configs" && offset === 60) {
          announceNextImageRead();
          await externalNextImageReadGate;
        }
        return reply(imagesPayload(activeRoot, offset));
      }
      return route.fallback();
    },
  );

  await page.goto("/next/captioning");
  const datasetSelect = page.getByLabel("数据集预设");
  await expect(datasetSelect.locator("option", { hasText: "configs dataset" })).toHaveCount(1);
  await datasetSelect.selectOption(datasetFile);
  const imageGroup = page.getByLabel("图片组");
  await expect(imageGroup.locator("option")).toHaveText("1 · images/configs/studio");
  await page.getByRole("button", { name: "扫描图片" }).click();
  await expect(page.getByRole("img", { name: "configs-image.png" })).toBeVisible();

  await page.getByRole("link", { name: "全局设置", exact: true }).click();
  await page.getByRole("textbox", { name: /配置根目录/ }).fill("external-configs");
  await page.getByRole("button", { name: "保存设置", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("现有文件未迁移");
  expect(settingsWrites).toEqual([{
    configs_root: "external-configs",
    revision: "settings-configs",
  }]);

  await page.getByRole("link", { name: "打标工作台", exact: true }).click();
  await expect(datasetSelect.locator("option", { hasText: "external-configs dataset" })).toHaveCount(1);
  await datasetSelect.selectOption(datasetFile);
  await externalPresetReadStarted;
  await expect(imageGroup.locator("option")).toHaveCount(0);
  await expect(page.getByRole("img", { name: "configs-image.png" })).toHaveCount(0);
  releasePresetRead();
  await expect(imageGroup.locator("option")).toHaveText("1 · images/external-configs/studio");

  await page.getByRole("button", { name: "扫描图片" }).click();
  await externalImageReadStarted;
  await expect(page.getByRole("img", { name: "configs-image.png" })).toHaveCount(0);
  await expect(page.getByRole("img", { name: "external-configs-image.png" })).toHaveCount(0);
  releaseImageRead();
  await expect(page.getByRole("img", { name: "external-configs-image.png" })).toBeVisible();
  await page.getByRole("button", { name: "下一页图片" }).click();
  try {
    await externalNextImageReadStarted;
    await expect(page.getByRole("img", { name: "external-configs-image.png" })).toHaveCount(0);
  } finally {
    releaseNextImageRead();
  }
  await expect(page.getByRole("img", { name: "external-configs-image-61.png" })).toBeVisible();

  expect(libraryRoots).toContain("configs");
  expect(libraryRoots.at(-1)).toBe("external-configs");
  expect(presetReads).toEqual([
    { root: "configs", file: datasetFile },
    { root: "external-configs", file: datasetFile },
  ]);
  expect(imageReads).toEqual([
    { root: "configs", params: `file=${encodeURIComponent(datasetFile)}&dataset_index=0&source=source&limit=60&offset=0` },
    { root: "external-configs", params: `file=${encodeURIComponent(datasetFile)}&dataset_index=0&source=source&limit=60&offset=0` },
    { root: "external-configs", params: `file=${encodeURIComponent(datasetFile)}&dataset_index=0&source=source&limit=60&offset=60` },
  ]);
  expect(settingsWrites).toHaveLength(1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});
