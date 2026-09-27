import { expect, test } from "@playwright/test";
import { configFile, mockWorkspace } from "./fixtures";

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
  const preset = {
    path: datasetFile,
    filename: "studio.toml",
    label: `${root} dataset`,
    summary: { dataset_count: 1, repeat_total: 2 },
    locked: false,
  };
  return {
    ok: true,
    presets: [preset],
    groups: [{ id: "studio", label: `${root} datasets`, files: [preset] }],
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
      num_repeats: 2,
      settings: {},
    }],
    defaults: { resolution: 1024, batch_size: 1 },
    readonly: false,
    summary: { dataset_count: 1, repeat_total: 2 },
  };
}

function previewPayload(root: string) {
  const name = `${root}-picker.png`;
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
    count: 1,
    total: 1,
    limit: 8,
    images: [{
      file: `images/${root}/studio/${name}`,
      name,
      url: `/api/config/dataset-presets/image?file=${encodeURIComponent(name)}`,
      thumbnail_url: `/api/config/dataset-presets/image?file=${encodeURIComponent(name)}`,
      caption: {
        ok: true,
        file: name.replace(/\.png$/, ".txt"),
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
    }],
    row: {
      source_dir: `images/${root}/studio`,
      image_dir: `cache/${root}/studio`,
      num_repeats: 2,
      settings: {},
    },
    settings: { resolution: 1024, enable_bucket: true },
    message: "",
  };
}

function estimatePayload(root: string) {
  return {
    total_steps: 40,
    train_image_count: 1,
    repeated_image_count: 2,
    effective_batch_size: 1,
    steps_per_epoch: 2,
    train_batch_size: 1,
    gradient_accumulation_steps: 1,
    duration_mode: "steps",
    datasets: [{
      index: 0,
      source_dir: `images/${root}/estimate`,
      image_dir: `cache/${root}/estimate`,
      train_image_count: 1,
      num_repeats: 2,
      sample_ratio: 1,
      sampled_image_count: 1,
      sampled_weighted_image_count: 2,
      trigger_clone_sampled_weighted_image_count: 0,
      uses_preprocessed_images: false,
      bucket_distribution: {
        basis: "source_projection",
        status: "ready",
        image_count: 1,
        unreadable_count: 0,
        buckets: [{ width: 1024, height: 1024, count: 1 }],
      },
    }],
  };
}

test("config root switch refreshes estimate and picker preview before preflight", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let activeRoot = "configs";
  const settingsWrites: unknown[] = [];
  const estimateReads: Array<{ root: string; configFile: string | null }> = [];
  const presetReads: Array<{ root: string; file: string | null }> = [];
  const previewReads: Array<{ root: string; params: string }> = [];
  const preflightReads: Array<{ root: string; body: Record<string, unknown> }> = [];

  let announceEstimateRead = () => {};
  const externalEstimateReadStarted = new Promise<void>((resolve) => {
    announceEstimateRead = resolve;
  });
  let releaseEstimateRead = () => {};
  const externalEstimateReadGate = new Promise<void>((resolve) => {
    releaseEstimateRead = resolve;
  });
  let announcePreviewRead = () => {};
  const externalPreviewReadStarted = new Promise<void>((resolve) => {
    announcePreviewRead = resolve;
  });
  let releasePreviewRead = () => {};
  const externalPreviewReadGate = new Promise<void>((resolve) => {
    releasePreviewRead = resolve;
  });

  await page.addInitScript(() => {
    window.confirm = () => true;
  });
  await page.route(
    (url) =>
      url.pathname === "/api/settings/global" ||
      url.pathname === "/api/config/steps" ||
      url.pathname === "/api/config/dataset-presets" ||
      url.pathname === "/api/config/dataset-presets/read" ||
      url.pathname === "/api/config/dataset-presets/images" ||
      url.pathname === "/api/training/preflight",
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
          return reply({ ...settingsPayload(activeRoot), ok: true, requires_reload: true });
        }
        return route.fallback();
      }

      if (url.pathname === "/api/training/preflight" && method === "POST") {
        preflightReads.push({
          root: activeRoot,
          body: route.request().postDataJSON() as Record<string, unknown>,
        });
        return reply({
          ok: true,
          variant: "lora",
          preset: "default",
          methods_subdir: "imported",
          summary: { errors: 0, warnings: 0, checks: 1 },
          checks: [{ level: "ok", key: "model", message: "Fixture ready" }],
          errors: [],
          warnings: [],
        });
      }
      if (method !== "GET") return route.fallback();
      if (url.pathname === "/api/config/steps") {
        estimateReads.push({ root: activeRoot, configFile: url.searchParams.get("config_file") });
        if (activeRoot === "external-configs") {
          announceEstimateRead();
          await externalEstimateReadGate;
        }
        return reply(estimatePayload(activeRoot));
      }
      if (url.pathname === "/api/config/dataset-presets")
        return reply(datasetPayload(activeRoot));
      if (url.pathname === "/api/config/dataset-presets/read") {
        presetReads.push({ root: activeRoot, file: url.searchParams.get("file") });
        return reply(presetPayload(activeRoot));
      }
      if (url.pathname === "/api/config/dataset-presets/images") {
        previewReads.push({ root: activeRoot, params: url.searchParams.toString() });
        if (activeRoot === "external-configs") {
          announcePreviewRead();
          await externalPreviewReadGate;
        }
        return reply(previewPayload(activeRoot));
      }
      return route.fallback();
    },
  );

  await page.goto("/next/training");
  await page.getByRole("button", { name: "训练量估算", exact: true }).click();
  const estimateDialog = page.getByRole("dialog", { name: "训练量估算" });
  await expect(estimateDialog.getByText("images/configs/estimate", { exact: true })).toBeVisible();
  await estimateDialog.getByRole("button", { name: "关闭", exact: true }).click();

  await page.getByRole("button", { name: "选择与配置数据集", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "选择与配置数据集" });
  await expect(dialog.getByRole("button", { name: "预览子集 1 图像 configs-picker.png" })).toBeVisible();
  await dialog.getByRole("button", { name: "取消", exact: true }).click();
  await expect(dialog).not.toBeVisible();

  await page.getByRole("link", { name: "全局设置", exact: true }).click();
  await page.getByRole("textbox", { name: /配置根目录/ }).fill("external-configs");
  await page.getByRole("button", { name: "保存设置", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("现有文件未迁移");
  expect(settingsWrites).toEqual([{
    configs_root: "external-configs",
    revision: "settings-configs",
  }]);

  await page.getByRole("link", { name: "训练配置", exact: true }).click();
  await page.getByRole("button", { name: "训练量估算", exact: true }).click();
  const refreshedEstimateDialog = page.getByRole("dialog", { name: "训练量估算" });
  await externalEstimateReadStarted;
  await expect(refreshedEstimateDialog.getByText("images/configs/estimate", { exact: true })).toHaveCount(0);
  releaseEstimateRead();
  await expect(refreshedEstimateDialog.getByText("images/external-configs/estimate", { exact: true })).toBeVisible();
  await refreshedEstimateDialog.getByRole("button", { name: "关闭", exact: true }).click();

  await page.getByRole("button", { name: "选择与配置数据集", exact: true }).click();
  const refreshedDialog = page.getByRole("dialog", { name: "选择与配置数据集" });
  await externalPreviewReadStarted;
  await expect(refreshedDialog.getByRole("button", { name: "预览子集 1 图像 configs-picker.png" })).toHaveCount(0);
  await expect(refreshedDialog.getByRole("button", { name: "预览子集 1 图像 external-configs-picker.png" })).toHaveCount(0);
  releasePreviewRead();
  await expect(refreshedDialog.getByRole("button", { name: "预览子集 1 图像 external-configs-picker.png" })).toBeVisible();
  await refreshedDialog.getByRole("button", { name: "取消", exact: true }).click();
  await page.getByRole("button", { name: "运行预检测", exact: true }).click();
  const validationDialog = page.getByRole("dialog", { name: "预览与预检" });
  await expect(validationDialog.locator(".training-preflight-state")).toHaveText("可以继续");

  expect(estimateReads.some((read) => read.root === "configs" && read.configFile === configFile.path)).toBe(true);
  expect(estimateReads.at(-1)).toEqual({ root: "external-configs", configFile: configFile.path });
  expect(presetReads.some((read) => read.root === "configs" && read.file === datasetFile)).toBe(true);
  expect(presetReads.at(-1)).toEqual({ root: "external-configs", file: datasetFile });
  expect(presetReads.every((read) => read.file === datasetFile)).toBe(true);
  const expectedPreviewParams = `file=${encodeURIComponent(datasetFile)}&dataset_index=0&source=source&limit=8`;
  expect(previewReads.some((read) => read.root === "configs")).toBe(true);
  expect(previewReads.at(-1)).toEqual({ root: "external-configs", params: expectedPreviewParams });
  expect(previewReads.every((read) => read.params === expectedPreviewParams)).toBe(true);
  expect(preflightReads).toHaveLength(1);
  expect(preflightReads[0]).toMatchObject({
    root: "external-configs",
    body: {
      config_file: configFile.path,
      variant: "lora",
      preset: "default",
      methods_subdir: "imported",
    },
  });
  expect(settingsWrites).toHaveLength(1);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});
