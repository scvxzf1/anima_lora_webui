import type { Page } from "@playwright/test";

export const configFile = {
  path: "configs/imported/studio-portrait.toml",
  filename: "studio-portrait.toml",
  label: "Studio portrait",
  method: "lora",
  methods_subdir: "imported",
  trainable: true,
  locked: false,
};
const model = {
  id: "model-a",
  name: "Krea-2 Studio",
  model_family: "krea2_raw",
  pretrained_model_name_or_path:
    "models/diffusion_models/krea2_raw.safetensors",
  qwen3: "models/text_encoders/qwen3vl.safetensors",
  vae: "models/vae/qwen.safetensors",
  complete: true,
};
const config = {
  ...model,
  id: undefined,
  name: undefined,
  complete: undefined,
  output_name: "studio-portrait",
  network_dim: 32,
  network_alpha: 32,
  train_batch_size: 1,
  gradient_accumulation_steps: 4,
  learning_rate: 0.00002,
  max_train_steps: 1600,
  dataset_config: "configs/datasets/studio.toml",
  gradient_checkpointing: true,
  base_compute: "nf4",
  network_args: ["custom_extension=preserved"],
};
const metrics = Array.from({ length: 160 }, (_, i) => ({
  step: i * 10,
  loss: 0.19 * Math.exp(-i / 55) + 0.05 + Math.sin(i) * 0.01,
  lr: 0.00002,
  timestamp: i,
}));
const task = {
  id: "fixture-run",
  name: "Studio portrait / rank 32",
  job: "training",
  state: "idle",
  group: "Portrait studies",
  history_source_config_file: configFile.path,
  started_at_text: "2026-09-09 10:30",
  run_dir: "output/runs/studio-portrait",
  metric_count: 160,
  log_count: 1000,
};
const profiles = {
  active_profile_id: "provider-a",
  profiles: [
    {
      id: "provider-a",
      name: "Studio captions",
      provider: "openai_compatible",
      kind: "external",
      status: "可用",
      available: true,
      api_key_configured: true,
      config: {
        base_url: "https://example.invalid/v1",
        model: "fixture-model",
      },
    },
  ],
  provider_types: [
    { id: "openai_compatible", label: "OpenAI Compatible", kind: "external" },
  ],
};
const dataset = {
  path: "configs/datasets/studio.toml",
  filename: "studio.toml",
  label: "Studio / 48 images",
  summary: { dataset_count: 1, repeat_total: 48 },
  locked: false,
};

export async function mockWorkspace(page: Page) {
  const writes: { path: string; body: unknown }[] = [];
  const unhandled: string[] = [];
  const imageData = await page.evaluate(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 480;
    canvas.height = 480;
    const ctx = canvas.getContext("2d")!;
    ctx.fillStyle = "#d4e0e1";
    ctx.fillRect(0, 0, 480, 480);
    ctx.fillStyle = "#4c7b78";
    ctx.fillRect(30, 30, 200, 420);
    ctx.fillStyle = "#1c343a";
    ctx.beginPath();
    ctx.moveTo(240, 80);
    ctx.lineTo(450, 390);
    ctx.lineTo(245, 390);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = "#c7656d";
    ctx.fillRect(70, 130, 115, 205);
    ctx.fillStyle = "#fff";
    ctx.font = "20px sans-serif";
    ctx.fillText("STUDIO / 01", 270, 445);
    return canvas.toDataURL("image/png").split(",")[1];
  });
  const job = {
    id: "caption-1",
    state: "done",
    profile_name: "Studio captions",
    profile_id: "provider-a",
    dataset_file: dataset.path,
    total: 3,
    completed: 3,
    failed: 0,
    items: Array.from({ length: 3 }, (_, i) => ({
      id: `image-${i}`,
      name: `studio-${i + 1}.png`,
      file: `studio-${i + 1}.png`,
      url: `/api/config/dataset-presets/image?image=${i}`,
      state: "ready",
      caption: "abstract still life, geometric composition",
      proposed_caption:
        "Studio composition with crisp shapes, muted teal and a red accent.",
    })),
  };
  await page.routeWebSocket("**/ws/training", (ws) => {
    ws.onMessage(() => {});
  });
  await page.route(
    (url) => url.pathname.startsWith("/api/"),
    async (route) => {
      const url = new URL(route.request().url());
      const path = url.pathname;
      const method = route.request().method();
      const reply = (data: unknown, status = 200) =>
        route.fulfill({
          status,
          contentType: "application/json",
          body: JSON.stringify(data),
        });
      if (method !== "GET") {
        const body = route.request().postDataJSON();
        writes.push({ path, body });
        if (path === "/api/config/raw")
          return reply({ ok: false, error: "Fixture: readonly disk" }, 409);
        if (path === "/api/training/preflight")
          return reply({
            ok: true,
            summary: { errors: 0, warnings: 0, checks: 1 },
            checks: [{ level: "ok", key: "model", message: "Fixture ready" }],
          });
        return reply(
          { ok: false, error: `Fixture blocked command: ${path}` },
          409,
        );
      }
      if (path === "/api/config/dataset-presets/cover")
        return reply({ ok: true, image: `data:image/png;base64,${imageData}`, reason: "" });
      if (
        path.endsWith("/dataset-presets/image") ||
        path === "/api/preview/image"
      )
        return route.fulfill({
          contentType: "image/png",
          body: Buffer.from(imageData, "base64"),
        });
      if (path === "/api/config/file-groups")
        return reply([
          { id: "imported", label: "Studio presets", files: [configFile] },
        ]);
      if (path === "/api/presets") return reply(["default", "low_vram"]);
      if (path === "/api/config/merged") return reply(config);
      if (path === "/api/config/raw")
        return reply({
          file: configFile.path,
          content: 'output_name = "studio-portrait"\nnetwork_dim = 32\n',
          meta: configFile,
        });
      if (path === "/api/config/model-families")
        return reply({
          items: [
            { name: "anima", supported_attention_modes: ["flash", "torch"] },
            {
              name: "krea2_raw",
              supported_attention_modes: ["flash", "torch", "sdpa"],
            },
            { name: "z_image", supported_attention_modes: ["flash", "torch"] },
          ],
        });
      if (path === "/api/config/steps")
        return reply({
          total_steps: 1600,
          steps_per_epoch: 12,
          train_image_count: 48,
          effective_batch_size: 4,
        });
      if (path === "/api/config/sample-prompts")
        return reply({
          file: "configs/sample_prompts.txt",
          content: "# Studio\nabstract still life --w 1024 --h 1024\n",
          prompts: ["abstract still life"],
        });
      if (path === "/api/settings/model-configs")
        return reply({
          items: [model],
          groups: [
            { id: "group-a", label: "Local models", item_ids: [model.id] },
          ],
          revision: "fixture-revision",
          default_id: model.id,
        });
      if (path === "/api/settings/global")
        return reply({
          output_root: "output/runs",
          ui_scale: 100,
          tagging_max_retained_jobs: 50,
          path_overrides: {
            configs_root: "",
            history_root: "",
            queue_root: "",
          },
          effective_paths: {
            configs_root: "/workspace/configs",
            history_root: "/workspace/history",
            queue_root: "/workspace/queue",
          },
          defaults: { output_root: "output/runs", ui_scale: 100 },
        });
      if (path === "/api/config/dataset-presets")
        return reply({
          ok: true,
          presets: [dataset],
          groups: [{ id: "studio", label: "Studio", files: [dataset] }],
        });
      if (path === "/api/config/dataset-presets/read")
        return reply({
          ok: true,
          file: dataset.path,
          name: "studio",
          content: "",
          datasets: [
            {
              source_dir: "images/studio",
              image_dir: "cache/studio",
              num_repeats: 1,
              settings: {},
            },
          ],
          defaults: { resolution: 1024, batch_size: 1 },
          readonly: false,
          summary: dataset.summary,
        });
      if (path === "/api/config/dataset-presets/images")
        return reply({
          ok: true,
          total: 3,
          images: job.items.map((item) => ({
            ...item,
            caption: { text: item.caption },
          })),
        });
      if (path === "/api/training/status")
        return reply({
          status: "running",
          task_id: "current-fixture",
          variant: "lora",
          preset: "default",
          job: "training",
          latest_progress: {
            current: 840,
            total: 1600,
            loss: 0.094,
            lr: 0.00002,
            rate: "2.73s/it",
          },
          latest_system: {
            vram_used_gb: 11.3,
            vram_total_gb: 24,
            gpu_temp: 67,
            gpu_util: 98,
          },
        });
      if (path === "/api/training/metrics") return reply(metrics);
      if (path === "/api/training/logs")
        return reply({
          records: [{ id: 1, line: "[training] step 840 / 1600 · loss=0.094" }],
        });
      if (path === "/api/training/gpus")
        return reply({
          gpus: [{ index: 0, name: "Fixture GPU", memory_total_gb: 24 }],
        });
      if (path === "/api/training/queue")
        return reply({
          ok: true,
          paused: true,
          status: "idle",
          summary: { total: 3, queued: 2, error: 1 },
          items: [0, 1, 2].map((i) => ({
            id: `queue-${i}`,
            state: i === 2 ? "error" : "queued",
            variant: "lora",
            preset: "default",
            source_config_file: configFile.path,
            runtime_config_file: `output/runs/queue-${i}/runtime.toml`,
            attempt: 1,
            max_attempts: 2,
            message:
              i === 2 ? "Out of memory: review the memory configuration" : "",
          })),
        });
      if (path === "/api/training/history")
        return reply({ ok: true, tasks: [task] });
      if (path === "/api/training/history/collections/settings")
        return reply({
          collection_order: ["Portrait studies"],
          config_group_order: {},
        });
      if (path === "/api/training/history/fixture-run")
        return reply({
          ok: true,
          task,
          metrics,
          logs: [{ line: "[training] completed 1600 steps" }],
          system: [],
          config_toml: "network_dim = 32",
        });
      if (path === "/api/training/history/fixture-run/artifacts")
        return reply({ ok: true, task_id: "fixture-run", artifacts: [
          { key: "logs", name: "logs.jsonl", state: "available", size_bytes: 1000 },
          { key: "runtime-config", state: "missing", message: "未保存或文件已不存在" },
        ] });
      if (path === "/api/preview/images")
        return reply({
          images: [{ name: "step-1000.png", file: "step-1000.png" }],
        });
      if (path === "/api/preview/weights")
        return reply({
          weights: [
            {
              file: "model.safetensors",
              name: "model.safetensors",
              size_bytes: 96000000,
            },
          ],
        });
      if (path === "/api/captioning/profiles") return reply(profiles);
      if (path === "/api/captioning/jobs") return reply({ jobs: [job] });
      if (path === "/api/captioning/jobs/caption-1")
        return reply({ ok: true, job });
      if (path === "/api/captioning/logs") return reply({ lines: [] });
      if (path === "/api/captioning/prompt-presets")
        return reply({
          presets: [
            {
              id: "prompt",
              name: "Studio captions",
              system_prompt: "",
              user_prompt: "Describe the image.",
              builtin: true,
            },
          ],
        });
      if (path === "/api/captioning/model-assets")
        return reply({
          assets: [
            {
              id: "fixture-asset",
              label: "Local caption model",
              state: "installed",
              installed: true,
              total_size: 400000000,
              repo_id: "fixture/model",
              license: "Apache 2.0",
            },
          ],
          downloads: [],
        });
      if (path === "/api/captioning/tag-dictionary")
        return reply({
          state: "missing",
          installed: false,
          source_name: "Tag dictionary",
          entry_count: 0,
          download_size: 40000000,
        });
      unhandled.push(`${method} ${path}`);
      return reply({ ok: false, error: `Unhandled fixture: ${path}` }, 501);
    },
  );
  return { writes, unhandled };
}
