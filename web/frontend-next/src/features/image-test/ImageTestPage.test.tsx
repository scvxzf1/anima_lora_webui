import { cleanup, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { jsonResponse, renderInApp } from "../../test/renderInApp";
import { ImageTestPage } from "./ImageTestPage";
import { IMAGE_TEST_STORAGE_KEY } from "./storage";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); localStorage.clear(); });

function apiFetcher(config: Record<string, unknown> = { model_family: "anima", pretrained_model_name_or_path: "model.safetensors" }) {
  return vi.fn(async (input: RequestInfo | URL) => {
    const path = String(input);
    if (path.includes("/api/config/file-groups")) return jsonResponse([{ id: "local", label: "Local", files: [{ path: "configs/gui-methods/test.toml", label: "Test", method: "lora", methods_subdir: "gui-methods" }] }]);
    if (path === "/api/presets") return jsonResponse(["default"]);
    if (path.includes("/api/config/merged")) return jsonResponse(config);
    if (path === "/api/image-test/status") return jsonResponse({ ok: true, status: "idle", running: false, output_count: 0 });
    if (path === "/api/analysis/weights") return jsonResponse({ weights: [{ file: "lora.safetensors", abs_path: "/models/lora.safetensors", name: "LoRA" }] });
    if (path.startsWith("/api/preview/images")) return jsonResponse({ ok: true, images: [] });
    if (path === "/api/training/gpus") return jsonResponse({ gpus: [{ index: 2, name: "GPU 2" }] });
    throw new Error(`Unexpected ${path}`);
  });
}

it("starts inference with a saved merged configuration", async () => {
  const fetcher = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input);
    if (path.includes("/api/config/file-groups")) return jsonResponse([{ id: "local", label: "Local", files: [{ path: "configs/gui-methods/test.toml", label: "Test", method: "lora", methods_subdir: "gui-methods" }] }]);
    if (path === "/api/presets") return jsonResponse(["default"]);
    if (path.includes("/api/config/merged")) return jsonResponse({ model_family: "krea2_raw", pretrained_model_name_or_path: "model.safetensors", attn_mode: "sdpa", sample_sampler: "er_sde", flow_shift: 3 });
    if (path === "/api/image-test/status") return jsonResponse({ ok: true, status: "idle", running: false, output_count: 0 });
    if (path === "/api/analysis/weights") return jsonResponse({ weights: [] });
    if (path.startsWith("/api/preview/images")) return jsonResponse({ ok: true, images: [] });
    if (path === "/api/training/gpus") return jsonResponse({ gpus: [] });
    if (path === "/api/image-test/start" && init?.method === "POST") return jsonResponse({ ok: true, status: "running", running: true });
    throw new Error(`Unexpected ${path}`);
  });
  vi.stubGlobal("fetch", fetcher);
  renderInApp(<ImageTestPage />);
  const user = userEvent.setup();
  await screen.findByRole("option", { name: "Test" }, { timeout: 10000 });
  await user.selectOptions(screen.getByLabelText("训练配置"), "configs/gui-methods/test.toml");
  await screen.findByRole("button", { name: "开始生成" });
  await waitFor(() => expect(screen.getByRole("button", { name: "开始生成" })).toBeEnabled());
  await waitFor(() => expect(screen.getByLabelText("注意力后端")).toHaveValue("torch"));
  expect(screen.getByLabelText("采样器")).toHaveValue("euler");
  expect(screen.getByLabelText("采样器")).toBeDisabled();
  const attention = screen.getByLabelText("注意力后端") as HTMLSelectElement;
  expect(Array.from(attention.options, (option) => option.value)).toEqual(["flash", "torch"]);
  await user.type(screen.getByLabelText("正向提示词"), "test prompt");
  await user.click(screen.getByRole("button", { name: "开始生成" }));
  await waitFor(() => expect(fetcher).toHaveBeenCalledWith("/api/image-test/start", expect.objectContaining({ method: "POST" })));
  const call = fetcher.mock.calls.find(([path]) => path === "/api/image-test/start")!;
  const body = JSON.parse(call[1]!.body as string);
  expect(body.config.model_family).toBe("krea2_raw");
  expect(body.sampler).toBe("euler");
  expect(body.attn_mode).toBe("torch");
  expect(body.flow_shift).toBe("");
  expect(body.prompt).toBe("test prompt");
});

it("restores draft and history range after a real page remount", async () => {
  vi.stubGlobal("fetch", apiFetcher({ model_family: "anima" }));
  const first = renderInApp(<ImageTestPage />);
  const user = userEvent.setup();
  await screen.findByRole("option", { name: "Test" });
  await user.selectOptions(screen.getByLabelText("训练配置"), "configs/gui-methods/test.toml");
  await user.type(screen.getByLabelText("正向提示词"), "persist me");
  await user.selectOptions(screen.getByLabelText("时间范围"), "30");
  await waitFor(() => expect(JSON.parse(localStorage.getItem(IMAGE_TEST_STORAGE_KEY)!).file_path).toBe("configs/gui-methods/test.toml"));
  await waitFor(() => expect(localStorage.getItem(IMAGE_TEST_STORAGE_KEY)).toContain('"history_range":"30"'));
  first.unmount();

  renderInApp(<ImageTestPage />);
  await screen.findByRole("option", { name: "Test" });
  expect(await screen.findByLabelText("正向提示词")).toHaveValue("persist me");
  expect(screen.getByLabelText("训练配置")).toHaveValue("configs/gui-methods/test.toml");
  expect(screen.getByLabelText("时间范围")).toHaveValue("30");
});

it("keeps restored inputs when configuration and option lists resolve later", async () => {
  localStorage.setItem(IMAGE_TEST_STORAGE_KEY, JSON.stringify({
    version: 2, file_path: "configs/gui-methods/test.toml", preset: "default", history_range: "14", dirty_fields: ["prompt"],
    draft: { prompt: "restored prompt", width: "768", gpu_index: "2", weight_path: "/models/lora.safetensors", sampler: "lcm", runtime_dtype: "bf16" },
  }));
  let resolveConfig!: (response: Response) => void;
  let resolveGpus!: (response: Response) => void;
  let resolveWeights!: (response: Response) => void;
  const fetcher = vi.fn((input: RequestInfo | URL) => {
    const path = String(input);
    if (path.includes("/api/config/merged")) return new Promise<Response>((resolve) => { resolveConfig = resolve; });
    if (path === "/api/training/gpus") return new Promise<Response>((resolve) => { resolveGpus = resolve; });
    if (path === "/api/analysis/weights") return new Promise<Response>((resolve) => { resolveWeights = resolve; });
    if (path.includes("/api/config/file-groups")) return Promise.resolve(jsonResponse([{ id: "local", files: [{ path: "configs/gui-methods/test.toml", label: "Test" }] }]));
    if (path === "/api/presets") return Promise.resolve(jsonResponse(["default"]));
    if (path === "/api/image-test/status") return Promise.resolve(jsonResponse({ ok: true, status: "idle", running: false }));
    if (path.startsWith("/api/preview/images")) return Promise.resolve(jsonResponse({ ok: true, images: [] }));
    throw new Error(`Unexpected ${path}`);
  });
  vi.stubGlobal("fetch", fetcher);
  renderInApp(<ImageTestPage />);
  await waitFor(() => expect(resolveConfig).toBeDefined());
  expect(screen.getByLabelText("正向提示词")).toHaveValue("restored prompt");
  expect(screen.getByLabelText("宽度")).toHaveValue(768);
  await waitFor(() => expect(resolveGpus).toBeDefined());
  await waitFor(() => expect(resolveWeights).toBeDefined());
  resolveConfig(jsonResponse({ model_family: "anima", resolution: 512, sample_steps: 40, sample_sampler: "euler", attn_mode: "torch", precision_preference: "fp16" }));
  resolveGpus(jsonResponse({ gpus: [{ index: 2, name: "GPU 2" }] }));
  resolveWeights(jsonResponse({ weights: [{ file: "lora.safetensors", abs_path: "/models/lora.safetensors", name: "LoRA" }] }));
  await screen.findByRole("option", { name: "Test" });
  await waitFor(() => expect(screen.getByLabelText("GPU")).toHaveValue("2"));
  expect(screen.getByLabelText("正向提示词")).toHaveValue("restored prompt");
  expect(screen.getByLabelText("宽度")).toHaveValue(512);
  expect(screen.getByLabelText("GPU")).toHaveValue("2");
  expect(screen.getByPlaceholderText("可选；从列表选择或输入路径")).toHaveValue("/models/lora.safetensors");
  expect(screen.getByLabelText("时间范围")).toHaveValue("14");
});

it("keeps a custom weight path when it is absent from the candidate list", async () => {
  localStorage.setItem(IMAGE_TEST_STORAGE_KEY, JSON.stringify({ version: 2, file_path: "configs/gui-methods/test.toml", draft: { prompt: "go", weight_path: "/custom/not-listed.safetensors" }, dirty_fields: ["prompt", "weight_path"] }));
  const fetcher = apiFetcher();
  vi.stubGlobal("fetch", fetcher);
  renderInApp(<ImageTestPage />);
  const weight = await screen.findByPlaceholderText("可选；从列表选择或输入路径");
  await waitFor(() => expect(weight).toHaveValue("/custom/not-listed.safetensors"));
  await waitFor(() => expect(screen.getByRole("button", { name: "开始生成" })).toBeEnabled());
});

it("applies the selected configuration defaults after remount when fields were never edited", async () => {
  const fetcher = apiFetcher({ model_family: "anima", resolution: 512, sample_steps: 50 });
  vi.stubGlobal("fetch", fetcher);
  const first = renderInApp(<ImageTestPage />);
  await screen.findByRole("option", { name: "Test" });
  await userEvent.setup().selectOptions(screen.getByLabelText("训练配置"), "configs/gui-methods/test.toml");
  await waitFor(() => expect(screen.getByLabelText("宽度")).toHaveValue(512));
  first.unmount();
  renderInApp(<ImageTestPage />);
  await waitFor(() => expect(screen.getByLabelText("宽度")).toHaveValue(512));
  expect(screen.getByLabelText("采样步数")).toHaveValue(50);
});

it("releases derived-field edits when switching configurations and restores the new defaults on remount", async () => {
  const fetcher = vi.fn(async (input: RequestInfo | URL) => {
    const path = String(input);
    if (path.includes("/api/config/file-groups")) return jsonResponse([{ id: "local", files: [
      { path: "configs/gui-methods/a.toml", label: "Config A" }, { path: "configs/gui-methods/b.toml", label: "Config B" },
    ] }]);
    if (path === "/api/presets") return jsonResponse(["default"]);
    if (path.includes("/api/config/merged")) return jsonResponse({ model_family: "anima", resolution: path.includes("b.toml") ? 768 : 512, sample_steps: path.includes("b.toml") ? 60 : 40 });
    if (path === "/api/image-test/status") return jsonResponse({ ok: true, status: "idle", running: false });
    if (path === "/api/analysis/weights") return jsonResponse({ weights: [] });
    if (path.startsWith("/api/preview/images")) return jsonResponse({ ok: true, images: [] });
    if (path === "/api/training/gpus") return jsonResponse({ gpus: [{ index: 2, name: "GPU 2" }] });
    throw new Error(`Unexpected ${path}`);
  });
  vi.stubGlobal("fetch", fetcher);
  const first = renderInApp(<ImageTestPage />);
  const user = userEvent.setup();
  await screen.findByRole("option", { name: "Config A" });
  await user.selectOptions(screen.getByLabelText("训练配置"), "configs/gui-methods/a.toml");
  await waitFor(() => expect(screen.getByLabelText("宽度")).toHaveValue(512));
  await user.clear(screen.getByLabelText("宽度"));
  await user.type(screen.getByLabelText("宽度"), "900");
  await user.selectOptions(screen.getByLabelText("训练配置"), "configs/gui-methods/b.toml");
  await waitFor(() => expect(screen.getByLabelText("宽度")).toHaveValue(768));
  expect(screen.getByLabelText("采样步数")).toHaveValue(60);
  first.unmount();
  renderInApp(<ImageTestPage />);
  await waitFor(() => expect(screen.getByLabelText("宽度")).toHaveValue(768));
});

it("does not overwrite a field edited while its configuration response is pending", async () => {
  localStorage.setItem(IMAGE_TEST_STORAGE_KEY, JSON.stringify({ version: 2, file_path: "configs/gui-methods/test.toml", draft: { prompt: "go" }, dirty_fields: ["prompt"] }));
  let resolveConfig!: (response: Response) => void;
  const fetcher = vi.fn((input: RequestInfo | URL) => {
    const path = String(input);
    if (path.includes("/api/config/merged")) return new Promise<Response>((resolve) => { resolveConfig = resolve; });
    if (path.includes("/api/config/file-groups")) return Promise.resolve(jsonResponse([{ id: "local", files: [{ path: "configs/gui-methods/test.toml", label: "Test" }] }]));
    if (path === "/api/presets") return Promise.resolve(jsonResponse(["default"]));
    if (path === "/api/image-test/status") return Promise.resolve(jsonResponse({ ok: true, status: "idle", running: false }));
    if (path === "/api/analysis/weights") return Promise.resolve(jsonResponse({ weights: [] }));
    if (path === "/api/training/gpus") return Promise.resolve(jsonResponse({ gpus: [{ index: 2, name: "GPU 2" }] }));
    if (path.startsWith("/api/preview/images")) return Promise.resolve(jsonResponse({ ok: true, images: [] }));
    throw new Error(`Unexpected ${path}`);
  });
  vi.stubGlobal("fetch", fetcher);
  renderInApp(<ImageTestPage />);
  await waitFor(() => expect(resolveConfig).toBeDefined());
  await userEvent.setup().clear(screen.getByLabelText("宽度"));
  await userEvent.setup().type(screen.getByLabelText("宽度"), "900");
  resolveConfig(jsonResponse({ model_family: "anima", resolution: 512, sample_steps: 50 }));
  await waitFor(() => expect(screen.getByLabelText("宽度")).toHaveValue(900));
});

it("preserves unavailable GPUs and cannot bypass lookup failure by selecting automatic", async () => {
  localStorage.setItem(IMAGE_TEST_STORAGE_KEY, JSON.stringify({ version: 2, file_path: "configs/gui-methods/test.toml", draft: { prompt: "go", gpu_index: "9" }, dirty_fields: ["prompt", "gpu_index"] }));
  const fetcher = vi.fn(async (input: RequestInfo | URL) => {
    const path = String(input);
    if (path === "/api/training/gpus") throw new Error("offline");
    if (path.includes("/api/config/file-groups")) return jsonResponse([{ id: "local", files: [{ path: "configs/gui-methods/test.toml", label: "Test" }] }]);
    if (path === "/api/presets") return jsonResponse(["default"]);
    if (path.includes("/api/config/merged")) return jsonResponse({ model_family: "anima" });
    if (path === "/api/image-test/status") return jsonResponse({ ok: true, status: "idle", running: false });
    if (path === "/api/analysis/weights") return jsonResponse({ weights: [] });
    if (path.startsWith("/api/preview/images")) return jsonResponse({ ok: true, images: [] });
    throw new Error(`Unexpected ${path}`);
  });
  vi.stubGlobal("fetch", fetcher);
  renderInApp(<ImageTestPage />);
  expect(await screen.findByText(/GPU 列表读取失败/)).toBeInTheDocument();
  expect(screen.getByLabelText("GPU")).toHaveValue("9");
  expect(screen.getByRole("button", { name: "开始生成" })).toBeDisabled();
  await userEvent.setup().selectOptions(screen.getByLabelText("GPU"), "");
  expect(screen.getByRole("button", { name: "开始生成" })).toBeDisabled();
});

it("retries a failed GPU lookup from the visible error", async () => {
  let gpuRequests = 0;
  const fetcher = vi.fn(async (input: RequestInfo | URL) => {
    const path = String(input);
    if (path.startsWith("/api/training/gpus")) {
      gpuRequests++;
      if (gpuRequests === 1) throw new Error("temporary failure");
      expect(path).toBe("/api/training/gpus?refresh=1");
      return jsonResponse({ gpus: [{ index: 2, name: "GPU 2" }] });
    }
    if (path.includes("/api/config/file-groups")) return jsonResponse([{ id: "local", files: [{ path: "configs/gui-methods/test.toml", label: "Test" }] }]);
    if (path === "/api/presets") return jsonResponse(["default"]);
    if (path.includes("/api/config/merged")) return jsonResponse({ model_family: "anima" });
    if (path === "/api/image-test/status") return jsonResponse({ ok: true, status: "idle", running: false });
    if (path === "/api/analysis/weights") return jsonResponse({ weights: [] });
    if (path.startsWith("/api/preview/images")) return jsonResponse({ ok: true, images: [] });
    throw new Error(`Unexpected ${path}`);
  });
  vi.stubGlobal("fetch", fetcher);
  renderInApp(<ImageTestPage />);
  await screen.findByText(/GPU 列表读取失败/);
  await userEvent.setup().click(screen.getByRole("button", { name: "重试" }));
  await waitFor(() => expect(screen.getByLabelText("GPU").querySelectorAll("option")).toHaveLength(2));
  expect(gpuRequests).toBe(2);
});

it.each(["", "2"])("blocks stale HTTP-success GPU snapshots for selection %s until a fresh retry", async (gpuIndex) => {
  localStorage.setItem(IMAGE_TEST_STORAGE_KEY, JSON.stringify({ version: 2, file_path: "configs/gui-methods/test.toml", draft: { prompt: "go", gpu_index: gpuIndex }, dirty_fields: ["prompt", "gpu_index"] }));
  let gpuRequests = 0;
  const base = apiFetcher();
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    if (String(input).startsWith("/api/training/gpus")) {
      gpuRequests++;
      if (gpuRequests > 1) expect(String(input)).toBe("/api/training/gpus?refresh=1");
      return jsonResponse({ gpus: [{ index: 2, name: "GPU 2" }], stale: gpuRequests === 1 });
    }
    return base(input);
  }));
  renderInApp(<ImageTestPage />);
  await screen.findByText(/采样未更新/);
  expect(screen.getByLabelText("GPU")).toHaveValue(gpuIndex);
  expect(screen.getByRole("button", { name: "开始生成" })).toBeDisabled();
  await userEvent.setup().selectOptions(screen.getByLabelText("GPU"), "");
  expect(screen.getByRole("button", { name: "开始生成" })).toBeDisabled();
  await userEvent.setup().click(screen.getByRole("button", { name: "重试" }));
  await waitFor(() => expect(screen.getByRole("button", { name: "开始生成" })).toBeEnabled());
  expect(gpuRequests).toBe(2);
});
