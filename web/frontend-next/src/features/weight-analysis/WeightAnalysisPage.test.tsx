import { cleanup, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { jsonResponse, renderInApp } from "../../test/renderInApp";
import { WeightAnalysisPage } from "./WeightAnalysisPage";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it("inspects a selected safetensors file through the analysis API", async () => {
  const fetcher = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    if (String(input) === "/api/analysis/weights") return jsonResponse({ weights: [{ file: "output/a.safetensors", name: "a.safetensors" }], count: 1 });
    if (String(input) === "/api/analysis/inspect" && init?.method === "POST") return jsonResponse({ ok: true, file: { name: "a.safetensors", path: "output/a.safetensors" }, adapter_type: "lora", summary: { layer_count: 2, block_count: 1, total_energy: 1.5 }, layers: [{ block: 2 }, { block: null }], component_summary: [{ label: "attention", layer_count: 2, fro_norm: 1.2 }], block_summary: [], style_top20: [], character_top20: [], heatmap: { blocks: [2], components: ["attention"], matrix: [[1.2]], max_value: 1.2, cells: [{ block: 2, component: "attention", fro_norm: 1.2, layer_count: 1, top_layer: "layers_2_attention", intensity: 1 }] } });
    throw new Error(`Unexpected ${input}`);
  });
  vi.stubGlobal("fetch", fetcher);
  renderInApp(<WeightAnalysisPage />);
  const user = userEvent.setup();
  await screen.findByText("a.safetensors", {}, { timeout: 10000 });
  await user.click(screen.getByRole("button", { name: "权重 A选择已有权重" }));
  expect(screen.getByRole("dialog", { name: "权重 A权重列表" })).toBeInTheDocument();
  await user.type(screen.getByRole("textbox", { name: "搜索权重 A权重" }), "a.safe");
  await user.click(within(screen.getByRole("dialog", { name: "权重 A权重列表" })).getByRole("button", { name: /a.safetensors/ }));
  expect(screen.getByRole("textbox", { name: "权重 A路径" })).toHaveValue("output/a.safetensors");
  await user.click(screen.getByRole("button", { name: "开始分析" }));
  const heatmap = await screen.findByRole("region", { name: "区块 × 组件热力图" });
  expect(within(heatmap).getByRole("columnheader", { name: "attention" })).toBeInTheDocument();
  expect(within(heatmap).getByRole("cell", { name: /Block 2.*范数: 1.2/ })).toBeInTheDocument();
  expect(screen.getByText(/纳入 1\/2 层/)).toBeInTheDocument();
  await user.click(screen.getByRole("tab", { name: "组件" }));
  expect(screen.getByText("attention")).toBeInTheDocument();
  expect(fetcher).toHaveBeenCalledWith("/api/analysis/inspect", expect.objectContaining({ method: "POST" }));
});

it("accepts a local safetensors file without exposing the native file input", async () => {
  const fetcher = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    if (String(input) === "/api/analysis/weights") return jsonResponse({ weights: [], count: 0 });
    if (String(input) === "/api/analysis/inspect-upload" && init?.body instanceof FormData) return jsonResponse({ ok: true, file: { name: "local.safetensors", path: "uploaded://local.safetensors" }, adapter_type: "lora", summary: { layer_count: 1 }, layers: [], component_summary: [], block_summary: [], style_top20: [], character_top20: [], heatmap: { blocks: [], components: [], matrix: [], max_value: 0, cells: [] } });
    throw new Error(`Unexpected ${input}`);
  });
  vi.stubGlobal("fetch", fetcher);
  renderInApp(<WeightAnalysisPage />);
  const user = userEvent.setup();
  await user.upload(screen.getByLabelText("上传权重 A文件"), new File(["test"], "local.safetensors"));
  expect(screen.getByRole("textbox", { name: "权重 A路径" })).toHaveValue("local.safetensors");
  await user.click(screen.getByRole("button", { name: "开始分析" }));
  await waitFor(() => expect(fetcher).toHaveBeenCalledWith("/api/analysis/inspect-upload", expect.objectContaining({ method: "POST" })));
  expect(await screen.findByText(/没有识别到带编号的 Block/)).toBeInTheDocument();
});

it("uses a shared color scale for both weights in comparison mode", async () => {
  const result = (name: string, value: number) => ({
    ok: true,
    file: { name, path: `output/${name}` },
    adapter_type: "LoRA",
    summary: { layer_count: 1, block_count: 1, total_energy: value ** 2 },
    layers: [{ block: 2 }],
    component_summary: [], block_summary: [], style_top20: [], character_top20: [],
    heatmap: { blocks: [2], components: ["attention"], matrix: [[value]], max_value: value, cells: [{ block: 2, component: "attention", fro_norm: value, layer_count: 1, top_layer: name, intensity: 1 }] },
  });
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    if (String(input) === "/api/analysis/weights") return jsonResponse({ weights: [], count: 0 });
    if (String(input) === "/api/analysis/inspect" && init?.body) {
      const path = JSON.parse(String(init.body)).path as string;
      return jsonResponse(path === "output/a.safetensors" ? result("a.safetensors", 2) : result("b.safetensors", 4));
    }
    throw new Error(`Unexpected ${input}`);
  }));
  renderInApp(<WeightAnalysisPage />);
  const user = userEvent.setup();
  await user.type(screen.getByRole("textbox", { name: "权重 A路径" }), "output/a.safetensors");
  await user.click(screen.getByRole("checkbox", { name: "A/B 对比" }));
  await user.type(screen.getByRole("textbox", { name: "权重 B路径" }), "output/b.safetensors");
  await user.click(screen.getByRole("button", { name: "开始分析" }));

  const first = await screen.findByRole("region", { name: "权重 A热力图" });
  const second = screen.getByRole("region", { name: "权重 B热力图" });
  expect(screen.getAllByText(/A\/B 共同最大值归一化/)).toHaveLength(2);
  expect(within(first).getByRole("cell").getAttribute("style")).toBe("background-color: color-mix(in srgb, var(--primary) 34%, var(--surface));");
  expect(within(second).getByRole("cell").getAttribute("style")).toBe("background-color: color-mix(in srgb, var(--primary) 58%, var(--surface));");
});
