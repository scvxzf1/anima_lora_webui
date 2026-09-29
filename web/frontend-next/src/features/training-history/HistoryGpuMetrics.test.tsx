import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { HistoryGpuMetrics } from "./HistoryGpuMetrics";

const { useMetricsCanvas } = vi.hoisted(() => ({ useMetricsCanvas: vi.fn() }));
vi.mock("../../components/useMetricsCanvas", () => ({ useMetricsCanvas }));

afterEach(() => {
  cleanup();
  useMetricsCanvas.mockClear();
  Object.defineProperty(HTMLElement.prototype, "clientWidth", { configurable: true, value: 0 });
  vi.unstubAllGlobals();
});

function mockChartWidth() {
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  Object.defineProperty(HTMLElement.prototype, "clientWidth", { configurable: true, get: () => 600 });
}

const points = [
  { ts: 100, vram_used_gb: 4, gpu_util: 20, gpu_temp: 60, per_gpu: [
    { index: 0, uuid: "a", name: "Card A", vram_used_gb: 3, gpu_util: 10, gpu_temp: 50 },
    { index: 1, uuid: "b", name: "Card B", vram_used_gb: 1, gpu_util: 5 },
  ] },
  { ts: 200, vram_used_gb: 16, gpu_util: 80, gpu_temp: 90, per_gpu: [
    { index: 0, uuid: "a", name: "Card A", vram_used_gb: 12, gpu_util: 70, gpu_temp: 85 },
    { index: 1, uuid: "b", name: "Card B", vram_used_gb: 8, gpu_util: 40 },
  ] },
];

it("plots available GPU metrics on time axes and switches from aggregate to the selected device", async () => {
  mockChartWidth();
  render(<HistoryGpuMetrics points={points} total={10} whitelist={[0]} />);

  expect(await screen.findByRole("group", { name: /显存 \(GB\)/ })).toBeInTheDocument();
  expect(screen.getByRole("group", { name: /GPU 利用率/ })).toBeInTheDocument();
  expect(screen.getByRole("group", { name: /GPU 温度/ })).toBeInTheDocument();
  const latestCall = (metricName: string) => useMetricsCanvas.mock.calls.filter((call) => call[3] === metricName).at(-1)!;
  expect(latestCall("显存 (GB)")).toMatchObject([expect.anything(), [[100000, 4], [200000, 16]], true, "显存 (GB)", "time", true, true]);
  expect(screen.getByRole("group", { name: "GPU 资源范围" })).toBeInTheDocument();

  fireEvent.click(screen.getByRole("button", { name: "GPU 0" }));
  expect(latestCall("显存 (GB)")[1]).toEqual([[100000, 3], [200000, 12]]);
  expect(screen.getByText("GPU 0 · Card A · 任务已选")).toBeInTheDocument();

  fireEvent.click(screen.getByRole("button", { name: "GPU 1" }));
  expect(screen.getByRole("checkbox", { name: "显存 (GB)" })).toBeInTheDocument();
  expect(screen.getByRole("checkbox", { name: "GPU 利用率 (%)" })).toBeInTheDocument();
  expect(screen.queryByRole("checkbox", { name: "GPU 温度 (°C)" })).not.toBeInTheDocument();
  expect(latestCall("显存 (GB)")[1]).toEqual([[100000, 1], [200000, 8]]);
  expect(screen.getByText("GPU 1 · Card B · 未选用")).toBeInTheDocument();
});

it("shows a labeled empty state when no GPU metrics are available", () => {
  render(<HistoryGpuMetrics points={[{ ts: 100, per_gpu: [] }]} />);
  expect(screen.getByRole("region", { name: "GPU 资源历史" })).toBeInTheDocument();
  expect(screen.getByText("暂无 GPU 资源记录")).toBeInTheDocument();
  expect(screen.queryByRole("group", { name: /显存/ })).not.toBeInTheDocument();
});

it("shows latest and peak GPU readings, recorded total VRAM, and joint values at one sample", () => {
  render(<HistoryGpuMetrics points={[
    { ts: 100, vram_used_gb: 3, vram_total_gb: 8, gpu_util: 20, gpu_temp: 50 },
    { ts: 200, vram_used_gb: 6, vram_total_gb: 8, gpu_util: 80, gpu_temp: 70 },
  ]} />);

  const summary = screen.getByLabelText("GPU 资源摘要");
  expect(summary).toBeInTheDocument();
  expect(summary).toHaveTextContent("总显存");
  expect(summary).toHaveTextContent("6 GB");
  expect(summary).toHaveTextContent("已读取峰值 80 %");
  const sampleSelect = screen.getByRole("combobox", { name: "同一采样点" });
  fireEvent.change(sampleSelect, { target: { value: (sampleSelect as HTMLSelectElement).options[0].value } });
  const inspected = screen.getByRole("group", { name: "所选采样点 GPU 数值" });
  expect(inspected).toHaveTextContent("3 GB");
  expect(inspected).toHaveTextContent("8 GB");
  expect(inspected).toHaveTextContent("20 %");
  expect(inspected).toHaveTextContent("50 °C");
});

it("keeps a selected sample by identity as the recent window moves, then falls back to latest", () => {
  const initial = Array.from({ length: 1000 }, (_, index) => ({ ts: index + 1, vram_used_gb: index + 1 }));
  const view = render(<HistoryGpuMetrics points={initial} />);
  const sampleSelect = screen.getByRole("combobox", { name: "同一采样点" }) as HTMLSelectElement;
  fireEvent.change(sampleSelect, { target: { value: sampleSelect.options[1].value } });
  expect(screen.getByRole("group", { name: "所选采样点 GPU 数值" })).toHaveTextContent("2 GB");

  view.rerender(<HistoryGpuMetrics points={[...initial, { ts: 1001, vram_used_gb: 1001 }]} />);
  expect(screen.getByRole("group", { name: "所选采样点 GPU 数值" })).toHaveTextContent("2 GB");

  view.rerender(<HistoryGpuMetrics points={[...initial, { ts: 1001, vram_used_gb: 1001 }, { ts: 1002, vram_used_gb: 1002 }]} />);
  expect(screen.getByRole("group", { name: "所选采样点 GPU 数值" })).toHaveTextContent("1002 GB");
});

it("shows summary and joint samples when only total VRAM is recorded", () => {
  render(<HistoryGpuMetrics points={[{ ts: 100, vram_total_gb: 8 }, { ts: 200, vram_total_gb: 12 }]} />);

  expect(screen.queryByText("暂无 GPU 资源记录")).not.toBeInTheDocument();
  expect(screen.getByLabelText("GPU 资源摘要")).toHaveTextContent("总显存");
  expect(screen.getByLabelText("GPU 资源摘要")).toHaveTextContent("已读取峰值 12 GB");
  expect(screen.getByRole("combobox", { name: "同一采样点" })).toBeInTheDocument();
  expect(screen.queryByRole("group", { name: /显存 \(GB\)/ })).not.toBeInTheDocument();
  expect(screen.queryByText("已隐藏所有 GPU 指标")).not.toBeInTheDocument();
});

it("offers one choice for indistinguishable legacy samples", () => {
  render(<HistoryGpuMetrics points={[{ vram_used_gb: 3 }, { vram_used_gb: 3 }, { vram_used_gb: 4 }]} />);
  const sampleSelect = screen.getByRole("combobox", { name: "同一采样点" }) as HTMLSelectElement;
  expect(sampleSelect.options).toHaveLength(2);
  expect(sampleSelect.options[0].value).not.toBe(sampleSelect.options[1].value);
  fireEvent.change(sampleSelect, { target: { value: sampleSelect.options[0].value } });
  expect(screen.getByRole("group", { name: "所选采样点 GPU 数值" })).toHaveTextContent("3 GB");
});

it("leaves total VRAM unknown when historical samples do not include it", () => {
  render(<HistoryGpuMetrics points={[{ ts: 100, vram_used_gb: 3, gpu_util: 20 }]} />);
  expect(screen.getByLabelText("GPU 资源摘要")).not.toHaveTextContent("总显存");
  expect(screen.getByRole("group", { name: "所选采样点 GPU 数值" })).toHaveTextContent("总显存—");
});

it("hides individual GPU charts, shows an all-hidden state, and preserves choices across devices", async () => {
  mockChartWidth();
  render(<HistoryGpuMetrics points={points} total={10} whitelist={[0]} />);

  const vram = screen.getByRole("checkbox", { name: "显存 (GB)" });
  const utilization = screen.getByRole("checkbox", { name: "GPU 利用率 (%)" });
  const temperature = screen.getByRole("checkbox", { name: "GPU 温度 (°C)" });
  fireEvent.click(vram);
  expect(screen.queryByRole("group", { name: /显存 \(GB\)/ })).not.toBeInTheDocument();
  expect(screen.getByRole("group", { name: /GPU 利用率/ })).toBeInTheDocument();

  fireEvent.click(utilization);
  fireEvent.click(temperature);
  expect(screen.getByText("已隐藏所有 GPU 指标")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "GPU 0" }));
  expect(screen.getByText("已隐藏所有 GPU 指标")).toBeInTheDocument();
  expect(screen.getByRole("checkbox", { name: "显存 (GB)" })).not.toBeChecked();

  fireEvent.click(screen.getByRole("button", { name: "GPU 1" }));
  expect(screen.getByRole("checkbox", { name: "GPU 利用率 (%)" })).not.toBeChecked();
  expect(screen.queryByRole("checkbox", { name: "GPU 温度 (°C)" })).not.toBeInTheDocument();
  expect(screen.getByText("已隐藏所有 GPU 指标")).toBeInTheDocument();

  fireEvent.click(screen.getByRole("button", { name: "GPU 0" }));
  fireEvent.click(screen.getByRole("checkbox", { name: "GPU 温度 (°C)" }));
  expect(screen.getByRole("group", { name: /GPU 温度/ })).toBeInTheDocument();
  expect(screen.queryByText("已隐藏所有 GPU 指标")).not.toBeInTheDocument();
  expect(useMetricsCanvas.mock.calls.some((call) => call[3] === "GPU 温度 (°C)" && call[5] === true)).toBe(true);
});
