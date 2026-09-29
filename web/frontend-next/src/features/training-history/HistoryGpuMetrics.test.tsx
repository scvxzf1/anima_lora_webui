import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { HistoryGpuMetrics } from "./HistoryGpuMetrics";

const setOption = vi.fn();
vi.mock("../../components/metricsChartRuntime", () => ({
  init: () => ({ setOption, resize: vi.fn(), dispose: vi.fn() }),
}));

afterEach(() => {
  cleanup();
  setOption.mockClear();
  Object.defineProperty(HTMLElement.prototype, "clientWidth", { configurable: true, value: 0 });
  vi.unstubAllGlobals();
});

function mockChartWidth() {
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  Object.defineProperty(HTMLElement.prototype, "clientWidth", { configurable: true, get: () => 600 });
}

const points = [
  { ts: 100, vram_used_gb: 4, gpu_util: 20, gpu_temp: 60, per_gpu: [{ index: 0, uuid: "a", name: "Card A", vram_used_gb: 3, gpu_util: 10, gpu_temp: 50 }] },
  { ts: 200, vram_used_gb: 16, gpu_util: 80, gpu_temp: 90, per_gpu: [{ index: 0, uuid: "a", name: "Card A", vram_used_gb: 12, gpu_util: 70, gpu_temp: 85 }] },
];

it("plots available GPU metrics on time axes and switches from aggregate to the selected device", async () => {
  mockChartWidth();
  render(<HistoryGpuMetrics points={points} total={10} whitelist={[0]} />);

  expect(await screen.findByRole("group", { name: /显存 \(GB\)/ })).toBeInTheDocument();
  expect(screen.getByRole("group", { name: /GPU 利用率/ })).toBeInTheDocument();
  expect(screen.getByRole("group", { name: /GPU 温度/ })).toBeInTheDocument();
  await waitFor(() => expect(setOption).toHaveBeenCalled());
  let option = setOption.mock.calls.at(-1)![0];
  expect(option.xAxis.type).toBe("time");
  expect(option.series[0].data).toEqual([[100000, 4], [200000, 16]]);
  expect(option.tooltip.formatter([{ value: [200000, 16] }]).textContent).toContain("时间:");
  expect(option.tooltip.formatter([{ value: [200000, 16] }]).textContent).toContain("显存 (GB): 16.00");
  expect(screen.getByRole("group", { name: "GPU 资源范围" })).toBeInTheDocument();

  fireEvent.click(screen.getByRole("button", { name: "GPU 0" }));
  await waitFor(() => {
    option = setOption.mock.calls.at(-1)![0];
    expect(option.series[0].data).toEqual([[100000, 3], [200000, 12]]);
  });
  expect(screen.getByText("GPU 0 · Card A · 任务已选")).toBeInTheDocument();
});

it("shows a labeled empty state when no GPU metrics are available", () => {
  render(<HistoryGpuMetrics points={[{ ts: 100, per_gpu: [] }]} />);
  expect(screen.getByRole("region", { name: "GPU 资源历史" })).toBeInTheDocument();
  expect(screen.getByText("暂无 GPU 资源记录")).toBeInTheDocument();
  expect(screen.queryByRole("group", { name: /显存/ })).not.toBeInTheDocument();
});
