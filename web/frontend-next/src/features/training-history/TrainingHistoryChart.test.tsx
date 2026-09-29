import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { TrainingMetricsCharts } from "../../components/TrainingMetricsCharts";

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

it("excludes validation samples from loss, applies EMA smoothing, and exposes keyboard inspection", async () => {
  mockChartWidth();
  const points = [
    { step: 1, loss: 1 },
    { step: 2, loss: 9, kind: "val", cmmd: 0.4 },
    { step: 3, loss: 0.5 },
  ];
  render(<TrainingMetricsCharts points={points} />);

  const lossChart = await screen.findByRole("group", { name: "Loss 趋势，2 个点" });
  expect(screen.getByRole("group", { name: "验证 CMMD，1 个点" })).toBeInTheDocument();
  await waitFor(() => expect(setOption).toHaveBeenCalled());
  const tooltip = setOption.mock.calls.find(([option]) => option.series[0].data.length === 2)![0].tooltip;
  expect(tooltip.formatter([{ value: [3, 0.5] }]).textContent).toContain("STEP: 3");

  fireEvent.change(within(lossChart.closest("section")!).getByRole("slider"), { target: { value: "0.5" } });
  await waitFor(() => {
    const option = setOption.mock.calls.at(-1)![0];
    expect(option.series[0].data).toEqual([[1, 1], [3, 0.75]]);
  });
  fireEvent.keyDown(lossChart, { key: "ArrowRight" });
  expect(screen.getByText("检查点 1/2 · STEP: 1 · Loss: 1.000")).toBeInTheDocument();
  fireEvent.keyDown(lossChart, { key: "ArrowRight" });
  expect(screen.getByText("检查点 2/2 · STEP: 3 · Loss: 0.7500")).toBeInTheDocument();
});

it("reports an accessible empty state for missing training metrics", () => {
  render(<TrainingMetricsCharts points={[{ step: 1, loss: Number.NaN }]} />);
  expect(screen.getByRole("region", { name: "Loss 趋势" })).toBeInTheDocument();
  expect(screen.getByText("暂无指标记录")).toBeInTheDocument();
});

it("shows an aligned, keyboard-readable learning-rate chart and filters validation LR", async () => {
  mockChartWidth();
  const points = [
    { step: 1, loss: 0.5, lr: 0.001 },
    { step: 2, loss: 0.4, lr: 0.9, kind: "val" },
    { step: 3, loss: 0.3, lr: 0.0001 },
  ];
  render(<TrainingMetricsCharts points={points} />);

  const learningRateChart = await screen.findByRole("group", { name: "学习率趋势，2 个点" });
  fireEvent.keyDown(learningRateChart, { key: "ArrowRight" });
  expect(screen.getByText("检查点 1/2 · STEP: 1 · 学习率: 1.000e-3")).toBeInTheDocument();
});

it("omits the learning-rate chart when no training LR is available", () => {
  render(<TrainingMetricsCharts points={[{ step: 1, loss: 0.5 }, { step: 2, lr: 0.9, ev: "val" }]} />);
  expect(screen.queryByRole("region", { name: "学习率趋势" })).not.toBeInTheDocument();
});
