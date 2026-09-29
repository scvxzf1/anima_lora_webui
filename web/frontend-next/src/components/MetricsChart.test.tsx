import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { MetricsChart } from "./MetricsChart";
import { TrainingMetricsCharts } from "./TrainingMetricsCharts";

const setOption = vi.fn();
const originalClientWidth = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "clientWidth");
vi.mock("./metricsChartRuntime", () => ({
  init: () => ({ setOption, resize: vi.fn(), dispose: vi.fn() }),
}));

afterEach(() => {
  cleanup();
  setOption.mockClear();
  Object.defineProperty(HTMLElement.prototype, "clientWidth", originalClientWidth ?? { configurable: true, value: 0 });
  vi.unstubAllGlobals();
});

function mockChartWidth() {
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  Object.defineProperty(HTMLElement.prototype, "clientWidth", { configurable: true, get: () => 600 });
}

it("shows accessible axis and metric labels in a safe HTML tooltip", async () => {
  mockChartWidth();
  render(<MetricsChart points={[{ step: 1, loss: 0.5 }]} />);

  await waitFor(() => expect(setOption).toHaveBeenCalled());
  const tooltip = setOption.mock.calls[0][0].tooltip;
  expect(tooltip).toMatchObject({ trigger: "axis", renderMode: "html", confine: true });
  const content = tooltip.formatter([{ value: [1, 0.5] }]);
  expect(content.textContent).toContain("STEP: 1");
  expect(content.textContent).toContain("Loss: 0.5000");
  expect(content.querySelector("img")).toBeNull();

});

it("writes custom metric names as text, not markup", async () => {
  mockChartWidth();
  render(<MetricsChart points={[{ step: 3, "<img src=x onerror=alert(1)>": 7 }]} metric="<img src=x onerror=alert(1)>" label="<img src=x onerror=alert(1)> 趋势" />);

  await waitFor(() => expect(setOption).toHaveBeenCalled());
  const tooltip = setOption.mock.calls[0][0].tooltip;
  const content = tooltip.formatter([{ value: [3, 7] }]);
  expect(content.textContent).toContain("<img src=x onerror=alert(1)>: 7.000");
  expect(content.querySelector("img")).toBeNull();

});

it("labels fallback coordinates and exposes keyboard point inspection", async () => {
  mockChartWidth();
  const { getByRole, getByText } = render(<MetricsChart points={[{ loss: 0.5 }, { loss: 0.25 }]} />);
  await waitFor(() => expect(setOption).toHaveBeenCalled());
  const tooltip = setOption.mock.calls[0][0].tooltip;
  const content = tooltip.formatter([{ value: [1, 0.5] }]);
  expect(content.textContent).toContain("采样序号: 1");
  const chart = getByRole("group", { name: /Loss 趋势，2 个点/ });
  fireEvent.keyDown(chart, { key: "ArrowRight" });
  expect(getByText("检查点 1/2 · 采样序号: 1 · Loss: 0.5000")).toBeTruthy();
  fireEvent.keyDown(chart, { key: "ArrowRight" });
  expect(getByText("检查点 2/2 · 采样序号: 2 · Loss: 0.2500")).toBeTruthy();
});

it("hides inspected metric values without hiding the chart or its coordinates", async () => {
  mockChartWidth();
  const { getByRole, getByText, queryByText } = render(<MetricsChart points={[{ step: 1, loss: 0.5 }]} valueVisible={false} />);

  await waitFor(() => expect(setOption).toHaveBeenCalled());
  const tooltip = setOption.mock.calls.at(-1)?.[0].tooltip;
  const content = tooltip.formatter([{ value: [1, 0.5] }]);
  expect(content.textContent).toContain("STEP: 1");
  expect(content.textContent).not.toContain("Loss:");

  fireEvent.keyDown(getByRole("group", { name: /Loss 趋势，1 个点/ }), { key: "ArrowRight" });
  expect(getByText("检查点 1/1 · STEP: 1")).toBeTruthy();
  expect(queryByText(/Loss: 0.5000/)).toBeNull();
});

it("draws learning-rate data and announces inspected values in scientific notation", async () => {
  mockChartWidth();
  const { getByRole, getByText } = render(<MetricsChart points={[{ step: 4, lr: 0.00001234 }]} metric="lr" label="学习率趋势" />);
  await waitFor(() => expect(setOption).toHaveBeenCalled());
  expect(setOption.mock.calls[0][0].series[0].data).toEqual([[4, 0.00001234]]);
  const chart = getByRole("group", { name: /学习率趋势，1 个点/ });
  fireEvent.keyDown(chart, { key: "ArrowRight" });
  expect(getByText("检查点 1/1 · STEP: 4 · 学习率: 1.234e-5")).toBeTruthy();
});

it("shows and exposes an older LR point when the initial window includes all loaded points", async () => {
  mockChartWidth();
  const points = Array.from({ length: 2001 }, (_, index) => ({ step: index + 1, ...(index === 0 ? { lr: 1e-5 } : {}) }));
  const { getByRole, getByText, getByLabelText } = render(<MetricsChart points={points} metric="lr" label="学习率趋势" initialLimit={0} />);
  expect(getByLabelText("学习率趋势数据窗口")).toHaveValue("0");
  await waitFor(() => expect(setOption).toHaveBeenCalled());
  expect(setOption.mock.calls.at(-1)?.[0].series[0].data).toEqual([[1, 1e-5]]);
  const chart = getByRole("group", { name: /学习率趋势，1 个点/ });
  fireEvent.keyDown(chart, { key: "ArrowRight" });
  expect(getByText("检查点 1/1 · STEP: 1 · 学习率: 1.000e-5")).toBeTruthy();
});

it("expands the LR window when updated data leaves only an older readable point", async () => {
  const recent = Array.from({ length: 2001 }, (_, index) => ({ step: index + 1, ...(index === 2000 ? { lr: 1e-6 } : {}) }));
  const older = Array.from({ length: 2001 }, (_, index) => ({ step: index + 1, ...(index === 0 ? { lr: 1e-5 } : {}) }));
  const { rerender } = render(<TrainingMetricsCharts points={recent} />);
  expect(screen.getByLabelText("学习率趋势数据窗口")).toHaveValue("2000");

  rerender(<TrainingMetricsCharts points={older} />);
  await waitFor(() => expect(screen.getByLabelText("学习率趋势数据窗口")).toHaveValue("0"));
  const chart = screen.getByRole("group", { name: "学习率趋势，1 个点" });
  fireEvent.keyDown(chart, { key: "ArrowRight" });
  expect(screen.getByText("检查点 1/1 · STEP: 1 · 学习率: 1.000e-5")).toBeTruthy();
});

it("preserves a manually selected empty LR window when older points exist", async () => {
  const points = Array.from({ length: 2001 }, (_, index) => ({ step: index + 1, ...(index === 0 ? { lr: 1e-5 } : {}) }));
  render(<MetricsChart points={points} metric="lr" label="学习率趋势" autoExpandEmptyWindow />);
  const window = screen.getByLabelText("学习率趋势数据窗口");
  fireEvent.change(window, { target: { value: "500" } });

  expect(window).toHaveValue("500");
  expect(screen.getByText("暂无指标记录")).toBeTruthy();
});

it("preserves a manually selected non-empty LR window as new points arrive", async () => {
  const points = Array.from({ length: 2001 }, (_, index) => ({ step: index + 1, ...(index === 1501 ? { lr: 1e-5 } : {}) }));
  const { rerender } = render(<MetricsChart points={points} metric="lr" label="学习率趋势" autoExpandEmptyWindow />);
  const window = screen.getByLabelText("学习率趋势数据窗口");
  fireEvent.change(window, { target: { value: "500" } });
  expect(window).toHaveValue("500");
  expect(screen.getByRole("group", { name: "学习率趋势，1 个点" })).toBeTruthy();

  const appended = [...points, ...Array.from({ length: 502 }, (_, index) => ({ step: 2002 + index }))];
  rerender(<MetricsChart points={appended} metric="lr" label="学习率趋势" autoExpandEmptyWindow />);

  expect(window).toHaveValue("500");
  expect(screen.getByText("暂无指标记录")).toBeTruthy();
});
