import { cleanup, fireEvent, render, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { MetricsChart } from "./MetricsChart";

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
