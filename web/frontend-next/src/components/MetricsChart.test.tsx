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
