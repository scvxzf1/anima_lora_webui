import { cleanup, render, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { MetricsChart } from "./MetricsChart";

const setOption = vi.fn();
vi.mock("./metricsChartRuntime", () => ({
  init: () => ({ setOption, resize: vi.fn(), dispose: vi.fn() }),
}));

afterEach(() => {
  cleanup();
  setOption.mockClear();
});

it("shows accessible axis and metric labels in a safe HTML tooltip", async () => {
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  const width = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "clientWidth");
  Object.defineProperty(HTMLElement.prototype, "clientWidth", { configurable: true, get: () => 600 });
  render(<MetricsChart points={[{ step: 1, loss: 0.5 }]} />);

  await waitFor(() => expect(setOption).toHaveBeenCalled());
  const tooltip = setOption.mock.calls[0][0].tooltip;
  expect(tooltip).toMatchObject({ trigger: "axis", renderMode: "html" });
  const content = tooltip.formatter([{ value: [1, 0.5] }]);
  expect(content.textContent).toContain("STEP: 1");
  expect(content.textContent).toContain("Loss: 0.5000");
  expect(content.querySelector("img")).toBeNull();

  Object.defineProperty(HTMLElement.prototype, "clientWidth", width ?? { configurable: true, value: 0 });
  vi.unstubAllGlobals();
});

it("writes custom metric names as text, not markup", async () => {
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  const width = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "clientWidth");
  Object.defineProperty(HTMLElement.prototype, "clientWidth", { configurable: true, get: () => 600 });
  render(<MetricsChart points={[{ step: 3, "<img src=x onerror=alert(1)>": 7 }]} metric="<img src=x onerror=alert(1)>" label="<img src=x onerror=alert(1)> 趋势" />);

  await waitFor(() => expect(setOption).toHaveBeenCalled());
  const tooltip = setOption.mock.calls[0][0].tooltip;
  const content = tooltip.formatter([{ value: [3, 7] }]);
  expect(content.textContent).toContain("<img src=x onerror=alert(1)>: 7.000");
  expect(content.querySelector("img")).toBeNull();

  Object.defineProperty(HTMLElement.prototype, "clientWidth", width ?? { configurable: true, value: 0 });
  vi.unstubAllGlobals();
});
