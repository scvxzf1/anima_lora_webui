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

it("keeps an axis tooltip on the training trend chart", async () => {
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  const width = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "clientWidth");
  Object.defineProperty(HTMLElement.prototype, "clientWidth", { configurable: true, get: () => 600 });
  render(<MetricsChart points={[{ step: 1, loss: 0.5 }]} />);

  await waitFor(() => expect(setOption).toHaveBeenCalled());
  expect(setOption.mock.calls[0][0].tooltip).toMatchObject({ trigger: "axis" });

  Object.defineProperty(HTMLElement.prototype, "clientWidth", width ?? { configurable: true, value: 0 });
  vi.unstubAllGlobals();
});
