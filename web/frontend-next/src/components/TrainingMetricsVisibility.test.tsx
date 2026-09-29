import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { TrainingMetricsCharts } from "./TrainingMetricsCharts";

const useMetricsCanvas = vi.hoisted(() => vi.fn());
vi.mock("./useMetricsCanvas", () => ({ useMetricsCanvas }));

afterEach(() => {
  cleanup();
  useMetricsCanvas.mockClear();
});

it("preserves a real MetricsChart's window, smoothing, and keyboard inspection while hidden", () => {
  const points = [
    { step: 1, loss: 0.8, lr: 0.001, cmmd: 0.4 },
    { step: 2, loss: 0.6, lr: 0.0005, cmmd: 0.3 },
  ];
  render(<TrainingMetricsCharts points={points} />);

  const section = document.querySelector<HTMLElement>('section[aria-label="学习率趋势"]')!;
  const controls = within(section);
  const learningRateCanvasCall = () => useMetricsCanvas.mock.calls.filter((call) => call[3] === "学习率").at(-1);
  fireEvent.change(controls.getByLabelText("学习率趋势数据窗口"), { target: { value: "0" } });
  fireEvent.change(controls.getByLabelText("平滑"), { target: { value: "0.5" } });
  fireEvent.keyDown(screen.getByRole("group", { name: "学习率趋势，2 个点" }), { key: "ArrowRight" });
  expect(screen.getByText("检查点 1/2 · STEP: 1 · 学习率: 1.000e-3")).toBeInTheDocument();

  fireEvent.click(screen.getByRole("checkbox", { name: "学习率趋势" }));
  expect(section).toHaveAttribute("hidden");
  expect(document.querySelector('section[aria-label="学习率趋势"]')).toBe(section);
  expect(learningRateCanvasCall()?.[5]).toBe(false);

  fireEvent.click(screen.getByRole("checkbox", { name: "学习率趋势" }));
  expect(section).not.toHaveAttribute("hidden");
  expect(controls.getByLabelText("学习率趋势数据窗口")).toHaveValue("0");
  expect(controls.getByLabelText("平滑")).toHaveValue("0.5");
  expect(screen.getByText("检查点 1/2 · STEP: 1 · 学习率: 1.000e-3")).toBeInTheDocument();
  expect(learningRateCanvasCall()?.[5]).toBe(true);
});
