import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { TrainingMetricsCharts } from "./TrainingMetricsCharts";

vi.mock("./MetricsChart", () => ({
  MetricsChart: ({ metric, initialLimit, hidden }: { metric?: string; initialLimit?: number; hidden?: boolean }) => <div data-testid={`chart-${metric ?? "loss"}`} data-initial-limit={initialLimit} hidden={hidden} />,
}));

afterEach(cleanup);

it("shows older learning-rate data with the full window when the recent window has none", () => {
  const olderLearningRate: Record<string, unknown>[] = Array.from({ length: 2001 }, (_, index) => ({ step: index + 1, loss: 1 }));
  olderLearningRate[0] = { ...olderLearningRate[0], lr: 1e-5 };
  const { queryByTestId, rerender } = render(<TrainingMetricsCharts points={olderLearningRate} />);
  expect(queryByTestId("chart-lr")?.getAttribute("data-initial-limit")).toBe("0");

  olderLearningRate[2000] = { ...olderLearningRate[2000], lr: 1e-6 };
  rerender(<TrainingMetricsCharts points={olderLearningRate} />);
  expect(queryByTestId("chart-lr")?.getAttribute("data-initial-limit")).toBe("2000");
});

it("does not offer LR when its only point has no drawable step in a mixed-step window", () => {
  const points: Record<string, unknown>[] = Array.from({ length: 2000 }, (_, index) => ({ step: index + 1, loss: 1 }));
  points[1999] = { ...points[1999], lr: 1e-5 };
  points[0] = { loss: 1 };
  points[1999] = { lr: 1e-5 };
  const { queryByTestId } = render(<TrainingMetricsCharts points={points} />);
  expect(queryByTestId("chart-lr")).toBeNull();
});

it("does not offer LR when no point has a drawable value", () => {
  const points: Record<string, unknown>[] = Array.from({ length: 2001 }, (_, index) => ({ step: index + 1, loss: 1 }));
  const { queryByTestId } = render(<TrainingMetricsCharts points={points} />);
  expect(queryByTestId("chart-lr")).toBeNull();
});

it("uses sample indices when no points have steps", () => {
  const points = Array.from({ length: 2001 }, (_, index) => ({ loss: 1, ...(index === 2000 ? { lr: 1e-5 } : {}) }));
  const { queryByTestId } = render(<TrainingMetricsCharts points={points} />);
  expect(queryByTestId("chart-lr")).toBeTruthy();
});

it("allows training charts to be hidden individually and restored after all are hidden", () => {
  const points = [{ step: 1, loss: 0.5, lr: 0.001, cmmd: 0.4 }];
  render(<TrainingMetricsCharts points={points} />);

  const loss = screen.getByRole("checkbox", { name: "Loss 趋势" });
  const learningRate = screen.getByRole("checkbox", { name: "学习率趋势" });
  const cmmd = screen.getByRole("checkbox", { name: "验证 CMMD" });
  fireEvent.click(learningRate);
  expect(screen.getByTestId("chart-lr")).toHaveAttribute("hidden");
  expect(screen.getByTestId("chart-loss")).toBeInTheDocument();
  expect(screen.getByTestId("chart-cmmd")).toBeInTheDocument();

  fireEvent.click(loss);
  fireEvent.click(cmmd);
  expect(screen.getByText("已隐藏所有训练指标")).toBeInTheDocument();
  fireEvent.click(learningRate);
  expect(screen.getByTestId("chart-lr")).not.toHaveAttribute("hidden");
  expect(screen.queryByText("已隐藏所有训练指标")).not.toBeInTheDocument();
});
