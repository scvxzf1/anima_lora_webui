import { cleanup, render } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { TrainingMetricsCharts } from "./TrainingMetricsCharts";

vi.mock("./MetricsChart", () => ({
  MetricsChart: ({ metric, initialLimit }: { metric?: string; initialLimit?: number }) => <div data-testid={`chart-${metric ?? "loss"}`} data-initial-limit={initialLimit} />,
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
