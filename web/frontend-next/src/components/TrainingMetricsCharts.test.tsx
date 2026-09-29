import { cleanup, render } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { TrainingMetricsCharts } from "./TrainingMetricsCharts";

vi.mock("./MetricsChart", () => ({
  MetricsChart: ({ metric }: { metric?: string }) => <div data-testid={`chart-${metric ?? "loss"}`} />,
}));

afterEach(cleanup);

it("only offers a learning-rate chart when the default recent window can draw it", () => {
  const olderLearningRate = Array.from({ length: 2001 }, (_, index) => ({ step: index + 1, loss: 1 }));
  olderLearningRate[0] = { ...olderLearningRate[0], lr: 1e-5 };
  const { queryByTestId, rerender } = render(<TrainingMetricsCharts points={olderLearningRate} />);
  expect(queryByTestId("chart-lr")).toBeNull();

  olderLearningRate[2000] = { ...olderLearningRate[2000], lr: 1e-6 };
  rerender(<TrainingMetricsCharts points={olderLearningRate} />);
  expect(queryByTestId("chart-lr")).toBeTruthy();
});

it("does not offer LR when its only point has no drawable step in a mixed-step window", () => {
  const points = Array.from({ length: 2000 }, (_, index) => ({ step: index + 1, loss: 1 }));
  points[1999] = { ...points[1999], lr: 1e-5 };
  points[0] = { loss: 1 };
  points[1999] = { lr: 1e-5 };
  const { queryByTestId } = render(<TrainingMetricsCharts points={points} />);
  expect(queryByTestId("chart-lr")).toBeNull();
});

it("uses sample indices when no points have steps", () => {
  const points = Array.from({ length: 2001 }, (_, index) => ({ loss: 1, ...(index === 2000 ? { lr: 1e-5 } : {}) }));
  const { queryByTestId } = render(<TrainingMetricsCharts points={points} />);
  expect(queryByTestId("chart-lr")).toBeTruthy();
});
