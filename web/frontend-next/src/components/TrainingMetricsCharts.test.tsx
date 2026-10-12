import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { TrainingMetricsCharts } from "./TrainingMetricsCharts";

vi.mock("./MetricsChart", () => ({
  MetricsChart: ({ metric, initialLimit, hidden }: { metric?: string; initialLimit?: number; hidden?: boolean }) => <div data-testid={`chart-${metric ?? "loss"}`} data-initial-limit={initialLimit} hidden={hidden} />,
}));

afterEach(cleanup);

it("ignores non-training STEP when discovering legacy learning-rate samples", () => {
  render(<TrainingMetricsCharts points={[
    { ev: "sample", step: 99, loss: 999, lr: 9 },
    { loss: 0.5, lr: 0.000012345 },
  ]} />);
  expect(screen.getByTestId("combined")).toHaveAttribute("data-lr", "true");
  expect(screen.getByText("最新 Loss: 0.5000")).toBeVisible();
  expect(screen.getByText("最新 学习率: 1.234e-5")).toBeVisible();
});

vi.mock("./CombinedTrainingChart", () => ({
  CombinedTrainingChart: ({ visible }: { visible: boolean[] }) => <div data-testid="combined" data-loss={String(visible[0])} data-lr={String(visible[1])} />,
}));

it("combines older learning-rate data with loss in one shared chart", () => {
  const points = Array.from({ length: 2001 }, (_, index) => ({ step: index + 1, loss: 1, ...(index === 0 ? { lr: 1e-5 } : {}) }));
  render(<TrainingMetricsCharts points={points} />);
  expect(screen.getByTestId("combined")).toHaveAttribute("data-lr", "true");
});

it("does not offer LR when its only point has no drawable step in a mixed-step window", () => {
  const points: Record<string, unknown>[] = Array.from({ length: 2000 }, (_, index) => ({ step: index + 1, loss: 1 }));
  points[1999] = { ...points[1999], lr: 1e-5 };
  points[0] = { loss: 1 };
  points[1999] = { lr: 1e-5 };
  const { queryByTestId } = render(<TrainingMetricsCharts points={points} />);
  expect(queryByTestId("combined")).toHaveAttribute("data-lr", "false");
});

it("does not offer LR when no point has a drawable value", () => {
  const points: Record<string, unknown>[] = Array.from({ length: 2001 }, (_, index) => ({ step: index + 1, loss: 1 }));
  const { queryByTestId } = render(<TrainingMetricsCharts points={points} />);
  expect(queryByTestId("combined")).toHaveAttribute("data-lr", "false");
});

it("uses sample indices when no points have steps", () => {
  const points = Array.from({ length: 2001 }, (_, index) => ({ loss: 1, ...(index === 2000 ? { lr: 1e-5 } : {}) }));
  const { queryByTestId } = render(<TrainingMetricsCharts points={points} />);
  expect(queryByTestId("combined")).toHaveAttribute("data-lr", "true");
});

it("allows training charts to be hidden individually and restored after all are hidden", () => {
  const points = [{ step: 1, loss: 0.5, lr: 0.001, cmmd: 0.4 }];
  render(<TrainingMetricsCharts points={points} />);

  const loss = screen.getByRole("checkbox", { name: "Loss 趋势" });
  const learningRate = screen.getByRole("checkbox", { name: "学习率趋势" });
  const cmmd = screen.getByRole("checkbox", { name: "验证 CMMD" });
  fireEvent.click(learningRate);
  expect(screen.getByTestId("combined")).toHaveAttribute("data-lr", "false");
  expect(screen.getByTestId("combined")).toHaveAttribute("data-loss", "true");
  expect(screen.getByTestId("chart-cmmd")).toBeInTheDocument();

  fireEvent.click(loss);
  fireEvent.click(cmmd);
  expect(screen.getByText("已隐藏所有训练曲线")).toBeInTheDocument();
  fireEvent.click(learningRate);
  expect(screen.getByTestId("combined")).toHaveAttribute("data-lr", "true");
  expect(screen.queryByText("已隐藏所有训练曲线")).not.toBeInTheDocument();
});
