import { render, screen, cleanup } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import { MonitorSummary } from "./MonitorSummary";

afterEach(cleanup);

it("does not display validation CMMD as live loss", () => {
  const { rerender } = render(<MonitorSummary status={{ job: "training", latest_metric: { kind: "val", loss: 0.91 } }} />);
  expect(screen.getByText("Loss").parentElement).toHaveTextContent("未记录");
  rerender(<MonitorSummary status={{ job: "training", latest_metric: { kind: "val", loss: 0.91 }, latest_progress: { loss: 0.2 } }} />);
  expect(screen.getByText("Loss").parentElement).toHaveTextContent("0.2000");
});

it("does not expose training metrics for preprocessing", () => {
  render(<MonitorSummary status={{ job: "preprocess", latest_metric: { loss: 0.91 } }} />);
  expect(screen.queryByText("Loss")).not.toBeInTheDocument();
});

it("shows current training steps, percentage, and sampled rate", () => {
  render(<MonitorSummary status={{
    job: "training",
    latest_progress: { current: 4, total: 10, rate: "2it/s" },
  }} />);

  expect(screen.getByRole("progressbar", { name: "任务进度" })).toHaveAttribute("aria-valuenow", "40");
  expect(screen.getByText("40.0%")).toBeInTheDocument();
  expect(screen.getByText("步数").parentElement).toHaveTextContent("4 / 10");
  expect(screen.getByText("最近采样速度").parentElement).toHaveTextContent("2it/s");
});

it.each([
  [79, false],
  [80, true],
  [95, true],
  [Number.NaN, false],
])("marks GPU temperature %s with warning=%s", (temperature, warning) => {
  render(<MonitorSummary status={{ job: "training", latest_system: { gpu_temp: temperature } }} />);
  const metric = screen.getByText("采样最高温度").parentElement;
  if (warning) {
    expect(metric).toHaveTextContent("高温预警");
    expect(metric?.querySelector('[data-tone="warning"]')).toBeInTheDocument();
  } else {
    expect(metric).not.toHaveTextContent("高温预警");
    expect(metric?.querySelector('[data-tone="warning"]')).not.toBeInTheDocument();
  }
});
