import { render, screen, cleanup } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { MonitorSummary } from "./MonitorSummary";

afterEach(cleanup);
afterEach(() => vi.useRealTimers());

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

it("shows no ETA until a valid running rate is available", () => {
  render(<MonitorSummary status={{
    status: "running",
    job: "training",
    latest_progress: { current: 4, total: 10 },
  }} />);

  expect(screen.getByText("预计完成").parentElement).toHaveTextContent("待计算");
});

it("estimates completion time on the same day", () => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date(2026, 0, 1, 12, 0, 0));
  render(<MonitorSummary status={{
    status: "running",
    job: "training",
    latest_progress: { current: 4, total: 10, rate: "2it/s" },
  }} />);

  const eta = screen.getByText("预计完成").parentElement?.querySelector("strong");
  expect(eta).toHaveTextContent("12:00");
  expect(eta).toHaveAttribute("title", "按当前速度估算，剩余约 3 秒。");
});

it("labels an ETA that crosses midnight as tomorrow", () => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date(2026, 0, 1, 23, 59, 58));
  render(<MonitorSummary status={{
    status: "running",
    job: "training",
    latest_progress: { current: 4, total: 10, rate: "2it/s" },
  }} />);

  expect(screen.getByText("预计完成").parentElement?.querySelector("strong")).toHaveTextContent("明日 00:00");
});

it("shows a calendar date for an ETA beyond tomorrow", () => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date(2026, 0, 1, 12, 0, 0));
  render(<MonitorSummary status={{
    status: "running",
    job: "training",
    latest_progress: { current: 0, total: 172800, rate: "1s/it" },
  }} />);

  expect(screen.getByText("预计完成").parentElement).toHaveTextContent("2026-01-03 12:00");
});

it.each([
  ["500ms/it", "3 秒"],
  ["2s/step", "12 秒"],
])("estimates the remaining time from %s", (rate, remaining) => {
  render(<MonitorSummary status={{
    status: "running",
    job: "training",
    latest_progress: { current: 4, total: 10, rate },
  }} />);

  expect(screen.getByText("预计完成").parentElement?.querySelector("strong")).toHaveAttribute("title", `按当前速度估算，剩余约 ${remaining}。`);
});

it("marks reached progress as imminent while the job is still running", () => {
  render(<MonitorSummary status={{
    status: "running",
    job: "training",
    latest_progress: { current: 10, total: 10, rate: "2it/s" },
  }} />);

  expect(screen.getByText("预计完成").parentElement).toHaveTextContent("即将完成");
});

it.each(["running", "training", "compiling", "caching", "saving"])("estimates ETA for the %s state", (status) => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date(2026, 0, 1, 12, 0, 0));
  render(<MonitorSummary status={{
    status,
    job: "training",
    latest_progress: { current: 4, total: 10, rate: "2it/s" },
  }} />);

  expect(screen.getByText("预计完成").parentElement).toHaveTextContent("12:00");
});

it.each(["idle", "error", "failed", "interrupted", "unavailable"])("does not estimate ETA for the %s state", (status) => {
  render(<MonitorSummary status={{
    status,
    job: "training",
    latest_progress: { current: 4, total: 10, rate: "2it/s" },
  }} />);

  expect(screen.getByText("预计完成").parentElement).toHaveTextContent("待计算");
});

it("does not report an errored task as complete when its target was reached", () => {
  render(<MonitorSummary status={{
    status: "error",
    job: "training",
    latest_progress: { current: 10, total: 10 },
  }} />);

  expect(screen.getByText("预计完成").parentElement).toHaveTextContent("已达目标步数");
});

it("uses the displayed metric rate when progress has no rate", () => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date(2026, 0, 1, 12, 0, 0));
  render(<MonitorSummary status={{
    status: "running",
    job: "training",
    latest_progress: { current: 4, total: 10 },
    latest_metric: { kind: "train", rate: "2it/s" },
  }} />);

  expect(screen.getByText("最近采样速度").parentElement).toHaveTextContent("2it/s");
  expect(screen.getByText("预计完成").parentElement).toHaveTextContent("12:00");
});

it("falls back when a finite ETA timestamp exceeds the Date range", () => {
  render(<MonitorSummary status={{
    status: "running",
    job: "training",
    latest_progress: { current: 0, total: Number.MAX_VALUE, rate: "1s/it" },
  }} />);

  expect(screen.getByText("预计完成").parentElement).toHaveTextContent("待计算");
});

it("does not estimate completion from an invalid rate", () => {
  render(<MonitorSummary status={{
    status: "running",
    job: "training",
    latest_progress: { current: 4, total: 10, rate: "0it/s" },
  }} />);

  expect(screen.getByText("预计完成").parentElement).toHaveTextContent("待计算");
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
