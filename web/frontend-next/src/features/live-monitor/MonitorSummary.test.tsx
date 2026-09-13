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
