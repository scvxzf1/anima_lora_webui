import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import { LogViewer } from "./LogViewer";

afterEach(cleanup);

it("makes the scrollable log content keyboard focusable and named", () => {
  render(<LogViewer lines={[{ id: 1, line: "step 1" }]} />);

  const log = screen.getByRole("log", { name: "日志内容" });
  expect(log).toHaveAttribute("tabindex", "0");
  log.focus();
  expect(log).toHaveFocus();
});
