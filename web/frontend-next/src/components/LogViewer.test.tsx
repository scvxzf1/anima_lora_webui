import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
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

it("renders untrusted log markup as text instead of interpreting it", () => {
  render(<LogViewer lines={[{ id: 1, line: '<script>window.pwned = true</script>' }]} />);

  const log = screen.getByRole("log", { name: "日志内容" });
  expect(log).toHaveTextContent("<script>window.pwned = true</script>");
  expect(log.querySelector("script")).toBeNull();
});

it("searches the loaded log window and clears only the current view", async () => {
  const lines = Array.from({ length: 300 }, (_, index) => ({ id: index + 1, line: `line ${index}` }));
  const { rerender } = render(<LogViewer lines={lines} total={600} />);
  const user = userEvent.setup();
  await user.type(screen.getByRole("textbox", { name: "搜索日志" }), "line 19");
  expect(screen.getByRole("log", { name: "日志内容" })).toHaveTextContent("line 19");
  expect(screen.getByRole("log", { name: "日志内容" })).not.toHaveTextContent("line 20");
  expect(screen.getByText(/当前视图 11 行/)).toHaveTextContent("共 600 行（前段未读取）");

  await user.click(screen.getByRole("button", { name: "清空当前视图" }));
  expect(screen.getByRole("log", { name: "日志内容" })).toHaveTextContent("当前视图无匹配日志");
  await user.clear(screen.getByRole("textbox", { name: "搜索日志" }));
  rerender(<LogViewer lines={[...lines, { id: 301, line: "line 300" }]} total={600} />);
  expect(screen.getByRole("log", { name: "日志内容" })).toHaveTextContent("line 300");
  expect(screen.getByRole("log", { name: "日志内容" })).not.toHaveTextContent("line 19");
});
