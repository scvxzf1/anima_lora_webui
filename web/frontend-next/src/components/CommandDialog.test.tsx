import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { CommandDialog } from "./CommandDialog";

afterEach(cleanup);

it("does not close while busy, including Escape", () => {
  const onClose = vi.fn();
  render(
    <CommandDialog title="正在保存" busy onClose={onClose}>
      <button type="button">内容</button>
    </CommandDialog>,
  );

  fireEvent.click(screen.getByRole("button", { name: "关闭" }));
  fireEvent.keyDown(document, { key: "Escape" });

  expect(onClose).not.toHaveBeenCalled();
});

it("closes from the button when idle", () => {
  const onClose = vi.fn();
  render(
    <CommandDialog title="编辑" onClose={onClose}>
      <p>内容</p>
    </CommandDialog>,
  );

  fireEvent.click(screen.getByRole("button", { name: "关闭" }));

  expect(onClose).toHaveBeenCalledTimes(1);
});
