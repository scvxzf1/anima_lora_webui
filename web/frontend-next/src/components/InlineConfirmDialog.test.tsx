import { useState } from "react";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { InlineConfirmDialog } from "./InlineConfirmDialog";

afterEach(cleanup);

function DialogHarness({ onConfirm = vi.fn() }: { onConfirm?: () => void | Promise<unknown> }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button type="button" onClick={() => setOpen(true)}>打开确认</button>
      {open && (
        <InlineConfirmDialog
          message="此操作不可恢复。"
          confirmLabel="执行操作"
          onConfirm={() => {
            const result = onConfirm();
            if (result && typeof result.then === "function") {
              return result.then(() => setOpen(false));
            }
            setOpen(false);
            return result;
          }}
          onCancel={() => setOpen(false)}
        />
      )}
    </>
  );
}

it("starts on cancel and restores focus after confirmation and cancellation unmounts", async () => {
  const user = userEvent.setup();
  render(<DialogHarness />);
  const trigger = screen.getByRole("button", { name: "打开确认" });

  await user.click(trigger);
  const cancel = screen.getByRole("button", { name: "取消" });
  expect(cancel).toHaveFocus();
  expect(screen.getByRole("dialog")).toHaveAttribute("aria-modal", "true");
  await user.click(cancel);
  await waitFor(() => expect(trigger).toHaveFocus());

  await user.click(trigger);
  await user.click(screen.getByRole("button", { name: "执行操作" }));
  await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  expect(trigger).toHaveFocus();
});

it("restores focus to its trigger when Escape closes the dialog", async () => {
  const user = userEvent.setup();
  render(<DialogHarness />);
  const trigger = screen.getByRole("button", { name: "打开确认" });
  await user.click(trigger);
  await user.keyboard("{Escape}");
  await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  expect(trigger).toHaveFocus();
});

it("keeps Escape and repeated confirmation from bypassing an in-flight action", async () => {
  const user = userEvent.setup();
  let resolveAction!: () => void;
  const onConfirm = vi.fn(() => new Promise<void>((resolve) => { resolveAction = resolve; }));
  render(<DialogHarness onConfirm={onConfirm} />);
  await user.click(screen.getByRole("button", { name: "打开确认" }));

  const confirm = screen.getByRole("button", { name: "执行操作" });
  await user.click(confirm);
  expect(onConfirm).toHaveBeenCalledTimes(1);
  expect(confirm).toBeDisabled();
  const cancel = screen.getByRole("button", { name: "取消" });
  expect(cancel).toHaveAttribute("aria-disabled", "true");
  cancel.focus();
  await user.keyboard("{Tab}");
  expect(cancel).toHaveFocus();
  await user.keyboard("{Tab}");
  expect(cancel).toHaveFocus();
  await user.keyboard("{Escape}");
  expect(screen.getByRole("dialog")).toBeInTheDocument();
  expect(onConfirm).toHaveBeenCalledTimes(1);

  await waitFor(() => expect(resolveAction).toBeTypeOf("function"));
  resolveAction();
  await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
});

it("allows another synchronous confirmation when the parent keeps the dialog mounted", async () => {
  const user = userEvent.setup();
  const onConfirm = vi.fn();
  render(
    <InlineConfirmDialog
      message="此操作不可恢复。"
      onConfirm={onConfirm}
      onCancel={vi.fn()}
    />,
  );

  const confirm = screen.getByRole("button", { name: "确定" });
  await user.click(confirm);
  expect(confirm).toBeEnabled();
  await user.click(confirm);
  expect(onConfirm).toHaveBeenCalledTimes(2);
});
