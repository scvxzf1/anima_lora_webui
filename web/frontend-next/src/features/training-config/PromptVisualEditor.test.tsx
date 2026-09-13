import { useState } from "react";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { PromptVisualEditor } from "./PromptVisualEditor";
afterEach(cleanup);
it("edits graphically and retains original comments in raw mode", async () => {
  function Test() {
    const [content, setContent] = useState("# saved\nhello --w 512 --custom yes\n");
    return <PromptVisualEditor content={content} onChange={setContent} onEditing={vi.fn()} disabled={false} />;
  }
  render(<Test />);
  const user = userEvent.setup();
  await user.click(screen.getByRole("button", { name: "样张 1: hello" }));
  await user.clear(screen.getByLabelText("宽度"));
  await user.type(screen.getByLabelText("宽度"), "768");
  await user.click(screen.getByRole("button", { name: "应用样张" }));
  await user.click(screen.getByRole("button", { name: "原文" }));
  expect(screen.getByLabelText("样张提示词内容")).toHaveValue("# saved\nhello --w 768 --custom yes\n");
});
