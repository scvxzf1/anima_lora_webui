import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { CaptionBatchToolbar } from "./CaptionBatchToolbar";
import type { CaptionItem } from "./api";
afterEach(cleanup);
for (const change of ["server", "draft"]) it(`rejects a preview after ${change} changes`, async () => {
  const items = [{id:"1", name:"one.png", proposed_caption:"red, red"}] as CaptionItem[];
  const onApply = vi.fn();
  const props = {items, drafts:{}, checked:["1"], filtered:items, filter:"", onFilter:vi.fn(), allowed:true, disabled:false, onApply, onSave:vi.fn(), undo:vi.fn(), redo:vi.fn(), canUndo:false, canRedo:false};
  const view = render(<CaptionBatchToolbar {...props} />), user = userEvent.setup();
  await user.selectOptions(screen.getByLabelText("批量操作"), "dedupe");
  await user.click(screen.getByRole("button", {name:"预览差异 (1)"}));
  view.rerender(<CaptionBatchToolbar {...props} items={change === "server" ? [{...items[0], proposed_caption:"server"}] : items} drafts={change === "draft" ? {"1":"local"} : {}} />);
  await user.click(screen.getByRole("button", {name:"应用到草稿"}));
  expect(onApply).not.toHaveBeenCalled();
  expect(screen.getByRole("alert")).toHaveTextContent("重新预览");
});
