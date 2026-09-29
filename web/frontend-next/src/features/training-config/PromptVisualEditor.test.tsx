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

it("applies shared sample values without losing per-row extras", async () => {
  function Test() {
    const [content, setContent] = useState("# saved\nfirst --w 512 --custom yes\nsecond --w 768\n");
    return <PromptVisualEditor content={content} onChange={setContent} onEditing={vi.fn()} disabled={false} />;
  }
  render(<Test />);
  const user = userEvent.setup();
  await user.click(screen.getByText("统一参数"));
  await user.type(screen.getByLabelText("统一宽度"), "1024");
  await user.click(screen.getByRole("button", { name: "应用统一参数" }));
  await user.click(screen.getByRole("button", { name: "原文" }));
  expect(screen.getByLabelText("样张提示词内容")).toHaveValue("# saved\nfirst --w 1024 --custom yes\nsecond --w 1024\n");
});

it("defaults new Qwen samples to Edit with Euler and CFG one", async () => {
  render(<PromptVisualEditor content="" onChange={vi.fn()} onEditing={vi.fn()} disabled={false} modelFamily="qwen_image_2_1" supportedPreviewTasks={["t2i", "edit"]} defaultTask="edit" />);
  await userEvent.click(screen.getByRole("button", { name: "新增样张" }));
  expect(screen.getByLabelText("样张任务")).toHaveValue("edit");
  expect(screen.getByLabelText("CFG")).toHaveValue(1);
  expect(screen.getByLabelText("采样器")).toHaveValue("euler");
  expect(screen.queryByLabelText("Flow shift")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "应用样张" })).toBeDisabled();
});

it("blocks unsupported edit rows without automatically rewriting content", async () => {
  const onChange = vi.fn();
  render(<PromptVisualEditor content={JSON.stringify({ prompt: "edit me", sample_task: "edit", reference_image: "/tmp/a.png" })} onChange={onChange} onEditing={vi.fn()} disabled={false} modelFamily="unknown" supportedPreviewTasks={[]} />);
  await userEvent.click(screen.getByRole("button", { name: "样张 1 · 编辑 · 1 图: edit me" }));
  expect(screen.getByRole("alert")).toHaveTextContent("不支持");
  expect(screen.getByRole("button", { name: "应用样张" })).toBeDisabled();
  expect(onChange).not.toHaveBeenCalled();
});

it("edits a legacy single-reference row into an ordered array", async () => {
  const source = JSON.stringify({ prompt: "edit me", sample_task: "edit", reference_image: "/a.png", custom: true });
  const onChange = vi.fn();
  render(<PromptVisualEditor content={source} onChange={onChange} onEditing={vi.fn()} disabled={false} modelFamily="qwen_image_2_1" supportedPreviewTasks={["t2i", "edit"]} maxPreviewReferences={4} />);
  await userEvent.click(screen.getByRole("button", { name: "样张 1 · 编辑 · 1 图: edit me" }));
  await userEvent.click(screen.getByRole("button", { name: "移除参考图 1" }));
  expect(screen.getByRole("button", { name: "应用样张" })).toBeDisabled();
  await userEvent.click(screen.getByRole("button", { name: "取消编辑" }));
  expect(onChange).not.toHaveBeenCalled();
});

it("clears original JSON reference keys when switching an edit sample to t2i", async () => {
  const onChange = vi.fn();
  render(<PromptVisualEditor content={JSON.stringify({ prompt: "test", sample_task: "edit", reference_image: "/a.png", reference_images: ["/b.png"], custom: true })} onChange={onChange} onEditing={vi.fn()} disabled={false} modelFamily="qwen_image_2_1" supportedPreviewTasks={["t2i", "edit"]} />);
  await userEvent.click(screen.getByRole("button", { name: "样张 1 · 编辑 · 1 图: test" }));
  expect(screen.getByRole("button", { name: "应用样张" })).toBeDisabled();
  expect(screen.getByRole("alert")).toHaveTextContent("不能同时设置");
  await userEvent.selectOptions(screen.getByLabelText("样张任务"), "t2i");
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "应用样张" }));
  expect(JSON.parse(onChange.mock.lastCall![0])).toEqual({ prompt: "test", sample_task: "t2i", custom: true });
});
