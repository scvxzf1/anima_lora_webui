import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { TrainingSamplePrompts } from "./TrainingExtras";
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

for (const kind of ["new", "existing", "reverted", "dirty"]) {
  it(`handles ${kind} row close without native confirmation`, async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ ok: true, exists: true, file: "configs/sample_prompts.txt", content: "old prompt --w 512\n", prompts: [] }))));
    const nativeConfirm = vi.fn();
    vi.stubGlobal("confirm", nativeConfirm);
    const onClose = vi.fn();
    render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <TrainingSamplePrompts file={{ path: "configs/imported/test.toml" }} promptFile="configs/sample_prompts.txt" onClose={onClose} onSaved={async () => {}} />
    </QueryClientProvider>);
    await screen.findByRole("button", { name: "样张 1: old prompt" });
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: kind === "new" ? "新增样张" : "样张 1: old prompt" }));
    if (kind === "dirty" || kind === "reverted") {
      await user.type(screen.getByLabelText("正向提示词"), "x");
      if (kind === "reverted") await user.keyboard("{Backspace}");
    }
    await user.click(screen.getByRole("button", { name: "取消" }));
    if (kind === "dirty") {
      expect(onClose).not.toHaveBeenCalled();
      expect(screen.getByRole("dialog", { name: "放弃样张提示词修改？" })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "继续编辑" })).toHaveFocus();
      await user.keyboard("{Escape}");
      expect(screen.getByLabelText("正向提示词")).toHaveValue("old promptx");
      await user.click(screen.getByRole("button", { name: "关闭" }));
      await user.click(screen.getByRole("button", { name: "放弃修改" }));
    }
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(nativeConfirm).not.toHaveBeenCalled();
  });
}

it("loads the existing fork without writes and retains content after a failed reference save", async () => {
  const content = "# original\nold prompt --w 512\n";
  const fetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    if (init?.method === "PATCH") return new Response(JSON.stringify({ error: "readonly" }), { status: 409 });
    return new Response(JSON.stringify({ ok: true, exists: true, file: "configs/sample-prompts/imported/test.txt", content, prompts: ["old prompt --w 512"] }));
  });
  vi.stubGlobal("fetch", fetch);
  const onClose = vi.fn();
  render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    <TrainingSamplePrompts file={{ path: "configs/imported/test.toml", method: "lora" }} promptFile="" onClose={onClose} onSaved={async () => {}} />
  </QueryClientProvider>);
  await screen.findByRole("button", { name: "样张 1: old prompt" });
  expect(String(fetch.mock.calls[0][0])).toContain("configs%2Fsample-prompts%2Fimported%2Ftest.txt");
  expect(fetch.mock.calls.every(([, init]) => !init?.method)).toBe(true);
  const user = userEvent.setup();
  await user.click(screen.getByRole("button", { name: "原文" }));
  expect(screen.getByLabelText("样张提示词内容")).toHaveValue(content);
  await user.click(screen.getByRole("button", { name: "保存提示词与配置引用" }));
  await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("readonly"));
  expect(onClose).not.toHaveBeenCalled();
  expect(screen.getByLabelText("样张提示词内容")).toHaveValue(content);
});
