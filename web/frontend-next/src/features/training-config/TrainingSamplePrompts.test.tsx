import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
const apiRequestMock = vi.hoisted(() => vi.fn());
vi.mock("../../api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../api/client")>()),
  apiRequest: apiRequestMock,
}));
import { TrainingSamplePrompts } from "./TrainingExtras";
import { ApiError } from "../../api/client";
afterEach(() => { cleanup(); apiRequestMock.mockReset(); vi.unstubAllGlobals(); });

describe.sequential("training sample prompts", () => {
  for (const kind of ["new", "existing", "reverted", "dirty"]) {
    it(`handles ${kind} row close without native confirmation`, async () => {
      apiRequestMock.mockResolvedValue({ ok: true, exists: true, file: "configs/sample_prompts.txt", content: "old prompt --w 512\n", prompts: [] });
      const nativeConfirm = vi.fn();
      vi.stubGlobal("confirm", nativeConfirm);
      const onClose = vi.fn();
      render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <TrainingSamplePrompts file={{ path: "configs/imported/test.toml" }} promptFile="configs/sample_prompts.txt" onClose={onClose} onSaved={async () => {}} />
      </QueryClientProvider>);
      await screen.findByRole("button", { name: "样张 1: old prompt" }, { timeout: 5000 });
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
    apiRequestMock.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (init?.method === "PATCH") throw new Error("readonly");
      if (init?.method === "PUT") return { ok: true, exists: true, file: "configs/sample-prompts/imported/test.txt", content, prompts: ["old prompt --w 512"] };
      return { ok: true, exists: true, file: "configs/sample-prompts/imported/test.txt", content, prompts: ["old prompt --w 512"] };
    });
    const onClose = vi.fn();
  render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    <TrainingSamplePrompts file={{ path: "configs/imported/test.toml", method: "lora" }} promptFile="" onClose={onClose} onSaved={async () => {}} />
  </QueryClientProvider>);
  await screen.findByRole("button", { name: "样张 1: old prompt" }, { timeout: 5000 });
  expect(String(apiRequestMock.mock.calls[0][0])).toContain("configs%2Fsample-prompts%2Fimported%2Ftest.txt");
  expect(apiRequestMock.mock.calls.every(([, init]) => !init?.method)).toBe(true);
  const user = userEvent.setup();
  await user.click(screen.getByRole("button", { name: "原文" }));
  expect(screen.getByLabelText("样张提示词内容")).toHaveValue(content);
  await user.click(screen.getByRole("button", { name: "保存提示词与配置引用" }));
  await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("readonly"));
  expect(onClose).not.toHaveBeenCalled();
  expect(screen.getByLabelText("样张提示词内容")).toHaveValue(content);
  });

  it("keeps the draft and blocks a stale prompt revision after 409", async () => {
    apiRequestMock.mockImplementation(async (_input: RequestInfo | URL, init?: RequestInit) => {
      if (init?.method === "PUT") throw new ApiError("文件已在其他位置修改", 409, {});
      return { ok: true, exists: true, file: "configs/sample_prompts.txt", content: "old prompt\n", prompts: ["old prompt"], revision: "old-revision" };
    });
    render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <TrainingSamplePrompts file={{ path: "configs/imported/test.toml" }} promptFile="configs/sample_prompts.txt" onClose={() => {}} onSaved={async () => {}} />
    </QueryClientProvider>);
    const user = userEvent.setup();
    await screen.findByRole("button", { name: "样张 1: old prompt" });
    await user.click(screen.getByRole("button", { name: "原文" }));
    const textarea = screen.getByLabelText("样张提示词内容");
    await user.clear(textarea);
    await user.type(textarea, "my draft\n");
    await user.click(screen.getByRole("button", { name: "保存提示词与配置引用" }));
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("文件已在其他位置修改"));
    expect(textarea).toHaveValue("my draft\n");
    expect(screen.getByRole("button", { name: "保存提示词与配置引用" })).toBeDisabled();
    const writes = apiRequestMock.mock.calls.filter(([, init]) => init?.method === "PUT");
    expect(writes).toHaveLength(1);
    expect(JSON.parse(writes[0][1].body as string).revision).toBe("old-revision");
  });
});
