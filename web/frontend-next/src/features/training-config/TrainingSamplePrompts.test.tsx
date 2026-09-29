import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
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
  it("applies common parameters to existing and newly added rows before saving the config link", async () => {
    const writes: Array<{ method: string; body: Record<string, unknown> }> = [];
    const content = "# keep\nstyle one --w 512 --h 512 --s 20 --g 3 --custom keep\nstyle two --w 768 --h 512 --s 30 --g 5\n";
    apiRequestMock.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input) === "/api/config/model-families") return { items: [] };
      if (init?.method) {
        const body = JSON.parse(String(init.body)) as Record<string, unknown>;
        writes.push({ method: init.method, body });
        if (init.method === "PUT") return { ok: true, file: "configs/sample-prompts/imported/saved.txt", content: String(body.content), prompts: [] };
        return { ok: true };
      }
      return { ok: true, exists: true, file: "configs/sample-prompts/imported/original.txt", content, prompts: [] };
    });
    const onClose = vi.fn();
    render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <TrainingSamplePrompts file={{ path: "configs/imported/training-config.toml" }} promptFile="configs/sample-prompts/imported/original.txt" onClose={onClose} onSaved={async () => {}} />
    </QueryClientProvider>);
    const user = userEvent.setup();
    await screen.findByRole("button", { name: "样张 2: style two" });
    expect(screen.getByRole("button", { name: "保存提示词与配置引用" })).toBeDisabled();
    await user.click(screen.getByText("统一参数"));
    for (const [label, value] of [["统一宽度", "1024"], ["统一高度", "1024"], ["统一步数", "28"], ["统一CFG", "4"]]) {
      await user.type(screen.getByLabelText(label), value);
    }
    await user.click(screen.getByRole("button", { name: "应用统一参数" }));
    await user.click(screen.getByRole("button", { name: "新增样张" }));
    expect(screen.getByLabelText("宽度")).toHaveValue(1024);
    expect(screen.getByLabelText("步数")).toHaveValue(28);
    await user.type(screen.getByLabelText("正向提示词"), "style three");
    await user.click(screen.getByRole("button", { name: "应用样张" }));
    await user.click(screen.getByRole("button", { name: "保存提示词与配置引用" }));
    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
    const put = writes.find((write) => write.method === "PUT")!;
    const saved = String(put.body.content);
    expect(put.body.train_config_file).toBe("configs/imported/training-config.toml");
    expect(saved).toContain("# keep\n");
    expect(saved).toContain("--custom keep");
    expect(saved.match(/--w 1024 --h 1024 --s 28 --g 4/g)).toHaveLength(3);
    expect(writes.find((write) => write.method === "PATCH")?.body.values).toEqual({ sample_prompts: "configs/sample-prompts/imported/saved.txt" });
  });

  it("ignores an older load after the dialog closes and reopens", async () => {
    const pending: Array<(value: unknown) => void> = [];
    apiRequestMock.mockImplementation((input: RequestInfo | URL) => {
      if (String(input) === "/api/config/model-families") return Promise.resolve({ items: [] });
      return new Promise((resolve) => pending.push(resolve));
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const dialog = () => <TrainingSamplePrompts file={{ path: "configs/imported/test.toml" }} promptFile="configs/sample-prompts/race.txt" onClose={() => {}} onSaved={async () => {}} />;
    const { rerender } = render(<QueryClientProvider client={client}>{dialog()}</QueryClientProvider>);
    await waitFor(() => expect(pending).toHaveLength(1));
    rerender(<QueryClientProvider client={client}>{null}</QueryClientProvider>);
    rerender(<QueryClientProvider client={client}>{dialog()}</QueryClientProvider>);
    await waitFor(() => expect(pending).toHaveLength(2));

    await act(async () => pending[1]({ ok: true, file: "configs/sample-prompts/new.txt", content: "new style --w 1024", prompts: [] }));
    expect(await screen.findByRole("button", { name: "样张 1: new style" })).toBeInTheDocument();
    await act(async () => pending[0]({ ok: true, file: "configs/sample-prompts/old.txt", content: "old style --w 512", prompts: [] }));
    expect(screen.getByRole("button", { name: "样张 1: new style" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "样张 1: old style" })).not.toBeInTheDocument();
  });

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
      expect(screen.getByRole("button", { name: "保存提示词与配置引用" })).toBeDisabled();
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
  expect(screen.getByRole("button", { name: "保存提示词与配置引用" })).toBeEnabled();
  expect(apiRequestMock.mock.calls.some(([input]) => String(input).includes("configs%2Fsample-prompts%2Fimported%2Ftest.txt"))).toBe(true);
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
