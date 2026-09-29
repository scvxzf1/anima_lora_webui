import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { HistorySnapshot } from "./HistorySnapshot";

const scrollIntoViewDescriptor = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "scrollIntoView");

describe("HistorySnapshot", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    if (scrollIntoViewDescriptor) Object.defineProperty(HTMLElement.prototype, "scrollIntoView", scrollIntoViewDescriptor);
    else Reflect.deleteProperty(HTMLElement.prototype, "scrollIntoView");
  });

  it("searches, counts, highlights and navigates matches without parsing markup", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ artifacts: [] }), { status: 200 })));
    const user = userEvent.setup();
    render(<QueryClientProvider client={new QueryClient()}><HistorySnapshot taskId="task/1" detail={{ config_toml: '[train]\nlearning_rate = 0.001\n# learning_rate' }} /></QueryClientProvider>);
    await user.type(screen.getByRole("searchbox", { name: "搜索配置快照" }), "learning_rate");
    expect(screen.getByText("1 / 2")).toBeInTheDocument();
    expect(document.querySelectorAll("mark")).toHaveLength(2);
    await user.click(screen.getByRole("button", { name: "下一个匹配项" }));
    expect(screen.getByText("2 / 2")).toBeInTheDocument();
    await user.clear(screen.getByRole("searchbox", { name: "搜索配置快照" }));
    expect(screen.getByText("未搜索")).toBeInTheDocument();
  });

  it("keeps match offsets aligned to the original text when lowercase expands", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ artifacts: [] }), { status: 200 })));
    const user = userEvent.setup();
    render(<QueryClientProvider client={new QueryClient()}><HistorySnapshot taskId="task/1" detail={{ config_toml: "İ=1" }} /></QueryClientProvider>);
    await user.type(screen.getByRole("searchbox", { name: "搜索配置快照" }), "=");
    const marks = document.querySelectorAll(".history-snapshot-code mark");
    expect(marks).toHaveLength(1);
    expect(marks[0]).toHaveTextContent("=");
  });

  it("matches contextual Unicode case and treats search punctuation literally", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ artifacts: [] }), { status: 200 })));
    const user = userEvent.setup();
    render(<QueryClientProvider client={new QueryClient()}><HistorySnapshot taskId="task/1" detail={{ config_toml: "ΟΣ = 1\nkey.* = 2" }} /></QueryClientProvider>);
    const search = screen.getByRole("searchbox", { name: "搜索配置快照" });
    await user.type(search, "ος");
    expect(screen.getByText("1 / 1")).toBeInTheDocument();
    expect(document.querySelector(".history-snapshot-code mark")).toHaveTextContent("ΟΣ");
    await user.clear(search);
    await user.type(search, ".*");
    expect(screen.getByText("1 / 1")).toBeInTheDocument();
    expect(document.querySelector(".history-snapshot-code mark")).toHaveTextContent(".*");
  });

  it("scrolls the first match on search and follows navigation", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ artifacts: [] }), { status: 200 })));
    vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => { callback(0); return 1; });
    vi.stubGlobal("cancelAnimationFrame", vi.fn());
    const scrolled: Element[] = [];
    Object.defineProperty(HTMLElement.prototype, "scrollIntoView", {
      configurable: true,
      value: vi.fn(function (this: Element) { scrolled.push(this); }),
    });
    const user = userEvent.setup();
    render(<QueryClientProvider client={new QueryClient()}><HistorySnapshot taskId="task/1" detail={{ config_toml: "x = x" }} /></QueryClientProvider>);
    await user.type(screen.getByRole("searchbox", { name: "搜索配置快照" }), "x");
    await waitFor(() => expect(scrolled.at(-1)).toBe(document.querySelector(".history-snapshot-hit-current")));
    const first = scrolled.at(-1);
    await user.click(screen.getByRole("button", { name: "下一个匹配项" }));
    await waitFor(() => expect(scrolled.at(-1)).toBe(document.querySelector(".history-snapshot-hit-current")));
    expect(scrolled.at(-1)).not.toBe(first);
  });

  it("copies the snapshot and retains the artifact whitelist download listing", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup();
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText } });
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ artifacts: [{ key: "config-snapshot", state: "available", name: "snapshot.toml" }] }), { status: 200 })));
    render(<QueryClientProvider client={new QueryClient()}><HistorySnapshot taskId="task/1" detail={{ config_toml: 'value = "<script>"' }} /></QueryClientProvider>);
    await user.click(screen.getByRole("button", { name: /复制全部/ }));
    expect(writeText).toHaveBeenCalledWith('value = "<script>"');
    expect(await screen.findByRole("link", { name: "snapshot.toml" })).toHaveAttribute("href", "/api/training/history/task%2F1/artifacts/config-snapshot?download=1");
    expect(document.querySelector(".history-snapshot-code script")).toBeNull();
  });
});
