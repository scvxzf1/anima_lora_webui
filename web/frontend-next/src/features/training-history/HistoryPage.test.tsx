import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { createMemoryRouter, RouterProvider } from "react-router-dom";
import { jsonResponse } from "../../test/renderInApp";
import { HistoryPage } from "./HistoryPage";
import { HistoryDetailPage } from "./HistoryDetailPage";
import { resetHistorySelection } from "./historyNavigation";
import { runHistoryDragConfirmation } from "./HistoryDrag";
import { batchUpdateHistoryTasks } from "./api";

const originalScrollIntoView = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "scrollIntoView");
const originalVisibilityState = Object.getOwnPropertyDescriptor(document, "visibilityState");

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  resetHistorySelection();
  vi.restoreAllMocks();
  if (originalVisibilityState) Object.defineProperty(document, "visibilityState", originalVisibilityState);
  else Reflect.deleteProperty(document, "visibilityState");
  if (originalScrollIntoView) Object.defineProperty(HTMLElement.prototype, "scrollIntoView", originalScrollIntoView);
  else delete (HTMLElement.prototype as Partial<HTMLElement>).scrollIntoView;
  vi.unstubAllGlobals();
});

it("rechecks cached history on mount while it is still inside the stale window", async () => {
  const fetcher = vi.fn(async () => jsonResponse({
    tasks: [{ id: "fresh", name: "Fresh from server", job: "training", state: "idle", archived: false }],
    total: 1,
    next_cursor: null,
  }));
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    if (String(input) === "/api/training/history/collections/settings") {
      return jsonResponse({ collection_order: [], config_group_order: {} });
    }
    if (String(input).startsWith("/api/training/history?")) return fetcher();
    throw new Error(`Unexpected request: ${String(input)}`);
  }));
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: 15_000 } },
  });
  client.setQueryData(["training-history", "list", 200, ""], {
    pages: [{ tasks: [{ id: "cached", name: "Cached history", job: "training", state: "idle", archived: false }], total: 1, next_cursor: null }],
    pageParams: [""],
  });
  const router = createMemoryRouter([{ path: "/history", element: <HistoryPage /> }], {
    initialEntries: ["/history?layout=list"],
  });
  render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);

  expect(await screen.findByRole("link", { name: /Fresh from server/ })).toBeInTheDocument();
  expect(fetcher).toHaveBeenCalledTimes(1);
});

it("refreshes active history only while visible and stops after the task terminates", async () => {
  vi.useFakeTimers({ toFake: ["setInterval", "clearInterval"] });
  let historyCalls = 0;
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url === "/api/training/history/collections/settings") {
      return jsonResponse({ collection_order: [], config_group_order: {} });
    }
    if (url.startsWith("/api/training/history?")) {
      historyCalls += 1;
      const active = historyCalls === 1;
      return jsonResponse({
        tasks: [{
          id: "active-task",
          name: active ? "Active task" : "Completed task",
          job: "training",
          state: active ? "running" : "idle",
          archived: false,
          log_count: active ? 3 : 8,
          metric_count: active ? 1 : 4,
        }],
        total: 1,
        next_cursor: null,
      });
    }
    throw new Error(`Unexpected request: ${url}`);
  }));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: Infinity } } });
  const router = createMemoryRouter([{ path: "/history", element: <HistoryPage /> }], {
    initialEntries: ["/history?layout=list"],
  });
  render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);
  expect(await screen.findByRole("link", { name: /Active task/ })).toBeInTheDocument();

  expect(screen.getByText(/3 日志/)).toBeInTheDocument();
  expect(screen.getByText(/1 指标/)).toBeInTheDocument();

  Object.defineProperty(document, "visibilityState", { configurable: true, value: "hidden" });
  act(() => document.dispatchEvent(new Event("visibilitychange")));
  expect(historyCalls).toBe(1);
  act(() => { vi.advanceTimersByTime(15_000); });
  expect(historyCalls).toBe(1);

  Object.defineProperty(document, "visibilityState", { configurable: true, value: "visible" });
  act(() => document.dispatchEvent(new Event("visibilitychange")));
  act(() => { vi.advanceTimersByTime(14_999); });
  expect(historyCalls).toBe(1);
  act(() => { vi.advanceTimersByTime(1); });
  await waitFor(() => expect(historyCalls).toBe(2));
  expect(await screen.findByRole("link", { name: /Completed task/ })).toBeInTheDocument();
  expect(screen.getByText(/8 日志/)).toBeInTheDocument();
  expect(screen.getByText(/4 指标/)).toBeInTheDocument();
  expect(historyCalls).toBe(2);
  act(() => { vi.advanceTimersByTime(15_000); });
  expect(historyCalls).toBe(2);
});

it("pauses automatic refresh while the runtime delete preview is open", async () => {
  vi.useFakeTimers({ toFake: ["setInterval", "clearInterval"] });
  const activeTask = { id: "active-task", name: "Active task", job: "training", state: "running", archived: false };
  let historyCalls = 0;
  const batchPayloads: Record<string, unknown>[] = [];
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url === "/api/training/history/collections/settings") return jsonResponse({ collection_order: [], config_group_order: {} });
    if (url === "/api/training/history/batch") {
      batchPayloads.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
      return jsonResponse({ ok: true, dry_run: true, task_count: 1, runtime_dir_count: 0, tasks: [activeTask], runtime_dirs: [], blocked: [] });
    }
    if (url.startsWith("/api/training/history?")) {
      historyCalls += 1;
      return jsonResponse({ tasks: [activeTask], total: 1, next_cursor: null });
    }
    throw new Error(`Unexpected request: ${url}`);
  }));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: Infinity } } });
  const router = createMemoryRouter([{ path: "/history", element: <HistoryPage /> }], {
    initialEntries: ["/history?layout=list"],
  });
  render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);
  const user = userEvent.setup();
  await user.click(await screen.findByRole("checkbox", { name: "选择 Active task" }));
  await user.click(screen.getByRole("button", { name: "删除记录及运行目录" }));
  await screen.findByText("Active task", { selector: "strong" });
  expect(batchPayloads).toEqual([{ action: "delete", task_ids: ["active-task"], delete_runtime_dirs: true, dry_run: true }]);

  act(() => { vi.advanceTimersByTime(15_000); });
  expect(historyCalls).toBe(1);
  expect(screen.getByRole("button", { name: "确认彻底删除" })).toBeInTheDocument();

  await user.click(screen.getByRole("button", { name: "取消" }));
  act(() => { vi.advanceTimersByTime(15_000); });
  await waitFor(() => expect(historyCalls).toBe(2));
});

it("rejects a captured drag confirmation after its selection scope changes", async () => {
  const batchPayloads: unknown[] = [];
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    if (String(input) === "/api/training/history/batch") {
      batchPayloads.push(JSON.parse(String(init?.body)));
      return jsonResponse({ ok: true });
    }
    throw new Error(`Unexpected request: ${String(input)}`);
  }));

  const capturedScope = "client:alpha";
  const currentScope = "client:beta";
  const staleConfirmation = () =>
    runHistoryDragConfirmation(capturedScope, currentScope, false, () => {
      void batchUpdateHistoryTasks({ action: "set_group", task_ids: ["alpha"], group: "target" });
    });

  expect(staleConfirmation()).toBe(false);
  await Promise.resolve();
  expect(batchPayloads).toEqual([]);
});

it("returns from a task detail to the filtered list and restores its task anchor", async () => {
  const task = {
    id: "task-1",
    name: "Needle training",
    job: "training",
    state: "error",
    archived: false,
  };
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url === "/api/training/history/collections/settings") {
      return jsonResponse({ collection_order: [], config_group_order: {} });
    }
    if (url === "/api/training/history/task-1") {
      return jsonResponse({ task, metrics: [] });
    }
    if (url.startsWith("/api/training/history?")) {
      return jsonResponse({ tasks: [task], total: 1, next_cursor: null });
    }
    throw new Error(`Unexpected request: ${url}`);
  }));

  const scrollIntoView = vi.fn();
  Object.defineProperty(HTMLElement.prototype, "scrollIntoView", {
    configurable: true,
    value: scrollIntoView,
  });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const router = createMemoryRouter([
    { path: "/history", element: <HistoryPage /> },
    { path: "/history/:taskId", element: <HistoryDetailPage /> },
  ], { initialEntries: ["/history?q=needle&state=error&layout=list"] });
  render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);

  await userEvent.setup().click(await screen.findByRole("checkbox", { name: "选择 Needle training" }));
  await userEvent.setup().click(screen.getByRole("link", { name: /Needle training/ }));
  expect(await screen.findByRole("heading", { name: "Needle training" })).toBeInTheDocument();
  await userEvent.setup().click(screen.getByRole("link", { name: "返回历史" }));

  await waitFor(() => expect(router.state.location.pathname).toBe("/history"));
  expect(router.state.location.search).toContain("q=needle");
  expect(router.state.location.search).toContain("state=error");
  expect(router.state.location.search).toContain("anchor=task-1");
  await waitFor(() => expect(scrollIntoView).toHaveBeenCalled());
  expect(screen.getByRole("link", { name: /Needle training/ })).toBeInTheDocument();
  expect(screen.getByRole("combobox", { name: "状态" })).toHaveValue("error");
  expect(screen.getByRole("checkbox", { name: "选择 Needle training" })).toBeChecked();
});

it("keeps cross-page selection during refresh and disables batch actions while fetching", async () => {
  const tasks = [
    { id: "task-1", name: "First training", job: "training", state: "idle", archived: false },
    { id: "task-2", name: "Second training", job: "training", state: "idle", archived: false },
  ];
  const refreshedSecond = deferred<Response>();
  let firstPageReads = 0;
  const batchPayloads: unknown[] = [];
  vi.stubGlobal("confirm", vi.fn(() => true));
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url === "/api/training/history/collections/settings") return jsonResponse({ collection_order: [], config_group_order: {} });
    if (url === "/api/training/history/batch") {
      batchPayloads.push(JSON.parse(String(init?.body)));
      return jsonResponse({ ok: true });
    }
    if (url.startsWith("/api/training/history?")) {
      const parsed = new URL(url, "http://localhost");
      if (parsed.searchParams.get("cursor") === "next") {
        if (firstPageReads > 1) return refreshedSecond.promise;
        return jsonResponse({ tasks: [tasks[1]], total: 2, next_cursor: null });
      }
      firstPageReads += 1;
      return jsonResponse({ tasks: [tasks[0]], total: 2, next_cursor: "next" });
    }
    throw new Error(`Unexpected request: ${url}`);
  }));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const router = createMemoryRouter([{ path: "/history", element: <HistoryPage /> }], {
    initialEntries: ["/history?layout=list"],
  });
  render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);
  const user = userEvent.setup();
  await user.click(await screen.findByRole("checkbox", { name: "选择 First training" }));
  await user.click(screen.getByRole("button", { name: "载入更多记录" }));
  await screen.findByRole("checkbox", { name: "选择 Second training" });
  await user.click(screen.getByRole("checkbox", { name: "选择 Second training" }));
  expect(screen.getByText("已选 2 项")).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "刷新" }));
  await waitFor(() => expect(firstPageReads).toBe(2));
  expect(screen.getByRole("button", { name: "归档已选" })).toBeDisabled();
  expect(screen.getByRole("button", { name: "刷新中" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "刷新中" }));
  expect(firstPageReads).toBe(2);
  await act(async () => refreshedSecond.resolve(jsonResponse({ tasks: [tasks[1]], total: 2, next_cursor: null })));
  await waitFor(() => expect(screen.getByRole("button", { name: "归档已选" })).toBeEnabled());
  expect(screen.getByText("已选 2 项")).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "归档已选" }));
  await waitFor(() => expect(batchPayloads).toEqual([{ action: "archive", task_ids: ["task-1", "task-2"] }]));
});

it("clears unverified selection after refresh failure and allows a retry", async () => {
  const task = { id: "task-1", name: "Retriable task", job: "training", state: "idle", archived: false };
  const failedRefresh = deferred<Response>();
  const batchPayloads: unknown[] = [];
  let reads = 0;
  vi.stubGlobal("confirm", vi.fn(() => true));
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url === "/api/training/history/collections/settings") return jsonResponse({ collection_order: [], config_group_order: {} });
    if (url === "/api/training/history/batch") {
      batchPayloads.push(JSON.parse(String(init?.body)));
      return jsonResponse({ ok: true });
    }
    if (url.startsWith("/api/training/history?")) {
      reads += 1;
      if (reads === 2) return failedRefresh.promise;
      return jsonResponse({ tasks: [task], total: 1, next_cursor: null });
    }
    throw new Error(`Unexpected request: ${url}`);
  }));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const router = createMemoryRouter([{ path: "/history", element: <HistoryPage /> }], { initialEntries: ["/history?layout=list"] });
  render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);
  const user = userEvent.setup();
  await user.click(await screen.findByRole("checkbox", { name: "选择 Retriable task" }));
  await user.click(screen.getByRole("button", { name: "刷新" }));
  await waitFor(() => expect(reads).toBe(2));
  await act(async () => failedRefresh.resolve(new Response("failure", { status: 500 })));
  await waitFor(() => expect(screen.queryByText("已选 1 项")).not.toBeInTheDocument());
  expect(screen.getByText(/已清除无法核对的选择/)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "刷新" })).toBeEnabled();
  await user.click(screen.getByRole("button", { name: "刷新" }));
  await waitFor(() => expect(reads).toBe(3));
  await user.click(await screen.findByRole("checkbox", { name: "选择 Retriable task" }));
  await user.click(screen.getByRole("button", { name: "归档已选" }));
  await waitFor(() => expect(batchPayloads).toEqual([{ action: "archive", task_ids: ["task-1"] }]));
});

it("clears selection when the refreshed second page fails", async () => {
  const first = { id: "first", name: "First page task", job: "training", state: "idle", archived: false };
  const second = { id: "second", name: "Second page task", job: "training", state: "idle", archived: false };
  const failedPage = deferred<Response>();
  let rootReads = 0;
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url === "/api/training/history/collections/settings") return jsonResponse({ collection_order: [], config_group_order: {} });
    if (url.startsWith("/api/training/history?")) {
      const cursor = new URL(url, "http://localhost").searchParams.get("cursor");
      if (cursor === "next") return rootReads > 1 ? failedPage.promise : jsonResponse({ tasks: [second], total: 2, next_cursor: null });
      rootReads += 1;
      return jsonResponse({ tasks: [first], total: 2, next_cursor: "next" });
    }
    throw new Error(`Unexpected request: ${url}`);
  }));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const router = createMemoryRouter([{ path: "/history", element: <HistoryPage /> }], { initialEntries: ["/history?layout=list"] });
  render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);
  const user = userEvent.setup();
  await user.click(await screen.findByRole("checkbox", { name: "选择 First page task" }));
  await user.click(screen.getByRole("button", { name: "载入更多记录" }));
  await user.click(await screen.findByRole("checkbox", { name: "选择 Second page task" }));
  await user.click(screen.getByRole("button", { name: "刷新" }));
  await waitFor(() => expect(rootReads).toBe(2));
  expect(screen.getByRole("button", { name: "刷新中" })).toBeDisabled();
  await act(async () => failedPage.resolve(new Response("failure", { status: 500 })));
  await waitFor(() => expect(screen.queryByText("已选 2 项")).not.toBeInTheDocument());
  expect(screen.getByText(/已清除无法核对的选择/)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "刷新" })).toBeEnabled();
});

it("does not apply an old refresh result after switching server search", async () => {
  const alpha = { id: "alpha", name: "Alpha task", job: "training", state: "idle", archived: false };
  const beta = { id: "beta", name: "Beta task", job: "training", state: "idle", archived: false };
  const oldRefresh = deferred<Response>();
  const batchPayloads: unknown[] = [];
  let alphaReads = 0;
  vi.stubGlobal("confirm", vi.fn(() => true));
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url === "/api/training/history/collections/settings") return jsonResponse({ collection_order: [], config_group_order: {} });
    if (url === "/api/training/history/batch") {
      batchPayloads.push(JSON.parse(String(init?.body)));
      return jsonResponse({ ok: true });
    }
    if (url.startsWith("/api/training/history?")) {
      if (new URL(url, "http://localhost").searchParams.get("q") === "alpha") {
        alphaReads += 1;
        return alphaReads === 2 ? oldRefresh.promise : jsonResponse({ tasks: [alpha], total: 1, next_cursor: null });
      }
      return jsonResponse({ tasks: [beta], total: 1, next_cursor: null });
    }
    throw new Error(`Unexpected request: ${url}`);
  }));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const router = createMemoryRouter([{ path: "/history", element: <HistoryPage /> }], { initialEntries: ["/history?q=alpha&layout=list"] });
  render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);
  const user = userEvent.setup();
  await user.click(await screen.findByRole("checkbox", { name: "选择 Alpha task" }));
  await user.click(screen.getByRole("button", { name: "刷新" }));
  await waitFor(() => expect(alphaReads).toBe(2));
  fireEvent.change(screen.getByRole("searchbox", { name: "搜索历史记录" }), { target: { value: "beta" } });
  await user.click(await screen.findByRole("checkbox", { name: "选择 Beta task" }));
  await act(async () => oldRefresh.resolve(jsonResponse({ tasks: [alpha], total: 1, next_cursor: null })));
  expect(screen.getByRole("checkbox", { name: "选择 Beta task" })).toBeChecked();
  await user.click(screen.getByRole("button", { name: "归档已选" }));
  await waitFor(() => expect(batchPayloads).toEqual([{ action: "archive", task_ids: ["beta"] }]));
});

it("reconciles selection when refresh ends before the previously loaded page depth", async () => {
  const tasks = [
    { id: "task-1", name: "Still here", job: "training", state: "idle", archived: false },
    { id: "task-2", name: "Middle task", job: "training", state: "idle", archived: false },
    { id: "task-3", name: "Removed task", job: "training", state: "idle", archived: false },
  ];
  let refreshed = false;
  let releaseRefresh!: () => void;
  const batchPayloads: unknown[] = [];
  vi.stubGlobal("confirm", vi.fn(() => true));
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url === "/api/training/history/collections/settings") return jsonResponse({ collection_order: [], config_group_order: {} });
    if (url === "/api/training/history/batch") {
      batchPayloads.push(JSON.parse(String(init?.body)));
      return jsonResponse({ ok: true });
    }
    if (url.startsWith("/api/training/history?")) {
      const cursor = new URL(url, "http://localhost").searchParams.get("cursor");
      if (!cursor && refreshed) await new Promise<void>((resolve) => { releaseRefresh = resolve; });
      if (cursor === "third") return jsonResponse({ tasks: [tasks[2]], total: 3, next_cursor: null });
      if (cursor === "second") return jsonResponse({ tasks: [tasks[1]], total: refreshed ? 2 : 3, next_cursor: refreshed ? null : "third" });
      return jsonResponse({ tasks: [tasks[0]], total: refreshed ? 2 : 3, next_cursor: "second" });
    }
    throw new Error(`Unexpected request: ${url}`);
  }));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const router = createMemoryRouter([{ path: "/history", element: <HistoryPage /> }], { initialEntries: ["/history?layout=list"] });
  render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);
  const user = userEvent.setup();
  await user.click(await screen.findByRole("checkbox", { name: "选择 Still here" }));
  await user.click(screen.getByRole("button", { name: "载入更多记录" }));
  await screen.findByRole("checkbox", { name: "选择 Middle task" });
  await user.click(screen.getByRole("checkbox", { name: "选择 Middle task" }));
  await user.click(screen.getByRole("button", { name: "载入更多记录" }));
  await user.click(await screen.findByRole("checkbox", { name: "选择 Removed task" }));
  refreshed = true;
  await user.click(screen.getByRole("button", { name: "刷新" }));
  expect(screen.getByRole("button", { name: "归档已选" })).toBeDisabled();
  await waitFor(() => expect(releaseRefresh).toBeDefined());
  releaseRefresh();
  await waitFor(() => expect(screen.getByText("已选 2 项")).toBeInTheDocument());
  expect(screen.getByRole("checkbox", { name: "选择 Still here" })).toBeChecked();
  expect(screen.getByRole("checkbox", { name: "选择 Middle task" })).toBeChecked();
  expect(screen.getByRole("button", { name: "归档已选" })).toBeEnabled();
  await user.click(screen.getByRole("button", { name: "归档已选" }));
  await waitFor(() => expect(batchPayloads).toEqual([{ action: "archive", task_ids: ["task-1", "task-2"] }]));
});

it("clears selected tasks when switching config groups", async () => {
  const tasks = [
    { id: "task-a", name: "Alpha task", job: "training", state: "idle", archived: false, group: "project", history_group_key: "alpha", history_group_label: "Alpha config" },
    { id: "task-b", name: "Beta task", job: "training", state: "idle", archived: false, group: "project", history_group_key: "beta", history_group_label: "Beta config" },
  ];
  const batchPayloads: unknown[] = [];
  vi.stubGlobal("confirm", vi.fn(() => true));
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url === "/api/training/history/collections/settings") return jsonResponse({ collection_order: ["project"], config_group_order: {} });
    if (url === "/api/training/history/batch") {
      batchPayloads.push(JSON.parse(String(init?.body)));
      return jsonResponse({ ok: true });
    }
    if (url.startsWith("/api/training/history?")) return jsonResponse({ tasks, total: 2, next_cursor: null });
    throw new Error(`Unexpected request: ${url}`);
  }));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const router = createMemoryRouter([{ path: "/history", element: <HistoryPage /> }], { initialEntries: ["/history?layout=list&collection=project"] });
  render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);
  const user = userEvent.setup();
  await user.click(await screen.findByRole("checkbox", { name: "选择 Alpha task" }));
  await user.click(screen.getByRole("button", { name: "Beta config" }));
  expect(screen.queryByText("已选 1 项")).not.toBeInTheDocument();
  await user.click(screen.getByRole("checkbox", { name: "选择 Beta task" }));
  await user.click(screen.getByRole("button", { name: "归档已选" }));
  await waitFor(() => expect(batchPayloads).toEqual([{ action: "archive", task_ids: ["task-b"] }]));
});

it("keeps the active collection and selection through a same-scope SPA detail round trip", async () => {
  const task = { id: "task-back", name: "Back navigation", job: "training", state: "idle", archived: false, group: "project" };
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url === "/api/training/history/collections/settings") return jsonResponse({ collection_order: ["project"], config_group_order: {} });
    if (url === "/api/training/history/task-back") return jsonResponse({ task, metrics: [] });
    if (url.startsWith("/api/training/history?")) return jsonResponse({ tasks: [task], total: 1, next_cursor: null });
    throw new Error(`Unexpected request: ${url}`);
  }));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const router = createMemoryRouter([
    { path: "/history", element: <HistoryPage /> },
    { path: "/history/:taskId", element: <HistoryDetailPage /> },
  ], { initialEntries: ["/history?layout=list&collection=project"] });
  render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);
  const user = userEvent.setup();
  await user.click(await screen.findByRole("checkbox", { name: "选择 Back navigation" }));
  await user.click(screen.getByRole("link", { name: /Back navigation/ }));
  await screen.findByRole("heading", { name: "Back navigation" });
  await router.navigate(-1);
  expect(await screen.findByRole("checkbox", { name: "选择 Back navigation" })).toBeChecked();
  expect(router.state.location.search).toContain("collection=project");
  const selectedCollection = document.querySelector(".object-library .object-row[data-selected='true']");
  expect(selectedCollection).toHaveTextContent("project");
});

it("drops selection across mounted search scopes and does not restore it on browser back", async () => {
  const alpha = { id: "alpha", name: "Alpha task", job: "training", state: "idle", archived: false };
  const beta = { id: "beta", name: "Beta task", job: "training", state: "idle", archived: false };
  const batchPayloads: unknown[] = [];
  vi.stubGlobal("confirm", vi.fn(() => true));
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url === "/api/training/history/collections/settings") return jsonResponse({ collection_order: [], config_group_order: {} });
    if (url === "/api/training/history/batch") {
      batchPayloads.push(JSON.parse(String(init?.body)));
      return jsonResponse({ ok: true });
    }
    if (url.startsWith("/api/training/history?")) {
      const task = new URL(url, "http://localhost").searchParams.get("q") === "alpha" ? alpha : beta;
      return jsonResponse({ tasks: [task], total: 1, next_cursor: null });
    }
    throw new Error(`Unexpected request: ${url}`);
  }));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const router = createMemoryRouter([
    { path: "/history", element: <HistoryPage /> },
    { path: "/outside", element: <p>Outside history</p> },
  ], { initialEntries: ["/history?q=alpha&layout=list"] });
  render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);
  const user = userEvent.setup();
  await user.click(await screen.findByRole("checkbox", { name: "选择 Alpha task" }));
  await act(async () => { await router.navigate("/outside"); });
  await screen.findByText("Outside history");
  await act(async () => { await router.navigate("/history?q=beta&layout=list"); });
  expect(await screen.findByRole("checkbox", { name: "选择 Beta task" })).not.toBeChecked();
  expect(screen.queryByText("已选 1 项")).not.toBeInTheDocument();
  await user.click(screen.getByRole("checkbox", { name: "选择 Beta task" }));
  await user.click(screen.getByRole("button", { name: "归档已选" }));
  await waitFor(() => expect(batchPayloads).toEqual([{ action: "archive", task_ids: ["beta"] }]));
  await act(async () => { await router.navigate(-1); });
  await act(async () => { await router.navigate(-1); });
  expect(await screen.findByRole("checkbox", { name: "选择 Alpha task" })).not.toBeChecked();
});

it.each(["refresh", "filter"])("closes an open runtime delete preview on %s before old IDs can submit", async (change) => {
  const alpha = { id: "alpha", name: "Alpha task", job: "training", state: "idle", archived: false };
  const beta = { id: "beta", name: "Beta task", job: "training", state: "idle", archived: false };
  const batchPayloads: Record<string, unknown>[] = [];
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url === "/api/training/history/collections/settings") return jsonResponse({ collection_order: [], config_group_order: {} });
    if (url === "/api/training/history/batch") {
      const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      batchPayloads.push(body);
      return jsonResponse({ ok: true, dry_run: true, task_count: 1, runtime_dir_count: 0, tasks: [alpha], runtime_dirs: [], blocked: [] });
    }
    if (url.startsWith("/api/training/history?")) {
      const task = new URL(url, "http://localhost").searchParams.get("q") === "beta" ? beta : alpha;
      return jsonResponse({ tasks: [task], total: 1, next_cursor: null });
    }
    throw new Error(`Unexpected request: ${url}`);
  }));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const router = createMemoryRouter([{ path: "/history", element: <HistoryPage /> }], { initialEntries: ["/history?q=alpha&layout=list"] });
  render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);
  const user = userEvent.setup();
  await user.click(await screen.findByRole("checkbox", { name: "选择 Alpha task" }));
  await user.click(screen.getByRole("button", { name: "删除记录及运行目录" }));
  await screen.findByText("Alpha task", { selector: "strong" });
  await user.click(screen.getByRole("checkbox", { name: "我已核对上述列表，并确认永久删除" }));
  expect(screen.getByRole("button", { name: "确认彻底删除" })).toBeEnabled();
  if (change === "refresh") await user.click(screen.getByRole("button", { name: "刷新" }));
  else fireEvent.change(screen.getByRole("searchbox", { name: "搜索历史记录" }), { target: { value: "beta" } });
  await waitFor(() => expect(screen.queryByRole("button", { name: "确认彻底删除" })).not.toBeInTheDocument());
  expect(batchPayloads).toEqual([{ action: "delete", task_ids: ["alpha"], delete_runtime_dirs: true, dry_run: true }]);
});

it("searches history on the server and resets pagination without clearing other filters", async () => {
  const requests: URL[] = [];
  const matchingTask = {
    id: "portrait-match",
    name: "Portrait training",
    job: "training",
    state: "error",
    archived: false,
    history_group_label: "portrait",
  };
  const unrelatedTask = {
    id: "landscape-unrelated",
    name: "Landscape training",
    job: "training",
    state: "error",
    archived: false,
    history_group_label: "landscape",
  };
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = new URL(String(input), "http://localhost");
    if (url.pathname === "/api/training/history/collections/settings") {
      return jsonResponse({ collection_order: [], config_group_order: {} });
    }
    if (url.pathname === "/api/training/history") {
      requests.push(url);
      const tasks = url.searchParams.get("q")?.trim()
        ? [matchingTask]
        : [matchingTask, unrelatedTask];
      return jsonResponse({ tasks, total: tasks.length, next_cursor: null });
    }
    throw new Error(`Unexpected request: ${url}`);
  }));

  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const router = createMemoryRouter([
    { path: "/history", element: <HistoryPage /> },
  ], { initialEntries: ["/history?state=error&layout=list&page=3&anchor=task-old"] });
  render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);

  await screen.findByRole("combobox", { name: "状态" });
  await userEvent.setup().type(screen.getByRole("searchbox", { name: "搜索历史记录" }), " portrait ");

  await waitFor(() => {
    expect(requests.some((url) => url.searchParams.get("q") === "portrait")).toBe(true);
  });
  expect(await screen.findByRole("link", { name: /Portrait training/ })).toBeInTheDocument();
  expect(screen.queryByRole("link", { name: /Landscape training/ })).not.toBeInTheDocument();
  const params = new URLSearchParams(router.state.location.search);
  expect(params.get("q")).toBe(" portrait ");
  expect(params.get("state")).toBe("error");
  expect(params.has("page")).toBe(false);
  expect(params.has("anchor")).toBe(false);
});

const advancedFilterCases = [
  { label: "训练变体", param: "variant", field: "training_variant", value: "lora", other: "loha" },
  { label: "预处理精度", param: "preprocess_precision", field: "preprocess_precision", value: "bf16", other: "fp16" },
  { label: "块交换精度", param: "swap", field: "block_swap_precision", value: "bf16", other: "fp16" },
  { label: "底模计算路径", param: "compute", field: "base_compute", value: "nf4", other: "bf16" },
  { label: "精度倾向", param: "precision", field: "precision_preference", value: "bf16", other: "fp16" },
] as const;

it.each(advancedFilterCases)("filters loaded tasks by $label and writes its URL parameter", async ({ label, param, field, value, other }) => {
  const tasks = [
    { id: "match", name: "Matching snapshot", job: "training", state: "idle", archived: false, [field]: value },
    { id: "other", name: "Other snapshot", job: "training", state: "idle", archived: false, [field]: other },
    { id: "missing", name: "Missing snapshot", job: "training", state: "idle", archived: false },
  ];
  const requests: URL[] = [];
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = new URL(String(input), "http://localhost");
    if (url.pathname === "/api/training/history/collections/settings") {
      return jsonResponse({ collection_order: [], config_group_order: {} });
    }
    if (url.pathname === "/api/training/history") {
      requests.push(url);
      return jsonResponse({ tasks, total: 30, next_cursor: "more" });
    }
    throw new Error(`Unexpected request: ${url}`);
  }));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const router = createMemoryRouter([{ path: "/history", element: <HistoryPage /> }], {
    initialEntries: ["/history?layout=list"],
  });
  render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);

  await userEvent.setup().click(await screen.findByText("高级筛选"));
  const select = await screen.findByRole("combobox", { name: label });
  await userEvent.setup().selectOptions(select, value);

  expect(await screen.findByRole("link", { name: /Matching snapshot/ })).toBeInTheDocument();
  expect(screen.queryByRole("link", { name: /Other snapshot/ })).not.toBeInTheDocument();
  expect(screen.queryByRole("link", { name: /Missing snapshot/ })).not.toBeInTheDocument();
  expect(new URLSearchParams(router.state.location.search).get(param)).toBe(value);
  expect(screen.getByText(/状态、集合及高级筛选仅作用于已读取记录/)).toBeInTheDocument();
  expect(requests).toHaveLength(1);
  expect(requests[0].searchParams.has("cursor")).toBe(false);
});

it("restores advanced filter selections from URL and does not match tasks without snapshot fields", async () => {
  const completeSnapshot = {
    id: "complete",
    name: "Complete snapshot",
    job: "training",
    state: "idle",
    archived: false,
    model_family: "krea2_raw",
    training_variant: "lora",
    preprocess_precision: "bf16",
    block_swap_precision: "bf16",
    base_compute: "nf4",
    precision_preference: "bf16",
  };
  const missingSnapshot = {
    id: "missing",
    name: "No snapshot fields",
    job: "training",
    state: "idle",
    archived: false,
  };
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = new URL(String(input), "http://localhost");
    if (url.pathname === "/api/training/history/collections/settings") {
      return jsonResponse({ collection_order: [], config_group_order: {} });
    }
    if (url.pathname === "/api/training/history") {
      return jsonResponse({ tasks: [completeSnapshot, missingSnapshot], total: 20, next_cursor: "more" });
    }
    throw new Error(`Unexpected request: ${url}`);
  }));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const router = createMemoryRouter([{ path: "/history", element: <HistoryPage /> }], {
    initialEntries: ["/history?layout=list&base=krea2_raw&variant=lora&preprocess_precision=bf16&swap=bf16&compute=nf4&precision=bf16"],
  });
  render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);

  await userEvent.setup().click(await screen.findByText("高级筛选"));
  for (const { label, param } of [
    { label: "基座模型", param: "base" },
    { label: "训练变体", param: "variant" },
    { label: "预处理精度", param: "preprocess_precision" },
    { label: "块交换精度", param: "swap" },
    { label: "底模计算路径", param: "compute" },
    { label: "精度倾向", param: "precision" },
  ]) {
    expect(screen.getByRole("combobox", { name: label })).toHaveValue(new URLSearchParams(router.state.location.search).get(param));
  }
  expect(await screen.findByRole("link", { name: /Complete snapshot/ })).toBeInTheDocument();
  expect(screen.queryByRole("link", { name: /No snapshot fields/ })).not.toBeInTheDocument();
  expect(screen.getByText(/状态、集合及高级筛选仅作用于已读取记录/)).toBeInTheDocument();
});

it("keeps an unloaded URL filter visible and deduplicates it when a later page contains the value", async () => {
  const firstPageTask = { id: "first", name: "First page", job: "training", state: "idle", archived: false, training_variant: "loha" };
  const laterTask = { id: "later", name: "Later match", job: "training", state: "idle", archived: false, training_variant: "lora" };
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = new URL(String(input), "http://localhost");
    if (url.pathname === "/api/training/history/collections/settings") {
      return jsonResponse({ collection_order: [], config_group_order: {} });
    }
    if (url.pathname === "/api/training/history") {
      return url.searchParams.get("cursor") === "more"
        ? jsonResponse({ tasks: [laterTask], total: 2, next_cursor: null })
        : jsonResponse({ tasks: [firstPageTask], total: 2, next_cursor: "more" });
    }
    throw new Error(`Unexpected request: ${url}`);
  }));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const router = createMemoryRouter([{ path: "/history", element: <HistoryPage /> }], {
    initialEntries: ["/history?layout=list&variant=lora"],
  });
  render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);

  await userEvent.setup().click(await screen.findByText("高级筛选"));
  const select = screen.getByRole("combobox", { name: "训练变体" });
  await waitFor(() => expect(select).toHaveValue("lora"));
  expect(screen.getByRole("option", { name: "lora（当前条件，尚未在已加载记录中出现）" })).toHaveValue("lora");
  expect(screen.getByText("已加载结果中没有符合筛选条件的记录，仍有结果未加载。")).toBeInTheDocument();
  expect(screen.getByText(/高级筛选仅作用于已读取记录/)).toBeInTheDocument();

  await userEvent.setup().click(screen.getByRole("button", { name: "载入更多记录" }));
  expect(await screen.findByRole("link", { name: /Later match/ })).toBeInTheDocument();
  expect(select).toHaveValue("lora");
  expect(screen.getAllByRole("option", { name: "lora" })).toHaveLength(1);
  expect(screen.queryByRole("option", { name: "lora（当前条件，尚未在已加载记录中出现）" })).not.toBeInTheDocument();
});
