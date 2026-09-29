import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { createMemoryRouter, RouterProvider } from "react-router-dom";
import { jsonResponse } from "../../test/renderInApp";
import { HistoryPage } from "./HistoryPage";
import { HistoryDetailPage } from "./HistoryDetailPage";
import { resetHistorySelection } from "./historyNavigation";

const originalScrollIntoView = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "scrollIntoView");

afterEach(() => {
  cleanup();
  resetHistorySelection();
  if (originalScrollIntoView) Object.defineProperty(HTMLElement.prototype, "scrollIntoView", originalScrollIntoView);
  else delete (HTMLElement.prototype as Partial<HTMLElement>).scrollIntoView;
  vi.unstubAllGlobals();
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
  let releaseSecondPage!: () => void;
  let holdSecondPage = false;
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url === "/api/training/history/collections/settings") return jsonResponse({ collection_order: [], config_group_order: {} });
    if (url.startsWith("/api/training/history?")) {
      const parsed = new URL(url, "http://localhost");
      if (parsed.searchParams.get("cursor") === "next") {
        if (holdSecondPage) await new Promise<void>((resolve) => { releaseSecondPage = resolve; });
        return jsonResponse({ tasks: [tasks[1]], total: 2, next_cursor: null });
      }
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
  holdSecondPage = true;
  await user.click(screen.getByRole("button", { name: "载入更多记录" }));
  expect(screen.getByRole("button", { name: "归档已选" })).toBeDisabled();
  releaseSecondPage();
  await screen.findByRole("checkbox", { name: "选择 Second training" });
  await user.click(screen.getByRole("checkbox", { name: "选择 Second training" }));
  expect(screen.getByText("已选 2 项")).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "刷新" }));
  await waitFor(() => expect(screen.getByText("已选 2 项")).toBeInTheDocument());
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

it("keeps selection through browser back and starts empty after a document reload", async () => {
  const task = { id: "task-back", name: "Back navigation", job: "training", state: "idle", archived: false };
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url === "/api/training/history/collections/settings") return jsonResponse({ collection_order: [], config_group_order: {} });
    if (url === "/api/training/history/task-back") return jsonResponse({ task, metrics: [] });
    if (url.startsWith("/api/training/history?")) return jsonResponse({ tasks: [task], total: 1, next_cursor: null });
    throw new Error(`Unexpected request: ${url}`);
  }));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const router = createMemoryRouter([
    { path: "/history", element: <HistoryPage /> },
    { path: "/history/:taskId", element: <HistoryDetailPage /> },
  ], { initialEntries: ["/history?layout=list"] });
  render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);
  const user = userEvent.setup();
  await user.click(await screen.findByRole("checkbox", { name: "选择 Back navigation" }));
  await user.click(screen.getByRole("link", { name: /Back navigation/ }));
  await screen.findByRole("heading", { name: "Back navigation" });
  await router.navigate(-1);
  expect(await screen.findByRole("checkbox", { name: "选择 Back navigation" })).toBeChecked();
  resetHistorySelection();
  await waitFor(() => expect(screen.getByRole("checkbox", { name: "选择 Back navigation" })).not.toBeChecked());
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
