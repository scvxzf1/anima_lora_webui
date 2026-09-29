import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { createMemoryRouter, RouterProvider } from "react-router-dom";
import { jsonResponse } from "../../test/renderInApp";
import { HistoryPage } from "./HistoryPage";
import { HistoryDetailPage } from "./HistoryDetailPage";

afterEach(() => {
  cleanup();
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

  await userEvent.setup().click(await screen.findByRole("link", { name: /Needle training/ }));
  expect(await screen.findByRole("heading", { name: "Needle training" })).toBeInTheDocument();
  await userEvent.setup().click(screen.getByRole("link", { name: "返回历史" }));

  await waitFor(() => expect(router.state.location.pathname).toBe("/history"));
  expect(router.state.location.search).toContain("q=needle");
  expect(router.state.location.search).toContain("state=error");
  expect(router.state.location.search).toContain("anchor=task-1");
  await waitFor(() => expect(scrollIntoView).toHaveBeenCalled());
  expect(screen.getByRole("link", { name: /Needle training/ })).toBeInTheDocument();
  expect(screen.getByRole("combobox", { name: "状态" })).toHaveValue("error");
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
