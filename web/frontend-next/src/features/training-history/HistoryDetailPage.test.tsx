import { cleanup, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createMemoryRouter, RouterProvider } from "react-router-dom";
import { render } from "@testing-library/react";
import { jsonResponse } from "../../test/renderInApp";
import { HistoryDetailPage } from "./HistoryDetailPage";

describe("history detail navigation", () => {
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it.each(["", "?view=unknown"])("falls back to overview for search %s", async (search) => {
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({
      task: { id: "task-1", job: "training", state: "done", name: "History task" }, metrics: [],
    })));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const router = createMemoryRouter([
      { path: "/history/:taskId", element: <HistoryDetailPage /> },
    ], { initialEntries: [`/history/task-1${search}`] });
    render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);

    expect(await screen.findByRole("link", { name: /^概览$/ })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("main")).toHaveAttribute("data-view", "overview");
  });

  it("encodes a special-character task ID in the detail request", async () => {
    const taskId = "task #&?";
    const fetchMock = vi.fn(async () => jsonResponse({
      task: { id: taskId, job: "training", state: "done", name: "History task" }, metrics: [],
    }));
    vi.stubGlobal("fetch", fetchMock);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const router = createMemoryRouter([
      { path: "/history/:taskId", element: <HistoryDetailPage /> },
    ], { initialEntries: [`/history/${encodeURIComponent(taskId)}?view=overview`] });
    render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);

    await screen.findByRole("heading", { name: "History task" });
    expect(fetchMock).toHaveBeenCalledWith(
      `/api/training/history/${encodeURIComponent(taskId)}`,
      expect.anything(),
    );
  });

  it("loads the log viewer only after navigating to the logs tab", async () => {
    vi.stubGlobal("ResizeObserver", class {
      observe() {}
      disconnect() {}
    });
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/logs")) return jsonResponse({ total: 0, lines: [] });
      return jsonResponse({ task: { id: "task-1", job: "training", state: "done", name: "History task" }, metrics: [] });
    }));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const router = createMemoryRouter([
      { path: "/history/:taskId", element: <HistoryDetailPage /> },
    ], { initialEntries: ["/history/task-1?view=overview"] });
    render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);

    await screen.findByRole("heading", { name: "History task" });
    expect(screen.queryByRole("log")).not.toBeInTheDocument();
    await userEvent.setup().click(screen.getByRole("link", { name: /^日志$/ }));
    expect(await screen.findByRole("log")).toBeInTheDocument();
  });

  it("switches tabs through the URL and opens the resume shortcut", async () => {
    const detailReads: string[] = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/artifacts")) return jsonResponse({ artifacts: [] });
      if (url.includes("/api/preview/images")) return jsonResponse({ images: [], total: 0 });
      if (url.includes("/api/preview/weights")) return jsonResponse({ weights: [], total: 0 });
      if (url.includes("resume-options")) {
        return jsonResponse({ checkpoints: [], default_checkpoint: "", message: "无检查点" });
      }
      detailReads.push(url);
      return jsonResponse({ task: { id: "task-1", job: "training", state: "done", name: "History task" }, metrics: [] });
    }));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const router = createMemoryRouter([
      { path: "/history/:taskId", element: <HistoryDetailPage /> },
    ], { initialEntries: ["/history/task-1?view=metrics"] });
    render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);

    const metrics = await screen.findByRole("link", { name: /^指标$/ });
    expect(metrics).toHaveAttribute("aria-current", "page");
    await userEvent.setup().click(screen.getByRole("link", { name: /^产物$/ }));
    expect(router.state.location.search).toBe("?view=artifacts");
    expect(screen.getByRole("link", { name: /^产物$/ })).toHaveAttribute("aria-current", "page");
    expect(detailReads).toHaveLength(1);

    await userEvent.setup().click(screen.getByRole("button", { name: /^检查点续训$/ }));
    expect(await screen.findByRole("dialog", { name: "从历史检查点续训" })).toBeInTheDocument();
    expect(await screen.findByText("无检查点")).toBeInTheDocument();
  });
});
