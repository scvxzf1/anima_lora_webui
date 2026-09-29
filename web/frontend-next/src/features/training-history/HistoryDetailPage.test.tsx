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

    const metrics = await screen.findByRole("link", { name: "指标", exact: true });
    expect(metrics).toHaveAttribute("aria-current", "page");
    await userEvent.setup().click(screen.getByRole("link", { name: "产物", exact: true }));
    expect(router.state.location.search).toBe("?view=artifacts");
    expect(screen.getByRole("link", { name: "产物", exact: true })).toHaveAttribute("aria-current", "page");
    expect(detailReads).toHaveLength(1);

    await userEvent.setup().click(screen.getByRole("button", { name: "检查点续训", exact: true }));
    expect(await screen.findByRole("dialog", { name: "从历史检查点续训" })).toBeInTheDocument();
    expect(await screen.findByText("无检查点")).toBeInTheDocument();
  });
});
