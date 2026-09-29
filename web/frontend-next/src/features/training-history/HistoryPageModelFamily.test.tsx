import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { createMemoryRouter, RouterProvider } from "react-router-dom";
import { expect, it, vi } from "vitest";
import { jsonResponse } from "../../test/renderInApp";
import { HistoryPage } from "./HistoryPage";

it("filters history by model family in the advanced filter", async () => {
  const tasks = [
    { id: "anima", name: "Anima run", job: "training", state: "idle", archived: false, model_family: "anima" },
    { id: "krea2", name: "Krea-2 run", job: "training", state: "idle", archived: false, model_family: "krea2_raw" },
    { id: "z-image", name: "Z-Image run", job: "training", state: "idle", archived: false, model_family: "z_image" },
  ];
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url === "/api/training/history/collections/settings") {
      return jsonResponse({ collection_order: [], config_group_order: {} });
    }
    if (url.startsWith("/api/training/history?")) {
      return jsonResponse({ tasks, total: tasks.length, next_cursor: null });
    }
    throw new Error(`Unexpected request: ${url}`);
  }));

  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const router = createMemoryRouter(
    [{ path: "/history", element: <HistoryPage /> }],
    { initialEntries: ["/history?layout=list&base=krea2_raw"] },
  );
  render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);

  expect(await screen.findByRole("link", { name: /Krea-2 run/ })).toBeInTheDocument();
  expect(screen.queryByRole("link", { name: /Anima run/ })).not.toBeInTheDocument();
  expect(screen.queryByRole("link", { name: /Z-Image run/ })).not.toBeInTheDocument();
  expect(screen.getByRole("combobox", { name: "基座模型" })).toHaveValue("krea2_raw");
});
