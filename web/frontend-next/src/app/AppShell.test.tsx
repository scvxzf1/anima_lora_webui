import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { createMemoryRouter, Link, RouterProvider } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppShell } from "./AppShell";

const { fetchSettings } = vi.hoisted(() => ({ fetchSettings: vi.fn() }));

vi.mock("../features/settings/api", () => ({
  fetchGlobalSettings: fetchSettings,
  settingsKeys: { global: ["settings", "global"] },
}));

vi.mock("./Topbar", () => ({
  Topbar: () => (
    <nav>
      <Link to="/datasets">数据集</Link>
    </nav>
  ),
}));

function renderShell() {
  const router = createMemoryRouter(
    [
      {
        element: <AppShell />,
        children: [
          { path: "/training", element: <h1>训练</h1> },
          { path: "/datasets", element: <h1>数据集工作区</h1> },
        ],
      },
    ],
    { initialEntries: ["/training"] },
  );
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  );
}

describe("AppShell UI scale", () => {
  afterEach(() => vi.unstubAllGlobals());

  beforeEach(() => {
    fetchSettings.mockResolvedValue({
      ui_scale: 150,
      ui_scale_config: 125,
      ui_scale_datasets: 80,
    });
    vi.stubGlobal("matchMedia", () => ({
      matches: false,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    }));
  });

  it("applies global and route scales to rendered elements after navigation", async () => {
    renderShell();

    const layout = document.querySelector<HTMLElement>(".next-layout")!;
    const content = document.getElementById("workspace")!;
    await waitFor(() => expect(layout.style.zoom).toBe("1.5"));
    expect(content.style.zoom).toBe("0.8333333333333334");
    expect(layout.style.height).toBe("calc(66.6667dvh)");

    fireEvent.click(screen.getByRole("link", { name: "数据集" }));
    expect(
      await screen.findByRole("heading", { name: "数据集工作区" }),
    ).toBeInTheDocument();
    await waitFor(() => expect(content.style.zoom).toBe("0.5333333333333333"));
    expect(layout.style.zoom).toBe("1.5");
  });
});
