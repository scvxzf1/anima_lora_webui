import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { QueuePage } from "./QueuePage";

afterEach(() => vi.unstubAllGlobals());

it("keeps the last queue snapshot readable after a failed refresh", async () => {
  const snapshot = {
    ok: true,
    revision: "revision-1",
    status: "running",
    paused: true,
    summary: { total: 3, queued: 1, running: 1, done: 1 },
    items: [
      { id: "run-1", state: "running", variant: "lora", preset: "default" },
      { id: "queue-2", state: "queued", variant: "loha", preset: "default" },
      { id: "done-3", state: "done", variant: "lokr", preset: "default" },
    ],
  };
  const fetchMock = vi
    .fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify(snapshot), { status: 200 }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ ok: false, error: "backend offline" }), {
        status: 200,
      }),
    );
  vi.stubGlobal("fetch", fetchMock);
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const view = render(
    <QueryClientProvider client={client}>
      <QueuePage />
    </QueryClientProvider>,
  );

  await screen.findByText("lora · default");
  fireEvent.click(screen.getByRole("button", { name: "刷新" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("backend offline");
  expect(screen.getByText(/保留上次数据/)).toBeInTheDocument();
  expect(screen.getByText("状态待确认")).toBeInTheDocument();
  expect(screen.getByText("lora · default")).toBeInTheDocument();
  expect(screen.getByText("loha · default")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "继续队列" })).toBeDisabled();

  fireEvent.click(screen.getByRole("button", { name: /^完成\s*1$/ }));
  const worklist = screen
    .getByRole("heading", { name: "完成" })
    .closest("section")!;
  expect(within(worklist).getByText("lokr · default")).toBeInTheDocument();
  expect(
    within(worklist).queryByText("lora · default"),
  ).not.toBeInTheDocument();
  await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
  view.unmount();
  client.clear();
});
