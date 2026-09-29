import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { HistoryLogs } from "./HistoryLogs";

afterEach(() => vi.unstubAllGlobals());

it("aborts an in-flight global search when the log viewer unmounts", async () => {
  let searchSignal: AbortSignal | undefined;
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      disconnect() {}
    },
  );
  vi.stubGlobal(
    "fetch",
    vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).includes("/logs/search")) {
        searchSignal = init?.signal ?? undefined;
        return new Promise<Response>(() => {});
      }
      return Promise.resolve(
        new Response(
          JSON.stringify({
            total: 1,
            offset: 0,
            logs: [{ line: "needle row" }],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      );
    }),
  );
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const view = render(
    <QueryClientProvider client={client}>
      <HistoryLogs taskId="fixture-run" />
    </QueryClientProvider>,
  );

  await screen.findByText("needle row");
  fireEvent.change(screen.getByLabelText("搜索全部日志"), {
    target: { value: "needle" },
  });
  fireEvent.click(screen.getByLabelText("执行全局搜索"));
  await waitFor(() => expect(searchSignal).toBeDefined());

  view.unmount();
  expect(searchSignal?.aborted).toBe(true);
  client.clear();
});
