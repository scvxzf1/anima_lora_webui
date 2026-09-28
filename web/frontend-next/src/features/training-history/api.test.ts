import { afterEach, describe, expect, it, vi } from "vitest";

import {
  batchUpdateHistoryTasks,
  fetchHistoryTaskDetail,
  fetchHistoryTasks,
  fetchHistoryTimeline,
} from "./api";

describe("training history API", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("reads history tasks including archived records", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify({ ok: true, tasks: [] }), { status: 200 }),
      );
    vi.stubGlobal("fetch", fetchMock);

    await fetchHistoryTasks(200);

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/training/history?limit=200&include_archived=1",
      expect.objectContaining({ signal: undefined }),
    );
  });

  it("encodes task ids when reading history details", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify({ ok: true, task: {} }), { status: 200 }),
      );
    vi.stubGlobal("fetch", fetchMock);

    await fetchHistoryTaskDetail("abc/123");

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/training/history/abc%2F123",
      expect.objectContaining({ signal: undefined }),
    );
  });

  it("sends selected timeline task ids in order and includes archived records", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ ok: true }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    await fetchHistoryTimeline(["task-b", "task/a"]);
    const url = new URL(fetchMock.mock.calls[0][0], "http://localhost");
    expect(url.pathname).toBe("/api/training/history/config-group/timeline");
    expect(url.searchParams.getAll("task_id")).toEqual(["task-b", "task/a"]);
    expect(url.searchParams.get("include_archived")).toBe("1");
  });

  it("passes an opaque cursor and search without interpreting the cursor", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ tasks: [], next_cursor: null, total: 0 })));
    vi.stubGlobal("fetch", fetchMock);
    await fetchHistoryTasks(200, undefined, "portrait group", "v1.cursor+/");
    const url = new URL(fetchMock.mock.calls[0][0], "http://localhost");
    expect(url.searchParams.get("q")).toBe("portrait group");
    expect(url.searchParams.get("cursor")).toBe("v1.cursor+/");
    expect(url.searchParams.get("limit")).toBe("200");
  });

  it("batches archive and delete actions with the required payload", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify({ ok: true, message: "ok" }), {
          status: 200,
        }),
      );
    vi.stubGlobal("fetch", fetchMock);

    await batchUpdateHistoryTasks({ action: "archive", task_ids: ["a", "b"] });

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/training/history/batch",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ action: "archive", task_ids: ["a", "b"] }),
      }),
    );
  });
});
