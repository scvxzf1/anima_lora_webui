import { afterEach, describe, expect, it, vi } from "vitest";
import { deletePreviewImages, fetchPreviewImages, makePreviewGroups } from "./api";

afterEach(() => vi.unstubAllGlobals());

describe("preview workspace API", () => {
  it("constructs a task-scoped image request with day filter", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ images: [] })));
    vi.stubGlobal("fetch", fetchMock);
    await fetchPreviewImages("training", "task", "task/a", undefined, "14");
    const url = new URL(fetchMock.mock.calls[0][0], "http://localhost");
    expect(url.pathname).toBe("/api/preview/images");
    expect(url.searchParams.get("task_id")).toBe("task/a");
    expect(url.searchParams.get("days")).toBe("14");
  });

  it("passes deletion scope and filenames to the bounded server endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ ok: true })));
    vi.stubGlobal("fetch", fetchMock);
    await deletePreviewImages("training", ["sample.png"], "task-1");
    expect(fetchMock.mock.calls[0][0]).toBe("/api/preview/images?task_id=task-1");
    expect(fetchMock.mock.calls[0][1]).toMatchObject({
      method: "DELETE", body: JSON.stringify({ source: "training", files: ["sample.png"] }),
    });
  });

  it("groups only training tasks and preserves grouped task references", () => {
    const groups = makePreviewGroups([
      { id: "1", job: "training", methods_subdir: "gui-methods", variant: "lora", preset: "default" },
      { id: "2", job: "training", methods_subdir: "gui-methods", variant: "lora", preset: "default" },
      { id: "3", job: "inference", methods_subdir: "gui-methods", variant: "lora" },
    ]);
    expect(groups).toHaveLength(1);
    expect(groups[0].tasks.map((task) => task.id)).toEqual(["1", "2"]);
  });
});
