import { afterEach, describe, expect, it, vi } from "vitest";
import { deletePreviewImages, fetchPreviewImages, fetchPreviewTasks, fetchPreviewWeights, makePreviewGroups, savePreviewSettings } from "./api";

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

  it("uses stable history cursors and asset offsets for later pages", async () => {
    const fetchMock = vi.fn().mockImplementation(async () => new Response(JSON.stringify({ tasks: [], images: [], weights: [] })));
    vi.stubGlobal("fetch", fetchMock);
    await fetchPreviewTasks("cursor/next");
    await fetchPreviewImages("training", "task", "task-1", undefined, "all", 200);
    await fetchPreviewWeights("task", "task-1", undefined, 100);
    const urls = fetchMock.mock.calls.map(([url]) => new URL(url, "http://localhost"));
    expect(urls[0].searchParams.get("cursor")).toBe("cursor/next");
    expect(urls[0].searchParams.get("include_archived")).toBe("1");
    expect(urls[1].searchParams.get("offset")).toBe("200");
    expect(urls[2].searchParams.get("offset")).toBe("100");
    expect(urls[2].searchParams.get("sort")).toBe("recent");
  });

  it("includes cursors for later config-group image pages", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ images: [] })));
    vi.stubGlobal("fetch", fetchMock);
    await fetchPreviewImages("training", "group", "", {
      key: "group", historyGroupKey: "", methodsSubdir: "gui-methods", variant: "lora",
      preset: "default", label: "group", tasks: [],
    }, "all", "cursor/next");
    const url = new URL(fetchMock.mock.calls[0][0], "http://localhost");
    expect(url.searchParams.get("mode")).toBe("config_group");
    expect(url.searchParams.get("cursor")).toBe("cursor/next");
    expect(url.searchParams.has("offset")).toBe(false);
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

  it("sends the loaded settings revision with a save", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ ok: true, revision: "new" })));
    vi.stubGlobal("fetch", fetchMock);
    await savePreviewSettings({ training_dir: "output/ckpt", inference_dir: "output/tests", custom_dir: "", revision: "loaded" });
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toMatchObject({ revision: "loaded" });
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
