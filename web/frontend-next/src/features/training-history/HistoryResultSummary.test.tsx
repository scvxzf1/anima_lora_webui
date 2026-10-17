import { cleanup, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { renderInApp } from "../../test/renderInApp";
import { HistoryResultSummary } from "./HistoryResultSummary";

describe("history result summary final model label", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("shows four latest sample thumbnails beside the overview summary", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      const payload = url.includes("/api/preview/images")
        ? { images: [1, 2, 3, 4].map((id) => ({ file: `sample-${id}.png`, name: `sample-${id}.png` })), total: 4 }
        : { weights: [], total: 0 };
      return new Response(JSON.stringify(payload), { status: 200, headers: { "Content-Type": "application/json" } });
    }));
    renderInApp(<HistoryResultSummary taskId="task-1" />);
    const thumbnails = await screen.findAllByRole("img", { name: /训练样张：sample-/ });
    expect(thumbnails).toHaveLength(4);
    expect(thumbnails[0]).toHaveAttribute("src", "/api/preview/image?task_id=task-1&file=sample-1.png");
  });

  it("labels the latest weight when backend metadata marks it final", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      const payload = url.includes("/api/preview/weights")
        ? { weights: [{ file: "adapter.safetensors", name: "adapter.safetensors", kind: "final", size_bytes: 1024, scope_label: "本任务" }], total: 1 }
        : { images: [], total: 0 };
      return new Response(JSON.stringify(payload), { status: 200, headers: { "Content-Type": "application/json" } });
    }));

    renderInApp(<HistoryResultSummary taskId="task-1" />);

    expect(await screen.findByText("Final Model")).toBeInTheDocument();
    expect(screen.getByText(/adapter\.safetensors/)).toBeInTheDocument();
  });

  it("does not infer final from names or scan older items when the latest weight is not final", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      const payload = url.includes("/api/preview/weights")
        ? { weights: [
            { file: "adapter-final.safetensors", name: "adapter-final.safetensors", kind: "checkpoint", size_bytes: 1024, scope_label: "本任务" },
            { file: "older.safetensors", name: "older.safetensors", kind: "final", size_bytes: 1024, scope_label: "本任务" },
          ], total: 2 }
        : { images: [], total: 0 };
      return new Response(JSON.stringify(payload), { status: 200, headers: { "Content-Type": "application/json" } });
    }));

    renderInApp(<HistoryResultSummary taskId="task-1" />);

    expect(await screen.findByText(/adapter-final\.safetensors/)).toBeInTheDocument();
    expect(screen.queryByText("Final Model")).not.toBeInTheDocument();
  });
});
