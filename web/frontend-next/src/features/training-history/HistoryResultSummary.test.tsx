import { cleanup, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { renderInApp } from "../../test/renderInApp";
import { HistoryResultSummary } from "./HistoryResultSummary";

describe("history result summary final model label", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("labels a final model only when its returned filename says final or last", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      const payload = url.includes("/api/preview/weights")
        ? { weights: [{ file: "adapter-final.safetensors", name: "adapter-final.safetensors", size_bytes: 1024, scope_label: "本任务" }], total: 1 }
        : { images: [], total: 0 };
      return new Response(JSON.stringify(payload), { status: 200, headers: { "Content-Type": "application/json" } });
    }));

    renderInApp(<HistoryResultSummary taskId="task-1" />);

    expect(await screen.findByText("Final Model")).toBeInTheDocument();
    expect(screen.getByText(/adapter-final\.safetensors/)).toBeInTheDocument();
  });

  it("does not infer a final model from a checkpoint filename", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      const payload = url.includes("/api/preview/weights")
        ? { weights: [{ file: "step-2500.safetensors", name: "step-2500.safetensors", size_bytes: 1024, scope_label: "本任务" }], total: 1 }
        : { images: [], total: 0 };
      return new Response(JSON.stringify(payload), { status: 200, headers: { "Content-Type": "application/json" } });
    }));

    renderInApp(<HistoryResultSummary taskId="task-1" />);

    expect(await screen.findByText(/step-2500\.safetensors/)).toBeInTheDocument();
    expect(screen.queryByText("Final Model")).not.toBeInTheDocument();
  });
});
