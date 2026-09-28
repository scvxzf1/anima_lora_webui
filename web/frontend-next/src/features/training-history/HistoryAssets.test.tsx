import { cleanup, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { renderInApp } from "../../test/renderInApp";
import { HistoryAssets } from "./HistoryAssets";

describe("history asset path copying", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("copies the local weight path while keeping its download action", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText } });
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      const payload = url.includes("/artifacts")
        ? { artifacts: [] }
        : url.includes("/api/preview/weights")
        ? { weights: [{ file: "step-10.safetensors", abs_path: "/output/task/step-10.safetensors", name: "step-10.safetensors", size_bytes: 1024, scope_label: "本任务" }] }
        : { images: [] };
      return new Response(JSON.stringify(payload), { status: 200, headers: { "Content-Type": "application/json" } });
    }));
    renderInApp(<HistoryAssets taskId="task-1" />);

    const copy = await screen.findByRole("button", { name: "复制 step-10.safetensors 的本地路径" });
    await user.click(copy);
    expect(writeText).toHaveBeenCalledWith("/output/task/step-10.safetensors");
    expect(screen.getByRole("status")).toHaveTextContent("step-10.safetensors的路径已复制");
    expect(screen.getByRole("link", { name: "step-10.safetensors" })).toHaveAttribute("download");
    expect(screen.getByText("/output/task/step-10.safetensors")).toBeInTheDocument();
  });

  it("shows an accessible failure message when clipboard access is denied", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn().mockRejectedValue(new Error("permission denied"));
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText } });
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      const payload = url.includes("/artifacts")
        ? { artifacts: [] }
        : url.includes("/api/preview/weights")
        ? { weights: [{ file: "step-10.safetensors", name: "step-10.safetensors", abs_path: "/output/task/step-10.safetensors", size_bytes: 1024, scope_label: "本任务" }] }
        : { images: [] };
      return new Response(JSON.stringify(payload), { status: 200, headers: { "Content-Type": "application/json" } });
    }));
    renderInApp(<HistoryAssets taskId="task-1" />);

    await user.click(await screen.findByRole("button", { name: "复制 step-10.safetensors 的本地路径" }));
    expect(writeText).toHaveBeenCalledWith("/output/task/step-10.safetensors");
    expect(screen.getByRole("status")).toHaveTextContent("无法复制step-10.safetensors的路径");
  });

  it("does not offer copying a weight file without the backend absolute path", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      const payload = url.includes("/artifacts")
        ? { artifacts: [] }
        : url.includes("/api/preview/weights")
        ? { weights: [{ file: "step-10.safetensors", name: "step-10.safetensors", size_bytes: 1024, scope_label: "本任务" }] }
        : { images: [] };
      return new Response(JSON.stringify(payload), { status: 200, headers: { "Content-Type": "application/json" } });
    }));
    renderInApp(<HistoryAssets taskId="task-1" />);

    expect(await screen.findByRole("button", { name: "复制 step-10.safetensors 的本地路径" })).toBeDisabled();
    expect(screen.getByRole("link", { name: "step-10.safetensors" })).toBeInTheDocument();
  });
});
