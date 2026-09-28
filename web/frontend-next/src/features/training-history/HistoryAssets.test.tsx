import { act, cleanup, screen, waitFor, within } from "@testing-library/react";
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

  it("navigates the open sample set with controls and arrow keys, respecting both ends", async () => {
    const user = userEvent.setup();
    const samples = ["sample-1.png", "sample-2.png", "sample-3.png"].map((file) => ({
      file,
      name: file,
      sample: { step: Number(file.match(/\d/)?.[0]) },
    }));
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      const payload = url.includes("/artifacts")
        ? { artifacts: [] }
        : url.includes("/api/preview/weights")
        ? { weights: [] }
        : { images: samples, total: samples.length };
      return new Response(JSON.stringify(payload), { status: 200, headers: { "Content-Type": "application/json" } });
    }));
    renderInApp(<HistoryAssets taskId="task-1" />);

    await user.click(await screen.findByRole("button", { name: "查看 Step 1 sample-1.png 的生成参数" }));
    const dialog = within(screen.getByRole("dialog"));
    expect(screen.getByRole("button", { name: "上一张样张" })).toBeDisabled();
    expect(dialog.getByRole("img", { name: "sample-1.png" })).toBeInTheDocument();
    expect(screen.getByText("1 / 3")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "下一张样张" }));
    expect(dialog.getByRole("img", { name: "sample-2.png" })).toBeInTheDocument();
    await user.keyboard("{ArrowRight}");
    expect(dialog.getByRole("img", { name: "sample-3.png" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "下一张样张" })).toBeDisabled();
    await user.keyboard("{ArrowRight}");
    expect(dialog.getByRole("img", { name: "sample-3.png" })).toBeInTheDocument();
    await user.keyboard("{ArrowLeft}");
    expect(dialog.getByRole("img", { name: "sample-2.png" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "关闭" }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("shows a recorded step on the thumbnail and uses the filename when the step is unknown", async () => {
    const samples = [
      { file: "sample-step.png", name: "sample-step.png", sample: { step: 2500 } },
      { file: "sample-unknown.png", name: "sample-unknown.png" },
    ];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      const payload = url.includes("/artifacts") ? { artifacts: [] }
        : url.includes("/api/preview/weights") ? { weights: [] }
        : { images: samples, total: samples.length };
      return new Response(JSON.stringify(payload), { status: 200, headers: { "Content-Type": "application/json" } });
    }));
    renderInApp(<HistoryAssets taskId="task-1" />);

    expect(await screen.findByText("Step 2500", { selector: ".history-image-grid span" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "查看 Step 2500 sample-step.png 的生成参数" })).toBeInTheDocument();
    expect(screen.getByText("sample-unknown.png", { selector: ".history-image-grid span" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "查看 sample-unknown.png sample-unknown.png 的生成参数" })).toBeInTheDocument();
  });

  it("distinguishes same-step variants by filename and tags each final weight", async () => {
    const samples = ["variant-a.png", "variant-b.png"].map((name) => ({ file: name, name, sample: { step: 500 } }));
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      const payload = url.includes("/artifacts") ? { artifacts: [] }
        : url.includes("/api/preview/weights") ? { weights: [
          { file: "final.safetensors", name: "final.safetensors", kind: "final", size_bytes: 1024, scope_label: "本任务" },
          { file: "checkpoint.safetensors", name: "checkpoint.safetensors", kind: "checkpoint", size_bytes: 1024, scope_label: "本任务" },
        ] }
        : { images: samples, total: samples.length };
      return new Response(JSON.stringify(payload), { status: 200, headers: { "Content-Type": "application/json" } });
    }));
    renderInApp(<HistoryAssets taskId="task-1" />);

    expect(await screen.findByRole("button", { name: "查看 Step 500 variant-a.png 的生成参数" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "查看 Step 500 variant-b.png 的生成参数" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "final.safetensors" }).parentElement?.parentElement).toHaveTextContent("Final Model");
    expect(screen.getByRole("link", { name: "checkpoint.safetensors" }).parentElement?.parentElement).not.toHaveTextContent("Final Model");
  });

  it("keeps the selected file stable when refreshed images change order", async () => {
    const user = userEvent.setup();
    const samples = ["sample-1.png", "sample-2.png", "sample-3.png"].map((file) => ({ file, name: file }));
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      const payload = url.includes("/artifacts") ? { artifacts: [] }
        : url.includes("/api/preview/weights") ? { weights: [] }
        : { images: samples, total: samples.length };
      return new Response(JSON.stringify(payload), { status: 200, headers: { "Content-Type": "application/json" } });
    }));
    const { client } = renderInApp(<HistoryAssets taskId="task-1" />);

    await user.click(await screen.findByText("sample-2.png", { selector: ".history-image-grid span" }));
    expect(within(screen.getByRole("dialog")).getByRole("img", { name: "sample-2.png" })).toBeInTheDocument();
    act(() => client.setQueryData(["history-images", "task-1", 0], {
      images: [{ file: "new.png", name: "new.png" }, ...samples], total: 4,
    }));
    expect(within(screen.getByRole("dialog")).getByRole("img", { name: "sample-2.png" })).toBeInTheDocument();
    expect(await screen.findByText("3 / 4")).toBeInTheDocument();

    act(() => client.setQueryData(["history-images", "task-1", 0], {
      images: samples.filter((image) => image.file !== "sample-2.png"), total: 2,
    }));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  });
});
