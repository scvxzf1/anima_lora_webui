import { act, cleanup, render, renderHook, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { StrictMode, useState, type SetStateAction } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useTrainingContextStore } from "../../app/trainingContextStore";
import { useHotstartIntent } from "./hotstartIntent";
import { useTrainingHotstart } from "./useTrainingHotstart";
import type { TrainingDraft } from "./trainingForm";

const inspect = vi.fn();
vi.mock("../training-history/historyHotstartApi", async (importOriginal) => {
  const original = await importOriginal<typeof import("../training-history/historyHotstartApi")>();
  return { ...original, inspectHotstart: (...args: unknown[]) => inspect(...args) };
});

const file = { path: "configs/imported/current.toml", method: "lora", methods_subdir: "imported" };
const target = { configFile: file.path, preset: "default", variant: "lora", subdir: "imported" };

function Harness({ busy = false, unavailable = false, hydrated = true, initial = { output_name: "edited" }, resume = "" }: {
  busy?: boolean;
  unavailable?: boolean;
  hydrated?: boolean;
  initial?: Record<string, string>;
  resume?: string;
}) {
  const [draft, setDraft] = useState<TrainingDraft>(initial);
  const hotstart = useTrainingHotstart({ file, preset: "default", hydrated, unavailable,
    busy, dirty: true, draft, mergedConfig: { resume }, setDraft });
  return <>
    <span data-testid="status">{hotstart.status}</span>
    <span data-testid="draft">{JSON.stringify(draft)}</span>
    <span data-testid="error">{hotstart.error}</span>
    <button onClick={hotstart.apply}>应用</button>
    <button onClick={hotstart.retry}>重试</button>
    <button onClick={hotstart.cancel}>取消</button>
  </>;
}

describe("hotstart intent", () => {
  beforeEach(() => {
    inspect.mockReset();
    useTrainingContextStore.setState({ configFile: file.path, preset: "default" });
    useHotstartIntent.setState({ intent: null });
  });
  afterEach(cleanup);

  it("rechecks the bound config and applies only a draft change once", async () => {
    inspect.mockResolvedValue({ ok: true, compatible: true, abs_path: "/output/weight.safetensors", kind: "LoRA" });
    useHotstartIntent.getState().offer({ ...target, path: "/output/weight.safetensors" });
    const user = userEvent.setup();
    render(<Harness resume="/old/state" />);
    await waitFor(() => expect(screen.getByTestId("status")).toHaveTextContent("ready"));
    expect(inspect).toHaveBeenCalledWith("/output/weight.safetensors", target, expect.any(AbortSignal));
    await user.click(screen.getByText("应用"));
    expect(JSON.parse(screen.getByTestId("draft").textContent || "{}")).toEqual({ output_name: "edited", network_weights: "/output/weight.safetensors", resume: "" });
    expect(useHotstartIntent.getState().intent).toBeNull();
    expect(inspect).toHaveBeenCalledTimes(1);
  });

  it("blocks application after failed inspection and permits retry or cancel", async () => {
    inspect.mockRejectedValueOnce(new Error("不兼容")).mockResolvedValueOnce({ ok: true, compatible: true, abs_path: "/output/weight.safetensors" });
    useHotstartIntent.getState().offer({ ...target, path: "/output/weight.safetensors" });
    const user = userEvent.setup();
    render(<Harness />);
    await waitFor(() => expect(screen.getByTestId("status")).toHaveTextContent("error"));
    await user.click(screen.getByText("应用"));
    expect(screen.getByTestId("draft")).not.toHaveTextContent("network_weights");
    await user.click(screen.getByText("重试"));
    await waitFor(() => expect(screen.getByTestId("status")).toHaveTextContent("ready"));
    await user.click(screen.getByText("取消"));
    expect(useHotstartIntent.getState().intent).toBeNull();
  });

  it("waits while busy and rejects a changed context", async () => {
    inspect.mockResolvedValue({ ok: true, compatible: true, abs_path: "/output/weight.safetensors" });
    useHotstartIntent.getState().offer({ ...target, path: "/output/weight.safetensors" });
    const { rerender } = render(<Harness busy />);
    expect(inspect).not.toHaveBeenCalled();
    rerender(<Harness />);
    await waitFor(() => expect(screen.getByTestId("status")).toHaveTextContent("ready"));
    useTrainingContextStore.getState().selectPreset("other");
    await userEvent.setup().click(screen.getByText("应用"));
    expect(screen.getByTestId("draft")).not.toHaveTextContent("network_weights");
    expect(useHotstartIntent.getState().intent).toBeNull();
  });

  it("ignores an older retry response and rechecks after busy state", async () => {
    let resolveFirst!: (value: unknown) => void;
    inspect.mockImplementationOnce(() => new Promise((resolve) => { resolveFirst = resolve; }))
      .mockResolvedValue({ ok: true, compatible: true, abs_path: "/output/new.safetensors" });
    useHotstartIntent.getState().offer({ ...target, path: "/output/weight.safetensors" });
    const user = userEvent.setup();
    const view = render(<Harness />);
    await waitFor(() => expect(inspect).toHaveBeenCalledTimes(1));
    view.rerender(<Harness busy />);
    view.rerender(<Harness />);
    await waitFor(() => expect(screen.getByTestId("status")).toHaveTextContent("ready"));
    resolveFirst({ ok: true, compatible: true, abs_path: "/output/old.safetensors" });
    await user.click(screen.getByText("应用"));
    expect(screen.getByTestId("draft")).toHaveTextContent("/output/new.safetensors");
    expect(inspect).toHaveBeenCalledTimes(2);
  });

  it("does not let a stale retry response replace the newer inspection", async () => {
    let resolveFirst!: (value: unknown) => void;
    inspect.mockImplementationOnce(() => new Promise((resolve) => { resolveFirst = resolve; }))
      .mockResolvedValue({ ok: true, compatible: true, abs_path: "/output/retried.safetensors" });
    useHotstartIntent.getState().offer({ ...target, path: "/output/weight.safetensors" });
    const user = userEvent.setup();
    render(<Harness />);
    await waitFor(() => expect(inspect).toHaveBeenCalledTimes(1));
    await user.click(screen.getByText("重试"));
    await waitFor(() => expect(screen.getByTestId("status")).toHaveTextContent("ready"));
    resolveFirst({ ok: true, compatible: true, abs_path: "/output/stale.safetensors" });
    await user.click(screen.getByText("应用"));
    expect(screen.getByTestId("draft")).toHaveTextContent("/output/retried.safetensors");
  });

  it("cancels a pending check when the source becomes unavailable or unmounts", async () => {
    inspect.mockImplementation(() => new Promise(() => {}));
    useHotstartIntent.getState().offer({ ...target, path: "/output/weight.safetensors" });
    const view = render(<Harness />);
    await waitFor(() => expect(inspect).toHaveBeenCalledTimes(1));
    const firstSignal = inspect.mock.calls[0][2] as AbortSignal;
    view.rerender(<Harness unavailable />);
    expect(firstSignal.aborted).toBe(true);
    expect(useHotstartIntent.getState().intent).toBeNull();
    view.unmount();
  });

  it("aborts an inspection on unmount", async () => {
    inspect.mockImplementation(() => new Promise(() => {}));
    useHotstartIntent.getState().offer({ ...target, path: "/output/weight.safetensors" });
    const view = render(<Harness />);
    await waitFor(() => expect(inspect).toHaveBeenCalledTimes(1));
    const signal = inspect.mock.calls[0][2] as AbortSignal;
    view.unmount();
    expect(signal.aborted).toBe(true);
  });

  it("rechecks after StrictMode aborts the initial setup", async () => {
    inspect.mockResolvedValue({ ok: true, compatible: true, abs_path: "/output/strict.safetensors" });
    useHotstartIntent.getState().offer({ ...target, path: "/output/strict.safetensors" });
    render(<StrictMode><Harness /></StrictMode>);
    await waitFor(() => expect(screen.getByTestId("status")).toHaveTextContent("ready"));
    expect(inspect).toHaveBeenCalledTimes(2);
    expect((inspect.mock.calls[0][2] as AbortSignal).aborted).toBe(true);
    await userEvent.setup().click(screen.getByText("应用"));
    expect(screen.getByTestId("draft")).toHaveTextContent("/output/strict.safetensors");
  });

  it("captures the inspected path before a delayed draft updater executes", async () => {
    inspect.mockResolvedValue({ ok: true, compatible: true, abs_path: "/output/delayed.safetensors" });
    useHotstartIntent.getState().offer({ ...target, path: "/output/delayed.safetensors" });
    let pending: SetStateAction<TrainingDraft> | undefined;
    const { result } = renderHook(() => useTrainingHotstart({
      file, preset: "default", hydrated: true, unavailable: false, busy: false,
      dirty: false, draft: {}, mergedConfig: {}, setDraft: (updater) => { pending = updater; },
    }));
    await waitFor(() => expect(result.current.status).toBe("ready"));
    act(() => result.current.apply());
    await waitFor(() => expect(result.current.status).toBe("idle"));
    expect(typeof pending).toBe("function");
    const draft = (pending as (draft: TrainingDraft) => TrainingDraft)({ output_name: "kept" });
    expect(draft).toEqual({ output_name: "kept", network_weights: "/output/delayed.safetensors" });
  });
});
