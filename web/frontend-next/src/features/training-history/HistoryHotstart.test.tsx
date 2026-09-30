import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter, RouterProvider } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useTrainingContextStore } from "../../app/trainingContextStore";
import { useHotstartIntent } from "../training-config/hotstartIntent";
import { HistoryHotstart } from "./HistoryHotstart";

const weight = { file: "weight.safetensors", name: "weight.safetensors", abs_path: "/output/weight.safetensors", size_bytes: 42, scope_label: "本任务" };
const file = { path: "configs/imported/current.toml", method: "lora", methods_subdir: "imported", trainable: true };

function mount() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  const router = createMemoryRouter([
    { path: "/", element: <HistoryHotstart weight={weight} /> },
    { path: "/training", element: <p>训练草稿</p> },
  ]);
  return { client, ...render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>) };
}

describe("history weight hotstart", () => {
  beforeEach(() => {
    useTrainingContextStore.setState({ configFile: file.path, preset: "default" });
    useHotstartIntent.setState({ intent: null });
  });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it("inspects against the selected current config and preset before navigating", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).includes("file-groups")) return Response.json([{ id: "imported", files: [file] }]);
      if (String(input).includes("presets")) return Response.json(["default"]);
      expect(init?.method).toBe("POST");
      expect(JSON.parse(String(init?.body))).toEqual({ path: weight.abs_path, variant: "lora", preset: "default", methods_subdir: "imported", config_file: file.path });
      return Response.json({ ok: true, compatible: true, abs_path: weight.abs_path, kind: "LoRA" });
    });
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    mount();
    await user.click(await screen.findByRole("button", { name: "应用到训练草稿" }));
    expect(screen.getByText(/当前可训练配置：configs\/imported\/current.toml/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "检查权重并前往草稿" }));
    expect(await screen.findByText("训练草稿")).toBeInTheDocument();
    expect(useHotstartIntent.getState().intent).toMatchObject({ path: weight.abs_path, configFile: file.path, preset: "default", variant: "lora", subdir: "imported" });
  });

  it("keeps the dialog open and creates no intent for an incompatible weight", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      if (String(input).includes("file-groups")) return Response.json([{ id: "imported", files: [file] }]);
      if (String(input).includes("presets")) return Response.json(["default"]);
      return Response.json({ ok: true, compatible: false, abs_path: weight.abs_path, message: "类型不匹配" });
    }));
    const user = userEvent.setup();
    mount();
    await user.click(await screen.findByRole("button", { name: "应用到训练草稿" }));
    await user.click(screen.getByRole("button", { name: "检查权重并前往草稿" }));
    await waitFor(() => expect(screen.getByText(/检查失败：类型不匹配/)).toBeInTheDocument());
    expect(useHotstartIntent.getState().intent).toBeNull();
  });

  it.each(["deleted", "locked", "readonly", "preset removed"])("rejects a pending inspection when %s", async (change) => {
    let resolveInspect!: (response: Response) => void;
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      if (String(input).includes("file-groups")) return Response.json([{ id: "imported", files: [file] }]);
      if (String(input).includes("presets")) return Response.json(["default"]);
      return new Promise<Response>((resolve) => { resolveInspect = resolve; });
    });
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    const { client } = mount();
    await user.click(await screen.findByRole("button", { name: "应用到训练草稿" }));
    await user.click(screen.getByRole("button", { name: "检查权重并前往草稿" }));
    await waitFor(() => expect(resolveInspect).toBeTypeOf("function"));
    if (change === "preset removed") client.setQueryData(["training-context", "presets"], []);
    else client.setQueryData(["training-context", "files"], [{ id: "imported", files: change === "deleted" ? [] : [{ ...file, [change]: true }] }]);
    resolveInspect(Response.json({ ok: true, compatible: true, abs_path: weight.abs_path }));
    await waitFor(() => expect(useHotstartIntent.getState().intent).toBeNull());
    expect(screen.queryByText("训练草稿")).not.toBeInTheDocument();
  });

  it("rejects a malformed success response without navigating", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      if (String(input).includes("file-groups")) return Response.json([{ id: "imported", files: [file] }]);
      if (String(input).includes("presets")) return Response.json(["default"]);
      return Response.json({ ok: true, compatible: true, abs_path: "relative/weight.safetensors" });
    }));
    const user = userEvent.setup();
    mount();
    await user.click(await screen.findByRole("button", { name: "应用到训练草稿" }));
    await user.click(screen.getByRole("button", { name: "检查权重并前往草稿" }));
    expect(await screen.findByText(/权重检查响应无效/)).toBeInTheDocument();
    expect(useHotstartIntent.getState().intent).toBeNull();
  });

  it("rejects a pending result after the selected preset changes", async () => {
    let resolveInspect!: (response: Response) => void;
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      if (String(input).includes("file-groups")) return Response.json([{ id: "imported", files: [file] }]);
      if (String(input).includes("presets")) return Response.json(["default", "low_vram"]);
      return new Promise<Response>((resolve) => { resolveInspect = resolve; });
    }));
    const user = userEvent.setup();
    mount();
    await user.click(await screen.findByRole("button", { name: "应用到训练草稿" }));
    await user.click(screen.getByRole("button", { name: "检查权重并前往草稿" }));
    await waitFor(() => expect(resolveInspect).toBeTypeOf("function"));
    useTrainingContextStore.getState().selectPreset("low_vram");
    resolveInspect(Response.json({ ok: true, compatible: true, abs_path: weight.abs_path }));
    await waitFor(() => expect(useHotstartIntent.getState().intent).toBeNull());
    expect(screen.queryByText("训练草稿")).not.toBeInTheDocument();
  });
});
