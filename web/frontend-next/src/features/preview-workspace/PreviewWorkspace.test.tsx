import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { createMemoryRouter, RouterProvider } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { PreviewWorkspace } from "./PreviewWorkspace";

vi.mock("../../app/useTrainingContext", () => ({
  useTrainingContext: () => ({ selectedFile: { path: "configs/imported/lora.toml", method: "lora", methods_subdir: "gui-methods" }, selectedPreset: "default" }),
}));
vi.mock("./api", async (load) => {
  const actual = await load<typeof import("./api")>();
  return {
    ...actual,
    fetchPreviewSettings: vi.fn().mockResolvedValue({ training_dir: "sample", inference_dir: "tests", custom_dir: "" }),
    fetchPreviewTasks: vi.fn().mockResolvedValue({ tasks: [{ id: "run1", job: "training", name: "Run 1", methods_subdir: "gui-methods", variant: "lora" }] }),
    fetchPreviewImages: vi.fn().mockResolvedValue({ images: [], count: 0, message: "暂无图片" }),
    fetchPreviewWeights: vi.fn().mockResolvedValue({ weights: [] }),
  };
});
afterEach(() => vi.restoreAllMocks());

describe("PreviewWorkspace page", () => {
  it("renders the preview workflow and task-scoped filters", async () => {
    const router = createMemoryRouter([{ path: "/preview", element: <PreviewWorkspace /> }], { initialEntries: ["/preview"] });
    render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><RouterProvider router={router} /></QueryClientProvider>);
    expect(await screen.findByRole("heading", { name: "预览工作区" })).toBeInTheDocument();
    expect(screen.getByRole("group", { name: "预览来源" })).toBeInTheDocument();
    expect(screen.getByLabelText("训练范围")).toHaveValue("latest");
    fireEvent.change(screen.getByLabelText("训练范围"), { target: { value: "task" } });
    await waitFor(() => expect(screen.getByRole("option", { name: "Run 1" })).toBeInTheDocument());
    expect(screen.getByRole("heading", { name: "保存的权重" })).toBeInTheDocument();
  });
});
