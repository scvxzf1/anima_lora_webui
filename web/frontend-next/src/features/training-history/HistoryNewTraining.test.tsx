import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createMemoryRouter, RouterProvider } from "react-router-dom";
import { afterEach, expect, it, vi } from "vitest";
import { HistoryNewTraining } from "./HistoryNewTraining";
import { useTrainingContextStore } from "../../app/trainingContextStore";

const content = 'model_family = "qwen_image_2_1"\nnetwork_dim = 16';
afterEach(() => { cleanup(); vi.unstubAllGlobals(); useTrainingContextStore.setState({ configFile: "", preset: "default" }); });

it("creates a separate named config through save-as, never starting or resuming training", async () => {
  const user = userEvent.setup();
  const writes: Array<{ url: string; method: string; body: unknown }> = [];
  vi.stubGlobal("fetch", vi.fn(async (url: string, options: RequestInit) => {
    writes.push({ url, method: options.method!, body: JSON.parse(String(options.body)) });
    return Response.json({ ok: true, file: "configs/imported/history-copy.toml" });
  }));
  const router = createMemoryRouter([
    { path: "/", element: <HistoryNewTraining detail={{ config_toml: content }} disabled={false} /> },
    { path: "/training", element: <p>新训练配置</p> },
  ]);
  render(<QueryClientProvider client={new QueryClient()}><RouterProvider router={router} /></QueryClientProvider>);
  await user.click(screen.getByRole("button", { name: "以此配置新建训练" }));
  expect(await screen.findByRole("dialog", { name: "新建训练配置" })).toBeInTheDocument();
  expect(screen.getByRole("textbox", { name: /^TOML$/ })).toHaveValue(content);
  expect(writes).toEqual([]);
  await user.type(screen.getByRole("textbox", { name: "配置名称" }), "history-copy");
  await user.click(screen.getByRole("button", { name: "创建配置" }));
  expect(await screen.findByText("新训练配置")).toBeInTheDocument();
  expect(writes).toEqual([{ url: "/api/config/raw/save-as", method: "POST", body: { file: "configs/imported/history-copy.toml", content } }]);
  expect(useTrainingContextStore.getState().configFile).toBe("configs/imported/history-copy.toml");
});

it("disables creation if the historical snapshot is missing", () => {
  const router = createMemoryRouter([{ path: "/", element: <HistoryNewTraining detail={{}} disabled={false} /> }]);
  render(<QueryClientProvider client={new QueryClient()}><RouterProvider router={router} /></QueryClientProvider>);
  expect(screen.getByRole("button", { name: "以此配置新建训练" })).toBeDisabled();
});
