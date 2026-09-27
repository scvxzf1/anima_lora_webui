import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { jsonResponse, renderInApp } from "../../test/renderInApp";
import { ImageTestPage } from "./ImageTestPage";

afterEach(() => vi.unstubAllGlobals());

it("starts inference with a saved merged configuration", async () => {
  const fetcher = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input);
    if (path.includes("/api/config/file-groups")) return jsonResponse([{ id: "local", label: "Local", files: [{ path: "configs/gui-methods/test.toml", label: "Test", method: "lora", methods_subdir: "gui-methods" }] }]);
    if (path === "/api/presets") return jsonResponse(["default"]);
    if (path.includes("/api/config/merged")) return jsonResponse({ model_family: "krea2_raw", pretrained_model_name_or_path: "model.safetensors" });
    if (path === "/api/image-test/status") return jsonResponse({ ok: true, status: "idle", running: false, output_count: 0 });
    if (path === "/api/analysis/weights") return jsonResponse({ weights: [] });
    if (path.startsWith("/api/preview/images")) return jsonResponse({ ok: true, images: [] });
    if (path === "/api/training/gpus") return jsonResponse({ gpus: [] });
    if (path === "/api/image-test/start" && init?.method === "POST") return jsonResponse({ ok: true, status: "running", running: true });
    throw new Error(`Unexpected ${path}`);
  });
  vi.stubGlobal("fetch", fetcher);
  renderInApp(<ImageTestPage />);
  const user = userEvent.setup();
  await screen.findByRole("option", { name: "Test" }, { timeout: 10000 });
  await user.selectOptions(screen.getByLabelText("训练配置"), "configs/gui-methods/test.toml");
  await screen.findByRole("button", { name: "开始生成" });
  await waitFor(() => expect(screen.getByRole("button", { name: "开始生成" })).toBeEnabled());
  await user.type(screen.getByLabelText("正向提示词"), "test prompt");
  await user.click(screen.getByRole("button", { name: "开始生成" }));
  await waitFor(() => expect(fetcher).toHaveBeenCalledWith("/api/image-test/start", expect.objectContaining({ method: "POST" })));
  const call = fetcher.mock.calls.find(([path]) => path === "/api/image-test/start")!;
  const body = JSON.parse(call[1]!.body as string);
  expect(body.config.model_family).toBe("krea2_raw");
  expect(body.sampler).toBe("euler");
  expect(body.prompt).toBe("test prompt");
});
