import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  act,
  cleanup,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter, RouterProvider } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import { TrainingWorkspace } from "./TrainingWorkspace";
import { TRAINING_DETAILED_MANAGEMENT_KEY } from "./TrainingConfigLibrary";

function renderWorkspace() {
  const router = createMemoryRouter(
    [
      { path: "/training", element: <TrainingWorkspace /> },
      { path: "/other", element: <main>其他页面</main> },
    ],
    { initialEntries: ["/training"] },
  );
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return {
    router,
    ...render(
      <QueryClientProvider client={client}>
        <RouterProvider router={router} />
      </QueryClientProvider>,
    ),
  };
}

function createFetchMock() {
  let content = 'output_name = "dragon-run"\nmax_train_steps = 1600\n';
  let merged = {
    output_name: "dragon-run",
    model_family: "krea2_raw",
    max_train_steps: 1600,
    train_batch_size: 1,
    dataset_config: "configs/datasets/alpha.toml",
    gradient_checkpointing: true,
  } as Record<string, unknown>;
  const fetchMock = vi.fn(
    async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = init?.method || "GET";
      if (url === "/api/config/dataset-presets") return jsonResponse({ presets: [], groups: [] });
      if (url === "/api/training/gpus") return jsonResponse({ gpus: [{ index: 0, name: "Test GPU", memory_total_gb: 24 }] });
      if (url === "/api/config/model-families")
        return jsonResponse({
          items: [
            {
              name: "krea2_raw",
              supported_attention_modes: ["flash", "torch", "sdpa"],
            },
            { name: "anima" },
            { name: "z_image" },
          ],
        });
      if (url === "/api/settings/model-configs")
        return jsonResponse({
          items: [],
          groups: [],
          revision: "test",
          default_id: "",
        });
      if (url.startsWith("/api/config/steps?"))
        return jsonResponse({
          total_steps: 1600,
          train_image_count: 100,
          effective_batch_size: 1,
          steps_per_epoch: 100,
        });
      if (url === "/api/config/file-groups?kind=training")
        return jsonResponse([
          {
            id: "imported",
            label: "导入配置",
            methods_subdir: "imported",
            files: [
              {
                path: "configs/imported/train.toml",
                label: "train.toml",
                filename: "train.toml",
                method: "lora",
                methods_subdir: "imported",
                trainable: true,
                locked: false,
              },
            ],
          },
        ]);
      if (url === "/api/presets") return jsonResponse(["default", "low_vram"]);
      if (url.startsWith("/api/config/merged?")) {
        const preset = new URL(url, "http://localhost").searchParams.get(
          "preset",
        );
        return jsonResponse({
          ...merged,
          max_train_steps: preset === "low_vram" ? 800 : merged.max_train_steps,
        });
      }
      if (url.startsWith("/api/config/raw?") && method === "GET")
        return jsonResponse({
          file: "configs/imported/train.toml",
          content,
          meta: {
            path: "configs/imported/train.toml",
            label: "train.toml",
            method: "lora",
            methods_subdir: "imported",
            locked: false,
          },
        });
      if (url === "/api/config/raw/patch-preview" && method === "POST") {
        const body = requestJson(init);
        return jsonResponse(patchResult(body.file, body.values));
      }
      if (url === "/api/config/raw" && method === "PATCH") {
        const body = requestJson(init);
        const result = patchResult(body.file, body.values);
        content = result.content;
        merged = { ...merged, ...body.values };
        return jsonResponse(result);
      }
      if (url === "/api/config/raw/save-as" && method === "POST") {
        const body = requestJson(init);
        return jsonResponse({
          ok: true,
          file: body.file,
          message: "保存成功",
          warnings: [],
        });
      }
      if (url === "/api/training/preflight" && method === "POST")
        return jsonResponse({
          ok: false,
          variant: "lora",
          preset: "default",
          methods_subdir: "imported",
          summary: { errors: 1, warnings: 0, checks: 2 },
          checks: [
            {
              level: "error",
              key: "qwen3",
              message: "Qwen3 文本编码器 不存在",
              path: "models/qwen.safetensors",
            },
            {
              level: "ok",
              key: "source_image_dir",
              message: "源图像目录 存在",
              path: "image_dataset",
            },
          ],
          errors: [
            {
              level: "error",
              key: "qwen3",
              message: "Qwen3 文本编码器 不存在",
            },
          ],
          warnings: [],
        });
      return new Response(
        JSON.stringify({ ok: false, error: `unhandled ${method} ${url}` }),
        { status: 500 },
      );
    },
  );
  return fetchMock;
}

function patchResult(file: string, values: Record<string, unknown>) {
  const changed = Object.keys(values);
  const lines = Object.entries(values).map(([key, value]) =>
    typeof value === "string"
      ? `${key} = ${JSON.stringify(value)}`
      : `${key} = ${String(value)}`,
  );
  return {
    ok: true,
    file,
    message: "保存成功",
    content: `${lines.join("\n")}\n`,
    changed,
    warnings: [],
  };
}

function requestJson(init?: RequestInit) {
  return JSON.parse(String(init?.body || "{}"));
}

function jsonResponse(payload: unknown, status = 200) {
  return new Response(JSON.stringify(payload), { status });
}

describe("TrainingWorkspace", () => {
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    window.localStorage.clear();
  });

  it("shows source-aware merged values without exposing a start action", async () => {
    vi.stubGlobal("fetch", createFetchMock());
    renderWorkspace();

    expect(
      screen.getByRole("heading", { name: "训练配置" }),
    ).toBeInTheDocument();
    expect(
      await screen.findByLabelText("数据集配置"),
    ).toBeInTheDocument();
    await userEvent.click(screen.getByRole("tab", { name: "训练计划" }));
    expect(await screen.findByDisplayValue("dragon-run")).toBeInTheDocument();
    expect(screen.getByText("2 当前文件")).toBeInTheDocument();
    expect(screen.getAllByText("继承/预设").length).toBeGreaterThan(0);
    expect(
      screen.getByRole("button", { name: "立即启动" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "加入队列" }),
    ).toBeInTheDocument();
  });

  it("previews and saves only changed fields, then runs structured preflight", async () => {
    const fetchMock = createFetchMock();
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    renderWorkspace();
    await user.click(screen.getByRole("tab", { name: "训练计划" }));
    const outputName = await screen.findByLabelText("输出名称");
    await waitFor(() => expect(outputName).toBeEnabled());

    await user.clear(outputName);
    await user.type(outputName, "dragon-edited");
    expect(screen.getByText("有未保存修改")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "保存并预检" })).toBeEnabled();
    await user.click(screen.getByRole("button", { name: "预览变更" }));
    expect(await within(screen.getByRole("dialog")).findByText("1 项")).toBeInTheDocument();
    expect(
      screen.getByText("output_name", {
        selector: ".training-patch-preview code",
      }),
    ).toBeInTheDocument();
    expect(
      requestBody(fetchMock, "/api/config/raw/patch-preview", "POST").values,
    ).toEqual({ output_name: "dragon-edited" });

    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", { name: "关闭" }),
    );

    await user.click(screen.getByRole("button", { name: "保存配置" }));
    expect(await screen.findByText("保存成功")).toBeInTheDocument();
    expect(requestBody(fetchMock, "/api/config/raw", "PATCH").values).toEqual({
      output_name: "dragon-edited",
    });
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "运行预检测" })).toBeEnabled(),
    );
    await user.click(screen.getByRole("button", { name: "运行预检测" }));
    expect(await screen.findByText("需要处理")).toBeInTheDocument();
    expect(screen.getByText("Qwen3 文本编码器 不存在")).toBeInTheDocument();
    expect(screen.getByText("源图像目录 存在")).toBeInTheDocument();
  });

  it("keeps a stale training draft and blocks repeat saves after 409", async () => {
    const baseFetch = createFetchMock();
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.startsWith("/api/config/raw?") && !init?.method) {
        const response = await baseFetch(input, init);
        return jsonResponse({ ...(await response.json()), revision: "read-revision" });
      }
      if (url === "/api/config/raw" && init?.method === "PATCH")
        return new Response(JSON.stringify({ ok: false, error: "文件已在其他位置修改" }), { status: 409 });
      return baseFetch(input, init);
    });
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    renderWorkspace();
    await user.click(screen.getByRole("tab", { name: "训练计划" }));
    const outputName = await screen.findByLabelText("输出名称");
    await waitFor(() => expect(outputName).toBeEnabled());
    await user.clear(outputName);
    await user.type(outputName, "unsaved-draft");
    await user.click(screen.getByRole("button", { name: "保存配置" }));
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("文件已在其他位置修改"));
    expect(outputName).toHaveValue("unsaved-draft");
    expect(screen.getByRole("button", { name: "保存配置" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "重新加载配置" })).toBeInTheDocument();
    expect(requestBody(fetchMock, "/api/config/raw", "PATCH").revision).toBe("read-revision");
    expect(fetchMock.mock.calls.filter(([url, init]) => String(url) === "/api/config/raw" && init?.method === "PATCH")).toHaveLength(1);
  });

  it("opens auxiliary dialogs and persists the library toggle", async () => {
    vi.stubGlobal("fetch", createFetchMock());
    const user = userEvent.setup();
    const view = renderWorkspace();
    await waitFor(() => expect(screen.getByLabelText("数据集配置")).toHaveValue("configs/datasets/alpha.toml"));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "训练量估算" }));
    const dialog = screen.getByRole("dialog", { name: "训练量估算" });
    expect(await within(dialog).findByText("训练图片")).toBeInTheDocument();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "预览与预检" }));
    expect(
      screen.getByRole("dialog", { name: "预览与预检" }),
    ).toBeInTheDocument();
    await user.keyboard("{Escape}");
    const toggle = screen.getByRole("button", { name: "收起配置库" });
    expect(toggle.closest(".training-command-bar")).not.toBeNull();
    expect(within(screen.getByRole("complementary", { name: "训练配置库" }))
      .queryByRole("button", { name: "收起配置库" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "收起配置库" }));
    expect(document.getElementById("training-config-library")).not.toBeVisible();
    expect(localStorage.getItem("dragon-next.training-library-expanded")).toBe(
      "false",
    );
    view.unmount();
    renderWorkspace();
    expect(screen.getByRole("button", { name: "展开配置库" })).toHaveAttribute(
      "aria-expanded",
      "false",
    );
    await user.click(screen.getByRole("button", { name: "展开配置库" }));
    expect(screen.getByRole("complementary", { name: "训练配置库" })).toBeVisible();
    expect(localStorage.getItem("dragon-next.training-library-expanded")).toBe(
      "true",
    );
  });

  it("hides training library management by default and persists its switch", async () => {
    vi.stubGlobal("fetch", createFetchMock());
    const user = userEvent.setup();
    const view = renderWorkspace();
    const library = await screen.findByRole("complementary", { name: "训练配置库" });
    await within(library).findByText("train.toml");
    expect(within(library).getByRole("switch", { name: "详细管理" })).not.toBeChecked();
    expect(within(library).getByRole("button", { name: "新建分组" })).toBeInTheDocument();
    expect(within(library).queryByRole("button", { name: "重命名当前分组" })).not.toBeInTheDocument();
    expect(within(library).queryByRole("button", { name: "拖动排序 train.toml" })).not.toBeInTheDocument();
    expect(within(library).queryByRole("combobox", { name: "移动 train.toml 到分组" })).not.toBeInTheDocument();
    expect(library.querySelector(".training-library-item")).toBeInTheDocument();

    await user.click(within(library).getByRole("switch", { name: "详细管理" }));
    expect(localStorage.getItem(TRAINING_DETAILED_MANAGEMENT_KEY)).toBe("true");
    expect(within(library).getByRole("button", { name: "重命名当前分组" })).toBeInTheDocument();
    expect(within(library).getByRole("button", { name: "拖动排序 train.toml" })).toBeInTheDocument();

    view.unmount();
    renderWorkspace();
    const restored = await screen.findByRole("complementary", { name: "训练配置库" });
    await within(restored).findByText("train.toml");
    expect(within(restored).getByRole("switch", { name: "详细管理" })).toBeChecked();
    await user.click(within(restored).getByRole("switch", { name: "详细管理" }));
    expect(localStorage.getItem(TRAINING_DETAILED_MANAGEMENT_KEY)).toBe("false");
    expect(within(restored).queryByRole("button", { name: "拖动排序 train.toml" })).not.toBeInTheDocument();
  });

  it("guards dirty preset switching, browser unload, and SPA navigation", async () => {
    vi.stubGlobal("fetch", createFetchMock());
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    const user = userEvent.setup();
    const { router } = renderWorkspace();
    await user.click(screen.getByRole("tab", { name: "训练计划" }));
    const outputName = await screen.findByLabelText("输出名称");
    await waitFor(() => expect(outputName).toBeEnabled());
    await user.type(outputName, "-dirty");

    const unload = new Event("beforeunload", { cancelable: true });
    expect(window.dispatchEvent(unload)).toBe(false);
    await user.selectOptions(screen.getByLabelText("当前硬件预设"), "low_vram");
    expect(confirm).toHaveBeenCalledWith(
      "当前训练配置有未保存修改。切换硬件预设会丢失这些修改，是否继续？",
    );
    expect(screen.getByLabelText("当前硬件预设")).toHaveValue("default");

    await act(async () => {
      await router.navigate("/other");
    });
    expect(confirm).toHaveBeenCalledWith(
      "当前训练配置有未保存修改，离开会丢失这些修改。是否继续？",
    );
    expect(router.state.location.pathname).toBe("/training");
  });

  it("blocks execution after a failed save and retains the draft", async () => {
    const base = createFetchMock();
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) =>
      String(input) === "/api/config/raw" && init?.method === "PATCH"
        ? Promise.resolve(jsonResponse({ ok: false, error: "磁盘只读" }, 409))
        : base(input, init),
    );
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    renderWorkspace();
    await user.click(screen.getByRole("tab", { name: "训练计划" }));
    const output = await screen.findByLabelText("输出名称");
    await waitFor(() => expect(output).toBeEnabled());
    await user.type(output, "-dirty");
    await user.click(screen.getByRole("button", { name: "保存并启动" }));
    expect(await screen.findByText("磁盘只读")).toBeInTheDocument();
    expect(output).toHaveValue("dragon-run-dirty");
    expect(
      fetchMock.mock.calls.some(([input, init]) =>
        String(input).startsWith("/api/training/") && init?.method === "POST",
      ),
    ).toBe(false);
  });

  it("renders one category and preserves drafts through keyboard and filtered switches", async () => {
    const fetchMock = createFetchMock();
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    renderWorkspace();
    await waitFor(() => expect(screen.getByLabelText("数据集配置")).toHaveValue("configs/datasets/alpha.toml"));
    expect(screen.getAllByRole("tabpanel")).toHaveLength(1);
    expect(screen.queryByLabelText("输出名称")).not.toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: "训练计划" }));
    await user.type(screen.getByLabelText("输出名称"), "-draft");
    await user.click(screen.getByRole("tab", { name: "输入准备" }));
    expect(screen.queryByLabelText("输出名称")).not.toBeInTheDocument();
    await user.keyboard("{ArrowRight}");
    expect(screen.getByRole("tab", { name: "方法配置" })).toHaveFocus();
    expect(screen.getByLabelText("LoRA rank")).toBeInTheDocument();
    await user.keyboard("{End}");
    expect(screen.getByRole("tabpanel", { name: "设备与性能" })).toBeInTheDocument();
    await user.type(screen.getByLabelText("搜索参数"), "output_name");
    expect(within(screen.getByRole("tabpanel")).getByRole("status")).toHaveTextContent("当前分类没有符合筛选条件");
    expect(screen.getByRole("tab", { name: /训练计划/ })).toHaveTextContent("(1)");
    await user.click(screen.getByRole("tab", { name: /训练计划/ }));
    expect(screen.getByLabelText("输出名称")).toHaveValue("dragon-run-draft");
    expect(screen.getAllByRole("tabpanel")).toHaveLength(1);
    await user.clear(screen.getByLabelText("搜索参数"));
    await user.selectOptions(screen.getByLabelText("参数视图"), "changed");
    expect(screen.getByLabelText("输出名称")).toHaveValue("dragon-run-draft");
    expect(screen.queryByLabelText("最大训练步数")).not.toBeInTheDocument();
    expect(fetchMock.mock.calls.some(([, init]) => init?.method && init.method !== "GET")).toBe(false);
  });

  it("saves a patched copy into imported configs without overwriting", async () => {
    const fetchMock = createFetchMock();
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    renderWorkspace();
    await user.click(screen.getByRole("tab", { name: "训练计划" }));
    const outputName = await screen.findByLabelText("输出名称");
    await waitFor(() => expect(outputName).toBeEnabled());
    await user.clear(outputName);
    await user.type(outputName, "copy-value");
    await user.click(screen.getByRole("button", { name: "另存配置" }));
    const dialog = screen.getByRole("dialog", { name: "另存训练配置" });
    await user.clear(within(dialog).getByRole("textbox", { name: "配置名称" }));
    await user.type(
      within(dialog).getByRole("textbox", { name: "配置名称" }),
      "dragon copy",
    );
    await user.click(within(dialog).getByRole("button", { name: "确认另存" }));

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/config/raw/save-as",
        expect.objectContaining({ method: "POST" }),
      ),
    );
    const body = requestBody(fetchMock, "/api/config/raw/save-as", "POST");
    expect(body.file).toBe("configs/imported/dragon_copy.toml");
    expect(body.content).toContain('output_name = "copy-value"');
  });
});

function requestBody(
  fetchMock: ReturnType<typeof vi.fn>,
  url: string,
  method: string,
) {
  const call = fetchMock.mock.calls.find(
    ([input, init]) =>
      String(input) === url && (init?.method || "GET") === method,
  );
  if (!call) throw new Error(`missing ${method} ${url}`);
  return requestJson(call[1]);
}
