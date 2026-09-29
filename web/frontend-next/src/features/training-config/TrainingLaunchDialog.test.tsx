import { cleanup, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { jsonResponse, renderInApp } from "../../test/renderInApp";
import { TrainingLaunchDialog } from "./TrainingLaunchDialog";

const file = {
  path: "configs/imported/fixture.toml",
  method: "lora",
  methods_subdir: "imported",
};
const preflight = (ok = true) => ({
  ok,
  variant: "lora",
  preset: "default",
  methods_subdir: "imported",
  summary: { errors: ok ? 0 : 1, warnings: 0, checks: 1 },
  checks: [
    {
      level: ok ? "ok" : "error",
      key: "model",
      message: ok ? "检查通过" : "模型不存在",
    },
  ],
  errors: ok ? [] : [{ level: "error", key: "model", message: "模型不存在" }],
  warnings: [],
});

describe("launch command boundary", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });
  it("requires preflight and explicit confirmation, then enqueues paused exactly once", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) =>
      jsonResponse(
        String(input) === "/api/training/gpus"
          ? { gpus: [] }
          : String(input).endsWith("preflight")
            ? preflight()
            : { ok: true, message: "已入队" },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);
    renderInApp(
      <TrainingLaunchDialog
        file={file}
        preset="default"
        mode="queue"
        gpuIds={["1", "2"]}
        deviceSummary="数据并行 · GPU 1 / GPU 2"
        deviceIssue=""
        onClose={vi.fn()}
      />,
    );
    expect(
      screen.getByRole("dialog", { name: "确认加入队列" }),
    ).toBeInTheDocument();
    expect(screen.getByText(file.path)).toBeInTheDocument();
    expect(screen.getByText("default")).toBeInTheDocument();
    expect(screen.getByText("数据并行 · GPU 1 / GPU 2")).toBeInTheDocument();
    const button = screen.getByRole("button", { name: "确认入队" });
    expect(button).toBeDisabled();
    await screen.findByText("检查通过");
    expect(screen.getByText("可以继续")).toHaveAttribute(
      "data-tone",
      "success",
    );
    expect(button).toBeDisabled();
    expect(
      fetchMock.mock.calls.some(
        ([url]) => String(url) === "/api/training/queue",
      ),
    ).toBe(false);
    const user = userEvent.setup();
    await user.click(screen.getByRole("checkbox"));
    await user.dblClick(button);
    await screen.findByText("已入队");
    expect(screen.getByRole("status")).toHaveTextContent("已入队");
    const calls = (
      fetchMock.mock.calls as unknown as [string, RequestInit][]
    ).filter(([url]) => url === "/api/training/queue");
    expect(calls).toHaveLength(1);
    expect(JSON.parse(String(calls[0][1].body))).toMatchObject({
      config_file: file.path,
      start_paused: true,
      confirmed: true,
      confirm_preprocess: true,
      gpu_whitelist: ["1", "2"],
    });
  });
  it("does not start after failed preflight even with the checkbox checked", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) =>
      jsonResponse(
        String(input).endsWith("gpus") ? { gpus: [] } : preflight(false),
      ),
    );
    vi.stubGlobal("fetch", fetchMock);
    renderInApp(
      <TrainingLaunchDialog
        file={file}
        preset="default"
        mode="start"
        gpuIds={["1"]}
        deviceSummary="单卡 · GPU 1"
        deviceIssue=""
        onClose={vi.fn()}
      />,
    );
    await screen.findByText("模型不存在");
    await userEvent.setup().click(screen.getByRole("checkbox"));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "确认启动" })).toBeDisabled(),
    );
    expect(
      fetchMock.mock.calls.some(
        ([url]) => String(url) === "/api/training/start",
      ),
    ).toBe(false);
  });

  it("shows warning status for a successful preflight with warnings", async () => {
    const warningResult = {
      ...preflight(),
      summary: { errors: 0, warnings: 1, checks: 2 },
      checks: [
        { level: "warning", key: "output_dir", message: "输出目录将被创建" },
        { level: "ok", key: "model", message: "模型检查通过" },
      ],
      warnings: [
        { level: "warning", key: "output_dir", message: "输出目录将被创建" },
      ],
    };
    const fetchMock = vi.fn(async (input: RequestInfo | URL) =>
      jsonResponse(
        String(input).endsWith("gpus")
          ? { gpus: [] }
          : String(input).endsWith("preflight")
            ? warningResult
            : { ok: true, message: "已启动" },
      ),
    );
    vi.stubGlobal(
      "fetch",
      fetchMock,
    );
    renderInApp(
      <TrainingLaunchDialog
        file={file}
        preset="default"
        mode="start"
        gpuIds={[]}
        deviceSummary="自动选择"
        deviceIssue=""
        onClose={vi.fn()}
      />,
    );

    expect(await screen.findByText("输出目录将被创建")).toBeInTheDocument();
    const status = screen.getByText("存在警告");
    expect(status).toHaveAttribute("data-tone", "warning");
    const confirm = screen.getByRole("button", { name: "确认启动" });
    expect(confirm).toBeDisabled();
    const user = userEvent.setup();
    await user.click(screen.getByRole("checkbox"));
    expect(confirm).toBeEnabled();
    await user.click(confirm);
    expect(await screen.findByRole("status")).toHaveTextContent("已启动");
    expect(
      fetchMock.mock.calls.filter(
        ([url]) => String(url) === "/api/training/start",
      ),
    ).toHaveLength(1);
  });

  it("orders mixed preflight checks by severity and preserves ties", async () => {
    const mixed = {
      ...preflight(false),
      summary: { errors: 1, warnings: 1, checks: 5 },
      checks: [
        { level: "ok", key: "ok-first", message: "通过一" },
        { level: "info", key: "info", message: "提示" },
        { level: "warning", key: "warning", message: "警告" },
        { level: "error", key: "error", message: "错误" },
        { level: "ok", key: "ok-second", message: "通过二" },
      ],
      errors: [{ level: "error", key: "error", message: "错误" }],
      warnings: [{ level: "warning", key: "warning", message: "警告" }],
    };
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) =>
        jsonResponse(String(input).endsWith("gpus") ? { gpus: [] } : mixed),
      ),
    );
    renderInApp(
      <TrainingLaunchDialog
        file={file}
        preset="default"
        mode="start"
        gpuIds={[]}
        deviceSummary="自动选择"
        deviceIssue=""
        onClose={vi.fn()}
      />,
    );

    expect(await screen.findAllByText("错误")).toHaveLength(3);
    expect(screen.getByText("需要处理")).toHaveAttribute("data-tone", "danger");
    const levels = Array.from(
      document.querySelectorAll(".training-preflight-checks li"),
      (item) => item.getAttribute("data-level"),
    );
    expect(levels).toEqual(["error", "warning", "info", "ok", "ok"]);
    const successes = screen.getAllByText(/^通过[一二]$/);
    expect(successes.map((item) => item.textContent)).toEqual(["通过一", "通过二"]);
  });

  it("renders untrusted preflight and completion payloads as text", async () => {
    const hostile = '<img src=x onerror="alert(1)">';
    const fetchMock = vi.fn(async (input: RequestInfo | URL) =>
      jsonResponse(
        String(input).endsWith("gpus")
          ? { gpus: [] }
          : String(input).endsWith("preflight")
            ? {
                ...preflight(),
                checks: [
                  {
                    level: "warning",
                    key: "output_dir",
                    message: hostile,
                    path: hostile,
                  },
                ],
              }
            : { ok: true, message: hostile },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);
    renderInApp(
      <TrainingLaunchDialog
        file={file}
        preset="default"
        mode="start"
        gpuIds={[]}
        deviceSummary="自动选择"
        deviceIssue=""
        onClose={vi.fn()}
      />,
    );

    await screen.findAllByText(hostile);
    expect(screen.getAllByText(hostile)).toHaveLength(2);
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    const confirm = screen.getByRole("checkbox");
    const user = userEvent.setup();
    await user.click(confirm);
    await user.click(screen.getByRole("button", { name: "确认启动" }));

    const status = await screen.findByRole("status");
    expect(status).toHaveTextContent(hostile);
    expect(status.querySelector("img")).toBeNull();
  });

  it("disables dialog commands while the request is pending", async () => {
    let resolveAction!: (response: Response) => void;
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      if (String(input).endsWith("gpus")) return jsonResponse({ gpus: [] });
      if (String(input).endsWith("preflight")) return jsonResponse(preflight());
      return new Promise<Response>((resolve) => {
        resolveAction = resolve;
      });
    });
    vi.stubGlobal("fetch", fetchMock);
    renderInApp(
      <TrainingLaunchDialog
        file={file}
        preset="default"
        mode="start"
        gpuIds={[]}
        deviceSummary="自动选择"
        deviceIssue=""
        onClose={vi.fn()}
      />,
    );
    await screen.findByText("检查通过");
    const user = userEvent.setup();
    await user.click(screen.getByRole("checkbox"));
    await user.click(screen.getByRole("button", { name: "确认启动" }));

    expect(await screen.findByRole("button", { name: "正在提交" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "取消" })).toBeDisabled();
    resolveAction(jsonResponse({ ok: true }));
    await screen.findByRole("status");
  });
});
