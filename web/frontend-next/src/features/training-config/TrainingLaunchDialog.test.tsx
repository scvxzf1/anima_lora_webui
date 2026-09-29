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
  summary: { errors: ok ? 0 : 1, warnings: 0, checks: 1 },
  checks: [
    {
      level: ok ? "ok" : "error",
      key: "model",
      message: ok ? "检查通过" : "模型不存在",
    },
  ],
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
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      if (String(input).endsWith("gpus")) return jsonResponse({ gpus: [] });
      if (String(input).endsWith("preflight")) return jsonResponse(preflight());
      return new Promise<Response>((resolve) => {
        window.setTimeout(() => resolve(jsonResponse({ ok: true })), 100);
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
    await screen.findByRole("status");
  });
});
