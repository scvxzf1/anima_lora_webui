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
    const button = screen.getByRole("button", { name: "确认入队" });
    expect(button).toBeDisabled();
    await screen.findByText("检查通过");
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
});
