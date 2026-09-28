import { cleanup, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { renderInApp } from "../../test/renderInApp";
import type { HistoryTaskSummary } from "./api";
import { HistoryOverview } from "./HistoryOverview";

describe("history overview checkpoint source", () => {
  afterEach(cleanup);

  it("copies available task paths and reports clipboard failures accessibly", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText } });
    renderInApp(<HistoryOverview detail={{ task: {
      job: "training", state: "idle", run_dir_abs: "/runs/task-1", project_root_abs: "/project",
      runtime_config_file: "config.runtime.toml", history_dir_abs: "/external/history/task-1",
      logs_path: "configs/web-training-history/task-1/logs.jsonl",
      config_snapshot: "configs/web-training-history/task-1/config.snapshot.toml",
      absolute_paths: {
        run_dir_abs: "/runs/task-1",
        history_dir_abs: "/external/history/task-1",
        runtime_config_file: "/custom-output/runs/task-1/config.runtime.toml",
        logs_path: "/external/history/task-1/logs.jsonl",
        config_snapshot: "/external/history/task-1/config.snapshot.toml",
      },
    } as HistoryTaskSummary & Record<string, unknown> }} />);

    await user.click(screen.getByRole("button", { name: "复制基础目录" }));
    expect(writeText).toHaveBeenCalledWith("/runs/task-1");
    expect(screen.getByRole("status")).toHaveTextContent("基础目录已复制");

    await user.click(screen.getByRole("button", { name: "复制实际运行配置" }));
    expect(writeText).toHaveBeenLastCalledWith("/custom-output/runs/task-1/config.runtime.toml");

    writeText.mockRejectedValueOnce(new Error("permission denied"));
    await user.click(screen.getByRole("button", { name: "复制历史日志文件" }));
    expect(writeText).toHaveBeenLastCalledWith("/external/history/task-1/logs.jsonl");
    expect(screen.getByRole("status")).toHaveTextContent("无法复制历史日志文件");
    await user.click(screen.getByRole("button", { name: "复制历史 TOML 快照" }));
    expect(writeText).toHaveBeenLastCalledWith("/external/history/task-1/config.snapshot.toml");
  });

  it("copies only the absolute path supplied by the history API", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText } });
    renderInApp(<HistoryOverview detail={{ task: {
      job: "training", state: "idle", project_root_abs: "/project", run_dir_abs: "/project/output/runs/task-1",
      runtime_config_file: "output/runs/task-1/config.runtime.toml",
      absolute_paths: { runtime_config_file: "/custom-output/task-1/config.runtime.toml" },
    } as HistoryTaskSummary & Record<string, unknown> }} />);

    await user.click(screen.getByRole("button", { name: "复制实际运行配置" }));
    expect(writeText).toHaveBeenCalledWith("/custom-output/task-1/config.runtime.toml");
  });

  it("does not offer copying for a relative path with no known base", () => {
    renderInApp(<HistoryOverview detail={{ task: {
      job: "training", state: "idle", runtime_config_file: "config.runtime.toml",
    } as HistoryTaskSummary & Record<string, unknown> }} />);

    expect(screen.getByText("config.runtime.toml")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "复制实际运行配置" })).toBeDisabled();
  });

  it("still copies an absolute path from an older history API response", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText } });
    renderInApp(<HistoryOverview detail={{ task: {
      job: "training", state: "idle", runtime_config_file: "/external/run/config.runtime.toml",
    } as HistoryTaskSummary & Record<string, unknown> }} />);

    await user.click(screen.getByRole("button", { name: "复制实际运行配置" }));
    expect(writeText).toHaveBeenCalledWith("/external/run/config.runtime.toml");
  });

  it.each<[HistoryTaskSummary["resume_from"], string]>([
    [{}, "未记录"],
    [undefined, "未记录"],
    [null, "未记录"],
    ["", "未记录"],
    ["output/legacy-state", "output/legacy-state"],
    [{ checkpoint_name: "epoch-2-state", checkpoint: "output/epoch-2-state", checkpoint_step: 42 }, "epoch-2-state · step 42"],
    [{ checkpoint: "output/step-0-state", checkpoint_step: 0 }, "output/step-0-state · step 0"],
    [{ checkpoint_step: 12 }, "step 12"],
    [{ checkpoint_name: "epoch-2-state", checkpoint_step: null }, "epoch-2-state"],
  ])("renders resume metadata %j without passing objects to React", (resume_from, expected) => {
    renderInApp(<HistoryOverview detail={{
      task: { job: "training", state: "idle", message: "训练完成", resume_from, returncode: 0 },
    }} />);
    expect(screen.getByRole("heading", { name: "配置指纹" })).toBeInTheDocument();
    expect(screen.getByText("来源检查点").nextElementSibling).toHaveTextContent(expected);
    expect(screen.getByText("退出码").nextElementSibling).toHaveTextContent("0");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
