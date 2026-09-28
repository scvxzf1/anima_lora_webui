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
      logs_path: "output/logs/task.log",
    } as HistoryTaskSummary & Record<string, unknown> }} />);

    await user.click(screen.getByRole("button", { name: "复制基础目录" }));
    expect(writeText).toHaveBeenCalledWith("/runs/task-1");
    expect(screen.getByRole("status")).toHaveTextContent("基础目录已复制");

    writeText.mockRejectedValueOnce(new Error("permission denied"));
    await user.click(screen.getByRole("button", { name: "复制历史日志文件" }));
    expect(writeText).toHaveBeenLastCalledWith("/project/output/logs/task.log");
    expect(screen.getByRole("status")).toHaveTextContent("无法复制历史日志文件");
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
