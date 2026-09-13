import { cleanup, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { renderInApp } from "../../test/renderInApp";
import type { HistoryTaskSummary } from "./api";
import { HistoryOverview } from "./HistoryOverview";

describe("history overview checkpoint source", () => {
  afterEach(cleanup);

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
