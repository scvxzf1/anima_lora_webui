import { describe, expect, it } from "vitest";
import { orderHistoryTimeline, timelineLogText } from "./timelineData";

describe("ordered history timeline", () => {
  it("follows requested task order and marks the first metric in each stage", () => {
    const result = orderHistoryTimeline({
      tasks: [{ id: "later", label: "Later" }, { id: "earlier", label: "Earlier" }, { id: "empty" }],
      segments: [{ task: { id: "later" }, index: 1 }, { task: { id: "earlier" }, index: 2 }, { task: { id: "empty" }, index: 3 }],
      metrics: [
        { source_task_id: "later", source_task_index: 1, visual_step: 1, loss: 0.3 },
        { source_task_id: "earlier", source_task_index: 2, visual_step: 2, loss: 0.2 },
      ],
      logs: [{ source_task_id: "later", line: "second" }, { source_task_id: "earlier", line: "first" }],
    }, ["earlier", "empty", "later"]);

    expect(result.tasks?.map((task) => task.id)).toEqual(["earlier", "empty", "later"]);
    expect(result.segments?.map((segment) => segment.task?.id)).toEqual(["earlier", "empty", "later"]);
    expect(result.metrics?.map((point) => [point.source_task_id, point.visual_step, point.stage_break_before])).toEqual([
      ["earlier", 1, false], ["later", 2, true],
    ]);
    expect(result.logs?.map((log) => log.line)).toEqual(["first", "second"]);
    expect(timelineLogText(result.logs![0])).toBe("[任务1] first");
  });

  it("handles empty metrics and unknown tasks without fabricating points", () => {
    const result = orderHistoryTimeline({ tasks: [{ id: "a" }], metrics: [], logs: [] }, ["a", "b"]);
    expect(result.metrics).toEqual([]);
    expect(result.logs).toEqual([]);
  });
});
