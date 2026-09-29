import { describe, expect, it } from "vitest";
import { captionJobStateLabel } from "./captionJobState";

describe("captionJobStateLabel", () => {
  it("uses only the provider recorded in the job snapshot", () => {
    expect(captionJobStateLabel("running", { settings: { provider: "cltagger" } })).toBe("正在本地打标");
    expect(captionJobStateLabel("running", { settings: { provider: "WD14" } })).toBe("正在本地打标");
    expect(captionJobStateLabel("running", { settings: { provider: "openai_compatible" } })).toBe("正在调用外部 API");
    expect(captionJobStateLabel("running", { settings: { provider: "future_provider" } })).toBe("正在处理");
    expect(captionJobStateLabel("running", { profile_id: "local-profile" } as never)).toBe("正在处理");
    expect(captionJobStateLabel("queued", { settings: { provider: "cltagger" } })).toBe("本地任务待处理");
  });

  it("localizes terminal states without treating partial or unknown states as success", () => {
    expect(captionJobStateLabel("queued")).toBe("待处理");
    expect(captionJobStateLabel("completed")).toBe("已完成");
    expect(captionJobStateLabel("partial")).toBe("部分完成");
    expect(captionJobStateLabel("failed")).toBe("任务失败");
    expect(captionJobStateLabel("canceled")).toBe("任务已停止");
    expect(captionJobStateLabel("cancelled")).toBe("任务已停止");
    expect(captionJobStateLabel("future_state")).toBe("future_state");
    expect(captionJobStateLabel(undefined)).toBe("未知状态");
  });
});
