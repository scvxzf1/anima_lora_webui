import { describe, expect, it } from "vitest";
import { historySummary, snapshotFields, taskOutcome } from "./historySummary";
import { finiteNumber, formatDuration, formatLearningRate, formatLoss } from "../../components/trainingNumbers";
import { readStackPages } from "./historyNavigation";

describe("history summary contract", () => {
  it("uses persisted step/loss, never metric count", () => {
    expect(historySummary({ job: "training", last_step: 6400, final_loss: 0.08, metric_count: 73 })).toMatchObject({ step: 6400, loss: 0.08 });
    expect(historySummary({ job: "training", metric_count: 73 }).step).toBeUndefined();
    expect(historySummary({ job: "training", last_step: 6400, final_loss: 0.08 }, [{ step: 8000, loss: 0.2 }])).toMatchObject({ step: 6400, loss: 0.08 });
  });
  it("finds last valid fields independently and preserves zero", () => {
    expect(historySummary({ job: "training" }, [{ step: 20, loss: 0, lr: 2e-7 }, { step: 30, loss: null }, { loss: NaN, lr: "" }])).toEqual({ step: 30, loss: 0, lr: 2e-7 });
    expect(historySummary({ job: "training", last_step: 0, final_loss: 0 }, [{ step: 1, loss: 1 }])).toMatchObject({ step: 0, loss: 0 });
  });
  it("does not invent training metrics for preprocessing or unknown jobs", () => {
    for (const job of ["preprocess", undefined]) expect(historySummary({ job, last_step: 5, metric_count: 7 }, [{ step: 99, loss: 1 }])).toEqual({ step: undefined, loss: undefined, lr: undefined });
  });
  it("never fills training summary fields with validation CMMD", () => {
    const validation = [{ kind: "val", step: 11, loss: 0.91 }, { ev: "val", step: 12, loss: 0.8 }];
    expect(historySummary({ job: "training" }, [{ step: 10, loss: 0.2 }, ...validation])).toMatchObject({ step: 10, loss: 0.2 });
    expect(historySummary({ job: "training" }, validation)).toEqual({ step: undefined, loss: undefined, lr: undefined });
  });
  it.each([null, undefined, "", " ", true, [], {}, NaN, Infinity, "Infinity"])("rejects invalid numeric value %s", (value) => expect(finiteNumber(value)).toBeUndefined());
  it("formats small LR, loss, unknown and durations accurately", () => {
    expect(formatLearningRate(2e-7)).toBe("2.00e-7");
    expect(formatLearningRate(0)).toBe("0");
    expect(formatLoss(0.08)).toBe("0.0800");
    expect(formatLoss(null)).toBe("未记录");
    expect(formatDuration(0, 3600)).toBe("1 时 0 分");
    expect(formatDuration(1, 0)).toBe("未记录");
    expect(formatDuration(0, 59.9)).toBe("1 分 0 秒");
  });
  it("parses TOML rather than guessing task-name fields", () => {
    expect(snapshotFields({ config_toml: 'network_dim = 16\ntorch_compile = false' }).values).toMatchObject({ network_dim: 16, torch_compile: false });
    expect(snapshotFields({ config_toml: "broken = [" }).invalid).toBe(true);
    expect(taskOutcome({ message: "CUDA failure" })).toBe("CUDA failure");
  });
  it("validates navigation state", () => {
    expect(readStackPages('{"group":2,"invalid":-1,"text":"0"}')).toEqual({ group: 2 });
    expect(readStackPages("[]")).toEqual({});
    expect(readStackPages("bad")).toEqual({});
  });
});
