import { describe, expect, it } from "vitest";
import { checkpointLosses, commonWeightDirectory, groupHistorySamples } from "./historyAssetPresentation";
import type { HistoryWeight } from "./api";

const weight = (abs_path?: string): HistoryWeight => ({ abs_path, file: "w", name: "w", size_bytes: 1, scope_label: "本任务" });

describe("history asset presentation", () => {
  it("finds only contiguous shared parent segments and handles Windows paths", () => {
    expect(commonWeightDirectory([weight("/runs/task/a.safetensors"), weight("/runs/task/b.safetensors")])).toBe("/runs/task");
    expect(commonWeightDirectory([weight("/runs/a/shared/a"), weight("/runs/b/shared/b")])).toBe("/runs");
    expect(commonWeightDirectory([weight("C:\\runs\\a"), weight("C:\\runs\\b")])).toBe("C:\\runs");
    expect(commonWeightDirectory([weight("/a/w"), weight("/b/w")])).toBeUndefined();
    expect(commonWeightDirectory([weight("/runs/a"), weight()])).toBeUndefined();
    expect(commonWeightDirectory([])).toBeUndefined();
  });

  it("keeps edit references and seeds distinct while ordering steps chronologically", () => {
    const image = (file: string, step: number, seed = 42, reference_file = "ref-a") => ({ file, name: file, sample: { prompt_index: 0, prompt: "portrait", step, seed, reference_file } });
    const groups = groupHistorySamples([image("late", 1000), image("early", 200), image("seed", 200, 43), image("ref", 200, 42, "ref-b")]);
    expect(groups).toHaveLength(3);
    expect(groups[0].images.map((entry) => entry.file)).toEqual(["early", "late"]);
    expect(groups[0].label).toBe("Prompt 1 · Seed 42");
  });

  it("matches exact training steps only, excluding validation and invalid values", () => {
    const losses = checkpointLosses([{ step: 100, loss: .2 }, { step: 100, loss: .9, kind: "val" }, { step: 200, loss: .1 }, { step: 300, loss: NaN }]);
    expect(losses.get(100)).toBe(.2);
    expect(losses.get(150)).toBeUndefined();
    expect(losses.get(200)).toBe(.1);
    expect(losses.get(300)).toBeUndefined();
  });
});
