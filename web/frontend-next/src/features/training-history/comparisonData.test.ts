import { describe, expect, it } from "vitest";
import { comparisonLines, comparisonParameters, type ComparisonEntry } from "./comparisonData";

const entry = (id: string, points: Record<string, unknown>[], config_toml = ""): ComparisonEntry => ({ id, name: id, detail: { task: { job: "training" }, metrics: points, config_toml } });

describe("history comparison data", () => {
  it("compares actual snapshots including nested values without inventing defaults", () => {
    const rows = comparisonParameters([
      entry("a", [], 'network_dim = 16\nlearning_rate = 0.0000002\noptimizer = { a = 1, b = 2 }'),
      entry("b", [], 'network_dim = 32\noptimizer = { b = 2, a = 1 }'),
    ]);
    expect(rows.find((row) => row.key === "network_dim")).toMatchObject({ values: ["16", "32"], different: true });
    expect(rows.find((row) => row.key === "learning_rate")).toMatchObject({ values: ["2e-7", undefined], different: true });
    expect(rows.find((row) => row.key === "optimizer")?.different).toBe(false);
    expect(comparisonParameters([entry("a", [], "invalid = ["), entry("b", [])])).toEqual([]);
  });

  it("uses one real step domain and excludes invalid, missing and preprocess coordinates", () => {
    const items = [entry("a", [{ step: 100, loss: 0.2 }, { step: 0, loss: 0 }, { loss: 0.1 }, { step: 50, loss: NaN }, { kind: "val", step: 110, loss: 99 }]), entry("b", [{ current: 50, loss: 0.3 }, { step: 200, loss: 0.1 }])];
    const all = comparisonLines(items, false);
    expect(all).toMatchObject({ start: 0, end: 200 });
    expect(all.lines[0].data).toEqual([[0, 0], [100, 0.2]]);
    const overlap = comparisonLines(items, true);
    expect(overlap).toMatchObject({ start: 50, end: 100 });
    expect(overlap.lines.map((line) => line.data)).toEqual([[[100, 0.2]], [[50, 0.3]]]);
    expect(comparisonLines([{ ...items[0], detail: { ...items[0].detail, task: { job: "preprocess" } } }], false).lines[0].data).toEqual([]);
  });

  it("keeps no-overlap and no-step data explicit instead of using sample indices", () => {
    const result = comparisonLines([entry("a", [{ step: 1, loss: 1 }]), entry("b", [{ step: 2, loss: 2 }]), entry("c", [{ loss: 0.1 }])], true);
    expect(result.hasOverlap).toBe(false);
    expect(result.lines.every(({ data }) => data.length === 0)).toBe(true);
  });
});
