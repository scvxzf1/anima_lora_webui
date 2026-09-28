import { describe, expect, it } from "vitest";
import { metricSeries } from "./metricSeries";

describe("metric semantics", () => {
  const points = [
    { step: 10, loss: 0.2 },
    { kind: "val", step: 10, loss: 0.91, cmmd: 0.91 },
    { ev: "val", step: 11, loss: 0.8, cmmd: 0.8 },
    { step: 12, loss: 0.1 },
  ];
  it("excludes validation before smoothing training loss", () => {
    expect(metricSeries(points, "loss", 0, 0.5, false).data).toEqual([[10, 0.2], [12, 0.15000000000000002]]);
    expect(metricSeries(points.slice(1, 3), "loss", 0, 0, false).data).toEqual([]);
  });
  it("smooths every valid training point without changing series length", () => {
    const result = metricSeries(
      [{ step: 1, loss: 1 }, { step: 2, loss: 0 }, { step: 3, loss: 1 }],
      "loss",
      0,
      0.5,
      false,
    );
    expect(result.data).toEqual([[1, 1], [2, 0.5], [3, 0.75]]);
  });
  it("retains independently named CMMD and other metrics", () => {
    expect(metricSeries(points, "cmmd", 0, 0, false).data).toEqual([[10, 0.91], [11, 0.8]]);
    expect(metricSeries([{ ts: 1, vram: 0 }], "vram", 0, 0, true).data).toEqual([[1000, 0]]);
  });
  it("preserves legacy loss points and the bounded source window", () => {
    expect(metricSeries(points, "loss", 2, 0, false)).toMatchObject({ data: [[12, 0.1]], count: 2 });
  });
});
