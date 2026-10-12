import { expect, it } from "vitest";
import { adaptiveTrainingData, combinedTrainingSeries, type TrainingChartPoint } from "./combinedTrainingSeries";
it("shares 2000-sample windows, filters events and retains true STEP/current", () => {
  const points = [{ ev: "sample", loss: 999 }, ...Array.from({ length: 7889 }, (_, i) => ({ current: 17 + i * 3, loss: i, lr: i / 10000 }))];
  const window = combinedTrainingSeries(points, 2000, 0);
  expect(window).toMatchObject({ loaded: 7889, valid: [2000, 2000], axis: "step" });
  expect(window.data).toHaveLength(2000);
  expect(window.data[0]).toEqual([6017, 2000, 0.2]);
  expect(combinedTrainingSeries(points, 6000, 0).data).toHaveLength(1889);
});
it("retains both curves peaks at aligned indices with bounded rendering", () => {
  const data: TrainingChartPoint[] = Array.from({ length: 90000 }, (_, i) => [i, 1, 0.0001]);
  data[45555][1] = 999; data[41111][1] = -999;
  data[12345][2] = 0.1; data[65432][2] = -0.1;
  const sampled = adaptiveTrainingData(data);
  expect(sampled.length).toBeLessThanOrEqual(2000);
  for (const i of [0, 89999, 45555, 41111, 12345, 65432]) expect(sampled).toContainEqual(data[i]);
});
it("resets smoothing and preserves missing metrics independently", () => {
  const result = combinedTrainingSeries([{ loss: 10, lr: 1 }, { lr: 2 }, { loss: 40 }, { loss: 0, lr: 4 }], null, 0.5);
  expect(result.data[1][1]).toBeNaN(); expect(result.data[1][2]).toBe(1.5);
  expect(result.data[2][1]).toBe(40); expect(result.data[2][2]).toBeNaN();
  expect(result.data[3]).toEqual([4, 20, 4]);
});
it("retains independent gap separators between sampled extrema", () => {
  const data: TrainingChartPoint[] = Array.from({ length: 9000 }, (_, i) => [i, 1, 1]);
  data[5][1] = 999; data[8][1] = NaN; data[10][1] = -999;
  data[6][2] = 999; data[9][2] = NaN; data[11][2] = -999;
  const sampled = adaptiveTrainingData(data);
  expect(sampled.length).toBeLessThanOrEqual(2000);
  for (const column of [1, 2]) {
    const peak = sampled.findIndex((p) => p[column] === 999), trough = sampled.findIndex((p) => p[column] === -999);
    expect(sampled.slice(peak + 1, trough).some((p) => Number.isNaN(p[column]))).toBe(true);
  }
});
