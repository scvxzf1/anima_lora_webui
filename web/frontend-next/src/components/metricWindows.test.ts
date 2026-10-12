import { expect, it } from "vitest";
import { adaptiveMetricData, metricWindows, segmentedMetricSeries } from "./metricWindows";
it("uses half-open 2000-sample windows with a precise final range", () => {
  expect(metricWindows(7889)).toEqual([{ start: 0, end: 2000 }, { start: 2000, end: 4000 }, { start: 4000, end: 6000 }, { start: 6000, end: 7889 }]);
  const points = Array.from({ length: 7889 }, (_, i) => ({ ts: i + 1, gpu_util: i }));
  expect(segmentedMetricSeries(points, "gpu_util", 2000, 0).data).toHaveLength(2000);
  expect(segmentedMetricSeries(points, "gpu_util", 2000, 0).data[0]).toEqual([2001000, 2000]);
  expect(segmentedMetricSeries(points, "gpu_util", 6000, 0).data.at(-1)).toEqual([7889000, 7888]);
});
it("adapts all-loaded rendering while retaining endpoints and spikes", () => {
  const data: [number, number][] = Array.from({ length: 90000 }, (_, i) => [i, i === 45555 ? 999 : i === 41111 ? -999 : 1]);
  const sampled = adaptiveMetricData(data);
  expect(sampled.length).toBeLessThanOrEqual(2000);
  expect(sampled[0]).toEqual(data[0]); expect(sampled.at(-1)).toEqual(data.at(-1));
  expect(sampled).toContainEqual([45555, 999]); expect(sampled).toContainEqual([41111, -999]);
});
it("smooths before downsampling and resets across missing GPU records", () => {
  const result = segmentedMetricSeries([{ ts: 1, gpu_util: 10 }, { ts: 2 }, { ts: 3, gpu_util: 40 }, { ts: 4, gpu_util: 0 }], "gpu_util", null, 0.5);
  expect(result.valid).toBe(3); expect(result.data[1][1]).toBeNaN();
  expect(result.data[2]).toEqual([3000, 40]); expect(result.data[3]).toEqual([4000, 20]);
});
it("retains extrema next to gaps without inventing continuity between retained points", () => {
  const data: [number, number][] = Array.from({ length: 9000 }, (_, i) => [i, 1]);
  data[5][1] = 999; data[8][1] = NaN; data[10][1] = -999;
  const sampled = adaptiveMetricData(data);
  expect(sampled.length).toBeLessThanOrEqual(2000);
  expect(sampled).toContainEqual([5, 999]); expect(sampled).toContainEqual([10, -999]);
  const peak = sampled.findIndex(([x]) => x === 5), trough = sampled.findIndex(([x]) => x === 10);
  expect(sampled.slice(peak + 1, trough).some(([, y]) => Number.isNaN(y))).toBe(true);
});

it("windows long training samples rather than STEP values or non-training events", () => {
  const training = Array.from({ length: 7889 }, (_, i) => ({ current: i * 3 + 17, loss: i, lr: i / 10000, ev: "step" }));
  const points = [{ ev: "preprocess", loss: 99999, lr: 9 }, ...training.slice(0, 2000), { kind: "sample", loss: 99999, lr: 9 }, ...training.slice(2000), { ev: "val", loss: 99999 }];
  for (const metric of ["loss", "lr"]) {
    const result = segmentedMetricSeries(points, metric, 2000, 0, false);
    expect(result).toMatchObject({ axis: "step", valid: 2000, loaded: 7889 });
    expect(result.data[0]).toEqual([6017, metric === "loss" ? 2000 : 0.2]);
    expect(result.data.at(-1)?.[0]).toBe(12014);
    const all = segmentedMetricSeries(points, metric, null, 0, false);
    expect(all.valid).toBe(7889); expect(all.data.length).toBeLessThanOrEqual(2000);
    expect(all.data[0][0]).toBe(17); expect(all.data.at(-1)?.[0]).toBe(23681);
  }
});
it("retains original fallback indices and resets smoothing at missing training values", () => {
  const result = segmentedMetricSeries([{ loss: 10 }, { ev: "sample", loss: 999 }, {}, { loss: 40 }, { loss: 0 }], "loss", null, 0.5, false);
  expect(result).toMatchObject({ axis: "index", loaded: 4, valid: 3 });
  expect(result.data[1][1]).toBeNaN();
  expect(result.data[2]).toEqual([4, 40]); expect(result.data[3]).toEqual([5, 20]);
});
