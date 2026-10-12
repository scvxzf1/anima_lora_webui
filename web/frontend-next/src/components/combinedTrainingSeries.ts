import { finiteNumber } from "./trainingNumbers";
import { metricSamples, METRIC_WINDOW_SIZE } from "./metricWindows";

export type TrainingChartPoint = [number, number, number];
export const trainingMetricNames = ["Loss", "学习率"] as const;
export const formatTrainingValue = (index: number, value: number) => Number.isFinite(value)
  ? index === 1 ? value.toExponential(3) : value.toPrecision(4) : "无记录";

// Keep both curves on exactly the same sample indices, including gap separators.
export function adaptiveTrainingData(data: TrainingChartPoint[], budget = 2000) {
  if (data.length <= budget) return data;
  const width = Math.ceil((data.length - 2) / Math.max(1, Math.floor((budget - 2) / 18)));
  const indices = new Set([0, data.length - 1]);
  for (let start = 1; start < data.length - 1; start += width) {
    for (const column of [1, 2]) {
      let min: number | undefined, max: number | undefined, gap: number | undefined;
      for (let i = start; i < Math.min(start + width, data.length - 1); i++) {
        if (!Number.isFinite(data[i][column])) { gap ??= i; continue; }
        if (min === undefined || data[i][column] < data[min][column]) min = i;
        if (max === undefined || data[i][column] > data[max][column]) max = i;
      }
      for (const index of [min, max, gap]) if (index !== undefined) indices.add(index);
    }
  }
  const selected = [...indices].sort((a, b) => a - b);
  // A retained gap must occur between finite extrema even across bucket boundaries.
  const result: TrainingChartPoint[] = [];
  selected.forEach((index, ordinal) => {
    const previous = selected[ordinal - 1];
    if (previous !== undefined) {
      const gaps = new Set<number>();
      for (const column of [1, 2]) {
        if (!Number.isFinite(data[previous][column]) || !Number.isFinite(data[index][column])) continue;
        for (let i = previous + 1; i < index; i++) {
          if (!Number.isFinite(data[i][column])) { gaps.add(i); break; }
        }
      }
      [...gaps].sort((a, b) => a - b).forEach((i) => result.push(data[i]));
    }
    result.push(data[index]);
  });
  return result;
}

export function combinedTrainingSeries(points: Record<string, unknown>[], start: number | null, smoothing: number) {
  const samples = metricSamples(points, "loss");
  const hasSteps = samples.some(({ point }) => finiteNumber(point.step ?? point.current) !== undefined);
  const end = start === null ? samples.length : Math.min(samples.length, start + METRIC_WINDOW_SIZE);
  const previous: (number | undefined)[] = [undefined, undefined];
  const valid = [0, 0];
  const data: TrainingChartPoint[] = [];
  for (let ordinal = start ?? 0; ordinal < end; ordinal++) {
    const { point, index } = samples[ordinal];
    const coordinate = hasSteps ? finiteNumber(point.step ?? point.current) : index + 1;
    const values = ["loss", "lr"].map((key, column) => {
      const value = coordinate === undefined ? undefined : finiteNumber(point[key]);
      if (value === undefined) { previous[column] = undefined; return NaN; }
      valid[column]++;
      return previous[column] = previous[column] === undefined ? value : smoothing * previous[column]! + (1 - smoothing) * value;
    });
    data.push([coordinate ?? (data.at(-1)?.[0] ?? 0) + 1, values[0], values[1]]);
  }
  return { data: start === null ? adaptiveTrainingData(data) : data, valid, loaded: samples.length, axis: hasSteps ? "step" : "index" };
}
