import { finiteNumber } from "./trainingNumbers";
import { isTrainingMetric } from "./metricSemantics";

export const METRIC_WINDOW_SIZE = 2000;
export function metricWindows(total: number) {
  return Array.from({ length: Math.ceil(total / METRIC_WINDOW_SIZE) }, (_, index) => {
    const start = index * METRIC_WINDOW_SIZE;
    return { start, end: Math.min(total, start + METRIC_WINDOW_SIZE) };
  });
}

// Ordered bucket extrema preserve narrow spikes and troughs, plus endpoints.
export function adaptiveMetricData(data: [number, number][], budget = 2000): [number, number][] {
  if (data.length <= budget) return data;
  // Leave room for a missing-value separator between retained finite points.
  const buckets = Math.max(1, Math.floor((budget - 3) / 4));
  const width = Math.ceil((data.length - 2) / buckets);
  const indices = new Set([0, data.length - 1]);
  for (let start = 1; start < data.length - 1; start += width) {
    let min: number | undefined, max: number | undefined;
    for (let i = start; i < Math.min(start + width, data.length - 1); i++) {
      if (!Number.isFinite(data[i][1])) continue;
      if (min === undefined || data[i][1] < data[min][1]) min = i;
      if (max === undefined || data[i][1] > data[max][1]) max = i;
    }
    if (min !== undefined) indices.add(min);
    if (max !== undefined) indices.add(max);
  }
  const selected = [...indices].sort((a, b) => a - b);
  const sampled: [number, number][] = [];
  selected.forEach((index, ordinal) => {
    const previous = selected[ordinal - 1];
    if (previous !== undefined && Number.isFinite(data[previous][1]) && Number.isFinite(data[index][1])) {
      for (let i = previous + 1; i < index; i++) {
        if (!Number.isFinite(data[i][1])) { sampled.push(data[i]); break; }
      }
    }
    sampled.push(data[index]);
  });
  return sampled;
}

// Filter events before counting windows, while retaining original fallback coordinates.
export function metricSamples(points: Record<string, unknown>[], metric: string) {
  return points.flatMap((point, index) =>
    (metric === "loss" || metric === "lr") && !isTrainingMetric(point) ? [] : [{ point, index }]);
}

export function segmentedMetricSeries(points: Record<string, unknown>[], metric: string, start: number | null, smoothing: number, timeAxis = true) {
  const samples = metricSamples(points, metric);
  const hasSteps = samples.some(({ point }) => finiteNumber(point.step ?? point.current) !== undefined);
  const axis = timeAxis ? "time" : hasSteps ? "step" : "index";
  const end = start === null ? samples.length : Math.min(samples.length, start + METRIC_WINDOW_SIZE);
  const begin = start ?? 0;
  let previous: number | undefined;
  let valid = 0;
  const data: [number, number][] = [];
  for (let ordinal = begin; ordinal < end; ordinal++) {
    const { point, index } = samples[ordinal];
    const value = finiteNumber(point[metric]);
    const coordinate = timeAxis ? finiteNumber(point.ts) : hasSteps ? finiteNumber(point.step ?? point.current) : index + 1;
    const x = coordinate === undefined ? undefined : timeAxis ? coordinate * 1000 : coordinate;
    if (value === undefined || x === undefined) {
      previous = undefined;
      if (data.length && Number.isFinite(data.at(-1)![1])) data.push([x ?? data.at(-1)![0] + 1, NaN]);
      continue;
    }
    previous = previous === undefined ? value : smoothing * previous + (1 - smoothing) * value;
    data.push([x, previous]); valid++;
  }
  return { data: start === null ? adaptiveMetricData(data) : data, axis, count: Math.max(0, end - begin), valid, loaded: samples.length };
}
