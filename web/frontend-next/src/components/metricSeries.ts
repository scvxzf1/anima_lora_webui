import { finiteNumber } from "./trainingNumbers";
import { isTrainingMetric } from "./metricSemantics";

export function metricSeries(points: Record<string, unknown>[], metric: string, limit: number, smoothing: number, timeAxis: boolean) {
  const window = limit ? points.slice(-limit) : points;
  const offset = points.length - window.length;
  const hasSteps = window.some((point) => finiteNumber(point.step ?? point.current) !== undefined);
  const axis = timeAxis ? "time" : hasSteps ? "step" : "index";
  let previous: number | undefined;
  const data: [number, number][] = [];
  window.forEach((point, index) => {
    if ((metric === "loss" || metric === "lr") && !isTrainingMetric(point)) return;
    const value = finiteNumber(point[metric]);
    const x = timeAxis ? finiteNumber(point.ts) : hasSteps ? finiteNumber(point.step ?? point.current) : offset + index + 1;
    if (value === undefined || x === undefined) return;
    previous = previous === undefined ? value : smoothing * previous + (1 - smoothing) * value;
    data.push([timeAxis ? x * 1000 : x, previous]);
  });
  return { data, axis, count: window.length };
}

export function metricRange(data: [number, number][], axis: string) {
  if (!data.length) return "";
  const show = (value: number) => axis === "time" ? new Date(value).toLocaleString() : String(value);
  const name = axis === "time" ? "时间" : axis === "step" ? "步数" : "采样序号";
  return `${name} ${show(data[0][0])} - ${show(data[data.length - 1][0])}`;
}
