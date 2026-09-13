import { finiteNumber } from "../../components/trainingNumbers";
import { isTrainingMetric } from "../../components/metricSemantics";
import type { HistoryTaskDetail } from "./api";
import { snapshotFields } from "./historySummary";

export type ComparisonEntry = { id: string; name: string; detail?: HistoryTaskDetail };
export type ComparisonLine = { id: string; name: string; data: [number, number][]; loaded: number; total?: number };

function stableValue(value: unknown): unknown {
  if (value instanceof Date) return value.toISOString();
  if (Array.isArray(value)) return value.map(stableValue);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).sort(([a], [b]) => a.localeCompare(b)).map(([key, item]) => [key, stableValue(item)]));
  }
  return value;
}

export function comparisonParameters(entries: ComparisonEntry[]) {
  const snapshots = entries.map(({ detail }) => detail ? snapshotFields(detail) : { values: {}, invalid: false });
  const keys = [...new Set(snapshots.flatMap(({ values }) => Object.keys(values)))].sort();
  return keys.map((key) => {
    const values = snapshots.map(({ values }) => Object.hasOwn(values, key) ? JSON.stringify(stableValue(values[key])) : undefined);
    return { key, values, different: new Set(values).size > 1 };
  });
}

export function comparisonLines(entries: ComparisonEntry[], overlap: boolean) {
  const lines: ComparisonLine[] = entries.map(({ id, name, detail }) => {
    const metrics = detail?.task?.job === "training" ? detail.metrics || [] : [];
    const data: [number, number][] = [];
    for (const point of metrics) {
      if (!isTrainingMetric(point)) continue;
      const step = finiteNumber(point.step) ?? finiteNumber(point.current);
      const loss = finiteNumber(point.loss);
      if (step !== undefined && step >= 0 && loss !== undefined) data.push([step, loss]);
    }
    data.sort((a, b) => a[0] - b[0]);
    return { id, name, data, loaded: metrics.length, total: finiteNumber(detail?.limits?.metrics_total) };
  });
  const valid = lines.filter(({ data }) => data.length);
  const starts = valid.map(({ data }) => data[0][0]);
  const ends = valid.map(({ data }) => data[data.length - 1][0]);
  const start = valid.length ? (overlap ? Math.max(...starts) : Math.min(...starts)) : undefined;
  const end = valid.length ? (overlap ? Math.min(...ends) : Math.max(...ends)) : undefined;
  const windowed = lines.map((line) => ({ ...line, data: line.data.filter(([step]) => start !== undefined && end !== undefined && step >= start && step <= end) }));
  return { lines: windowed, start, end, hasOverlap: start !== undefined && end !== undefined && start <= end };
}
