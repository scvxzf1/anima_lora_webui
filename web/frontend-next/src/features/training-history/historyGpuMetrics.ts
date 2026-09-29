import { finiteNumber } from "../../components/trainingNumbers";

export type HistoryGpuDevice = { key: string; index: number; name: string };

export const HISTORY_GPU_METRICS = [
  { key: "vram_used_gb", label: "显存占用", unit: "GB" },
  { key: "vram_total_gb", label: "总显存", unit: "GB" },
  { key: "gpu_util", label: "GPU 利用率", unit: "%" },
  { key: "gpu_temp", label: "GPU 温度", unit: "°C" },
] as const;

export type HistoryGpuMetricKey = typeof HISTORY_GPU_METRICS[number]["key"];
export type HistoryGpuSummary = Record<HistoryGpuMetricKey, { latest?: number; peak?: number }>;

export const HISTORY_GPU_MAX_POINTS = 1000;

export function historyGpuRecentPoints(points: Record<string, unknown>[]): Record<string, unknown>[] {
  return points.slice(-HISTORY_GPU_MAX_POINTS);
}

export function historyGpuSummary(points: Record<string, unknown>[]): HistoryGpuSummary {
  const summary = Object.fromEntries(HISTORY_GPU_METRICS.map(({ key }) => [key, {}])) as HistoryGpuSummary;
  for (const { key } of HISTORY_GPU_METRICS) {
    const values = points.map((point) => finiteNumber(point[key])).filter((value): value is number => value !== undefined);
    if (!values.length) continue;
    summary[key] = { latest: values[values.length - 1], peak: Math.max(...values) };
  }
  return summary;
}

export function historyGpuInspectionIndices(points: Record<string, unknown>[]): number[] {
  const byKey = new Map<string, number>();
  points.forEach((point, index) => {
    if (HISTORY_GPU_METRICS.some(({ key }) => finiteNumber(point[key]) !== undefined)) {
      byKey.set(historyGpuSampleKey(point), index);
    }
  });
  return [...byKey.values()].sort((a, b) => a - b);
}

export function historyGpuSampleKey(point: Record<string, unknown>): string {
  return JSON.stringify([point.ts ?? point.timestamp ?? null, ...HISTORY_GPU_METRICS.map(({ key }) => finiteNumber(point[key]) ?? null)]);
}

export function historyGpuSampleTime(point: Record<string, unknown>, index: number): string {
  const timestamp = finiteNumber(point.ts ?? point.timestamp);
  if (timestamp === undefined) return `采样 ${index + 1}`;
  const milliseconds = Math.abs(timestamp) < 1e12 ? timestamp * 1000 : timestamp;
  return `${new Date(milliseconds).toLocaleString()} · 采样 ${index + 1}`;
}

export function formatHistoryGpuValue(value: number | undefined, unit: string): string {
  if (value === undefined) return "—";
  const formatted = unit === "GB" ? value.toFixed(2).replace(/\.00$/, "").replace(/(\.\d)0$/, "$1") : String(Math.round(value));
  return `${formatted} ${unit}`;
}

function gpuRows(point: Record<string, unknown>): Record<string, unknown>[] {
  return Array.isArray(point.per_gpu)
    ? point.per_gpu.filter((row): row is Record<string, unknown> => Boolean(row) && typeof row === "object" && !Array.isArray(row))
    : [];
}

function gpuKey(row: Record<string, unknown>): string | undefined {
  const uuid = typeof row.uuid === "string" ? row.uuid.trim() : "";
  if (uuid) return `uuid:${uuid}`;
  const index = finiteNumber(row.index);
  return index !== undefined && Number.isInteger(index) && index >= 0 ? `index:${index}` : undefined;
}

export function historyGpuDevices(points: Record<string, unknown>[]): HistoryGpuDevice[] {
  const devices = new Map<string, HistoryGpuDevice>();
  for (const point of points) {
    for (const row of gpuRows(point)) {
      const key = gpuKey(row);
      const index = finiteNumber(row.index);
      if (!key || index === undefined || !Number.isInteger(index) || index < 0) continue;
      devices.set(key, { key, index, name: typeof row.name === "string" && row.name.trim() ? row.name : `GPU ${index}` });
    }
  }
  return [...devices.values()].sort((a, b) => a.index - b.index || a.key.localeCompare(b.key));
}

export function historyGpuChartPoints(points: Record<string, unknown>[], deviceKey?: string): Record<string, unknown>[] {
  return points.map((point) => {
    const ts = point.ts ?? point.timestamp;
    if (!deviceKey) return { ...point, ts };
    const row = gpuRows(point).find((gpu) => gpuKey(gpu) === deviceKey);
    return row ? { ...row, ts } : { ts };
  });
}
