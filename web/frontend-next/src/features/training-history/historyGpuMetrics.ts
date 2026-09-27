import { finiteNumber } from "../../components/trainingNumbers";

export type HistoryGpuDevice = { key: string; index: number; name: string };

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
