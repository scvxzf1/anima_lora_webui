export function finiteNumber(value: unknown): number | undefined {
  if (typeof value !== "number" && typeof value !== "string") return undefined;
  if (typeof value === "string" && !value.trim()) return undefined;
  const number = Number(value);
  return Number.isFinite(number) ? number : undefined;
}

export function latestNumber(points: Record<string, unknown>[], key: string) {
  for (let index = points.length - 1; index >= 0; index--) {
    const number = finiteNumber(points[index][key]);
    if (number !== undefined) return number;
  }
  return undefined;
}

export function formatLoss(value: unknown) {
  const number = finiteNumber(value);
  if (number === undefined) return "未记录";
  return number !== 0 && Math.abs(number) < 0.0001
    ? number.toExponential(2)
    : number.toFixed(4);
}

export function formatLearningRate(value: unknown) {
  const number = finiteNumber(value);
  return number === undefined ? "未记录" : number === 0 ? "0" : number.toExponential(2);
}

export function formatStep(value: unknown) {
  const number = finiteNumber(value);
  return number === undefined || number < 0 ? "未记录" : number.toFixed(0);
}

export function formatDuration(start: unknown, end: unknown) {
  const from = finiteNumber(start);
  const to = finiteNumber(end);
  if (from === undefined || to === undefined || to < from) return "未记录";
  const seconds = Math.round(to - from);
  if (seconds < 60) return `${seconds} 秒`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)} 分 ${seconds % 60} 秒`;
  return `${Math.floor(seconds / 3600)} 时 ${Math.floor(seconds % 3600 / 60)} 分`;
}
