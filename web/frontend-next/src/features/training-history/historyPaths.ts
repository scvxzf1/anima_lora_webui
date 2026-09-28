export type HistoryPathTask = Record<string, unknown>;

export type ResolvedHistoryPath = {
  value: string;
  absolute: boolean;
};

export function resolveHistoryPath(task: HistoryPathTask, field: string, fallback?: unknown): ResolvedHistoryPath {
  const raw = String(task[field] || fallback || "").trim();
  if (!raw) return { value: "", absolute: false };
  const absolutePaths = task.absolute_paths;
  const resolved = absolutePaths && typeof absolutePaths === "object"
    ? String((absolutePaths as Record<string, unknown>)[field] || "").trim()
    : "";
  if (resolved) return { value: resolved, absolute: true };
  return { value: raw, absolute: /^(?:[A-Za-z]:[\\/]|\\\\|\/)/.test(raw) };
}
