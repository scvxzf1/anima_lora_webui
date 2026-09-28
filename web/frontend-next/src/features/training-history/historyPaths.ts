export type HistoryPathTask = Record<string, unknown>;

export type ResolvedHistoryPath = {
  value: string;
  absolute: boolean;
};

const PROJECT_RELATIVE_ROOTS = /^(?:configs|image_dataset|library|logs|models|networks|output|post_image_dataset|scripts|tests|web)(?:[\\/]|$)/;
const HISTORY_RELATIVE_FIELDS = new Set(["logs_path", "metrics_path", "system_path", "config_snapshot"]);
const RUN_RELATIVE_FIELDS = new Set(["runtime_config_file", "original_config_file", "dataset_config_file"]);

export function resolveHistoryPath(task: HistoryPathTask, field: string, fallback?: unknown): ResolvedHistoryPath {
  const raw = String(task[field] || fallback || "").trim();
  if (!raw) return { value: "", absolute: false };
  if (isAbsolutePath(raw)) return { value: raw, absolute: true };

  const clean = raw.replace(/^(?:\.\/|\.\\)+/, "").replace(/^[\\/]+/, "");
  if (!clean) return { value: raw, absolute: false };

  if (HISTORY_RELATIVE_FIELDS.has(field)) {
    return joinKnownBase(task.history_dir_abs, clean, raw);
  }
  if (RUN_RELATIVE_FIELDS.has(field)) {
    return joinKnownBase(task.run_dir_abs, clean, raw);
  }
  if (field === "run_dir_abs" || field === "history_dir_abs") {
    return joinKnownBase(task.project_root_abs, clean, raw);
  }
  if (PROJECT_RELATIVE_ROOTS.test(clean)) {
    return joinKnownBase(task.project_root_abs, clean, raw);
  }
  return { value: raw, absolute: false };
}

function joinKnownBase(baseValue: unknown, child: string, fallback: string): ResolvedHistoryPath {
  const base = String(baseValue || "").trim().replace(/[\\/]+$/, "");
  if (!isAbsolutePath(base)) return { value: fallback, absolute: false };
  const separator = base.includes("\\") && !base.includes("/") ? "\\" : "/";
  return { value: `${base}${separator}${child.replace(/[\\/]+/g, separator)}`, absolute: true };
}

function isAbsolutePath(value: string): boolean {
  return /^(?:[A-Za-z]:[\\/]|\\\\|\/)/.test(value);
}
