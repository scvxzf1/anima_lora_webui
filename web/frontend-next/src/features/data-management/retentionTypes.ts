export type RetentionSortBy = "progress" | "mtime" | "size";
export type RetentionSortOrder = "desc" | "asc";
export type RetentionMode = "keep_top_n" | "delete_top_m" | "keep_first_last" | "delete_all_checkpoints";
export type RetentionAction = "keep" | "delete";
export type RetentionRole =
  | "checkpoint"
  | "final_weight"
  | "optimizer_state"
  | "data_cache"
  | "analysis_tensor"
  | "protected"
  | "other"
  | "symlink";

export type RetentionPolicy = {
  sort_by: RetentionSortBy;
  sort_order: RetentionSortOrder;
  mode: RetentionMode;
  keep_count: number;
  delete_count: number;
  keep_first: number;
  keep_last: number;
  keep_final_weights: boolean;
  keep_weight_patterns: string[];
  ensure_at_least_one_weight: boolean;
  clean_optimizer_states: boolean;
  clean_data_cache: boolean;
  clean_analysis_tensors: boolean;
};

export type RetentionItem = {
  relative_path: string;
  name: string;
  kind: string;
  role: RetentionRole;
  role_label: string;
  size_bytes: number;
  modified_at: string | null;
  progress: number | null;
  action: RetentionAction;
  reason: string;
  rank: number | null;
};

export type RetentionGroup = {
  relative_path: string;
  name: string;
  checkpoint_count: number;
  keep_count: number;
  delete_count: number;
  keep_size_bytes: number;
  delete_size_bytes: number;
  items: RetentionItem[];
  truncated: boolean;
  excluded: boolean;
};

export type RetentionBlockedItem = { relative_path: string; reason: string };

export type RetentionSummary = {
  scope_dirs: number;
  group_count: number;
  matched_group_count: number;
  keep_count: number;
  delete_count: number;
  keep_size_bytes: number;
  delete_size_bytes: number;
  blocked_count: number;
  excluded_group_count: number;
  truncated: boolean;
  scanned_files: number;
};

export type RetentionPreviewResponse = {
  ok: boolean;
  dry_run: true;
  root_id: string;
  path: string;
  policy: RetentionPolicy;
  scope: { dirs: string[]; truncated: boolean; scanned_files: number; scanned_dirs: number };
  groups: RetentionGroup[];
  summary: RetentionSummary;
  blocked: RetentionBlockedItem[];
  exclude_units: string[];
  plan_revision: string;
  generated_at: string;
};

export type RetentionCleanupResponse = {
  ok: boolean;
  dry_run: false;
  deleted: string[];
  missing: string[];
  blocked: RetentionBlockedItem[] | Array<Record<string, unknown>>;
  errors: Record<string, string>;
  message?: string;
};

export type RetentionScopeOption = {
  id: string;
  label: string;
  rootId: string;
  dirs: string[];
};

export const DEFAULT_RETENTION_POLICY: RetentionPolicy = {
  sort_by: "progress",
  sort_order: "desc",
  mode: "keep_top_n",
  keep_count: 2,
  delete_count: 3,
  keep_first: 1,
  keep_last: 1,
  keep_final_weights: true,
  keep_weight_patterns: [],
  ensure_at_least_one_weight: true,
  clean_optimizer_states: true,
  clean_data_cache: false,
  clean_analysis_tensors: false,
};
