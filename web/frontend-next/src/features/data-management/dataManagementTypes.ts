export type DataEntryKind = "file" | "directory" | "symlink" | "other" | "missing";
export type DataEntryStatus = "ready" | "missing" | "unavailable" | (string & {});
export type DataAccess = "read_only" | "cleanup" | (string & {});
export type DataSizeState = "ready" | "partial" | "unavailable" | "pending" | (string & {});

export type DataEntry = {
  relative_path: string;
  name: string;
  kind: DataEntryKind;
  size_bytes: number | null;
  modified_at: string | null;
  cleanup_allowed: boolean;
  cleanup_reason: string;
  status: DataEntryStatus;
};

export type DataRoot = {
  id: string;
  label: string;
  path: string;
  access: DataAccess;
  size_bytes: number | null;
  size_state: DataSizeState;
  file_count: number | null;
  directory_count: number | null;
  entries?: DataEntry[];
  truncated?: boolean;
  cache_state?: string;
};

export type DataRootsResponse = {
  ok: boolean;
  generated_at?: string;
  roots: DataRoot[];
};

export type DataEntriesResponse = {
  ok: boolean;
  generated_at?: string;
  root_id?: string;
  path?: string;
  entries: DataEntry[];
  file_count?: number | null;
  directory_count?: number | null;
  size_bytes?: number | null;
  size_state?: DataSizeState;
  cache_state?: string;
  truncated?: boolean;
};

export type CleanupPlanItem = {
  relative_path: string;
  name: string;
  path: string;
  kind: DataEntryKind;
  size_bytes: number | null;
  modified_at: string | null;
};

export type CleanupBlockedItem = {
  relative_path?: string;
  name?: string;
  path?: string;
  reason?: string;
} | string;

export type CleanupPreviewResponse = {
  ok: boolean;
  dry_run: true;
  root_id: string;
  items: CleanupPlanItem[];
  total_size_bytes: number | null;
  item_count: number;
  blocked: CleanupBlockedItem[];
  plan_revision: string;
};

export type CleanupResponse = {
  ok: boolean;
  dry_run: false;
  deleted: Array<CleanupPlanItem | string>;
  missing: Array<CleanupPlanItem | string>;
  blocked: CleanupBlockedItem[];
  errors: Record<string, string>;
  message?: string;
};
