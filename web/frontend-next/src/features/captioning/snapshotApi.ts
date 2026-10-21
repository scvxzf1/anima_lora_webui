import { apiRequest } from "../../api/client";

const root = "/api/captioning";

export type CaptionSnapshot = {
  id: string;
  created_at: string;
  created_at_ms: number;
  reason: string;
  label: string;
  dataset_file: string;
  dataset_index: number;
  source: string;
  scope_dir: string;
  file_count: number;
  bytes: number;
  broken?: boolean;
};

export type CaptionSnapshotRestore = {
  ok: true;
  id: string;
  written: number;
  removed: number;
  skipped: string[];
  snapshot?: CaptionSnapshot;
};

export const snapshotKeys = {
  list: (datasetFile: string, datasetIndex: number) =>
    ["captioning", "snapshots", datasetFile, datasetIndex] as const,
};

export const fetchCaptionSnapshots = (
  datasetFile: string,
  datasetIndex: number,
  signal?: AbortSignal,
) => {
  const query = new URLSearchParams({
    dataset_file: datasetFile,
    dataset_index: String(datasetIndex),
  });
  return apiRequest<{ ok: true; snapshots: CaptionSnapshot[] }>(
    `${root}/caption-snapshots?${query.toString()}`,
    { signal },
  );
};

export const createCaptionSnapshot = (payload: {
  dataset_file: string;
  dataset_index: number;
  source: string;
  label?: string;
}) =>
  apiRequest<{ ok: true; snapshot: CaptionSnapshot }>(
    `${root}/caption-snapshots`,
    { method: "POST", body: JSON.stringify(payload) },
  );

export const restoreCaptionSnapshot = (id: string) =>
  apiRequest<CaptionSnapshotRestore>(
    `${root}/caption-snapshots/${encodeURIComponent(id)}/restore`,
    { method: "POST" },
  );

export const deleteCaptionSnapshot = (id: string) =>
  apiRequest<{ ok: true; deleted: string }>(
    `${root}/caption-snapshots/${encodeURIComponent(id)}`,
    { method: "DELETE" },
  );
