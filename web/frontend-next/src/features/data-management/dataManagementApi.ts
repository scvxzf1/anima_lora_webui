import { apiRequest } from "../../api/client";
import type {
  CleanupPreviewResponse,
  CleanupResponse,
  DataEntriesResponse,
  DataRootsResponse,
} from "./dataManagementTypes";

const ROOT = "/api/data-management";

export function fetchDataRoots(signal?: AbortSignal, refresh = false) {
  const params = refresh ? "?refresh=true" : "";
  return apiRequest<DataRootsResponse>(`${ROOT}/roots${params}`, { signal });
}

export function fetchDataRootsMeta(signal?: AbortSignal) {
  return apiRequest<DataRootsResponse>(`${ROOT}/roots?mode=meta`, { signal });
}

export function fetchDataEntries(rootId: string, path: string, signal?: AbortSignal, refresh = false) {
  const params = new URLSearchParams({ root_id: rootId });
  if (path) params.set("path", path);
  if (refresh) params.set("refresh", "true");
  return apiRequest<DataEntriesResponse>(`${ROOT}/entries?${params}`, { signal });
}

export function previewCleanup(rootId: string, items: string[]) {
  return apiRequest<CleanupPreviewResponse>(`${ROOT}/cleanup/preview`, {
    method: "POST",
    body: JSON.stringify({ root_id: rootId, items }),
  });
}

export function cleanupData(
  rootId: string,
  items: string[],
  expectedItems: string[],
  planRevision: string,
) {
  return apiRequest<CleanupResponse>(`${ROOT}/cleanup`, {
    method: "POST",
    body: JSON.stringify({
      root_id: rootId,
      items,
      expected_items: expectedItems,
      plan_revision: planRevision,
      confirmed: true,
    }),
  });
}
