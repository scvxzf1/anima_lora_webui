import { apiRequest } from "../../api/client";
import type {
  RetentionCleanupResponse,
  RetentionPolicy,
  RetentionPreviewResponse,
} from "./retentionTypes";

const ROOT = "/api/data-management/retention";

export function previewRetention(
  rootId: string,
  dirs: string[],
  policy: RetentionPolicy,
  excludeUnits: string[],
  signal?: AbortSignal,
) {
  return apiRequest<RetentionPreviewResponse>(`${ROOT}/preview`, {
    method: "POST",
    body: JSON.stringify({ root_id: rootId, dirs, policy, exclude_units: excludeUnits }),
    signal,
  });
}

export function cleanupRetention(
  rootId: string,
  dirs: string[],
  policy: RetentionPolicy,
  expectedItems: string[],
  planRevision: string,
  excludeUnits: string[],
) {
  return apiRequest<RetentionCleanupResponse>(`${ROOT}/cleanup`, {
    method: "POST",
    body: JSON.stringify({
      root_id: rootId,
      dirs,
      policy,
      expected_items: expectedItems,
      plan_revision: planRevision,
      exclude_units: excludeUnits,
      confirmed: true,
    }),
  });
}
