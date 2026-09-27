import { ApiError, apiRequest } from "../../api/client";

export type ImageStatus = { ok: boolean; status: string; running: boolean; error?: string; output_dir?: string; output_count?: number; started_at_text?: string; finished_at_text?: string; logs?: string[]; last_request?: Record<string, unknown> };
export type ImageItem = { file: string; name: string; url: string; mtime_text?: string; width?: number; height?: number };
export type ImageListing = { ok: boolean; images: ImageItem[]; total?: number; message?: string };
export type WeightItem = { file: string; abs_path?: string; name: string; mtime_text?: string };
export const imageKeys = { status: ["image-test", "status"] as const, weights: ["image-test", "weights"] as const, images: (days: string) => ["image-test", "images", days] as const };
export async function fetchImageStatus(signal?: AbortSignal) {
  try { return await apiRequest<ImageStatus>("/api/image-test/status", { signal }); }
  catch (error) {
    if (error instanceof ApiError && error.status === 200 && typeof error.payload === "object" && error.payload && "status" in error.payload) return error.payload as ImageStatus;
    throw error;
  }
}
export const fetchImageWeights = (signal?: AbortSignal) => apiRequest<{ weights: WeightItem[] }>("/api/analysis/weights", { signal });
export const fetchImages = (days: string, signal?: AbortSignal) => apiRequest<ImageListing>(`/api/preview/images?${new URLSearchParams({ source: "inference", limit: "100", ...(days === "all" ? {} : { days }) })}`, { signal });
export const startImage = (payload: Record<string, unknown>) => apiRequest<ImageStatus>("/api/image-test/start", { method: "POST", body: JSON.stringify(payload) });
export const stopImage = () => apiRequest<ImageStatus>("/api/image-test/stop", { method: "POST" });
export const deleteImages = (files: string[]) => apiRequest<{ deleted_count: number }>("/api/image-test/images", { method: "DELETE", body: JSON.stringify({ files }) });
