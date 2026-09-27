import { ApiError, apiRequest } from "../../api/client";
export type Check = { level: string; key: string; message: string; group?: string; path?: string; hint?: string; detail?: string };
export type EnvironmentReport = { ok: boolean; platform: Record<string, unknown>; summary: Record<string, number>; checks: Check[]; groups?: string[]; errors?: Check[]; warnings?: Check[] };
export async function fetchEnvironment(signal?: AbortSignal) {
  try { return await apiRequest<EnvironmentReport>("/api/environment/check", { signal }); }
  catch (error) {
    if (error instanceof ApiError && error.status === 200 && typeof error.payload === "object" && error.payload && "checks" in error.payload) return error.payload as EnvironmentReport;
    throw error;
  }
}
