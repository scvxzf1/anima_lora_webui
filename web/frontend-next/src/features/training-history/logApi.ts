import { apiRequest } from "../../api/client";

export const LOG_PAGE_SIZE = 400;
export type LogRecord = { line?: string; message?: string; text?: string; type?: string };
export type LogPage = { logs: LogRecord[]; indices?: number[]; offset: number; total: number };
export type LogMatch = { match_index: number | null; match_ordinal: number; matches_total: number; total: number };
export const logUrl = (taskId: string) => `/api/training/history/${encodeURIComponent(taskId)}/logs`;
export function fetchLogPage(taskId: string, offset: number | undefined, limit: number, signal?: AbortSignal) {
  const params = new URLSearchParams({ limit: String(limit) });
  if (offset !== undefined) params.set("offset", String(offset));
  return apiRequest<LogPage>(`${logUrl(taskId)}?${params}`, { signal });
}
export function searchLogs(taskId: string, query: string, cursor: number, direction: string, signal?: AbortSignal) {
  const params = new URLSearchParams({ query, cursor: String(cursor), direction });
  return apiRequest<LogMatch>(`${logUrl(taskId)}/search?${params}`, { signal });
}
export const logText = (record: LogRecord) => String(record.line ?? record.message ?? record.text ?? "").replace(/[\r\n]+/g, " ↵ ");
