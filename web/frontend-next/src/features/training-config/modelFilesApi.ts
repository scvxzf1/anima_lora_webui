import { apiRequest, ApiError } from "../../api/client";

export type ModelFileEntry = {
  name: string;
  path: string;
  kind: "directory" | "file";
};
export type ModelFilesPage = {
  ok: true;
  roots: { name: string; path: string }[];
  directory: string;
  parent: string | null;
  entries: ModelFileEntry[];
  offset: number;
  next_offset: number | null;
  total: number;
  truncated?: boolean;
};

export function loadModelFiles(
  directory: string | undefined,
  offset: number,
  signal: AbortSignal,
) {
  const params = new URLSearchParams({ offset: String(offset), limit: "100" });
  if (directory) params.set("directory", directory);
  return apiRequest<ModelFilesPage>(`/api/settings/model-files?${params}`, {
    signal,
  });
}

export function modelFilesError(error: Error) {
  return error instanceof ApiError && [404, 501].includes(error.status)
    ? "目录浏览暂不可用，可手动填写路径或从模型库选择。"
    : `无法读取目录：${error.message}`;
}
