import { apiRequest } from "../../api/client";

export type SampleReference = { ok: true; reference_image: string; width: number; height: number; url: string };
const absolutePath = (path: string) => /^(?:\/|[A-Za-z]:[\\/])/.test(path.trim());

export function managedReferenceUrl(path: string) {
  const key = path.replace(/\\/g, "/").split("/").pop()?.match(/^([a-f0-9]{64})\.png$/)?.[1];
  return key ? `/api/config/sample-references/${key}` : "";
}

export async function importSampleReference(source: File | string, signal?: AbortSignal) {
  if (typeof source === "string" && !absolutePath(source)) {
    throw new Error("请输入服务器上的绝对路径。");
  }
  const body = typeof source === "string" ? JSON.stringify({ path: source.trim() }) : new FormData();
  if (body instanceof FormData) body.append("file", source);
  const result = await apiRequest<SampleReference>("/api/config/sample-references", { method: "POST", body, signal });
  if (typeof result.reference_image !== "string" || !absolutePath(result.reference_image) ||
    typeof result.url !== "string" || !/^\/api\/config\/sample-references\/[a-f0-9]{64}$/.test(result.url) ||
    !Number.isSafeInteger(result.width) || !Number.isSafeInteger(result.height) || result.width <= 0 || result.height <= 0) throw new Error("参考图响应不完整。");
  return result;
}
