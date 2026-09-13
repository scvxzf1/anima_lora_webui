import { apiRequest } from "../../api/client";

const root = "/api/captioning";
export type DownloadJob = {
  id: string;
  asset_id?: string;
  state: string;
  bytes_downloaded: number;
  total_bytes: number;
  current_file?: string;
  error?: string;
};
export type CaptionAsset = {
  id: string;
  label: string;
  repo_id: string;
  license: string;
  state: string;
  installed: boolean;
  total_size: number;
  requires_auth: boolean;
  auth_configured: boolean;
  auth_hint: string;
  download?: DownloadJob;
};
export type DictionaryStatus = {
  installed: boolean;
  state: string;
  source_name: string;
  entry_count: number;
  download_size: number;
  download?: DownloadJob;
};
export const assetKeys = {
  assets: ["captioning", "assets"] as const,
  dictionary: ["captioning", "dictionary"] as const,
};
export const fetchCaptionAssets = (signal?: AbortSignal) =>
  apiRequest<{ assets: CaptionAsset[]; downloads: DownloadJob[] }>(
    `${root}/model-assets`,
    { signal },
  );
export const fetchDictionary = (signal?: AbortSignal) =>
  apiRequest<DictionaryStatus>(`${root}/tag-dictionary`, { signal });
export const downloadAsset = (id: string) =>
  apiRequest<{ download: DownloadJob }>(
    `${root}/model-assets/${encodeURIComponent(id)}/download`,
    { method: "POST" },
  );
export const downloadDictionary = () =>
  apiRequest<{ download: DownloadJob }>(`${root}/tag-dictionary/download`, {
    method: "POST",
  });
export const cancelDownload = (id: string, dictionary = false) =>
  apiRequest<{ download: DownloadJob }>(
    `${root}/${dictionary ? "tag-dictionary/" : ""}downloads/${encodeURIComponent(id)}/cancel`,
    { method: "POST" },
  );
export const downloadActive = (state?: string) =>
  ["queued", "running", "downloading", "validating", "publishing"].includes(
    state || "",
  );
