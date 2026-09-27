import { apiRequest } from '../../api/client';
import type { DatasetPreviewResponse } from '../dataset-editor/types';

export type MaskPage = Omit<DatasetPreviewResponse, 'images'> & {
  images: Array<DatasetPreviewResponse['images'][number] & { has_mask: boolean; thumbnail_url?: string }>;
  mask_dir: string;
  mask_mode: string;
  config_revision: string;
  readonly: boolean;
  offset: number;
  next_offset: number;
  has_more_after: boolean;
};
export type MaskImage = {
  ok: true; revision: string; width: number; height: number;
  image_url: string; mask_url: string; has_mask: boolean; readonly: boolean;
  basis: 'training' | 'projected'; mask_dir: string; mask_file: string;
};
export function maskUrl(file: string, index: number, suffix = '', extra: Record<string, string> = {}) {
  return `/api/config/dataset-masks${suffix}?${new URLSearchParams({ file, dataset_index: String(index), ...extra })}`;
}
export function fetchMasks(file: string, index: number, offset: number, signal?: AbortSignal) {
  return apiRequest<MaskPage>(maskUrl(file, index, '', { offset: String(offset) }), { signal });
}
export function fetchMask(file: string, index: number, image: string, signal?: AbortSignal) {
  return apiRequest<MaskImage>(maskUrl(file, index, '/image', { image }), { signal });
}
export function saveMask(file: string, index: number, image: string, revision: string, body: Blob) {
  return apiRequest<{ ok: true; revision: string }>(maskUrl(file, index, '/image', { image }), {
    method: 'PUT', headers: { 'Content-Type': 'image/png', 'If-Match': revision }, body,
  });
}
export function applyMasks(
  file: string,
  index: number,
  revision: string,
  indices: number[] = [index],
) {
  return apiRequest<{ ok: true; message: string }>(maskUrl(file, index, '/apply'), {
    method: 'POST', headers: { 'If-Match': revision },
    body: JSON.stringify({ indices }),
  });
}
