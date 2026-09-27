import { apiRequest } from "../../api/client";

export type WeightCandidate = { file: string; abs_path?: string; name: string; mtime_text?: string; size_bytes?: number };
export type WeightListing = { weights: WeightCandidate[]; count: number; message?: string; analysis_note?: string };
export type AnalysisRow = { label?: string; name?: string; component?: string; block?: number | null; fro_norm?: number; contribution?: number; score?: number; layer_count?: number; reason?: string };
export type WeightHeatmapData = {
  blocks: number[];
  components: string[];
  matrix: number[][];
  max_value: number;
  cells: { block: number; component: string; fro_norm: number; layer_count: number; top_layer: string; intensity: number }[];
};
export type WeightResult = { ok: boolean; file: { name: string; path: string }; adapter_type: string; unsupported?: { unsupported?: boolean; reason?: string }; disclaimer?: string; summary: { layer_count?: number; block_count?: number; total_energy?: number; total_param_count?: number; top_layer?: string }; layers: AnalysisRow[]; component_summary: AnalysisRow[]; block_summary: AnalysisRow[]; style_top20: AnalysisRow[]; character_top20: AnalysisRow[]; heatmap?: WeightHeatmapData };
export const fetchWeightCandidates = (signal?: AbortSignal) => apiRequest<WeightListing>("/api/analysis/weights", { signal });
export const inspectWeight = (path: string) => apiRequest<WeightResult>("/api/analysis/inspect", { method: "POST", body: JSON.stringify({ path }) });
export function inspectWeightFile(file: File) { const body = new FormData(); body.append("file", file); return apiRequest<WeightResult>("/api/analysis/inspect-upload", { method: "POST", body }); }
