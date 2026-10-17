import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiRequest } from "../../api/client";
import type { TrainingConfigFile } from "../../api/trainingContext";
import { draftFromMerged, type TrainingFieldSpec } from "./trainingForm";
import { NETWORK_ARG_FIELD_MAP } from "./domain/defaults.js";

export function fetchTrainingInheritedDefaults(file: TrainingConfigFile, preset: string, signal?: AbortSignal) {
  const query = new URLSearchParams({
    variant: file.method || "lora",
    preset: preset || "default",
    methods_subdir: file.methods_subdir || "gui-methods",
  });
  return apiRequest<Record<string, unknown>>(`/api/config/merged?${query}`, { signal });
}

// Do not confuse displayConfigValue's UI fallbacks with values returned by the server.
export function inheritedDraft(config: Record<string, unknown>, fields: TrainingFieldSpec[]) {
  const args = new Set((Array.isArray(config.network_args) ? config.network_args : [])
    .filter((entry) => String(entry).indexOf("=") > 0)
    .map((entry) => String(entry).split("=", 1)[0].trim()).filter(Boolean));
  const knownFields = fields.filter(({ key, defaultValue }) => {
    if (Object.hasOwn(config, key) && config[key] != null && !(config[key] === "" && defaultValue !== undefined)) return true;
    if (key === "precision_preference") return ["bf16", "fp16", "no"].includes(String(config.mixed_precision)) || config.full_fp16 === true || config.full_bf16 === true;
    if (key === "lora_adapter_kind") return ["use_glora", "use_vera", "use_lokr", "use_loha"].some((flag) => config[flag] != null);
    const spec = NETWORK_ARG_FIELD_MAP.get(key);
    if (spec && args.has(spec.arg)) return true;
    return key === "sample_vae_decode" && config.qwen_sample_vae_decode != null && config.qwen_sample_vae_decode !== "";
  });
  return draftFromMerged(config, knownFields);
}

export function useTrainingInheritedDefaults(file: TrainingConfigFile | undefined, preset: string, fields: TrainingFieldSpec[]) {
  const query = useQuery({
    queryKey: ["training-config", "inherited", file?.path, file?.method || "lora", file?.methods_subdir || "gui-methods", preset],
    queryFn: ({ signal }) => fetchTrainingInheritedDefaults(file!, preset, signal),
    enabled: Boolean(file),
    retry: false,
    placeholderData: undefined,
  });
  return useMemo(() => query.isSuccess ? inheritedDraft(query.data, fields) : undefined, [query.isSuccess, query.data, fields]);
}
