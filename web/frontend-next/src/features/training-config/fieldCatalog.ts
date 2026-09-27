import {
  FIELD_LABEL_ZH,
  FIELD_OPTIONS,
} from "../../../../static/js/config/catalog/labels-options.js";
import { CONFIG_FIELD_CATALOG } from "../../../../static/js/dragon-ui/pages/config-field-catalog.js";
import { displayConfigValue } from "../../../../static/js/dragon-ui/pages/config-values.js";
import {
  configFieldAvailability,
  resolveConfigAdapterKind,
} from "../../../../static/js/dragon-ui/pages/config-field-availability.js";
import {
  configureModelFamilyCapabilities,
  modelFamilyOptionSupported,
} from "../../../../static/js/features/config-form/model-family.js?v=qwen-image-21-v2";
import { apiRequest } from "../../api/client";
import { TRAINING_FIELDS, type TrainingFieldSpec } from "./trainingForm";

// Reuse metadata only; none of the legacy DOM runtime is mounted in React.
export function fieldsForConfig(
  config: Record<string, unknown>,
): TrainingFieldSpec[] {
  const fields = new Map(TRAINING_FIELDS.map((field) => [field.key, field]));
  for (const key of new Set([
    ...Object.keys(CONFIG_FIELD_CATALOG),
    ...Object.keys(config),
  ])) {
    if (fields.has(key)) continue;
    if (key === "general" || key === "datasets" || key.includes(".")) continue;
    const value = displayConfigValue(key, config);
    const options = FIELD_OPTIONS[key];
    fields.set(key, {
      key,
      label: FIELD_LABEL_ZH[key] || key,
      group: CONFIG_FIELD_CATALOG[key]?.stage || "training",
      kind:
        typeof value === "boolean"
          ? "boolean"
          : typeof value === "number"
            ? "number"
            : value !== null && typeof value === "object"
              ? "json"
              : options?.every((v) => typeof v === "string")
                ? "select"
                : "text",
      options: options?.map(String),
      step: typeof value === "number" ? "any" : undefined,
    });
  }
  return [...fields.values()].map((field) => {
    if (CONFIG_FIELD_CATALOG[field.key]?.cluster === "preprocess") field = { ...field, group: "input" };
    const options = FIELD_OPTIONS[field.key];
    return field.kind === "select" &&
      options?.every((value) => typeof value === "string")
      ? { ...field, options: options as string[] }
      : field;
  });
}

export async function fetchFieldCapabilities(signal?: AbortSignal) {
  const payload = await apiRequest<{ items: unknown[] }>(
    "/api/config/model-families",
    { signal },
  );
  configureModelFamilyCapabilities(payload);
  return payload;
}

export function fieldAvailability(
  key: string,
  values: Record<string, unknown>,
  method: string,
) {
  return configFieldAvailability(key, {
    values,
    method,
    adapter: resolveConfigAdapterKind(values),
    modelFamily: values.model_family || "anima",
    baseCompute: values.base_compute || "bf16",
    maxTrainEpochsConfigured: Number(values.max_train_epochs) > 0,
    dimFromWeights: Boolean(values.dim_from_weights),
    networkWeights: String(values.network_weights || ""),
  });
}

export function availableFieldOptions(
  field: TrainingFieldSpec,
  family: string,
) {
  return field.options?.filter((option) =>
    modelFamilyOptionSupported(field.key, family, option),
  );
}
