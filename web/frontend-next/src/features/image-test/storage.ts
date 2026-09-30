export type ImageTestDraft = {
  prompt: string;
  negative_prompt: string;
  width: string;
  height: string;
  infer_steps: string;
  guidance_scale: string;
  seed: string;
  sampler: string;
  attn_mode: string;
  runtime_dtype: string;
  text_encoder_dtype: string;
  gpu_index: string;
  weight_path: string;
  lora_multiplier: string;
};

export const IMAGE_TEST_STORAGE_KEY = "dragon-next:image-test:draft:v2";
export const IMAGE_TEST_DEFAULT_DRAFT: ImageTestDraft = {
  prompt: "", negative_prompt: "", width: "1024", height: "1024", infer_steps: "28",
  guidance_scale: "4", seed: "", sampler: "euler", attn_mode: "flash", runtime_dtype: "bf16",
  text_encoder_dtype: "same", gpu_index: "", weight_path: "", lora_multiplier: "1",
};
export const IMAGE_TEST_HISTORY_RANGES = ["7", "14", "30", "all"] as const;
export type ImageTestHistoryRange = typeof IMAGE_TEST_HISTORY_RANGES[number];

const stringFields = Object.keys(IMAGE_TEST_DEFAULT_DRAFT) as (keyof ImageTestDraft)[];
export const IMAGE_TEST_CONFIG_FIELDS: (keyof ImageTestDraft)[] = ["width", "height", "infer_steps", "guidance_scale", "sampler", "attn_mode", "runtime_dtype"];
const optionValues: Partial<Record<keyof ImageTestDraft, readonly string[]>> = {
  sampler: ["euler", "er_sde", "lcm"],
  attn_mode: ["flash", "torch", "sdpa", "sageattn", "flex", "xformers"],
  runtime_dtype: ["bf16", "fp16", "fp32"],
  text_encoder_dtype: ["same", "bf16", "fp16", "fp32"],
};

export type ImageTestStoredState = {
  file_path: string;
  preset: string;
  draft: ImageTestDraft;
  history_range: ImageTestHistoryRange;
  dirty_fields: (keyof ImageTestDraft)[];
};

export function readImageTestState(): ImageTestStoredState {
  const fallback: ImageTestStoredState = {
    file_path: "", preset: "default", draft: { ...IMAGE_TEST_DEFAULT_DRAFT }, history_range: "7", dirty_fields: [],
  };
  try {
    const raw = localStorage.getItem(IMAGE_TEST_STORAGE_KEY);
    if (!raw) return fallback;
    const value: unknown = JSON.parse(raw);
    if (!value || typeof value !== "object" || (value as { version?: unknown }).version !== 2) return fallback;
    const source = value as Record<string, unknown>;
    const rawDraft = source.draft;
    const draft = { ...IMAGE_TEST_DEFAULT_DRAFT };
    const dirtyFields: (keyof ImageTestDraft)[] = [];
    if (rawDraft && typeof rawDraft === "object" && !Array.isArray(rawDraft)) {
      const fields = rawDraft as Record<string, unknown>;
      for (const key of stringFields) {
        const entry = fields[key];
        if (typeof entry !== "string" || entry.length > 100_000) continue;
        if (optionValues[key] && !optionValues[key]!.includes(entry)) continue;
        if (key === "gpu_index" && entry !== "" && !/^\d+$/.test(entry)) continue;
        draft[key] = entry;
      }
    }
    const dirty = Array.isArray(source.dirty_fields) ? source.dirty_fields : [];
    for (const key of dirty) if (typeof key === "string" && stringFields.includes(key as keyof ImageTestDraft)) dirtyFields.push(key as keyof ImageTestDraft);
    return {
      file_path: typeof source.file_path === "string" ? source.file_path : "",
      preset: typeof source.preset === "string" && source.preset ? source.preset : "default",
      draft,
      history_range: IMAGE_TEST_HISTORY_RANGES.includes(source.history_range as ImageTestHistoryRange)
        ? source.history_range as ImageTestHistoryRange : "7",
      dirty_fields: [...new Set(dirtyFields)],
    };
  } catch { return fallback; }
}

export function writeImageTestState(state: ImageTestStoredState) {
  try {
    localStorage.setItem(IMAGE_TEST_STORAGE_KEY, JSON.stringify({
      version: 2,
      file_path: state.file_path,
      preset: state.preset,
      draft: state.draft,
      history_range: state.history_range,
      dirty_fields: state.dirty_fields,
    }));
  } catch { /* Keep the current form usable when browser storage is unavailable. */ }
}
