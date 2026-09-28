export type SamplePromptRow = Record<
  | "prompt"
  | "negative_prompt"
  | "height"
  | "width"
  | "cfg"
  | "steps"
  | "seed"
  | "flow_shift"
  | "sample_sampler"
  | "extra"
  | "sample_task"
  | "reference_image",
  string
> & {
  reference_images: string[];
  json?: Record<string, unknown>;
  json_error?: string;
};

export function blankSamplePromptRow(): SamplePromptRow {
  return {
    prompt: "",
    negative_prompt: "",
    height: "",
    width: "",
    cfg: "",
    steps: "",
    seed: "",
    flow_shift: "",
    sample_sampler: "",
    extra: "",
    sample_task: "t2i",
    reference_image: "",
    reference_images: [],
  };
}

export function samplePromptsContentNeedsTextMode(content: unknown): boolean {
  const text = String(content || "");
  if (!text) return false;
  return text.split(/\r?\n/).some((line) => {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) return true;
    if (trimmed.startsWith("{")) return true;
    return serializeSamplePromptRow(parseSamplePromptLine(line)) !== trimmed;
  });
}

export function parseSamplePromptLine(line: unknown): SamplePromptRow {
  const text = String(line || "").trim();
  if (/^\{\s*"[^"\n]+"\s*:/.test(text)) return parseJsonSamplePrompt(text);
  const parts = String(line || "").trim().split(/\s+--/);
  const row = blankSamplePromptRow();
  row.prompt = (parts.shift() || "").trim();
  const extras: string[] = [];

  for (const rawPart of parts) {
    const part = rawPart.trim();
    let match = part.match(/^h\s+(\d+)$/i);
    if (match) {
      row.height = match[1];
      continue;
    }
    match = part.match(/^w\s+(\d+)$/i);
    if (match) {
      row.width = match[1];
      continue;
    }
    match = part.match(/^g\s+([\d.]+)$/i);
    if (match) {
      row.cfg = match[1];
      continue;
    }
    match = part.match(/^l\s+([\d.]+)$/i);
    if (match) {
      row.cfg = match[1];
      continue;
    }
    match = part.match(/^s\s+(\d+)$/i);
    if (match) {
      row.steps = match[1];
      continue;
    }
    match = part.match(/^d\s+(\d+)$/i);
    if (match) {
      row.seed = match[1];
      continue;
    }
    match = part.match(/^n\s+(.+)$/i);
    if (match) {
      row.negative_prompt = match[1].trim();
      continue;
    }
    match = part.match(/^ss\s+(.+)$/i);
    if (match) {
      row.sample_sampler = match[1].trim();
      continue;
    }
    match = part.match(/^fs\s+([\d.]+)$/i);
    if (match) {
      row.flow_shift = match[1];
      continue;
    }
    if (part) extras.push(`--${part}`);
  }
  row.extra = extras.join(" ");
  return row;
}

export function serializeSamplePromptRow(row: SamplePromptRow): string {
  if (row.json || row.sample_task === "edit" || row.reference_image || row.reference_images?.length) return serializeJsonSamplePrompt(row);
  if (!row.prompt) return "";
  const args: string[] = [];
  if (row.negative_prompt) args.push(`--n ${row.negative_prompt.trim()}`);
  if (row.width) args.push(`--w ${positiveIntegerText(row.width)}`);
  if (row.height) args.push(`--h ${positiveIntegerText(row.height)}`);
  if (row.steps) args.push(`--s ${positiveIntegerText(row.steps)}`);
  if (row.cfg) args.push(`--g ${positiveNumberText(row.cfg)}`);
  if (row.seed) args.push(`--d ${positiveIntegerText(row.seed)}`);
  if (row.flow_shift) args.push(`--fs ${positiveNumberText(row.flow_shift)}`);
  if (row.sample_sampler) args.push(`--ss ${row.sample_sampler.trim()}`);
  if (row.extra) args.push(row.extra.trim());
  return [row.prompt.trim(), ...args.filter(Boolean)].join(" ");
}

const JSON_FIELDS: Record<string, string> = {
  prompt: "prompt",
  negative_prompt: "negative_prompt",
  width: "width",
  height: "height",
  steps: "sample_steps",
  cfg: "guidance_scale",
  seed: "seed",
  flow_shift: "flow_shift",
  sample_sampler: "sample_sampler",
  sample_task: "sample_task",
  reference_image: "reference_image",
};
const NUMERIC_FIELDS = new Set(["width", "height", "steps", "cfg", "seed", "flow_shift"]);

function parseJsonSamplePrompt(text: string): SamplePromptRow {
  const row = blankSamplePromptRow();
  try {
    const data: unknown = JSON.parse(text);
    if (!data || Array.isArray(data) || typeof data !== "object" || typeof (data as Record<string, unknown>).prompt !== "string") throw new Error("prompt required");
    const source = data as Record<string, unknown>;
    row.json = source;
    row.sample_task = (source.sample_task as string) || (source.reference_image || (source.reference_images as unknown[] | undefined)?.length ? "edit" : "t2i");
    row.reference_images = Array.isArray(source.reference_images) ? source.reference_images.filter((path): path is string => typeof path === "string") :
      (typeof source.reference_image === "string" && source.reference_image ? [source.reference_image] : []);
    if (source.guidance_scale === undefined && source.scale !== undefined) row.cfg = String(source.scale);
    for (const [field, key] of Object.entries(JSON_FIELDS)) {
      const value = source[key];
      if (value !== undefined && value !== null) row[field as keyof SamplePromptRow] = String(value) as never;
    }
  } catch {
    row.prompt = text;
    row.json_error = "JSON 样张格式无效，请在原文模式修正。";
  }
  return row;
}

function serializeJsonSamplePrompt(row: SamplePromptRow): string {
  const data = { ...row.json };
  const originalReferences = Array.isArray(data.reference_images) ? data.reference_images :
    (typeof data.reference_image === "string" && data.reference_image ? [data.reference_image] : []);
  const references = Array.isArray(row.reference_images) ? row.reference_images : [];
  const referencesChanged = JSON.stringify(references) !== JSON.stringify(originalReferences);
  for (const [field, key] of Object.entries(JSON_FIELDS)) {
    if (field === "reference_image" && (referencesChanged || row.sample_task === "t2i" || Array.isArray(data.reference_images))) continue;
    const value = row[field as keyof SamplePromptRow] ?? "";
    if (String(data[key] ?? (field === "sample_task" ? "t2i" : "")) === value) continue;
    if (value === "") delete data[key];
    else data[key] = NUMERIC_FIELDS.has(field) ? Number(value) : value;
  }
  if (referencesChanged || row.sample_task === "t2i" || Array.isArray(data.reference_images)) {
    delete data.reference_image;
    if (row.sample_task === "edit" && references.length) data.reference_images = [...references];
    else delete data.reference_images;
  }
  if (row.cfg === "" && data.guidance_scale === undefined) delete data.scale;
  if (row.cfg !== "" && data.guidance_scale === undefined && data.scale !== undefined) delete data.scale;
  return JSON.stringify(data);
}

function positiveIntegerText(value: unknown): string {
  const n = Math.max(0, Math.floor(Number(value)));
  return Number.isFinite(n) ? String(n) : "";
}

function positiveNumberText(value: unknown): string {
  const text = String(value ?? "").trim();
  if (!/^\d+(?:\.\d+)?$/.test(text)) return "";
  return text;
}
