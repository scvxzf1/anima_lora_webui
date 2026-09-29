import { parseSamplePromptLine, serializeSamplePromptRow, type SamplePromptRow } from "./samplePromptCodec";

export function promptLines(content: string) {
  return content.split(/(?<=\n)/).map((raw, index) => ({ raw, index, row: parseSamplePromptLine(raw) }))
    .filter(({ raw }) => raw.trim() && !raw.trimStart().startsWith("#"));
}

export function updatePromptLine(content: string, index: number, row: SamplePromptRow) {
  const lines = content.split(/(?<=\n)/);
  if (lines[index] && JSON.stringify(parseSamplePromptLine(lines[index])) === JSON.stringify(row)) return content;
  const ending = lines[index]?.endsWith("\r\n") ? "\r\n" : lines[index]?.endsWith("\n") ? "\n" : "";
  lines[index] = serializeSamplePromptRow(row) + ending;
  return lines.join("");
}

export type UniformPromptValues = Partial<Pick<SamplePromptRow, "width" | "height" | "steps" | "cfg">>;

export function commonPromptValues(content: string): UniformPromptValues {
  const rows = promptLines(content).map(({ row }) => row);
  if (!rows.length) return {};
  const result: UniformPromptValues = {};
  for (const key of ["width", "height", "steps", "cfg"] as const) {
    if (rows[0][key] && rows.every((row) => row[key] === rows[0][key])) result[key] = rows[0][key];
  }
  return result;
}

export function applyUniformPromptValues(
  content: string,
  values: UniformPromptValues,
  modelFamily?: string,
  supportedPreviewTasks: readonly string[] = ["t2i"],
  maxPreviewReferences = 4,
): { content: string; error: string } {
  const entries = promptLines(content);
  const updates = entries.map(({ index, row }) => ({ index, row: { ...row, ...Object.fromEntries(
    Object.entries(values).filter(([, value]) => value !== undefined && value !== ""),
  ) } as SamplePromptRow }));
  for (const { index, row } of updates) {
    const error = promptRowError(row, modelFamily, supportedPreviewTasks, maxPreviewReferences);
    if (error) return { content, error: `样张 ${entries.findIndex((entry) => entry.index === index) + 1}：${error}` };
  }
  let next = content;
  for (const { index, row } of updates) next = updatePromptLine(next, index, row);
  return { content: next, error: "" };
}

export function isQwenSampleFamily(modelFamily?: string) {
  return ["qwen_image_2_1", "qwen_image_21", "qwen21"].includes(modelFamily || "");
}

export function supportsEditSamples(modelFamily?: string, supportedPreviewTasks: readonly string[] = ["t2i"]) {
  return Boolean(modelFamily) && supportedPreviewTasks.includes("edit");
}

export function promptRowError(row: SamplePromptRow, modelFamily?: string, supportedPreviewTasks: readonly string[] = ["t2i"], maxPreviewReferences = 4) {
  if (row.json_error) return row.json_error;
  if (!["t2i", "edit"].includes(row.sample_task)) return "未知样张任务，请在原文模式修正。";
  if (row.json && "reference_images" in row.json && (!Array.isArray(row.json.reference_images) || row.json.reference_images.some((path: unknown) => typeof path !== "string"))) return "reference_images 须为路径数组。";
  if (row.json && "reference_images" in row.json && (row.json.reference_images as unknown[]).length === 0) return "reference_images 不能为空数组。";
  if (row.json && typeof row.json.reference_image === "string" && row.json.reference_image.trim() &&
      Array.isArray(row.json.reference_images) && row.json.reference_images.length) return "reference_image 与 reference_images 不能同时设置。";
  if (!Array.isArray(row.reference_images) || row.reference_images.some((path) => typeof path !== "string" || !path.trim())) return "参考图路径无效。";
  if (row.reference_images.length > maxPreviewReferences) return `参考图最多 ${maxPreviewReferences} 张。`;
  if (row.sample_task === "t2i" && (row.reference_image.trim() || row.reference_images.length ||
      (row.json && (typeof row.json.reference_image === "string" && row.json.reference_image.trim() ||
        Array.isArray(row.json.reference_images) && row.json.reference_images.length)))) return "文生图样张不能携带参考图。";
  if (row.sample_task === "edit") {
    if (!supportsEditSamples(modelFamily, supportedPreviewTasks)) return "当前模型不支持图像编辑样张。";
    if (row.extra) return "其他 TXT 参数无法自动转换为 JSON，请先在原文模式确认这些参数。";
    if (!row.reference_images.length) return "请载入参考图。";
  }
  if (isQwenSampleFamily(modelFamily) && row.sample_sampler && row.sample_sampler !== "euler") return "Qwen 样张仅支持 Euler 采样器。";
  if (isQwenSampleFamily(modelFamily) && row.flow_shift) return "Qwen 样张自动计算 Flow shift，请在原文中移除旧参数。";
  if (isQwenSampleFamily(modelFamily) && row.cfg && Number(row.cfg) < 1) return "Qwen 样张 CFG 不能小于 1。";
  if (!row.prompt.trim()) return "正向提示词不能为空。";
  if (Object.values(row).some((value) => typeof value === "string" && /[\r\n]/.test(value))) return "每条样张须为单行提示词。";
  for (const key of ["width", "height", "steps", "seed", "cfg", "flow_shift"] as const) {
    if (!row[key]) continue;
    const n = Number(row[key]);
    if (!/^\d+(?:\.\d+)?$/.test(row[key]) || !Number.isFinite(n) || n < 0) return `${key} 必须为非负数。`;
    if (["width", "height", "steps", "seed"].includes(key) && !Number.isSafeInteger(n)) return `${key} 必须为整数。`;
    if (["width", "height"].includes(key) && n < 64) return "宽高不能小于 64。";
    if (isQwenSampleFamily(modelFamily) && ["width", "height"].includes(key) && (n > 2048 || n % 32 !== 0)) return "Qwen 样张宽高须为 64 至 2048 范围内的 32 倍数。";
    if (key === "steps" && (n < 1 || n > 1000)) return "步数须在 1 至 1000 之间。";
  }
  return "";
}

export function movePromptLine(content: string, index: number, direction: number) {
  const entries = promptLines(content);
  const position = entries.findIndex((line) => line.index === index);
  const target = entries[position + direction];
  if (!target) return content;
  const lines = content.split(/(?<=\n)/);
  const a = lines[index].replace(/[\r\n]+$/, "");
  const b = lines[target.index].replace(/[\r\n]+$/, "");
  lines[index] = lines[index].replace(a, b);
  lines[target.index] = lines[target.index].replace(b, a);
  return lines.join("");
}
