import { parseSamplePromptLine, serializeSamplePromptRow, type SamplePromptRow } from "../../../../static/js/features/sample-prompts/model.js";

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

export function promptRowError(row: SamplePromptRow) {
  if (!row.prompt.trim()) return "正向提示词不能为空。";
  if (Object.values(row).some((value) => /[\r\n]/.test(value))) return "每条样张须为单行提示词。";
  for (const key of ["width", "height", "steps", "seed", "cfg", "flow_shift"] as const) {
    if (!row[key]) continue;
    const n = Number(row[key]);
    if (!/^\d+(?:\.\d+)?$/.test(row[key]) || !Number.isFinite(n) || n < 0) return `${key} 必须为非负数。`;
    if (["width", "height", "steps", "seed"].includes(key) && !Number.isSafeInteger(n)) return `${key} 必须为整数。`;
    if (["width", "height"].includes(key) && n < 64) return "宽高不能小于 64。";
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
