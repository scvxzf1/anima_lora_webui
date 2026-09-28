import type { DatasetFormValues } from "./datasetForm";

export function editDatasetIssue(editEnabled: boolean, rows: DatasetFormValues["datasets"]): string | null {
  if (!editEnabled) {
    return rows.some((row) => row.reference_image_dir.trim())
      ? "数据集仍包含参考图目录（编辑前）；请启用编辑 LoRA 或移除该目录" : null;
  }
  if (rows.some((row) => row.edit_role === "normal")) return "编辑 LoRA 不能混用普通数据子集";
  const pairs = new Map<string, { before: number; after: number }>();
  for (const row of rows) {
    const id = row.edit_pair_id.trim();
    if (!id) return "编辑子集需要配对名称";
    const pair = pairs.get(id) || { before: 0, after: 0 };
    pair[row.edit_role === "before" ? "before" : "after"] += 1;
    pairs.set(id, pair);
  }
  if ([...pairs.values()].some((pair) => pair.before !== 1 || pair.after !== 1)) return "每组配对必须各有一个编辑前和编辑后子集";
  if (rows.some((row) => row.edit_role === "before" && !row.source_dir.trim())) return "编辑前图片目录不能为空";
  return null;
}
