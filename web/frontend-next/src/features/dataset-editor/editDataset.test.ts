import { describe, expect, it } from "vitest";
import { emptyDatasetRow } from "./datasetForm";
import { editDatasetIssue } from "./editDataset";

describe("model-independent edit dataset structure", () => {
  const before = { ...emptyDatasetRow(), edit_role: "before" as const, edit_pair_id: "color", source_dir: "before" };
  const after = { ...emptyDatasetRow(), edit_role: "after" as const, edit_pair_id: "color", source_dir: "after" };
  it("accepts a complete pair without knowing a model", () => {
    expect(editDatasetIssue(true, [before, after])).toBeNull();
  });
  it("requires a complete named pair", () => {
    expect(editDatasetIssue(true, [before])).toContain("各有一个");
    expect(editDatasetIssue(true, [before, { ...after, edit_pair_id: "" }])).toContain("配对名称");
    expect(editDatasetIssue(true, [before, { ...before }, after])).toContain("各有一个");
    expect(editDatasetIssue(true, [{ ...before, source_dir: "" }, after])).toContain("不能为空");
  });
  it("rejects mixed ordinary data and leftover references", () => {
    expect(editDatasetIssue(true, [before, after, emptyDatasetRow()])).toContain("不能混用");
    expect(editDatasetIssue(false, [{ ...emptyDatasetRow(), reference_image_dir: "before" }])).toContain("仍包含参考图目录");
  });
});
