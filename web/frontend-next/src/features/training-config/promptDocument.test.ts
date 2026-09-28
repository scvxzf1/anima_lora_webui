import { expect, it } from "vitest";
import { promptLines, updatePromptLine, movePromptLine, promptRowError } from "./promptDocument";
import { samplePromptsContentNeedsTextMode } from "../../../../static/js/features/sample-prompts/model.js";

it("preserves comments, blank lines, untouched prompts and unknown options", () => {
  const text = "# note\r\n\r\nfirst --w 512 --custom yes\r\n second --l 7  \r\n";
  const rows = promptLines(text);
  expect(rows).toHaveLength(2);
  const updated = updatePromptLine(text, rows[0].index, { ...rows[0].row, width: "768" });
  expect(updated).toBe("# note\r\n\r\nfirst --w 768 --custom yes\r\n second --l 7  \r\n");
  expect(movePromptLine("# header\na\n\nb", 1, 1)).toBe("# header\nb\n\na");
  expect(promptRowError({ ...rows[0].row, steps: "0" })).not.toBe("");
});

it("round-trips JSON references and unknown fields without rewriting untouched lines", () => {
  const source = { prompt: "修改颜色", sample_task: "edit", reference_image: "/图片 数据/参考 图.png", width: 768, guidance_scale: 1, custom: { nested: [1, "中文"] } };
  const content = `# note\r\n\r\n${JSON.stringify(source)}\r\n old --l 7  \n`;
  const { index, row } = promptLines(content)[0];
  expect(updatePromptLine(content, index, row)).toBe(content);
  const result = updatePromptLine(content, index, { ...row, width: "1024" });
  expect(JSON.parse(result.split("\r\n")[2])).toEqual({ ...source, width: 1024 });
  expect(result.endsWith("\r\n old --l 7  \n")).toBe(true);
  expect(samplePromptsContentNeedsTextMode(JSON.stringify(source))).toBe(true);
  expect(promptRowError(row, "qwen_image_2_1", ["t2i", "edit"])).toBe("");
  expect(promptRowError(row, "future_edit_model", ["t2i", "edit"])).toBe("");
  expect(promptRowError(row, "unknown", [])).toContain("不支持");
  expect(promptRowError(row, "qwen_image_2_1", ["t2i"])).toContain("不支持");
});

it("keeps malformed JSON out of graphical writes", () => {
  const row = promptLines('{"prompt": broken}')[0].row;
  expect(promptRowError(row)).toContain("JSON");
});

it("preserves legacy brace prompts and infers JSON reference tasks", () => {
  expect(promptLines("{soft focus} --w 512")[0].row.prompt).toBe("{soft focus}");
  const row = promptLines('{"prompt":"edit","reference_image":"/a b.png","scale":3}')[0].row;
  expect(row.sample_task).toBe("edit");
  expect(row.cfg).toBe("3");
  expect(row.reference_images).toEqual(["/a b.png"]);
});

it("serializes edited references as an array while retaining unknown fields", () => {
  const content = JSON.stringify({ prompt: "edit", reference_image: "/old.png", custom: { a: 1 } });
  const row = promptLines(content)[0].row;
  expect(JSON.parse(updatePromptLine(content, 0, { ...row, reference_images: ["/old.png", "/new.png"] }))).toEqual({ prompt: "edit", custom: { a: 1 }, reference_images: ["/old.png", "/new.png"], sample_task: "edit" });
  expect(promptRowError({ ...row, reference_images: ["/a", "/b"] }, "qwen_image_2_1", ["t2i", "edit"], 1)).toContain("最多 1 张");
  expect(promptRowError(promptLines('{"prompt":"edit","sample_task":"edit","reference_images":["/a",2]}')[0].row, "qwen_image_2_1", ["t2i", "edit"])).toContain("路径数组");
  expect(JSON.parse(updatePromptLine(content, 0, { ...row, sample_task: "t2i", reference_images: [], reference_image: "" }))).toEqual({ prompt: "edit", custom: { a: 1 } });
  const conflicting = JSON.stringify({ prompt: "edit", sample_task: "edit", reference_image: "/stale.png", reference_images: ["/canonical.png"] });
  const conflictingRow = promptLines(conflicting)[0].row;
  expect(updatePromptLine(conflicting, 0, conflictingRow)).toBe(conflicting);
  expect(JSON.parse(updatePromptLine(conflicting, 0, { ...conflictingRow, prompt: "new edit" }))).toEqual({ prompt: "new edit", sample_task: "edit", reference_images: ["/canonical.png"] });
});

it("rejects empty arrays, conflicting references, and t2i references", () => {
  const error = (source: object) => promptRowError(promptLines(JSON.stringify({ prompt: "test", ...source }))[0].row, "qwen_image_2_1", ["t2i", "edit"]);
  expect(error({ sample_task: "edit", reference_images: [] })).toContain("不能为空数组");
  expect(error({ sample_task: "edit", reference_image: "/a.png", reference_images: ["/b.png"] })).toContain("不能同时设置");
  expect(error({ sample_task: "t2i", reference_image: "/a.png" })).toContain("不能携带参考图");
  expect(error({ sample_task: "t2i", reference_images: ["/a.png"] })).toContain("不能携带参考图");
  expect(error({ sample_task: "t2i", reference_image: "" })).toBe("");
});
