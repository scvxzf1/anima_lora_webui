import { describe, expect, it } from "vitest";
import type { CaptionItem } from "./api";
import { applyDeltas, batchPreview, filterCandidates, tagCounts } from "./captionBatch";
const items: CaptionItem[] = Array.from({ length: 65 }, (_, index) => ({ id: String(index), name: `${index}.png`, file: "", url: "", state: "ready", caption: "", proposed_caption: " red, blue\nred " }));
describe("whole-job candidate tags", () => {
  it("counts each item once across pages and includes drafts in filters", () => {
    expect(tagCounts(items, { "64": "green" })).toEqual([["blue", 64], ["red", 64], ["green", 1]]);
    expect(filterCandidates(items, { "64": "green" }, "green").map((i) => i.id)).toEqual(["64"]);
  });
  it("fails closed for empty scope and preserves untouched raw text", () => {
    expect(batchPreview(items, {}, [], "dedupe", "", "")).toEqual([]);
    expect(batchPreview(items, {}, ["64"], "append", "red", "")).toEqual([]);
    expect(batchPreview(items, {}, ["64"], "delete", "red, blue", "")).toEqual([]);
    expect(batchPreview(items, { "64": "red" }, ["64"], "append", "blue, blue, red", "")[0].after).toBe("red, blue");
  });
  it("previews, applies, undoes and redoes only deltas with existing drafts", () => {
    const drafts = { "64": "red, blue, red" };
    const delta = batchPreview(items, drafts, ["64"], "front", "blue", "");
    expect(delta).toEqual([{ id: "64", before: drafts["64"], after: "blue, red, red" }]);
    expect(items[64].proposed_caption).toBe(" red, blue\nred ");
    const next = applyDeltas(items, drafts, delta);
    expect(applyDeltas(items, next, delta, true)).toEqual(drafts);
    expect(applyDeltas(items, drafts, delta)).toEqual(next);
    expect(batchPreview(items, drafts, ["64"], "dedupe", "", "")[0].after).toBe("red, blue");
    expect(batchPreview(items, drafts, ["64"], "replace", "red", "green")[0].after).toBe("green, blue, green");
  });
});
it("keeps case, spaces and underscores distinct", () => {
  const exact = [{ ...items[0], proposed_caption: "Blue_Hair, blue hair, Blue_Hair" }];
  expect(tagCounts(exact, {})).toHaveLength(2);
  expect(batchPreview(exact, {}, ["0"], "delete", "blue_hair", "")).toEqual([]);
});
