import { describe, expect, it } from "vitest";
import {
  appendCaptionTag,
  deleteCaptionTag,
  joinCaptionTags,
  moveCaptionTag,
  splitCaptionTags,
  updateCaptionTag,
} from "./captionTags";

describe("caption tag helpers", () => {
  it("matches legacy comma/newline splitting and canonical joining", () => {
    const source = "1girl, blue_hair\nsolo";
    expect(splitCaptionTags(source)).toEqual(["1girl", "blue_hair", "solo"]);
    expect(joinCaptionTags(splitCaptionTags(source))).toBe(
      "1girl, blue_hair, solo",
    );
  });

  it("edits, appends, removes, and reorders without mutating input", () => {
    const source = ["1girl", "blue_hair", "solo"];
    expect(moveCaptionTag(source, 2, 0)).toEqual([
      "solo",
      "1girl",
      "blue_hair",
    ]);
    expect(updateCaptionTag(source, 1, " white hair ")).toEqual([
      "1girl",
      "white hair",
      "solo",
    ]);
    expect(updateCaptionTag(source, 0, "")).toEqual(source);
    expect(deleteCaptionTag(source, 0)).toEqual(["blue_hair", "solo"]);
    expect(appendCaptionTag(source, "smile,")).toEqual([
      "1girl",
      "blue_hair",
      "solo",
      "smile",
    ]);
    expect(source).toEqual(["1girl", "blue_hair", "solo"]);
  });
});
