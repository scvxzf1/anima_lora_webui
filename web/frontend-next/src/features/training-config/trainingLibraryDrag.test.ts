import { describe, expect, it } from "vitest";
import { groupDropId, trainingDropAfter, trainingFilePlacement } from "./trainingLibraryDrag";

const groups = [
  { id: "a", label: "A", files: [{ path: "a1" }, { path: "a2" }] },
  { id: "b", label: "B", files: [{ path: "b1" }, { path: "b2" }] },
  { id: "empty", label: "Empty", files: [] },
  { id: "locked", label: "Locked", readonly: true, files: [{ path: "l1" }] },
];

describe("trainingDropAfter", () => {
  it("uses keyboard list direction within and across groups", () => {
    expect(trainingDropAfter(groups, "a1", "a2", undefined, undefined)).toBe(true);
    expect(trainingDropAfter(groups, "a2", "a1", undefined, undefined)).toBe(false);
    expect(trainingDropAfter(groups, "a1", "b2", undefined, undefined)).toBe(true);
    expect(trainingDropAfter(groups, "b2", "a1", undefined, undefined)).toBe(false);
    expect(trainingDropAfter(groups, "a1", "a1", undefined, undefined)).toBe(false);
    expect(trainingDropAfter(groups, "missing", "b2", undefined, undefined)).toBe(false);
  });
  it("keeps pointer placement tied to the target half", () => {
    const rect = { top: 100, height: 40 };
    expect(trainingDropAfter(groups, "a1", "a2", 110, rect)).toBe(false);
    expect(trainingDropAfter(groups, "a1", "a2", 120, rect)).toBe(false);
    expect(trainingDropAfter(groups, "a2", "a1", 130, rect)).toBe(true);
  });
});

describe("trainingFilePlacement", () => {
  it("places files before and after a file in another group", () => {
    expect(trainingFilePlacement(groups, "a1", "b1")).toEqual({
      target: "file",
      file: "a1",
      group: "b",
      index: 0,
    });
    expect(trainingFilePlacement(groups, "a1", "b1", true)?.index).toBe(1);
  });
  it("accepts empty and collapsed group targets", () => {
    expect(
      trainingFilePlacement(groups, "a1", groupDropId("empty"))?.index,
    ).toBe(0);
    expect(trainingFilePlacement(groups, "a1", groupDropId("b"))?.index).toBe(
      2,
    );
  });
  it("accounts for removing the source before same-group insertion", () => {
    expect(trainingFilePlacement(groups, "a1", "a2", true)?.index).toBe(1);
    expect(trainingFilePlacement(groups, "a2", "a1")?.index).toBe(0);
    expect(trainingFilePlacement(groups, "a1", "a2")).toBeNull();
    expect(trainingFilePlacement(groups, "a2", groupDropId("a"))).toBeNull();
  });
  it("rejects missing, same-file and readonly targets and sources", () => {
    for (const [file, over] of [
      ["a1", "a1"],
      ["a1", "missing"],
      ["missing", "b1"],
      ["a1", "l1"],
      ["l1", "a1"],
    ]) {
      expect(trainingFilePlacement(groups, file, over)).toBeNull();
    }
    expect(
      trainingFilePlacement(
        [{ ...groups[0], files: [{ path: "a1", locked: true }] }, groups[1]],
        "a1",
        "b1",
      ),
    ).toBeNull();
  });
});
