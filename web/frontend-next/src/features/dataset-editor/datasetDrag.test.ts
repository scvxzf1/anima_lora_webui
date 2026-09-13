import { describe, expect, it } from "vitest";
import { datasetDropOrder } from "./datasetDrag";
import type { DatasetLibraryGroup } from "./types";

const groups: DatasetLibraryGroup[] = [
  {
    id: "a",
    label: "A",
    kind: "dataset",
    movable: true,
    files: [{ path: "a" }],
  },
  {
    id: "b",
    label: "B",
    kind: "dataset",
    movable: true,
    files: [{ path: "b" }, { path: "c" }],
  },
  { id: "empty", label: "Empty", kind: "dataset", movable: true, files: [] },
];
describe("dataset drop placement", () => {
  it("places before and after the exact target, including same-group moves", () => {
    expect(
      datasetDropOrder(groups, "a", {
        groupId: "b",
        file: "b",
        position: "before",
      }),
    ).toEqual(["a", "b", "c"]);
    expect(
      datasetDropOrder(groups, "a", {
        groupId: "b",
        file: "b",
        position: "after",
      }),
    ).toEqual(["b", "a", "c"]);
    expect(
      datasetDropOrder(groups, "b", {
        groupId: "b",
        file: "c",
        position: "after",
      }),
    ).toEqual(["c", "b"]);
  });
  it("handles empty groups and rejects missing, self and locked targets", () => {
    expect(
      datasetDropOrder(groups, "a", { groupId: "empty", position: "after" }),
    ).toEqual(["a"]);
    expect(
      datasetDropOrder(groups, "a", {
        groupId: "a",
        file: "a",
        position: "after",
      }),
    ).toBeNull();
    expect(
      datasetDropOrder(groups, "a", {
        groupId: "b",
        file: "missing",
        position: "after",
      }),
    ).toBeNull();
    expect(
      datasetDropOrder(
        groups.map((group) => ({ ...group, locked: true })),
        "a",
        { groupId: "b", position: "after" },
      ),
    ).toBeNull();
  });
});
