import { describe, expect, it } from "vitest";
import { legacyHashPath, legacyHashTarget } from "./legacyRouteAdapter";

describe("legacy route adapter", () => {
  it.each([
    ["#dashboard", "/training"],
    ["#dataset-editor", "/datasets"],
    ["#page/live-training", "/monitor"],
    ["#queue", "/queue"],
    ["#model-config", "/models"],
    ["#tagging", "/captioning"],
    ["#history/task%2Fone", "/history/task%2Fone"],
  ])("maps %s to %s", (hash, expected) => expect(legacyHashPath(hash)).toBe(expected));

  it("leaves unknown hashes for the Next router", () => expect(legacyHashPath("#unknown")).toBeUndefined());

  it("keeps the history view separate from the task id", () => {
    expect(legacyHashTarget("#history/task%2Fone/metrics")).toEqual({ path: "/history/task%2Fone", view: "metrics" });
  });
});
