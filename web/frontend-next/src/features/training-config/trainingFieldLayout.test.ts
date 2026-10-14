import { describe, expect, it } from "vitest";
import { getTrainingFieldLayout, getTrainingGridColumns } from "./trainingFieldLayout";

describe("getTrainingFieldLayout", () => {
  it("keeps optimizer type and learning rate compact", () => {
    expect(getTrainingFieldLayout({ key: "optimizer_type", kind: "select" }, 2)).toEqual({ span: "single", width: "compact" });
    expect(getTrainingFieldLayout({ key: "learning_rate", kind: "number" }, 2)).toEqual({ span: "single", width: "compact" });
  });

  it("marks optimizer args and paths as dense-flow wide cards", () => {
    expect(getTrainingFieldLayout({ key: "optimizer_args", kind: "text" }, 6)).toEqual({ span: "wide", width: "wide" });
    expect(getTrainingFieldLayout({ key: "pretrained_model_name_or_path", kind: "text" }, 4).span).toBe("wide");
  });

  it("does not make a single field span beyond its capped container", () => {
    expect(getTrainingFieldLayout({ key: "optimizer_args", kind: "json" }, 1)).toEqual({ span: "single", width: "wide" });
  });

  it("chooses three tracks for a five-field group with one wide field", () => {
    expect(
      getTrainingGridColumns([
        { key: "optimizer_type", kind: "select" },
        { key: "learning_rate", kind: "number" },
        { key: "optimizer_args", kind: "text" },
        { key: "lr_scheduler", kind: "select" },
        { key: "lr_warmup_steps", kind: "number" },
      ]),
    ).toBe(3);
  });
});
