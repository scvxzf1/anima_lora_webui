import { describe, expect, it } from "vitest";
import {
  applySettingsDefaults,
  SETTINGS_GROUPS,
  settingsDraft,
  settingsPatch,
} from "./settingsForm";
import { addModel, deleteModel, moveModel, reorderModel } from "./modelLibrary";
import type { ModelConfigResponse } from "./api";

describe("global settings draft", () => {
  it("exposes the tagging retention limit with the backend range", () => {
    const field = SETTINGS_GROUPS.flatMap((group) => group.fields).find(
      ({ key }) => key === "tagging_max_retained_jobs",
    );
    expect(field).toMatchObject({ label: "打标任务保留数量", kind: "number", min: 1, max: 500 });
  });
  it("edits raw overrides, never effective paths, and sends only changed keys", () => {
    const baseline = settingsDraft({
      configs_root: "wrong",
      path_overrides: { configs_root: "", history_root: "", queue_root: "" },
      effective_paths: { configs_root: "/effective" },
      ui_scale: 100,
      output_root: "output/runs",
    });
    expect(baseline.configs_root).toBe("");
    expect(settingsPatch({ ...baseline, ui_scale: 125 }, baseline)).toEqual({
      ui_scale: 125,
    });
  });
  it("does not erase fields missing from a partial defaults response", () => {
    const draft = {
      output_root: "runs",
      ui_scale: 125,
      dragon_motion_enabled: true,
    };
    expect(applySettingsDefaults(draft, undefined)).toEqual(draft);
    expect(applySettingsDefaults(draft, { ui_scale: 100 })).toEqual({
      ...draft,
      ui_scale: 100,
    });
  });
});

describe("model library membership", () => {
  const library: ModelConfigResponse = {
    revision: "revision-1",
    default_id: "a",
    items: ["a", "b"].map((id) => ({
      id,
      name: id,
      model_family: "anima",
      pretrained_model_name_or_path: "dit",
      qwen3: "te",
      vae: "vae",
    })),
    groups: [
      { id: "g1", label: "G1", item_ids: ["a", "b"] },
      { id: "g2", label: "G2", item_ids: [] },
    ],
  };
  it("preserves revision and exactly one group membership through mutations", () => {
    const added = addModel(library);
    const moved = moveModel(added, "b", "g2");
    const ordered = reorderModel(moved, "a", 1);
    expect(ordered.revision).toBe("revision-1");
    const membership = ordered.groups!.flatMap((group) => group.item_ids);
    expect(membership.length).toBe(ordered.items.length);
    expect(new Set(membership).size).toBe(ordered.items.length);
    expect(library.groups![0].item_ids).toEqual(["a", "b"]);
  });
  it("protects default entries and refuses missing target groups", () => {
    expect(deleteModel(library, "a")).toBe(library);
    expect(moveModel(library, "b", "missing")).toBe(library);
  });
});
