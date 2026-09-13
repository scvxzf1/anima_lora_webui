import { describe, expect, it } from "vitest";

import {
  draftFromMerged,
  importedTrainingPath,
  rawConfigOwnKeys,
  trainingPatchValues,
  sameTrainingValue,
} from "./trainingForm";
import { fieldsForConfig } from "./fieldCatalog";
import { displayConfigValue } from "../../../../static/js/dragon-ui/pages/config-values.js";
import { CONFIG_FIELD_CATALOG } from "../../../../static/js/dragon-ui/pages/config-field-catalog.js";

describe("training config form domain", () => {
  it("ignores equivalent numeric and JSON formatting changes", () => {
    expect(sameTrainingValue("1.0", 1, "number")).toBe(true);
    expect(sameTrainingValue("", 0, "number")).toBe(false);
    expect(sameTrainingValue('[ "keep" ]', '["keep"]', "json")).toBe(true);
    expect(sameTrainingValue("[", "[]", "json")).toBe(false);
    expect(
      trainingPatchValues(
        { network_args: '[ "keep" ]' },
        { network_args: '["keep"]' },
        [{ key: "network_args", kind: "json", label: "Args", group: "method" }],
      ),
    ).toEqual({});
  });
  it("uses catalog defaults without writing unedited keys", () => {
    const baseline = draftFromMerged({
      output_name: "run",
      gradient_checkpointing: true,
    });
    expect(baseline.max_train_steps).toBe(
      displayConfigValue("max_train_steps", {}),
    );
    expect(baseline.network_dim).toBe(displayConfigValue("network_dim", {}));

    expect(
      trainingPatchValues({ ...baseline, output_name: "edited" }, baseline),
    ).toEqual({
      output_name: "edited",
    });
  });

  it("covers the live catalog once and retains unknown network arguments", () => {
    const original = {
      network_args: ["custom_extension=keep", "resampler_heads=8"],
    };
    const fields = fieldsForConfig(original);
    const keys = fields.map((field) => field.key);
    expect(new Set(keys).size).toBe(keys.length);
    for (const key of Object.keys(CONFIG_FIELD_CATALOG))
      expect(keys).toContain(key);
    const baseline = draftFromMerged(original, fields);
    expect(
      trainingPatchValues(
        { ...baseline, resampler_heads: 16 },
        baseline,
        fields,
        original,
      ),
    ).toEqual({
      network_args: ["custom_extension=keep", "resampler_heads=16"],
    });
    expect(trainingPatchValues(baseline, baseline, fields, original)).toEqual(
      {},
    );
  });

  it("preserves raw argument edits when also changing a mapped control", () => {
    const original = { network_args: ["unknown=old"] };
    const fields = fieldsForConfig(original);
    const baseline = draftFromMerged(original, fields);
    const patch = trainingPatchValues(
      { ...baseline, network_args: '["unknown=new"]', resampler_heads: 16 },
      baseline,
      fields,
      original,
    );
    expect(patch).toEqual({
      network_args: ["unknown=new", "resampler_heads=16"],
    });
  });

  it("tracks only top-level TOML ownership and normalizes imported save-as paths", () => {
    expect([
      ...rawConfigOwnKeys(
        [
          'output_name = "run"',
          "max_train_steps = 100",
          "[variant]",
          'family = "lora"',
        ].join("\n"),
      ),
    ]).toEqual(["output_name", "max_train_steps"]);
    expect(importedTrainingPath("dragon copy.toml")).toBe(
      "configs/imported/dragon_copy.toml",
    );
    expect(importedTrainingPath("../escape")).toBe(
      "configs/imported/escape.toml",
    );
  });
});
