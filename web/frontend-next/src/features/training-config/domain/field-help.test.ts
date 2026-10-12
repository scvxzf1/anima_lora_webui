import { describe, expect, it } from "vitest";
import { FIELD_HELP_ZH } from "./field-help.js";
import { FIELD_HELP_BASICS_ZH } from "./help/basics.js";
import { FIELD_HELP_ADAPTERS_ZH } from "./help/adapters.js";
import { FIELD_HELP_PLAN_ZH } from "./help/plan.js";
import { FIELD_HELP_INPUT_ZH } from "./help/input.js";
import { FIELD_HELP_RESOURCES_ZH } from "./help/resources.js";
import { FIELD_HELP_RUNTIME_ZH } from "./help/runtime.js";
import { FIELD_HELP_ADVANCED_ZH } from "./help/advanced.js";
import { FIELD_HELP_SUMMARY_ZH } from "./field-help-summary.js";
import { fieldsForConfig } from "../fieldCatalog";
import { resolveFieldHelp } from "./resolveFieldHelp";

const chapters = [FIELD_HELP_BASICS_ZH, FIELD_HELP_ADAPTERS_ZH, FIELD_HELP_PLAN_ZH,
  FIELD_HELP_INPUT_ZH, FIELD_HELP_RESOURCES_ZH, FIELD_HELP_RUNTIME_ZH, FIELD_HELP_ADVANCED_ZH];
const fields = fieldsForConfig({});

describe("Next field help catalog", () => {
  it("keeps summaries aligned with every detailed entry", () => {
    expect(Object.keys(FIELD_HELP_ZH).sort()).toEqual(Object.keys(FIELD_HELP_SUMMARY_ZH).sort());
    for (const [field, help] of Object.entries(FIELD_HELP_ZH)) {
      expect(FIELD_HELP_SUMMARY_ZH[field as keyof typeof FIELD_HELP_SUMMARY_ZH]).toBe(help.summary);
    }
  });

  it("covers the current field catalog and keeps migration advice for retired keys", () => {
    for (const key of [...fields.map((field) => field.key), "is_reg", "batch_size", "use_hydra", "use_sigma_router"]) {
      expect(resolveFieldHelp(key), key).toBeDefined();
    }
    const keys = chapters.flatMap((chapter) => Object.keys(chapter));
    expect(new Set(keys).size).toBe(keys.length);
    expect(Object.keys(FIELD_HELP_ZH).sort()).toEqual(keys.sort());
  });

  it("uses reviewed V2 content directly without the seven-section schema", () => {
    for (const [key, help] of Object.entries(FIELD_HELP_ZH)) {
      expect(resolveFieldHelp(key), key).toBe(help);
      expect(help.version, key).toBe(2);
      expect(help.summary.trim(), key).not.toBe("");
      expect(help.recommendation.trim(), key).not.toBe("");
      for (const legacy of ["fill", "benefit", "cost", "risk", "recommend", "ps"]) {
        expect(Object.hasOwn(help, legacy), `${key}.${legacy}`).toBe(false);
      }
    }
  });

  it("keeps related help and option values consistent with the shared catalog", () => {
    for (const [key, help] of Object.entries(FIELD_HELP_ZH)) {
      for (const related of help.relatedKeys ?? []) expect(resolveFieldHelp(related), `${key} → ${related}`).toBeDefined();
      const options = fields.find((field) => field.key === key)?.options;
      for (const option of help.options ?? []) {
        expect(option.when.trim(), `${key}.${option.value}`).not.toBe("");
        if (options) expect(options, key).toContain(option.value);
      }
      const values = help.options?.map((option) => option.value) ?? [];
      expect(new Set(values).size, key).toBe(values.length);
    }
  });

  it("resolves the old decode key without guessing unknown or inherited property names", () => {
    expect(resolveFieldHelp("qwen_sample_vae_decode")).toBe(FIELD_HELP_ZH.sample_vae_decode);
    for (const key of ["future_field", "toString", "__proto__", "constructor"]) {
      expect(resolveFieldHelp(key)).toBeUndefined();
    }
  });
});
