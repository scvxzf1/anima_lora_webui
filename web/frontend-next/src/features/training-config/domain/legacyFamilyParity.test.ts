import { describe, expect, it } from "vitest";
import { availableFieldOptions, fieldAvailability, fieldsForConfig } from "../fieldCatalog";

const field = (key: string) => {
  const spec = fieldsForConfig({}).find((candidate) => candidate.key === key);
  if (!spec) throw new Error(`Missing training field: ${key}`);
  return spec;
};

describe("legacy Krea-2 field constraints", () => {
  it("keeps the Krea-2 attention and selective-checkpoint choices", () => {
    expect(availableFieldOptions(field("attn_mode"), "krea2_raw")).toEqual([
      "torch",
      "flash",
      "sdpa",
    ]);
    expect(availableFieldOptions(field("selective_checkpoint"), "krea2")).toEqual([
      "off",
      "every_other",
    ]);
  });

  it("filters Krea-2 compile and V100 diagnostic options without affecting other families", () => {
    expect(availableFieldOptions(field("compile_inductor_mode"), "krea2_raw")).toEqual([
      "default",
    ]);
    expect(availableFieldOptions(field("v100_flash_stability"), "krea2_raw")).toEqual([
      "off",
    ]);
    expect(availableFieldOptions(field("compile_inductor_mode"), "anima")).toEqual(
      field("compile_inductor_mode").options,
    );
    expect(availableFieldOptions(field("selective_checkpoint"), "z_image")).toContain("mlp_only");
  });

  it("keeps unsupported Krea-2 values editable until corrected without mutating the draft", () => {
    const values = {
      model_family: "krea2_raw",
      torch_compile: true,
      compile_dynamic_seq: true,
      compile_seq_bands: true,
      compile_inductor_mode: "reduce-overhead",
      selective_checkpoint: "mlp_only",
      v100_flash_stability: "safe",
    };
    const original = structuredClone(values);

    for (const key of ["compile_dynamic_seq", "compile_seq_bands", "compile_inductor_mode", "selective_checkpoint", "v100_flash_stability"]) {
      expect(fieldAvailability(key, values, "lora").enabled, key).toBe(true);
    }
    expect(availableFieldOptions(field("compile_inductor_mode"), "krea2_raw")).toEqual(["default"]);
    expect(availableFieldOptions(field("v100_flash_stability"), "krea2_raw")).toEqual(["off"]);
    expect(availableFieldOptions(field("selective_checkpoint"), "krea2_raw")).toEqual(["off", "every_other"]);

    const correctedValues = {
      ...values,
      compile_dynamic_seq: false,
      compile_seq_bands: false,
      compile_inductor_mode: "default",
      selective_checkpoint: "every_other",
      v100_flash_stability: "off",
    };
    for (const [key, code] of [
      ["compile_dynamic_seq", "krea2-compile-dynamic-seq"],
      ["compile_seq_bands", "krea2-compile-seq-bands"],
      ["compile_inductor_mode", "krea2-compile-inductor-mode"],
      ["v100_flash_stability", "krea2-v100-flash-stability"],
    ]) {
      expect(fieldAvailability(key, correctedValues, "lora")).toMatchObject({ enabled: false, code });
    }
    expect(fieldAvailability("selective_checkpoint", correctedValues, "lora").enabled).toBe(true);
    expect(values).toEqual(original);
    expect(availableFieldOptions(field("selective_checkpoint"), "krea2_raw")).not.toContain("mlp_only");
  });

  it("keeps corrective Krea-2 compile choices available when torch.compile is off", () => {
    const values = {
      model_family: "krea2_raw",
      torch_compile: false,
      compile_dynamic_seq: true,
      compile_seq_bands: true,
      compile_inductor_mode: "reduce-overhead",
    };
    expect(fieldAvailability("compile_dynamic_seq", values, "lora").enabled).toBe(true);
    expect(fieldAvailability("compile_seq_bands", values, "lora").enabled).toBe(true);
    expect(fieldAvailability("compile_inductor_mode", values, "lora").enabled).toBe(true);
  });

  it("keeps unknown-family attention fail-closed without imposing Krea-2 rules elsewhere", () => {
    expect(fieldAvailability("attn_mode", { model_family: "future_family" }, "lora")).toMatchObject({
      enabled: false,
      code: "model-family-unknown",
    });
    expect(fieldAvailability("compile_dynamic_seq", { model_family: "anima" }, "lora").enabled).toBe(true);
    expect(fieldAvailability("compile_inductor_mode", { model_family: "z_image" }, "lora").enabled).toBe(true);
  });
});
