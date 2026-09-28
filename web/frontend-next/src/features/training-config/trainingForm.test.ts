import { describe, expect, it } from "vitest";
import { parse as parseToml } from "smol-toml";
// @ts-expect-error Vitest runs under Node; the frontend app tsconfig omits Node types.
import { readFileSync } from "node:fs";

import {
  draftFromMerged,
  importedTrainingPath,
  rawConfigOwnKeys,
  trainingPatchValues,
  sameTrainingValue,
  restoreKnownFormDefaults,
  type TrainingDraft,
} from "./trainingForm";
import {
  fieldAvailability,
  fieldsForConfig,
  filterTrainingFields,
} from "./fieldCatalog";
import { displayConfigValue } from "./domain/config-values.js";
import {
  CONFIG_FIELD_CATALOG,
  configFieldCatalogEntry,
} from "./domain/config-field-catalog.js";

function readConfig(relativePath: string) {
  return parseToml(
    readFileSync(new URL(relativePath, import.meta.url), "utf8"),
  ) as Record<string, unknown>;
}

function fieldManifest(config: Record<string, unknown>) {
  return new Map(
    fieldsForConfig(config).map((field) => {
      const owner = configFieldCatalogEntry(field.key);
      return [
        field.key,
        {
          kind: field.kind,
          group: field.group,
          owner: `${owner.location}/${owner.stage}/${owner.cluster}`,
        },
      ] as const;
    }),
  );
}

function diffFieldManifests(
  left: Map<string, unknown>,
  right: Map<string, unknown>,
) {
  const keys = new Set([...left.keys(), ...right.keys()]);
  return [...keys]
    .sort()
    .flatMap((key) => {
      const before = left.get(key);
      const after = right.get(key);
      if (before === undefined || after === undefined) return [];
      return JSON.stringify(before) === JSON.stringify(after)
        ? []
        : [{ key, before, after }];
    });
}

describe("training config form domain", () => {
  it("restores only known defaults for supplied editable fields and keeps unrelated draft keys", () => {
    const draft = {
      max_train_steps: 1600,
      adaptive_fp32_modules: '["custom"]',
      output_name: "keep me",
      unknown_extension: "untouched",
    };
    const restored = restoreKnownFormDefaults(
      draft,
      [
        { key: "max_train_steps", kind: "number", label: "steps", group: "training" },
        { key: "adaptive_fp32_modules", kind: "json", label: "modules", group: "resources" },
        { key: "unknown_extension", kind: "text", label: "extension", group: "training" },
      ],
      new Set(["max_train_steps", "adaptive_fp32_modules", "unknown_extension"]),
    );
    expect(restored).toEqual({
      ...draft,
      max_train_steps: 0,
      adaptive_fp32_modules: "[]",
    });
    expect(
      restoreKnownFormDefaults(draft, [
        { key: "max_train_steps", kind: "number", label: "steps", group: "training" },
      ], new Set()),
    ).toEqual(draft);
  });

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

  it("hydrates boolean fields as booleans and keeps a false weight-dimension switch inactive", () => {
    const source = {
      model_family: "anima",
      dim_from_weights: false,
      network_weights: "adapter.safetensors",
      network_dim: 16,
      cache_latents: true,
      weighted_captions: false,
      use_custom_down_autograd: false,
    };
    const fields = fieldsForConfig(source);
    for (const key of ["dim_from_weights", "cache_latents", "weighted_captions", "use_custom_down_autograd"]) {
      expect(fields.find((field) => field.key === key)?.kind).toBe("boolean");
    }
    expect(fieldsForConfig({}).find((field) => field.key === "dim_from_weights")?.kind).toBe("boolean");
    expect(fieldsForConfig({}).find((field) => field.key === "use_custom_down_autograd")?.kind).toBe("boolean");

    const baseline = draftFromMerged(source, fields);
    expect(baseline.dim_from_weights).toBe(false);
    expect(baseline.cache_latents).toBe(true);
    expect(baseline.weighted_captions).toBe(false);
    expect(fieldAvailability("network_dim", baseline, "lora").enabled).toBe(true);
    expect(fieldAvailability("network_dim", { ...baseline, dim_from_weights: "false" }, "lora").enabled).toBe(true);
    expect(fieldAvailability("network_dim", { ...baseline, dim_from_weights: true }, "lora").enabled).toBe(false);
    expect(trainingPatchValues(baseline, baseline, fields, source)).toEqual({});
    expect(draftFromMerged({ dim_from_weights: "false" }, fields).dim_from_weights).toBe(false);
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

  it("keeps key, type, group, and owner stable across representative TOML manifests", () => {
    const profiles = [
      ["anima", "../../../../../configs/methods/lora.toml"],
      ["krea2", "../../../../../configs/methods/krea2_lora.toml"],
      ["z_image", "../../../../../configs/methods/z_image_lora.toml"],
      ["qwen", "../../../../../configs/methods/qwen_image_2_1_lora.toml"],
    ] as const;
    const manifests = profiles.map(([name, path]) => [
      name,
      fieldManifest(readConfig(path)),
    ] as const);
    const reference = manifests[0];
    const differences = manifests.slice(1).flatMap(([name, manifest]) =>
      diffFieldManifests(reference[1], manifest).map((difference) => ({
        profile: name,
        ...difference,
      })),
    );

    expect(differences, JSON.stringify(differences, null, 2)).toEqual([]);
    expect(configFieldCatalogEntry("__future_runtime_field__")).toMatchObject({
      location: "audit_only",
      stage: "resources",
      cluster: "unclassified",
    });
    expect(CONFIG_FIELD_CATALOG["__future_runtime_field__"]).toBeUndefined();
    expect(
      fieldsForConfig({ __future_runtime_field__: 7 }).find(
        (field) => field.key === "__future_runtime_field__",
      ),
    ).toMatchObject({ kind: "number", group: "training" });
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

  it("keeps resolvable conflicts visible while disabling their controls", () => {
    const availability = fieldAvailability(
      "max_train_steps",
      { model_family: "krea2_raw", max_train_epochs: 43 },
      "lora",
    );
    expect(availability).toMatchObject({
      visible: true,
      enabled: false,
      code: "epochs-override-steps",
    });
    expect(availability.reason).toContain("max_train_epochs");
  });

  it("preserves manual and AUTO swap values while patching only the toggled mode", () => {
    const source = {
      model_family: "krea2_raw",
      auto_block_swap: false,
      auto_block_swap_mode: "dynamic",
      auto_block_swap_interval: 8,
      blocks_to_swap: 20,
    };
    const fields = fieldsForConfig(source);
    const manual = draftFromMerged(source, fields);
    const automatic: TrainingDraft = { ...manual, auto_block_swap: true };

    expect(fieldAvailability("auto_block_swap_mode", manual, "lora")).toMatchObject({
      enabled: false,
      code: "auto-block-swap-disabled",
    });
    expect(fieldAvailability("blocks_to_swap", automatic, "lora")).toMatchObject({
      visible: true,
      enabled: false,
      code: "auto-block-swap-enabled",
    });
    expect(automatic.blocks_to_swap).toBe(20);
    expect(trainingPatchValues(automatic, manual, fields, source)).toEqual({
      auto_block_swap: true,
    });

    const autoSource = { ...source, auto_block_swap: true, blocks_to_swap: 17 };
    const autoDraft = draftFromMerged(autoSource, fieldsForConfig(autoSource));
    const returnedToManual: TrainingDraft = {
      ...autoDraft,
      auto_block_swap: false,
    };
    expect(returnedToManual.blocks_to_swap).toBe(17);
    expect(
      trainingPatchValues(
        returnedToManual,
        autoDraft,
        fieldsForConfig(autoSource),
        autoSource,
      ),
    ).toEqual({ auto_block_swap: false });
  });

  it("preserves compile child values and patches only explicit parent changes", () => {
    const source = {
      torch_compile: false,
      compile_block_scope: "resident",
      compile_dynamic_seq: false,
      compile_seq_bands: true,
    };
    const fields = fieldsForConfig(source);
    const baseline = draftFromMerged(source, fields);
    const compileEnabled: TrainingDraft = { ...baseline, torch_compile: true };

    expect(fieldAvailability("compile_block_scope", baseline, "lora")).toMatchObject({
      visible: true,
      enabled: false,
      code: "torch-compile-disabled",
    });
    expect(
      fieldAvailability("compile_seq_bands", compileEnabled, "lora"),
    ).toMatchObject({ enabled: false, code: "compile-dynamic-seq-disabled" });
    expect(compileEnabled.compile_block_scope).toBe("resident");
    expect(compileEnabled.compile_seq_bands).toBe(true);
    expect(trainingPatchValues(compileEnabled, baseline, fields, source)).toEqual({
      torch_compile: true,
    });
  });

  it("hides unrelated method branches from the applicable view but keeps them auditable", () => {
    const availability = fieldAvailability(
      "lora_adapter_kind",
      { model_family: "anima" },
      "ip_adapter",
    );
    expect(availability).toMatchObject({
      visible: false,
      enabled: false,
      code: "method-context",
    });
    expect(availability.reason).toContain("IP-Adapter");
  });

  it("does not let search bypass the applicable view", () => {
    const fields = fieldsForConfig({
      model_family: "anima",
      lora_adapter_kind: "lora",
    });
    const draft = {
      model_family: "anima",
      lora_adapter_kind: "lora",
    } as TrainingDraft;
    const baseline = { ...draft };
    const applicable = filterTrainingFields(
      fields,
      draft,
      baseline,
      "lora_adapter_kind",
      "applicable",
      "ip_adapter",
    );
    expect(applicable.map((field) => field.key)).not.toContain("lora_adapter_kind");

    const audit = filterTrainingFields(
      fields,
      draft,
      baseline,
      "lora_adapter_kind",
      "all",
      "ip_adapter",
    );
    expect(audit.map((field) => field.key)).toContain("lora_adapter_kind");
  });

  it("keeps hidden branches and unknown keys in audit without making them editable", () => {
    const source = {
      model_family: "anima",
      use_ip_adapter: false,
      pe_lora_enabled: false,
      resampler_heads: 8,
      pe_lora_rank: 4,
      __future_runtime_field__: 7,
    };
    expect(fieldAvailability("use_ip_adapter", source, "ip_adapter")).toMatchObject({
      visible: true, enabled: true,
    });
    expect(fieldAvailability("resampler_heads", source, "ip_adapter")).toMatchObject({
      visible: false, enabled: false, code: "feature-disabled",
    });
    expect(fieldAvailability("pe_lora_rank", source, "ip_adapter")).toMatchObject({
      visible: false, enabled: false, code: "feature-disabled",
    });
    expect(fieldAvailability("__future_runtime_field__", source, "ip_adapter")).toMatchObject({
      visible: false, enabled: false, code: "audit-only",
    });
    expect(fieldAvailability("dit_path", source, "lora")).toMatchObject({
      visible: false, enabled: false,
    });
    expect(fieldAvailability("dit_path", source, "spd")).toMatchObject({
      visible: true, enabled: true,
    });
    expect(fieldAvailability("resampler_heads", { ...source, use_ip_adapter: true }, "ip_adapter")).toMatchObject({
      visible: true, enabled: true,
    });
    expect(fieldAvailability("pe_lora_rank", { ...source, use_ip_adapter: true }, "ip_adapter")).toMatchObject({
      visible: false, enabled: false, code: "feature-disabled",
    });

    const fields = fieldsForConfig(source);
    const baseline = draftFromMerged(source, fields);
    expect(filterTrainingFields(fields, baseline, baseline, "__future_runtime_field__", "applicable", "ip_adapter"))
      .toEqual([]);
    expect(filterTrainingFields(fields, baseline, baseline, "__future_runtime_field__", "all", "ip_adapter"))
      .toHaveLength(1);
    expect(filterTrainingFields(fields, { ...baseline, __future_runtime_field__: 8 }, baseline, "__future_runtime_field__", "changed", "ip_adapter"))
      .toHaveLength(1);
    expect(fieldAvailability("adaptive_precision", source, "ip_adapter")).toMatchObject({
      visible: true, enabled: true,
    });
  });

  it("owns the Qwen task selector and defaults old configs to t2i", () => {
    const qwenFields = fieldsForConfig({ model_family: "qwen_image_2_1" });
    expect(qwenFields.find((field) => field.key === "qwen_image_2_1_task")).toMatchObject({
      kind: "select",
      options: ["t2i", "edit"],
    });
    expect(draftFromMerged({ model_family: "qwen_image_2_1" }, qwenFields).qwen_image_2_1_task)
      .toBe("t2i");
    expect(fieldAvailability("qwen_image_2_1_task", { model_family: "qwen_image_2_1" }, "lora"))
      .toMatchObject({ visible: true, enabled: true });
    expect(fieldAvailability("qwen_image_2_1_task", { model_family: "anima" }, "lora"))
      .toMatchObject({ visible: false, enabled: false, code: "qwen-task-family" });
  });

  it("defaults and scopes the Qwen text encoder cache policy", () => {
    const fields = fieldsForConfig({ model_family: "qwen_image_2_1" });
    expect(fields.find((field) => field.key === "qwen_text_encoder_cache_policy")).toMatchObject({
      kind: "select",
      options: ["auto", "cpu_offload", "gpu", "cpu"],
      defaultValue: "auto",
    });
    expect(draftFromMerged({ model_family: "qwen_image_2_1" }, fields).qwen_text_encoder_cache_policy)
      .toBe("auto");
    expect(fieldAvailability("qwen_text_encoder_cache_policy", { model_family: "qwen_image_2_1", qwen_image_2_1_task: "edit" }, "lora"))
      .toMatchObject({ visible: true, enabled: true });
    expect(fieldAvailability("qwen_text_encoder_cache_policy", { model_family: "anima" }, "lora"))
      .toMatchObject({ visible: false, enabled: false, code: "qwen-cache-family" });
  });
});
