import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";

import { TrainingFieldEditor } from "../TrainingFieldEditor";
import { fieldAvailability, fieldsForConfig } from "../fieldCatalog";
import { FIELD_HELP_SUMMARY_ZH } from "./field-help-summary.js";
// @ts-expect-error Legacy domain JS modules do not yet ship TypeScript declarations.
import { FIELD_OPTIONS } from "./labels-options.js";
// @ts-expect-error Legacy domain JS modules do not yet ship TypeScript declarations.
import { collectConfigDraftChanges, displayConfigValue, prepareConfigPatch } from "./config-values.js";
// @ts-expect-error Legacy domain JS modules do not yet ship TypeScript declarations.
import { NETWORK_ARG_FIELD_MAP } from "./defaults.js";
// @ts-expect-error Legacy domain JS modules do not yet ship TypeScript declarations.
import { FORM_UI_PERSIST_DEFAULT_FIELDS, NETWORK_ARG_FIELD_SPECS } from "./defaults.js";

afterEach(cleanup);

it("keeps LoKr backend defaults, choices, and network args round-trippable", () => {
  const backendFields = [
    ["lokr_grouped_delta_backend", "triton"],
    ["lokr_grouped_delta_backward_backend", "triton_grad_w1_w2_grad_x"],
  ] as const;

  for (const [key, value] of backendFields) {
    expect(NETWORK_ARG_FIELD_MAP.get(key)?.default).toBe(value);
    expect(displayConfigValue(key, {})).toBe(value);
    expect(FIELD_HELP_SUMMARY_ZH[key]).toBeTruthy();
  }
  expect(FIELD_OPTIONS.lokr_grouped_delta_backend).toEqual(["eager", "triton"]);
  expect(FIELD_OPTIONS.lokr_grouped_delta_backward_backend).toEqual([
    "eager",
    "triton_grad_x",
    "triton_grad_w2_partial",
    "triton_grad_w2_grad_x",
    "triton_grad_w1_w2_grad_x",
  ]);

  const original = {
    network_args: [
      "lokr_factor_group_size=8",
      "lokr_grouped_delta_backend=eager",
      "unrelated_flag=keep",
    ],
  };
  expect(displayConfigValue("lokr_grouped_delta_backend", original)).toBe("eager");
  expect(
    prepareConfigPatch(
      { lokr_grouped_delta_backend: "triton", lokr_grouped_delta_backward_backend: "triton_grad_x" },
      original,
    ),
  ).toEqual({
    network_args: [
      "lokr_factor_group_size=8",
      "lokr_grouped_delta_backend=triton",
      "unrelated_flag=keep",
      "lokr_grouped_delta_backward_backend=triton_grad_x",
    ],
  });
});

it("gates LoKr and ConvRot fields by adapter and base compute", () => {
  expect(
    fieldAvailability(
      "lokr_grouped_delta_backend",
      { model_family: "anima", lora_adapter_kind: "lora", base_compute: "bf16" },
      "lora",
    ).enabled,
  ).toBe(false);
  expect(
    fieldAvailability(
      "lokr_grouped_delta_backend",
      { model_family: "anima", lora_adapter_kind: "lokr", base_compute: "bf16" },
      "lora",
    ).enabled,
  ).toBe(true);
  expect(
    fieldAvailability(
      "convrot_group_size",
      { model_family: "anima", base_compute: "bf16" },
      "lora",
  ).code,
  ).toBe("feature-disabled");
  expect(
    fieldAvailability(
      "convrot_group_size",
      { model_family: "anima", base_compute: "w8a16_convrot" },
      "lora",
    ).enabled,
  ).toBe(true);
});

it("restores only declared defaults and collects field-level draft changes", () => {
  expect(displayConfigValue("train_batch_size", {})).toBe(1);
  expect(displayConfigValue("lokr_grouped_delta_backend", {})).toBe("triton");
  expect(displayConfigValue("unregistered_extension", {})).toBe("");

  const baseline = { network_dim: 16, network_alpha: 16, custom_extension: "keep" };
  expect(
    collectConfigDraftChanges({
      scopeKeys: ["network_dim", "network_alpha"],
      baselineValues: baseline,
      draftValues: {
        network_dim: 32,
        network_alpha: 16,
        custom_extension: "changed-outside-scope",
      },
    }),
  ).toEqual({ network_dim: 32 });
});

it("derives and persists the precision preference through training precision fields", () => {
  expect(displayConfigValue("precision_preference", {})).toBe("bf16");
  expect(displayConfigValue("precision_preference", { mixed_precision: "fp16" })).toBe("fp16");
  expect(displayConfigValue("precision_preference", { full_fp16: true })).toBe("fp16");
  expect(displayConfigValue("precision_preference", { mixed_precision: "no" })).toBe("fp32");
  expect(FORM_UI_PERSIST_DEFAULT_FIELDS.has("precision_preference")).toBe(true);

  expect(
    collectConfigDraftChanges({
      scopeKeys: ["precision_preference"],
      baselineValues: { mixed_precision: "fp16" },
      draftValues: { precision_preference: "fp16" },
    }),
  ).toEqual({});
  expect(
    collectConfigDraftChanges({
      scopeKeys: ["precision_preference"],
      baselineValues: { mixed_precision: "fp16" },
      draftValues: { precision_preference: "fp32" },
    }),
  ).toEqual({ precision_preference: "fp32" });
  expect(
    prepareConfigPatch(
      { precision_preference: "fp32" },
      { full_fp16: true, full_bf16: true },
    ),
  ).toEqual({ mixed_precision: "no", full_fp16: false, full_bf16: false });
});

it("keeps Soft Tokens advanced defaults and choices in the config catalog", () => {
  const softTokenSpecs = NETWORK_ARG_FIELD_SPECS.filter(
    (spec: { family: string; key: string; default: unknown }) =>
      spec.family === "soft_tokens",
  );
  const specDefaults = Object.fromEntries(
    softTokenSpecs.map((spec: { key: string; default: unknown }) => [spec.key, spec.default]),
  );
  expect(specDefaults).toMatchObject({
    contrastive_objective: "infonce",
    contrastive_negative_mode: "shuffled",
    contrastive_every_n: 1,
    n_layers: 10,
    n_t_buckets: 100,
    splice_position: "end_of_sequence",
    softrank_softness: 0.1,
    softrank_method: "neuralsort",
    dual_bank: false,
  });
  for (const spec of softTokenSpecs as { key: string; default: unknown }[]) {
    expect(displayConfigValue(spec.key, {})).toBe(spec.default);
  }
  expect(FIELD_OPTIONS.contrastive_objective).toEqual(["infonce", "softrank"]);
  expect(FIELD_OPTIONS.softrank_method).toEqual(["neuralsort", "softsort"]);
  expect(FIELD_OPTIONS.dual_bank).toEqual([false, true]);
  const visibleKeys = new Set(
    fieldsForConfig({ model_family: "anima", method: "soft_tokens" }).map(
      (field) => field.key,
    ),
  );
  for (const key of ["softrank_softness", "softrank_method", "dual_bank"]) {
    expect(visibleKeys.has(key)).toBe(true);
  }
  expect(
    softTokenSpecs.some((spec: { key: string }) =>
      spec.key.toLowerCase().includes("agsm"),
    ),
  ).toBe(false);
  expect(Object.keys(FIELD_OPTIONS).some((key) => key.toLowerCase().includes("agsm"))).toBe(false);
});

it("renders the current field help summary in the Next editor", () => {
  const field = fieldsForConfig({ model_family: "anima", network_dim: 16 }).find(
    (candidate) => candidate.key === "network_dim",
  )!;
  render(
    <TrainingFieldEditor
      fields={[field]}
      draft={{ model_family: "anima", network_dim: 16 }}
      ownKeys={new Set(["network_dim"])}
      disabled={false}
      onChange={() => undefined}
      method="lora"
    />,
  );
  expect(screen.getByText(FIELD_HELP_SUMMARY_ZH.network_dim)).toBeInTheDocument();
});
