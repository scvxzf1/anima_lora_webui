import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { TRAINING_FIELDS, draftFromMerged, trainingPatchValues } from "./trainingForm";
import { fieldAvailability } from "./fieldCatalog";
import { TrainingFieldEditor } from "./TrainingFieldEditor";
import { configFieldDisclosure } from "./domain/config-field-disclosure-rules.js";

afterEach(cleanup);
const keys = ["qwen_sample_swap_mode", "qwen_sample_blocks_to_swap", "sample_vae_decode"];

it("shows Qwen preview controls only for Qwen and enables custom count conditionally", () => {
  for (const key of keys) {
    expect(configFieldDisclosure(key, { modelFamily: "anima" }).visible).toBe(key === "sample_vae_decode");
    expect(configFieldDisclosure(key, { modelFamily: "qwen_image_2_1" }).visible).toBe(true);
  }
  expect(fieldAvailability(keys[1], { model_family: "qwen_image_2_1", qwen_sample_swap_mode: "inherit" }, "lora").enabled).toBe(false);
  expect(fieldAvailability(keys[1], { model_family: "qwen_image_2_1", qwen_sample_swap_mode: "custom" }, "lora").enabled).toBe(true);
});

it("persists explicit zero, decode strategy and switching back to inheritance", () => {
  const fields = TRAINING_FIELDS.filter(field => keys.includes(field.key));
  const defaults = draftFromMerged({}, fields);
  expect(defaults).toEqual({ qwen_sample_swap_mode: "inherit", qwen_sample_blocks_to_swap: 0, sample_vae_decode: "tiled" });
  expect(draftFromMerged({ sample_vae_decode: "auto" }, fields).sample_vae_decode).toBe("auto");
  expect(draftFromMerged({ qwen_sample_vae_decode: "cpu" }, fields).sample_vae_decode).toBe("cpu");
  expect(draftFromMerged({ sample_vae_decode: "tiled", qwen_sample_vae_decode: "cpu" }, fields).sample_vae_decode).toBe("tiled");
  for (const modelFamily of ["anima", "krea2_raw", "z_image", "qwen_image_2_1"]) {
    expect(configFieldDisclosure("sample_vae_decode", { modelFamily }).visible).toBe(true);
  }
  const custom = { qwen_sample_swap_mode: "custom", qwen_sample_blocks_to_swap: 26, sample_vae_decode: "cpu" };
  expect(trainingPatchValues(custom, defaults, fields)).toEqual(custom);
  expect(trainingPatchValues({ ...custom, qwen_sample_blocks_to_swap: 0 }, custom, fields)).toEqual({ qwen_sample_blocks_to_swap: 0 });
  expect(trainingPatchValues({ ...custom, qwen_sample_swap_mode: "inherit" }, custom, fields)).toEqual({ qwen_sample_swap_mode: "inherit" });
});

it("renders Chinese mode and decode choices", () => {
  for (const key of [keys[0], keys[2]]) {
    const field = TRAINING_FIELDS.find(value => value.key === key)!;
    render(<TrainingFieldEditor fields={[field]}
      draft={{ model_family: "qwen_image_2_1", [key]: String(field.defaultValue) }}
      ownKeys={new Set()} disabled={false} onChange={vi.fn()} />);
  }
  expect(screen.getByRole("option", { name: "继承训练配置" })).toBeInTheDocument();
  expect(screen.getByRole("option", { name: "分块 GPU 解码（推荐）" })).toBeInTheDocument();
  expect(screen.getByRole("option", { name: "CPU 解码" })).toBeInTheDocument();
});
