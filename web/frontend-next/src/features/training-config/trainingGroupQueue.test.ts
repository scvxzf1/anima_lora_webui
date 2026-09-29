import { describe, expect, it } from "vitest";
import type { TrainingConfigGroup } from "../../api/trainingContext";
import { groupQueueItems, queueableTrainingFiles } from "./trainingGroupQueue";

describe("training group queue entries", () => {
  const group: TrainingConfigGroup = {
    id: "imported",
    label: "Imported",
    methods_subdir: "imported",
    files: [
      {
        path: "configs/imported/first.toml",
        filename: "first.toml",
        label: "First",
        method: "lora",
        trainable: true,
      },
      {
        path: "configs/imported/second.toml",
        filename: "second.toml",
        trainable: true,
      },
      { path: "configs/datasets/data.toml", trainable: true },
      { path: "configs/imported/readonly.toml", trainable: false },
    ],
  };

  it("keeps only trainable configs in group order and supplies a stable request", () => {
    const files = queueableTrainingFiles(group);
    expect(files.map((file) => file.path)).toEqual([
      "configs/imported/first.toml",
      "configs/imported/second.toml",
    ]);
    expect(groupQueueItems(files, "low_vram", group)).toEqual([
      {
        variant: "lora",
        preset: "low_vram",
        methods_subdir: "imported",
        config_file: "configs/imported/first.toml",
        filename: "first.toml",
        label: "First",
        confirm_preprocess: true,
      },
      {
        variant: "second",
        preset: "low_vram",
        methods_subdir: "imported",
        config_file: "configs/imported/second.toml",
        filename: "second.toml",
        label: "second.toml",
        confirm_preprocess: true,
      },
    ]);
  });
});
