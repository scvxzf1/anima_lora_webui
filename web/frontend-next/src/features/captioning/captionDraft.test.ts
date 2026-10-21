import { describe, expect, it } from "vitest";
import { buildCaptionJobPayload } from "./captionDraft";

const base = {
  datasetFile: "configs/datasets/example.toml",
  datasetIndex: 2,
  source: "training",
  profileId: "profile-1",
  prompt: "user prompt",
  systemPrompt: "system prompt",
  items: ["a.png", "b.png"],
};

describe("caption job payload", () => {
  it("carries the write policy and a trimmed trigger word", () => {
    const payload = buildCaptionJobPayload({
      ...base,
      onExisting: "append",
      triggerWord: "  MyStyle  ",
    });

    expect(payload.on_existing).toBe("append");
    expect(payload.trigger_word).toBe("MyStyle");
    expect(payload.items).toEqual([{ file: "a.png" }, { file: "b.png" }]);
    expect(payload.dataset_index).toBe(2);
  });

  it("defaults to overwrite and an empty trigger word", () => {
    const payload = buildCaptionJobPayload({ ...base, onExisting: "", triggerWord: "" });

    expect(payload.on_existing).toBe("overwrite");
    expect(payload.trigger_word).toBe("");
  });
});
