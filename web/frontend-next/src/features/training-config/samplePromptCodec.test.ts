import { expect, it } from "vitest";
import { blankSamplePromptRow, parseSamplePromptLine, serializeSamplePromptRow } from "./samplePromptCodec";

it("round-trips a structured sample row with negative prompt and sampling options", () => {
  const row = {
    ...blankSamplePromptRow(),
    prompt: "style preview",
    negative_prompt: "low quality",
    width: "1024",
    height: "768",
    cfg: "4",
    steps: "28",
    seed: "42",
    flow_shift: "3",
    sample_sampler: "euler",
  };
  expect(parseSamplePromptLine(serializeSamplePromptRow(row))).toEqual(row);
});

it("keeps unknown TXT options through a field update", () => {
  const row = parseSamplePromptLine("style one --w 512 --custom keep");
  expect(serializeSamplePromptRow({ ...row, width: "1024" })).toBe("style one --w 1024 --custom keep");
});
