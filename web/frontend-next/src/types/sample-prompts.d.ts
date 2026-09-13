declare module "*sample-prompts/model.js" {
  export type SamplePromptRow = Record<"prompt" | "negative_prompt" | "height" | "width" | "cfg" | "steps" | "seed" | "flow_shift" | "sample_sampler" | "extra", string>;
  export function parseSamplePromptLine(line: string): SamplePromptRow;
  export function serializeSamplePromptRow(row: SamplePromptRow): string;
}
