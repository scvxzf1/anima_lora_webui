declare module "*sample-prompts/model.js" {
  export type SamplePromptRow = Record<"prompt" | "negative_prompt" | "height" | "width" | "cfg" | "steps" | "seed" | "flow_shift" | "sample_sampler" | "extra" | "sample_task" | "reference_image", string> & { reference_images: string[]; json?: Record<string, unknown>; json_error?: string };
  export function samplePromptsContentNeedsTextMode(content: string): boolean;
  export function parseSamplePromptLine(line: string): SamplePromptRow;
  export function serializeSamplePromptRow(row: SamplePromptRow): string;
}
