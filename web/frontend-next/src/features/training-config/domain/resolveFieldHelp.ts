import { FIELD_HELP_ZH, type FieldHelpV2 } from "./field-help.js";

const HELP_ALIASES: Record<string, string> = {
  qwen_sample_vae_decode: "sample_vae_decode",
};

export function resolveFieldHelp(key: string): FieldHelpV2 | undefined {
  const canonicalKey = Object.hasOwn(HELP_ALIASES, key) ? HELP_ALIASES[key] : key;
  if (!Object.hasOwn(FIELD_HELP_ZH, canonicalKey)) return undefined;
  return FIELD_HELP_ZH[canonicalKey];
}
