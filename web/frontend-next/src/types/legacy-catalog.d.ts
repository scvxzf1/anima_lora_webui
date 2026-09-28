declare module "*config/catalog/labels-options.js" {
  export const FIELD_LABEL_ZH: Record<string, string>;
  export const FIELD_OPTIONS: Record<string, (string | number | boolean)[]>;
}
declare module "*pages/config-field-types.js?v=training-field-types-20260927" {
  export function configFieldInputKind(
    key: string,
    value: unknown,
    options?: unknown[] | null,
  ): "text" | "number" | "boolean" | "select" | "json";
  export function normalizeBooleanConfigValue(
    key: string,
    value: unknown,
    fallback?: boolean,
  ): boolean;
}
declare module "*config/catalog/field-help-summary.js" {
  export const FIELD_HELP_SUMMARY_ZH: Record<string, string>;
}
declare module "*pages/config-field-catalog.js" {
  export const CONFIG_STAGE_META: readonly { id: string; clusters: readonly { id: string; label: string }[] }[];
  export const CONFIG_FIELD_CATALOG: Record<
    string,
    {
      stage: "input" | "method" | "training" | "resources";
      location: string;
      cluster: string;
      siblingOrder: number;
      controlledBy: string[];
    }
  >;
  export function configFieldCatalogEntry(key: string): {
    location: string;
    stage: string;
    cluster: string;
  };
}
declare module "*pages/config-values.js" {
  export function displayConfigValue(
    key: string,
    config: Record<string, unknown>,
  ): unknown;
  export function prepareConfigPatch(
    changes: Record<string, unknown>,
    original: Record<string, unknown>,
  ): Record<string, unknown>;
}
declare module "*pages/config-field-availability.js" {
  export function configFieldAvailability(
    key: string,
    context: Record<string, unknown>,
  ): { enabled: boolean; reason: string; code: string | null };
  export function resolveConfigAdapterKind(
    values: Record<string, unknown>,
  ): string;
}
declare module "*pages/config-field-disclosure-rules.js?v=auto-block-swap-20260908-v3" {
  export function configFieldDisclosure(
    key: string,
    context: Record<string, unknown>,
  ): { visible: boolean; reason: string; code: string | null };
}
declare module "*config-form/model-family.js?v=qwen-image-21-v2" {
  export function configureModelFamilyCapabilities(payload: unknown): unknown[];
  export function normalizeModelFamily(value: unknown): string;
  export function modelFamilyOptionSupported(
    key: string,
    family: string,
    option: string,
  ): boolean;
}
