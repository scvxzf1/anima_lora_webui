export interface FieldHelpSection {
  title: string;
  paragraphs: string[];
}

export interface FieldHelpV2 {
  version: 2;
  summary: string;
  recommendation: string;
  options?: Array<{ value: string; when: string; detail?: string }>;
  guidance?: FieldHelpSection[];
  cautions?: string[];
  advanced?: FieldHelpSection[];
  relatedKeys?: string[];
}

export type FieldHelp = FieldHelpV2;

export const FIELD_HELP_ZH: Record<string, FieldHelp>;
