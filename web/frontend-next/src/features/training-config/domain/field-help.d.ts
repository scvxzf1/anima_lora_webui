export interface FieldHelp {
  summary: string;
  fill: string;
  benefit: string[];
  cost: string[];
  risk: string[];
  recommend: string | string[];
  ps?: string | string[];
}

export const FIELD_HELP_ZH: Record<string, FieldHelp>;
