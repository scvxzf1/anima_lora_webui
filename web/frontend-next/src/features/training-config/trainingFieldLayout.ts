import type { TrainingFieldSpec } from "./trainingForm";

export type TrainingFieldLayout = {
  /** `wide` cards consume two desktop grid tracks and dense-flow fills the gaps. */
  span: "single" | "wide";
  /** Used by the card layout to keep long paths/JSON compact but readable. */
  width: "compact" | "wide";
};

export type TrainingGridColumns = 1 | 2 | 3 | 4;

const WIDE_KEY_RE = /(?:_args|_weights|_path|_dir|_file|_root|_config)$/;
const WIDE_KEYS = new Set([
  "pretrained_model_name_or_path",
  "dataset_config",
  "network_args",
  "optimizer_args",
  "output_dir",
]);

/**
 * Infer a card's footprint from field metadata instead of maintaining a
 * per-field layout table. The group count is intentionally part of the input:
 * two-field groups stay balanced while larger groups can use dense placement.
 */
export function getTrainingFieldLayout(
  field: Pick<TrainingFieldSpec, "key" | "kind">,
  groupTotalCount: number,
): TrainingFieldLayout {
  const wide = field.kind === "json" || WIDE_KEYS.has(field.key) || WIDE_KEY_RE.test(field.key);
  if (!wide) return { span: "single", width: "compact" };
  // A lone field should not claim an arbitrary number of columns; its container
  // is capped by CSS. For groups, spanning two tracks lets short cards fill the
  // remaining track through `grid-auto-flow: row dense`.
  return {
    span: groupTotalCount > 1 ? "wide" : "single",
    width: "wide",
  };
}

function fieldSpan(field: Pick<TrainingFieldSpec, "key" | "kind">): number {
  return getTrainingFieldLayout(field, 2).span === "wide" ? 2 : 1;
}

/**
 * Choose the widest grid that keeps the dense placement balanced. The score
 * prefers fewer empty tracks, then fewer rows, and finally more columns so a
 * five-field group with one wide field becomes a compact 3 + 2 arrangement
 * instead of a 4 + 1 layout with a half-empty second row.
 */
export function getTrainingGridColumns(
  fields: Array<Pick<TrainingFieldSpec, "key" | "kind">>,
): TrainingGridColumns {
  if (fields.length <= 1) return 1;
  const spans = fields.map(fieldSpan);
  let best: { columns: TrainingGridColumns; cost: number; holes: number; rows: number } | undefined;
  for (const columns of [1, 2, 3, 4] as const) {
    const rows: boolean[][] = [];
    for (const span of spans) {
      const width = Math.min(span, columns);
      let placed = false;
      for (const row of rows) {
        for (let start = 0; start <= columns - width; start += 1) {
          if (row.slice(start, start + width).some(Boolean)) continue;
          row.fill(true, start, start + width);
          placed = true;
          break;
        }
        if (placed) break;
      }
      if (!placed) {
        const row = Array<boolean>(columns).fill(false);
        row.fill(true, 0, width);
        rows.push(row);
      }
    }
    const holes = rows.reduce((total, row) => total + row.filter((cell) => !cell).length, 0);
    const cost = holes * 3 + rows.length;
    if (!best || cost < best.cost || (cost === best.cost && columns > best.columns)) {
      best = { columns, cost, holes, rows: rows.length };
    }
  }
  return best!.columns;
}
