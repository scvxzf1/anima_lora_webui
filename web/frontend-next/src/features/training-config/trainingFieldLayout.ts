import type { TrainingFieldSpec } from "./trainingForm";

export type TrainingFieldLayout = {
  span: "single" | "wide";
  width: "compact" | "wide";
};
export type TrainingGridColumns = 12;

/** All text/JSON fields take half a desktop row, independent of group size. */
export function getTrainingFieldLayout(
  field: Pick<TrainingFieldSpec, "key" | "kind">,
  _groupTotalCount: number,
): TrainingFieldLayout {
  return field.kind === "text" || field.kind === "json"
    ? { span: "wide", width: "wide" }
    : { span: "single", width: "compact" };
}

export function getTrainingGridColumns(
  _fields: Array<Pick<TrainingFieldSpec, "key" | "kind">>,
): TrainingGridColumns {
  return 12;
}
