// Older validation events duplicate CMMD into `loss`; never chart it as training loss.
export const isTrainingMetric = (point: Record<string, unknown>) =>
  point.kind !== "val" && point.ev !== "val";
