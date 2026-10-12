// Untagged records are legacy training samples. Explicit events must be training.
const trainingEvents = new Set(["step", "train", "training"]);
export const isTrainingMetric = (point: Record<string, unknown>) =>
  [point.kind, point.ev].every((event) => event === undefined || event === null || event === "" || trainingEvents.has(String(event)));
