import { finiteNumber } from "../../components/trainingNumbers";
import { snapshotFields } from "./historySummary";
import type { HistoryTaskDetail } from "./api";

export function overviewProgress(detail: HistoryTaskDetail) {
  const { values } = snapshotFields(detail);
  const epochs = finiteNumber(values.max_train_epochs);
  const steps = finiteNumber(values.max_train_steps);
  const epochTarget = epochs !== undefined && Number.isSafeInteger(epochs) && epochs > 0 ? epochs : undefined;
  const stepTarget = !epochTarget && steps !== undefined && Number.isSafeInteger(steps) && steps > 0 ? steps : undefined;
  return { epochTarget, stepTarget };
}
