import { parse } from "smol-toml";
import { finiteNumber, latestNumber } from "../../components/trainingNumbers";
import { isTrainingMetric } from "../../components/metricSemantics";
import type { HistoryResumeSource, HistoryTaskDetail, HistoryTaskSummary } from "./api";

const STATE_LABELS: Record<string, string> = {
  idle: "完成", running: "运行中", error: "异常", failed: "异常",
  interrupted: "已中断", stopped: "已中断",
};
export const historyStateLabel = (state?: string) => STATE_LABELS[state || ""] || state || "未知";
export const historyTaskName = (task?: HistoryTaskSummary) =>
  String(task?.name || task?.history_run_label || task?.id || "历史任务");

export function historySummary(task?: HistoryTaskSummary, metrics: Record<string, unknown>[] = []) {
  // Persisted summary is shared with the list; series only fills missing values.
  const training = task?.job === "training";
  const trainingMetrics = metrics.filter(isTrainingMetric);
  return {
    step: training ? finiteNumber(task.last_step) ?? latestNumber(trainingMetrics, "step") : undefined,
    loss: training ? finiteNumber(task.final_loss) ?? latestNumber(trainingMetrics, "loss") : undefined,
    lr: training ? latestNumber(trainingMetrics, "lr") : undefined,
  };
}

export function snapshotFields(detail: HistoryTaskDetail) {
  try {
    return { values: parse(detail.config_toml || "") as Record<string, unknown>, invalid: false };
  } catch {
    return { values: {} as Record<string, unknown>, invalid: true };
  }
}

export function displayField(value: unknown): string {
  if (typeof value === "boolean") return value ? "开启" : "关闭";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "未记录";
  return typeof value === "string" && value.trim() ? value : "未记录";
}

export function historyCheckpointLabel(resume: HistoryResumeSource | string | null | undefined): string {
  if (typeof resume === "string") return displayField(resume);
  if (!resume) return "未记录";
  const name = displayField(resume.checkpoint_name);
  const path = displayField(resume.checkpoint);
  const checkpoint = name !== "未记录" ? name : path !== "未记录" ? path : "";
  const step = finiteNumber(resume.checkpoint_step);
  const stepLabel = step !== undefined && Number.isSafeInteger(step) && step >= 0 ? `step ${step}` : "";
  return [checkpoint, stepLabel].filter(Boolean).join(" · ") || "未记录";
}

export function taskOutcome(task?: HistoryTaskSummary) {
  if (task?.message?.trim()) return task.message;
  if (task?.state === "running") return "任务运行中，最新进度以当前监控为准。";
  if (["error", "failed", "interrupted", "stopped"].includes(task?.state || ""))
    return "未保存结束原因，可查看任务日志。";
  return task?.finished_at ? "执行已结束，产物与检查点的可用性需分别确认。" : "未记录结束时间。";
}
