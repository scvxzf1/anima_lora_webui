import type { HistoryTimeline } from "./api";

export function orderHistoryTimeline(payload: HistoryTimeline, taskIds: string[]): HistoryTimeline {
  const rank = new Map(taskIds.map((id, index) => [id, index]));
  const taskId = (item: { id?: string }) => String(item.id || "");
  const tasks = [...(payload.tasks || [])].sort((a, b) => (rank.get(taskId(a)) ?? Number.MAX_SAFE_INTEGER) - (rank.get(taskId(b)) ?? Number.MAX_SAFE_INTEGER));
  const taskRank = new Map(tasks.map((task, index) => [taskId(task), index + 1]));
  const segments = [...(payload.segments || [])].sort((a, b) => segmentRank(a, taskRank) - segmentRank(b, taskRank));
  const metricByTask = new Map<string, typeof payload.metrics>();
  for (const point of payload.metrics || []) {
    const id = String(point.source_task_id || "");
    metricByTask.set(id, [...(metricByTask.get(id) || []), point]);
  }
  let visualStep = 1;
  const metrics = tasks.flatMap((task, index) => {
    const points = metricByTask.get(taskId(task)) || [];
    return points.map((point, pointIndex) => ({
      ...point,
      source_task_index: index + 1,
      visual_step: visualStep++,
      stage_break_before: index > 0 && pointIndex === 0,
    }));
  });
  const logByTask = new Map<string, typeof payload.logs>();
  for (const log of payload.logs || []) {
    const id = String(log.source_task_id || "");
    logByTask.set(id, [...(logByTask.get(id) || []), log]);
  }
  const logs = tasks.flatMap((task, index) => (logByTask.get(taskId(task)) || []).map((log) => ({ ...log, source_task_index: index + 1 })));
  return { ...payload, tasks, segments, metrics, logs };
}

function segmentRank(segment: { task?: { id?: string } }, rank: Map<string, number>) {
  return rank.get(String(segment.task?.id || "")) ?? Number.MAX_SAFE_INTEGER;
}

export function timelineLogText(log: { line?: string; source_task_index?: number; source_task_label?: string }) {
  return `[${log.source_task_label || `任务${log.source_task_index || "?"}`}] ${log.line || ""}`;
}
