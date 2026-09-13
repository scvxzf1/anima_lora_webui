import { Link } from "react-router-dom";
import type { HistoryTaskSummary } from "./api";
import { HistoryDragItem } from "./HistoryDrag";
import { historyStateLabel, historySummary, historyTaskName } from "./historySummary";
import { formatLoss, formatStep } from "../../components/trainingNumbers";
import { rememberHistoryTask } from "./historyNavigation";
import { historyReturnSearch } from "./useHistoryRestore";
export { historyTaskName } from "./historySummary";

export type HistoryTaskCardProps = {
  task: HistoryTaskSummary;
  selected: string[];
  busy: boolean;
  listSearch: string;
  onToggle: (id: string) => void;
};
export function HistoryTaskCard({
  task,
  selected,
  busy,
  listSearch,
  onToggle,
}: HistoryTaskCardProps) {
  const name = historyTaskName(task);
  const summary = historySummary(task);
  const returnSearch = historyReturnSearch(listSearch, Number(new URLSearchParams(listSearch).get("depth")) || 1, String(task.id));
  return (
    <article className="history-card" data-history-task={task.id} data-state={task.state || "unknown"}>
      <HistoryDragItem
        id={`task:${task.id}`}
        disabled={busy || !task.id}
        data={{
          kind: "task",
          collection: task.group || "",
          label: name,
          taskIds: task.id ? [task.id] : [],
        }}
      >
        <input
          type="checkbox"
          aria-label={`选择 ${name}`}
          checked={selected.includes(String(task.id))}
          disabled={busy || !task.id}
          onChange={() => onToggle(String(task.id))}
        />
      </HistoryDragItem>
      <Link
        to={`/history/${encodeURIComponent(String(task.id))}?${new URLSearchParams({ from: returnSearch })}`}
        state={{ listSearch: returnSearch }}
        onClick={() => rememberHistoryTask(returnSearch, String(task.id))}
        className="history-card-link"
      >
        <span className="history-state" data-state={task.state || "unknown"}>
          {historyStateLabel(task.state)}
        </span>
        <h3>{name}</h3>
        <p>
          {task.history_source_config_file ||
            task.run_dir ||
            task.output_dir ||
            "—"}
        </p>
      </Link>
      <div className="history-card-meta">
        <span>{task.job === "preprocess" ? "预处理" : task.job === "training" ? "训练" : "任务类型未记录"}</span>
        <span>{task.group || "未分类"}</span>
        <span>{task.started_at_text || ""}</span>
        <span>
          {summary.step != null
            ? `STEP ${formatStep(summary.step)}`
            : `${task.log_count ?? 0} 日志`}{" "}
          ·{" "}
          {summary.loss != null
            ? `LOSS ${formatLoss(summary.loss)}`
            : `${task.metric_count ?? 0} 指标`}
        </span>
        {task.loss_preview?.length ? (
          <Sparkline values={task.loss_preview} />
        ) : null}
      </div>
    </article>
  );
}

function Sparkline({ values }: { values: number[] }) {
  const points = values.slice(-32);
  const min = Math.min(...points);
  const span = Math.max(...points) - min || 1;
  const polyline = points
    .map(
      (value, index) =>
        `${(index / Math.max(1, points.length - 1)) * 72},${20 - ((value - min) / span) * 16}`,
    )
    .join(" ");
  return (
    <svg
      className="history-task-sparkline"
      viewBox="0 0 72 20"
      role="img"
      aria-label={`损失趋势，${points.length} 个点`}
    >
      <polyline points={polyline} />
    </svg>
  );
}
