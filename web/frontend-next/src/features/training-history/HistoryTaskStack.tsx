import { useRef, useEffect } from "react";
import type { CSSProperties } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { HistoryDragItem } from "./HistoryDrag";
import { HistoryTaskCard, type HistoryTaskCardProps } from "./HistoryTaskCard";
import type { historyStacks } from "./historyOrder";
import { formatLoss, formatStep } from "../../components/trainingNumbers";
import { historySummary } from "./historySummary";
import { useHistoryStackState } from "./historyNavigation";

type Props = Omit<HistoryTaskCardProps, "task"> & {
  stack: ReturnType<typeof historyStacks>[number];
  onToggleGroup: (ids: string[]) => void;
};
export function HistoryTaskStack({ stack, onToggleGroup, ...props }: Props) {
  const { open, page, setPage, toggle } = useHistoryStackState(stack.id);
  const checkbox = useRef<HTMLInputElement>(null);
  const ids = stack.tasks.flatMap((task) => (task.id ? [task.id] : []));
  const selectedCount = ids.filter((id) => props.selected.includes(id)).length;
  useEffect(() => {
    if (checkbox.current)
      checkbox.current.indeterminate =
        selectedCount > 0 && selectedCount < ids.length;
  }, [selectedCount, ids.length]);
  const currentPage = Math.min(
    page,
    Math.max(0, Math.ceil(stack.tasks.length / 10) - 1),
  );
  const training = stack.tasks.filter((task) => task.job === "training").length;
  const errors = stack.tasks.filter((task) =>
    ["error", "interrupted", "failed", "stopped"].includes(task.state || ""),
  ).length;
  const statusCounts = [
    stack.tasks.filter((task) => task.state === "idle").length,
    stack.tasks.filter((task) => ["error", "failed"].includes(task.state || "")).length,
    stack.tasks.filter((task) => task.state === "running").length,
    stack.tasks.filter((task) => ["interrupted", "stopped"].includes(task.state || "")).length,
  ];
  const total = Math.max(1, stack.tasks.length);
  const statusGradient = `linear-gradient(to bottom, #2f9e62 0 ${statusCounts[0] / total * 100}%, #d34b4b ${statusCounts[0] / total * 100}% ${(statusCounts[0] + statusCounts[1]) / total * 100}%, #3d7edb ${(statusCounts[0] + statusCounts[1]) / total * 100}% ${(statusCounts[0] + statusCounts[1] + statusCounts[2]) / total * 100}%, #c99527 ${(statusCounts[0] + statusCounts[1] + statusCounts[2]) / total * 100}% 100%)`;
  const latest = stack.tasks.find(
    (task) => historySummary(task).step != null || historySummary(task).loss != null,
  );
  const summary = historySummary(latest);
  return (
    <section className="history-task-stack" style={{ "--history-status-gradient": statusGradient } as CSSProperties}>
      <HistoryDragItem
        id={`config-stack:${stack.id}`}
        disabled={props.busy}
        data={{
          kind: "config",
          collection: stack.collection,
          key: stack.key,
          region: "stack",
          label: stack.label,
          taskIds: ids,
        }}
      >
        <input
          ref={checkbox}
          type="checkbox"
          aria-label={`选择配置组 ${stack.label}`}
          disabled={props.busy || !ids.length}
          checked={ids.length > 0 && selectedCount === ids.length}
          onChange={() => onToggleGroup(ids)}
        />
        <button
          className="history-stack-toggle"
          type="button"
          aria-expanded={open}
          onClick={toggle}
        >
          {open ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
          <span>
            <strong>{stack.label}</strong>
            <small>
              {stack.tasks.length} 条 · {training} 训练 ·{" "}
              {stack.tasks.length - training} 预处理
              {errors ? ` · ${errors} 异常` : ""}
            </small>
          </span>
        </button>
        <span className="history-stack-metrics">
          {summary.step != null ? <b>STEP {formatStep(summary.step)}</b> : null}
          {summary.loss != null ? (
            <b>LOSS {formatLoss(summary.loss)}</b>
          ) : null}
          {latest?.loss_preview?.length ? (
            <MiniSparkline values={latest.loss_preview} />
          ) : null}
        </span>
        <span className="history-stack-collection">
          {stack.collection || "未分类"}
        </span>
      </HistoryDragItem>
      {open && (
        <div className="history-stack-tasks">
          {stack.tasks
            .slice(currentPage * 10, (currentPage + 1) * 10)
            .map((task) => (
              <HistoryTaskCard key={task.id} {...props} task={task} />
            ))}
          {stack.tasks.length > 10 && (
            <div className="toolbar">
              <button
                type="button"
                disabled={currentPage === 0}
                onClick={() => setPage(currentPage - 1)}
              >
                上一页任务
              </button>
              <span>
                第 {currentPage + 1} / {Math.ceil(stack.tasks.length / 10)} 页
              </span>
              <button
                type="button"
                disabled={(currentPage + 1) * 10 >= stack.tasks.length}
                onClick={() => setPage(currentPage + 1)}
              >
                下一页任务
              </button>
            </div>
          )}
        </div>
      )}
    </section>
  );
}

function MiniSparkline({ values }: { values: number[] }) {
  const points = values.slice(-24);
  const min = Math.min(...points);
  const span = Math.max(...points) - min || 1;
  const path = points
    .map(
      (value, index) =>
        `${(index / Math.max(1, points.length - 1)) * 64},${18 - ((value - min) / span) * 14}`,
    )
    .join(" ");
  return (
    <svg
      className="history-stack-sparkline"
      viewBox="0 0 64 18"
      role="img"
      aria-label={`配置组损失趋势，${points.length} 个点`}
    >
      <polyline points={path} />
    </svg>
  );
}
