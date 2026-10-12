import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { combinedTrainingSeries, formatTrainingValue, trainingMetricNames } from "./combinedTrainingSeries";
import { metricWindows } from "./metricWindows";
import { useCombinedTrainingCanvas } from "./useCombinedTrainingCanvas";

export const COMBINED_TRAINING_LABEL = "Loss 与学习率趋势";
export function CombinedTrainingChart({ points, total, visible, values }: {
  points: Record<string, unknown>[]; total?: number; visible: boolean[]; values: boolean[];
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [smoothing, setSmoothing] = useState(0);
  const [segment, setSegment] = useState<number | null>(null);
  const [focusedIndex, setFocusedIndex] = useState<number | null>(null);
  const series = useMemo(() => combinedTrainingSeries(points, segment, smoothing), [points, segment, smoothing]);
  const inspectionData = useMemo(() => series.data.filter((point) => visible.some((show, column) => show && Number.isFinite(point[column + 1]))), [series.data, visible]);
  const focusedPoint = focusedIndex === null ? undefined : inspectionData[focusedIndex];
  useEffect(() => setFocusedIndex(null), [points, segment, smoothing, visible[0], visible[1]]);
  useCombinedTrainingCanvas(ref, series.data, series.axis, visible, values);
  const inspect = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const current = focusedIndex ?? (event.key === "ArrowRight" ? -1 : inspectionData.length);
    setFocusedIndex(Math.max(0, Math.min(inspectionData.length - 1, current + (event.key === "ArrowRight" ? 1 : -1))));
  };
  const range = inspectionData.length ? `${series.axis === "step" ? "STEP" : "采样序号"} ${inspectionData[0][0]}–${inspectionData.at(-1)![0]}` : "无有效范围";
  return <section className="chart-section" aria-label={COMBINED_TRAINING_LABEL} hidden={!visible.some(Boolean)}>
    <div className="chart-heading">
      <h2>{COMBINED_TRAINING_LABEL}</h2>
      <div className="chart-controls">
        <label className="checkbox-row">窗口
          <select aria-label={`${COMBINED_TRAINING_LABEL}数据窗口`} value={segment === null ? "all" : segment} onChange={(event) => setSegment(event.target.value === "all" ? null : Number(event.target.value))}>
            <option value="all">全部已读取</option>
            {metricWindows(series.loaded).map(({ start, end }) => <option key={start} value={start}>{start}–{end} 点</option>)}
          </select>
        </label>
        <label className="checkbox-row">平滑<input aria-label={`${COMBINED_TRAINING_LABEL}平滑`} type="range" min="0" max="0.95" step="0.05" value={smoothing} onChange={(event) => setSmoothing(Number(event.target.value))} /></label>
      </div>
    </div>
    <p className="training-chart-legend">{visible[0] ? <span className="training-loss-legend">Loss · 左轴</span> : null}{visible[1] ? <span className="training-lr-legend">学习率 · 右轴</span> : null}</p>
    <p className="data-scope">{visible.flatMap((show, column) => show ? [`${trainingMetricNames[column]} ${series.valid[column]} 个有效点`] : []).join(" · ")} · 渲染 {series.data.length} 点 / 已读取 {series.loaded} 点{series.loaded !== points.length ? `（源记录 ${points.length} 条，已过滤非训练事件）` : ""}{total !== undefined && total > points.length ? ` / 共 ${total} 条源记录（部分未读取）` : ""} · {range}</p>
    {inspectionData.length ? <>
      <div ref={ref} className="metric-chart" role="group" tabIndex={0} aria-label={`${COMBINED_TRAINING_LABEL}，${series.data.length} 个点`} aria-keyshortcuts="ArrowLeft ArrowRight" onKeyDown={inspect} />
      <p className="chart-inspection" aria-live="polite">{focusedPoint ? `检查点 ${focusedIndex! + 1}/${inspectionData.length} · ${series.axis === "step" ? "STEP" : "采样序号"}: ${focusedPoint[0]}${visible.flatMap((show, column) => show && values[column] ? [` · ${trainingMetricNames[column]}: ${formatTrainingValue(column, focusedPoint[column + 1])}`] : []).join("")}` : ""}</p>
    </> : <p className="empty-state">暂无指标记录</p>}
  </section>;
}
