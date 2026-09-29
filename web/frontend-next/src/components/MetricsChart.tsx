import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { metricRange, metricSeries } from "./metricSeries";
import { useMetricsCanvas } from "./useMetricsCanvas";

export function MetricsChart({ points, metric = "loss", label = "Loss 趋势", timeAxis = false, total, initialLimit = 2000, autoExpandEmptyWindow = false }: {
  points: Record<string, unknown>[]; metric?: string; label?: string; timeAxis?: boolean; total?: number; initialLimit?: number; autoExpandEmptyWindow?: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [smoothing, setSmoothing] = useState(0);
  const [limit, setLimit] = useState(initialLimit);
  const [focusedIndex, setFocusedIndex] = useState<number | null>(null);
  const series = useMemo(() => metricSeries(points, metric, limit, smoothing, timeAxis), [points, metric, limit, smoothing, timeAxis]);
  useEffect(() => {
    if (autoExpandEmptyWindow && limit !== 0 && !series.data.length && metricSeries(points, metric, 0, smoothing, timeAxis).data.length) {
      setLimit(0);
    }
  }, [autoExpandEmptyWindow, limit, metric, points, series.data.length, smoothing, timeAxis]);
  const metricName = label.replace(/\s*趋势$/, "");
  useMetricsCanvas(ref, series.data, timeAxis, metricName, series.axis);
  const axisName = series.axis === "time" ? "时间" : series.axis === "step" ? "STEP" : "采样序号";
  const focusedPoint = focusedIndex === null ? undefined : series.data[focusedIndex];
  const inspectPoint = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const current = focusedIndex ?? (event.key === "ArrowRight" ? -1 : series.data.length);
    setFocusedIndex(Math.max(0, Math.min(series.data.length - 1, current + (event.key === "ArrowRight" ? 1 : -1))));
  };
  return <section className="chart-section" aria-label={label}>
    <div className="chart-heading">
      <h2>{label}</h2>
      <div className="chart-controls">
        <label className="checkbox-row">窗口
          <select aria-label={`${label}数据窗口`} value={limit} onChange={(e) => setLimit(Number(e.target.value))}>
            <option value={500}>最近 500 点</option><option value={2000}>最近 2000 点</option><option value={0}>全部已读取</option>
          </select>
        </label>
        <label className="checkbox-row">平滑
          <input type="range" min="0" max="0.95" step="0.05" value={smoothing} onChange={(e) => setSmoothing(Number(e.target.value))} />
        </label>
      </div>
    </div>
    <p className="data-scope">{series.data.length} 个有效点 / 已读取 {points.length} 点{total !== undefined && total > points.length ? ` / 共 ${total} 点（前段未读取）` : ""} · {metricRange(series.data, series.axis) || "无有效范围"}</p>
    {series.data.length ? <>
      <div ref={ref} className="metric-chart" role="group" tabIndex={0} aria-label={`${label}，${series.data.length} 个点`} aria-keyshortcuts="ArrowLeft ArrowRight" onKeyDown={inspectPoint} />
      <p className="chart-inspection" aria-live="polite">{focusedPoint ? `检查点 ${focusedIndex! + 1}/${series.data.length} · ${axisName}: ${series.axis === "time" ? new Date(focusedPoint[0]).toLocaleString() : focusedPoint[0]} · ${metricName}: ${formatMetricValue(metricName, focusedPoint[1])}` : ""}</p>
    </> : <p className="empty-state">暂无指标记录</p>}
  </section>;
}

function formatMetricValue(metricName: string, value: number) {
  return metricName === "学习率" ? value.toExponential(3) : value.toPrecision(4);
}
