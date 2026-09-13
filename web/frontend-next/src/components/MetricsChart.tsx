import { useMemo, useRef, useState } from "react";
import { metricRange, metricSeries } from "./metricSeries";
import { useMetricsCanvas } from "./useMetricsCanvas";

export function MetricsChart({ points, metric = "loss", label = "Loss 趋势", timeAxis = false, total }: {
  points: Record<string, unknown>[]; metric?: string; label?: string; timeAxis?: boolean; total?: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [smoothing, setSmoothing] = useState(0);
  const [limit, setLimit] = useState(2000);
  const series = useMemo(() => metricSeries(points, metric, limit, smoothing, timeAxis), [points, metric, limit, smoothing, timeAxis]);
  useMetricsCanvas(ref, series.data, timeAxis);
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
    {series.data.length ? <div ref={ref} className="metric-chart" role="img" aria-label={`${label}，${series.data.length} 个点，最新 ${series.data.at(-1)?.[1].toPrecision(4)}`} /> : <p className="empty-state">暂无指标记录</p>}
  </section>;
}
