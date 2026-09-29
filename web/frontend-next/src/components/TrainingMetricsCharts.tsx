import { useState } from "react";
import { MetricsChart } from "./MetricsChart";
import { metricSeries } from "./metricSeries";
import { finiteNumber } from "./trainingNumbers";
import "./TrainingMetricsCharts.css";

const METRIC_OPTIONS = [
  { key: "loss", label: "Loss 趋势" },
  { key: "lr", label: "学习率趋势" },
  { key: "cmmd", label: "验证 CMMD" },
] as const;

export function TrainingMetricsCharts({ points, total }: { points: Record<string, unknown>[]; total?: number }) {
  const hasLearningRate = metricSeries(points, "lr", 0, 0, false).data.length > 0;
  const hasRecentLearningRate = metricSeries(points, "lr", 2000, 0, false).data.length > 0;
  const availableMetrics = METRIC_OPTIONS.filter(({ key }) => key === "loss"
    || key === "lr" && hasLearningRate
    || key === "cmmd" && points.some((point) => finiteNumber(point.cmmd) !== undefined));
  const [hiddenMetrics, setHiddenMetrics] = useState<Set<string>>(() => new Set());
  const visibleMetrics = availableMetrics.filter(({ key }) => !hiddenMetrics.has(key));
  const toggleMetric = (key: string, visible: boolean) => setHiddenMetrics((hidden) => {
    const next = new Set(hidden);
    if (visible) next.delete(key);
    else next.add(key);
    return next;
  });
  return <>
    <div className="training-metric-visibility" role="group" aria-label="训练指标显隐">
      {availableMetrics.map(({ key, label }) => <label className="checkbox-row" key={key}>
        <input type="checkbox" checked={!hiddenMetrics.has(key)} onChange={(event) => toggleMetric(key, event.target.checked)} />
        {label}
      </label>)}
    </div>
    {availableMetrics.length ? availableMetrics.map(({ key, label }) => key === "loss"
      ? <MetricsChart key={key} points={points} total={total} hidden={hiddenMetrics.has(key)} />
      : key === "lr"
        ? <MetricsChart key={key} points={points} total={total} metric="lr" label={label} initialLimit={hasRecentLearningRate ? 2000 : 0} autoExpandEmptyWindow hidden={hiddenMetrics.has(key)} />
        : <MetricsChart key={key} points={points} total={total} metric="cmmd" label={label} hidden={hiddenMetrics.has(key)} />)
      : null}
    {!visibleMetrics.length ? <p className="training-metrics-empty">已隐藏所有训练指标</p> : null}
  </>;
}
