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
  const [hiddenValues, setHiddenValues] = useState<Set<string>>(() => new Set());
  const visibleMetrics = availableMetrics.filter(({ key }) => !hiddenMetrics.has(key));
  const toggleMetric = (key: string, visible: boolean) => setHiddenMetrics((hidden) => {
    const next = new Set(hidden);
    if (visible) next.delete(key);
    else next.add(key);
    return next;
  });
  const toggleValue = (key: string, visible: boolean) => setHiddenValues((hidden) => {
    const next = new Set(hidden);
    if (visible) next.delete(key);
    else next.add(key);
    return next;
  });
  const latestValues = (["loss", "lr"] as const).flatMap((key) => {
    if (!availableMetrics.some((metric) => metric.key === key) || hiddenValues.has(key)) return [];
    const data = metricSeries(points, key, 0, 0, false).data;
    if (!data.length) return [];
    const label = key === "loss" ? "Loss" : "学习率";
    const value = key === "lr" ? data.at(-1)![1].toExponential(3) : data.at(-1)![1].toPrecision(4);
    return [`最新 ${label}: ${value}`];
  });
  return <>
    <div className="training-metric-visibility" role="group" aria-label="训练曲线显隐">
      {availableMetrics.map(({ key, label }) => <label className="checkbox-row" key={key}>
        <input type="checkbox" checked={!hiddenMetrics.has(key)} onChange={(event) => toggleMetric(key, event.target.checked)} />
        {label}
      </label>)}
    </div>
    <div className="training-metric-values" role="group" aria-label="Loss 与学习率数值显隐">
      {availableMetrics.filter(({ key }) => key === "loss" || key === "lr").map(({ key }) => {
        const label = key === "loss" ? "Loss 数值" : "学习率数值";
        return <label className="checkbox-row" key={key}>
          <input type="checkbox" checked={!hiddenValues.has(key)} onChange={(event) => toggleValue(key, event.target.checked)} />
          {label}
        </label>;
      })}
      {latestValues.map((value) => <span className="training-metric-latest" key={value}>{value}</span>)}
    </div>
    {availableMetrics.length ? availableMetrics.map(({ key, label }) => key === "loss"
      ? <MetricsChart key={key} points={points} total={total} hidden={hiddenMetrics.has(key)} valueVisible={!hiddenValues.has(key)} />
      : key === "lr"
        ? <MetricsChart key={key} points={points} total={total} metric="lr" label={label} initialLimit={hasRecentLearningRate ? 2000 : 0} autoExpandEmptyWindow hidden={hiddenMetrics.has(key)} valueVisible={!hiddenValues.has(key)} />
        : <MetricsChart key={key} points={points} total={total} metric="cmmd" label={label} hidden={hiddenMetrics.has(key)} />)
      : null}
    {!visibleMetrics.length ? <p className="training-metrics-empty">已隐藏所有训练曲线</p> : null}
  </>;
}
