import { MetricsChart } from "./MetricsChart";
import { metricSeries } from "./metricSeries";
import { finiteNumber } from "./trainingNumbers";

export function TrainingMetricsCharts({ points, total }: { points: Record<string, unknown>[]; total?: number }) {
  const hasLearningRate = metricSeries(points, "lr", 0, 0, false).data.length > 0;
  const hasRecentLearningRate = metricSeries(points, "lr", 2000, 0, false).data.length > 0;
  return <>
    <MetricsChart points={points} total={total} />
    {hasLearningRate && <MetricsChart points={points} total={total} metric="lr" label="学习率趋势" initialLimit={hasRecentLearningRate ? 2000 : 0} />}
    {points.some((point) => finiteNumber(point.cmmd) !== undefined) &&
      <MetricsChart points={points} total={total} metric="cmmd" label="验证 CMMD" />}
  </>;
}
