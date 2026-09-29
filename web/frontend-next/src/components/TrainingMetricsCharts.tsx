import { MetricsChart } from "./MetricsChart";
import { finiteNumber } from "./trainingNumbers";

export function TrainingMetricsCharts({ points, total }: { points: Record<string, unknown>[]; total?: number }) {
  const hasLearningRate = points.some((point) => point.kind !== "val" && point.ev !== "val" && finiteNumber(point.lr) !== undefined);
  return <>
    <MetricsChart points={points} total={total} />
    {hasLearningRate && <MetricsChart points={points} total={total} metric="lr" label="学习率趋势" />}
    {points.some((point) => finiteNumber(point.cmmd) !== undefined) &&
      <MetricsChart points={points} total={total} metric="cmmd" label="验证 CMMD" />}
  </>;
}
