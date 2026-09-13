import { MetricsChart } from "./MetricsChart";
import { finiteNumber } from "./trainingNumbers";

export function TrainingMetricsCharts({ points, total }: { points: Record<string, unknown>[]; total?: number }) {
  return <>
    <MetricsChart points={points} total={total} />
    {points.some((point) => finiteNumber(point.cmmd) !== undefined) &&
      <MetricsChart points={points} total={total} metric="cmmd" label="验证 CMMD" />}
  </>;
}
