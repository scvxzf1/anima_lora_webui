import { HistoryArtifacts } from "./HistoryArtifacts";
import { HistorySampleGallery } from "./HistorySampleGallery";
import { HistoryWeights } from "./HistoryWeights";
import type { HistoryMetricPoint } from "./api";
import "./HistoryAssets.css";

export function HistoryAssets({ taskId, metrics = [] }: { taskId: string; metrics?: HistoryMetricPoint[] }) {
  return <section className="history-assets">
    <HistorySampleGallery key={`samples-${taskId}`} taskId={taskId} />
    <HistoryWeights key={`weights-${taskId}`} taskId={taskId} metrics={metrics} />
    <HistoryArtifacts taskId={taskId} />
  </section>;
}
