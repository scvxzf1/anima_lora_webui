import { finiteNumber, formatLearningRate, formatLoss, formatStep } from "../../components/trainingNumbers";
import type { TrainingStatus } from "./api";
import { isTrainingMetric } from "../../components/metricSemantics";

function quantity(value: unknown, unit: string, digits = 0) {
  const number = finiteNumber(value);
  return number == null ? "未记录" : `${number.toFixed(digits)}${unit}`;
}

export function MonitorSummary({ status }: { status: TrainingStatus }) {
  const progress = status.latest_progress || {};
  const latest = status.latest_metric || {};
  const metric = isTrainingMetric(latest) ? latest : {};
  const system = status.latest_system || {};
  const step = finiteNumber(progress.current) ?? finiteNumber(metric.step);
  const total = finiteNumber(progress.total) ?? finiteNumber(metric.total);
  const pct = step != null && total != null && total > 0
    ? Math.max(0, Math.min(100, step / total * 100)) : undefined;
  const fields = [
    ["Loss", formatLoss((isTrainingMetric(progress) ? finiteNumber(progress.loss) : undefined) ?? metric.loss)],
    ["学习率", formatLearningRate(finiteNumber(progress.lr) ?? metric.lr)],
    ["最近采样速度", String(progress.rate || metric.rate || "未记录")],
    ["显存", `${quantity(system.vram_used_gb, " GB", 1)} / ${quantity(system.vram_total_gb, " GB", 1)}`],
    ["GPU 温度", quantity(system.gpu_temp, "°C")],
    ["GPU 利用率", quantity(system.gpu_util, "%")],
  ];
  return <>
    <section className="monitor-progress-card" aria-label="任务进度">
      <div className="monitor-progress-copy">
        <div><span>进度</span><strong>{pct == null ? "未记录" : `${pct.toFixed(1)}%`}</strong></div>
        <div><span>{status.job === "preprocess" ? "处理进度" : "步数"}</span><strong>{formatStep(step)} / {formatStep(total)}</strong></div>
        <div><span>任务目录</span><strong>{status.output_dir || "未记录"}</strong></div>
      </div>
      {pct != null ? <div className="monitor-progress-track" role="progressbar" aria-label="任务进度" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
        <div style={{ width: `${pct}%` }} />
      </div> : null}
      {status.anomaly_message ? <p className="monitor-anomaly" role="status">{status.anomaly_message}</p> : null}
    </section>
    <section className="monitor-metrics" aria-label="实时指标">
      {fields.filter(([name]) => status.job !== "preprocess" || !["Loss", "学习率"].includes(name)).map(([label, value]) =>
        <div className="monitor-metric" key={label}><span>{label}</span><strong>{value}</strong></div>)}
    </section>
  </>;
}

export function GpuDetails({ gpu }: { gpu: Record<string, unknown> }) {
  return <dl className="monitor-gpu-card">
    <div><dt>设备</dt><dd>GPU {finiteNumber(gpu.index) ?? "?"}</dd></div>
    <div><dt>型号</dt><dd>{String(gpu.name || "未记录")}</dd></div>
    <div><dt>总显存</dt><dd>{quantity(gpu.memory_total_gb, " GB", 1)}</dd></div>
  </dl>;
}
