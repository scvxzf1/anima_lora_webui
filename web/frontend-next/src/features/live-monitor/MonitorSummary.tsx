import { finiteNumber, formatLearningRate, formatLoss, formatStep } from "../../components/trainingNumbers";
import type { GpuInfo, TrainingStatus } from "./api";
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
  const temperature = finiteNumber(system.gpu_temp);
  const highTemperature = temperature != null && temperature >= 80;
  const pct = step != null && total != null && total > 0
    ? Math.max(0, Math.min(100, step / total * 100)) : undefined;
  const fields = [
    ["Loss", formatLoss((isTrainingMetric(progress) ? finiteNumber(progress.loss) : undefined) ?? metric.loss)],
    ["学习率", formatLearningRate(finiteNumber(progress.lr) ?? metric.lr)],
    ["最近采样速度", String(progress.rate || metric.rate || "未记录")],
    ["采样显存", `${quantity(system.vram_used_gb, " GB", 1)} / ${quantity(system.vram_total_gb, " GB", 1)}`],
    ["采样最高温度", quantity(system.gpu_temp, "°C")],
    ["采样最高利用率", quantity(system.gpu_util, "%")],
  ];
  return <>
    <section className="monitor-progress-card" aria-label="任务进度">
      <div className="monitor-progress-heading">
        <h2>任务进度</h2>
        <strong>{pct == null ? "未记录" : `${pct.toFixed(1)}%`}</strong>
      </div>
      {pct != null ? <div className="monitor-progress-track" role="progressbar" aria-label="任务进度" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
        <div style={{ width: `${pct}%` }} />
      </div> : null}
      <div className="monitor-progress-copy">
        <div><span>{status.job === "preprocess" ? "处理进度" : "步数"}</span><strong>{formatStep(step)} / {formatStep(total)}</strong></div>
        <div><span>任务目录</span><strong>{status.output_dir || "未记录"}</strong></div>
      </div>
      {status.anomaly_message ? <p className="monitor-anomaly" role="status">{status.anomaly_message}</p> : null}
    </section>
    <section className="monitor-metrics" aria-label="实时指标">
      {fields.filter(([name]) => status.job !== "preprocess" || !["Loss", "学习率"].includes(name)).map(([label, value]) =>
        <div className="monitor-metric" key={label}><span>{label}</span><strong data-tone={label === "采样最高温度" && highTemperature ? "warning" : undefined}>{value}
          {label === "采样最高温度" && highTemperature ? <span className="monitor-temperature-warning">高温预警</span> : null}
        </strong></div>)}
    </section>
  </>;
}

export function GpuDetails({ gpu, participation }: { gpu: GpuInfo; participation: string }) {
  return <dl className="monitor-gpu-card">
    <div className="monitor-gpu-title"><dt>GPU {finiteNumber(gpu.index) ?? "?"}</dt><dd>{participation}</dd></div>
    <div className="monitor-gpu-name"><dt>型号</dt><dd>{String(gpu.name || "未记录")}</dd></div>
    <div><dt>显存</dt><dd>{quantity(gpu.memory_used_gb, " GB", 1)} / {quantity(gpu.memory_total_gb, " GB", 1)}</dd></div>
    <div><dt>利用率</dt><dd>{quantity(gpu.gpu_util, "%")}</dd></div>
    <div><dt>温度</dt><dd>{quantity(gpu.gpu_temp, "°C")}</dd></div>
  </dl>;
}
