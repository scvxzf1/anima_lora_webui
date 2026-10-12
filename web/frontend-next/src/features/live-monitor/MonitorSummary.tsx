import { finiteNumber, formatLearningRate, formatLoss, formatStep } from "../../components/trainingNumbers";
import type { GpuInfo, TrainingStatus } from "./api";
import { isTrainingMetric } from "../../components/metricSemantics";
import { Activity, Cpu } from "lucide-react";

const RUNNING_STATES = new Set(["running", "training", "compiling", "caching", "saving"]);

function quantity(value: unknown, unit: string, digits = 0) {
  const number = finiteNumber(value);
  return number == null ? "未记录" : `${number.toFixed(digits)}${unit}`;
}

function secondsPerStep(rate: unknown) {
  const match = String(rate ?? "").trim().toLowerCase().replace(/\s+/g, "").match(/^([\d.]+)(ms\/it|s\/it|s\/step|it\/s)$/);
  if (!match) return undefined;
  const amount = Number(match[1]);
  if (!Number.isFinite(amount) || amount <= 0) return undefined;
  const seconds = match[2] === "it/s" ? 1 / amount : match[2] === "ms/it" ? amount / 1000 : amount;
  return Number.isFinite(seconds) && seconds > 0 ? seconds : undefined;
}

function etaInfo(status: TrainingStatus, current: number | undefined, total: number | undefined) {
  if (current == null || total == null || total <= 0) return { text: "待计算" };
  if (current >= total) {
    return { text: RUNNING_STATES.has(status.status || "") ? "即将完成" : "已达目标步数" };
  }
  if (!RUNNING_STATES.has(status.status || "")) return { text: "待计算" };
  const speed = secondsPerStep(status.latest_progress?.rate || status.latest_metric?.rate);
  if (speed == null) return { text: "待计算" };
  const remainingSeconds = Math.ceil((total - current) * speed);
  if (!Number.isFinite(remainingSeconds) || remainingSeconds <= 0) return { text: "待计算" };
  const etaTimestamp = Date.now() + remainingSeconds * 1000;
  if (!Number.isFinite(etaTimestamp) || Math.abs(etaTimestamp) > 8.64e15) return { text: "待计算" };
  const eta = new Date(etaTimestamp);
  if (!Number.isFinite(eta.getTime())) return { text: "待计算" };
  const clock = `${String(eta.getHours()).padStart(2, "0")}:${String(eta.getMinutes()).padStart(2, "0")}`;
  const now = new Date(Date.now());
  const sameDay = eta.getFullYear() === now.getFullYear()
    && eta.getMonth() === now.getMonth()
    && eta.getDate() === now.getDate();
  const tomorrow = new Date(now.getFullYear(), now.getMonth(), now.getDate() + 1);
  const isTomorrow = eta.getFullYear() === tomorrow.getFullYear()
    && eta.getMonth() === tomorrow.getMonth()
    && eta.getDate() === tomorrow.getDate();
  const date = `${eta.getFullYear()}-${String(eta.getMonth() + 1).padStart(2, "0")}-${String(eta.getDate()).padStart(2, "0")}`;
  const text = sameDay ? clock : isTomorrow ? `明日 ${clock}` : `${date} ${clock}`;
  return { text, title: `按当前速度估算，剩余约 ${remainingSeconds} 秒。` };
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
  const eta = etaInfo(status, step, total);
  const fields = [
    ["预计完成", eta.text],
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
        <div className="monitor-progress-title-wrap">
          <Activity size={18} className="monitor-heading-icon" />
          <h2>任务进度</h2>
        </div>
        <strong>{pct == null ? "未记录" : `${pct.toFixed(1)}%`}</strong>
      </div>
      {pct != null ? <div className="monitor-progress-track" role="progressbar" aria-label="任务进度" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
        <div style={{ width: `${pct}%` }} />
      </div> : null}
      <div className="monitor-progress-copy">
        <div className="monitor-chip"><span>{status.job === "preprocess" ? "处理进度" : "步数"}</span><strong>{formatStep(step)} / {formatStep(total)}</strong></div>
        <div className="monitor-chip monitor-dir-chip"><span>任务目录</span><strong title={status.output_dir}>{status.output_dir || "未记录"}</strong></div>
      </div>
      {status.anomaly_message ? <p className="monitor-anomaly" role="status">{status.anomaly_message}</p> : null}
    </section>
    <section className="monitor-metrics" aria-label="实时指标">
      {fields.filter(([name]) => status.job !== "preprocess" || !["Loss", "学习率"].includes(name)).map(([label, value]) =>
        <div className="monitor-metric" key={label}><span>{label}</span><strong title={label === "预计完成" ? eta.title : undefined} data-tone={label === "采样最高温度" && highTemperature ? "warning" : undefined}>{value}
          {label === "采样最高温度" && highTemperature ? <span className="monitor-temperature-warning">高温预警</span> : null}
        </strong></div>)}
    </section>
  </>;
}

export function GpuDetails({ gpu, participation }: { gpu: GpuInfo; participation: string }) {
  const memUsed = finiteNumber(gpu.memory_used_gb);
  const memTotal = finiteNumber(gpu.memory_total_gb);
  const memPct = memUsed != null && memTotal != null && memTotal > 0
    ? Math.max(0, Math.min(100, (memUsed / memTotal) * 100)) : undefined;
  const util = finiteNumber(gpu.gpu_util);
  const isSelected = participation === "当前任务已选";
  const isUnselected = participation === "未选用";

  return <dl className="monitor-gpu-card" data-participation={isSelected ? "selected" : isUnselected ? "unselected" : "idle"}>
    <div className="monitor-gpu-title">
      <dt><Cpu size={14} className="monitor-gpu-icon" /> GPU {finiteNumber(gpu.index) ?? "?"}</dt>
      <dd><span className="monitor-gpu-badge" data-status={isSelected ? "selected" : isUnselected ? "unselected" : "idle"}>{participation}</span></dd>
    </div>
    <div className="monitor-gpu-name"><dt>型号</dt><dd>{String(gpu.name || "未记录")}</dd></div>
    <div className="monitor-gpu-stat">
      <dt>显存</dt>
      <dd>
        <span>{quantity(gpu.memory_used_gb, " GB", 1)} / {quantity(gpu.memory_total_gb, " GB", 1)}</span>
        {memPct != null ? <div className="monitor-mini-meter" title={`显存已用 ${memPct.toFixed(1)}%`}><div style={{ width: `${memPct}%` }} /></div> : null}
      </dd>
    </div>
    <div className="monitor-gpu-stat">
      <dt>利用率</dt>
      <dd>
        <span>{quantity(gpu.gpu_util, "%")}</span>
        {util != null ? <div className="monitor-mini-meter" title={`利用率 ${util}%`}><div style={{ width: `${Math.max(0, Math.min(100, util))}%` }} /></div> : null}
      </dd>
    </div>
    <div><dt>温度</dt><dd>{quantity(gpu.gpu_temp, "°C")}</dd></div>
  </dl>;
}
