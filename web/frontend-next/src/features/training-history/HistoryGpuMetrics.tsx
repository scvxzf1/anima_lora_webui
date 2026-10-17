import { useMemo, useState } from "react";
import { MetricsChart } from "../../components/MetricsChart";
import { finiteNumber } from "../../components/trainingNumbers";
import {
  formatHistoryGpuValue,
  HISTORY_GPU_METRICS,
  historyGpuChartPoints,
  historyGpuDevices,
  historyGpuInspectionIndices,
  historyGpuRecentPoints,
  historyGpuSampleKey,
  historyGpuSampleTime,
  historyGpuSummary,
} from "./historyGpuMetrics";
import { useHistorySystem } from "./useHistorySystem";
import "./HistoryGpuMetrics.css";

const METRICS = [
  { key: "vram_used_gb", label: "显存 (GB)" },
  { key: "gpu_util", label: "GPU 利用率 (%)" },
  { key: "gpu_temp", label: "GPU 温度 (°C)" },
] as const;

export function HistoryGpuMetrics({ points, total, whitelist = [], taskId }: {
  points: Record<string, unknown>[]; total?: number; whitelist?: number[]; taskId?: string;
}) {
  const history = useHistorySystem(taskId, points, total);
  const recentPoints = history.points;
  const devices = useMemo(() => historyGpuDevices(recentPoints), [recentPoints]);
  const [selectedKey, setSelectedKey] = useState("");
  const [selectedSample, setSelectedSample] = useState("");
  const [hiddenMetrics, setHiddenMetrics] = useState<Set<string>>(() => new Set());
  const device = devices.find((item) => item.key === selectedKey);
  const chartPoints = useMemo(() => historyGpuChartPoints(recentPoints, device?.key), [recentPoints, device?.key]);
  const metrics = METRICS.filter(({ key }) => chartPoints.some((point) => finiteNumber(point[key]) !== undefined));
  const visibleMetrics = metrics.filter(({ key }) => !hiddenMetrics.has(key));
  const summary = historyGpuSummary(chartPoints);
  const hasSummary = HISTORY_GPU_METRICS.some(({ key }) => summary[key].latest !== undefined || summary[key].peak !== undefined);
  const inspectionIndices = historyGpuInspectionIndices(historyGpuRecentPoints(chartPoints)).map((index) => index + Math.max(0, chartPoints.length - 1000));
  const latestInspection = inspectionIndices.at(-1);
  const requestedSample = inspectionIndices.find((index) => historyGpuSampleKey(chartPoints[index]) === selectedSample);
  const inspectionIndex = requestedSample ?? latestInspection;
  const inspectionPoint = inspectionIndex === undefined ? undefined : chartPoints[inspectionIndex];
  const toggleMetric = (key: string, visible: boolean) => setHiddenMetrics((hidden) => {
    const next = new Set(hidden);
    if (visible) next.delete(key);
    else next.add(key);
    return next;
  });
  const scope = device
    ? whitelist.length ? whitelist.includes(device.index) ? "任务已选" : "未选用" : "参与状态未确认"
    : devices.length ? whitelist.length ? `任务已选 GPU ${whitelist.join("、")}` : "参与设备未确认" : "仅有汇总记录";

  return <section className="history-gpu-metrics" aria-label="GPU 资源历史">
    {taskId && <div className="history-gpu-loading" role="status">已读取 {history.read} / {history.total ?? "未知"} 个采样点{history.loading ? <> · 正在读取 <button type="button" onClick={history.cancel}>取消读取</button></> : null}{history.error ? <><span role="alert"> · {history.error}</span> <button type="button" onClick={history.retry}>重新读取</button></> : null}</div>}
    <div className="history-gpu-heading">
      <div><h2>GPU 资源</h2><p>{device ? `GPU ${device.index} · ${device.name} · ${scope}` : scope}</p></div>
      {devices.length ? <div className="history-gpu-choices" role="group" aria-label="GPU 资源范围">
        <button type="button" aria-pressed={!device} onClick={() => { setSelectedKey(""); setSelectedSample(""); }}>汇总</button>
        {devices.map((item) => <button type="button" key={item.key} aria-pressed={device?.key === item.key}
          title={`${item.name} · ${item.key}`} onClick={() => { setSelectedKey(item.key); setSelectedSample(""); }}>
          GPU {item.index}{devices.some((other) => other !== item && other.index === item.index) ? ` · ${item.key.slice(-8)}` : ""}
        </button>)}
      </div> : null}
    </div>
    {metrics.length || hasSummary ? <div className="history-gpu-metric-content">
      {inspectionIndices.length ? <div className="history-gpu-inspection" aria-label="GPU 联合采样检查">
        <label htmlFor="history-gpu-inspection-point">同一采样点</label>
        <select id="history-gpu-inspection-point" value={inspectionPoint ? historyGpuSampleKey(inspectionPoint) : ""}
          onChange={(event) => setSelectedSample(event.target.value)}>
          {inspectionIndices.map((index) => <option value={historyGpuSampleKey(chartPoints[index])} key={index}>
            {historyGpuSampleTime(chartPoints[index], index)}
          </option>)}
        </select>
        {inspectionPoint ? <div className="history-gpu-kpi-wrap" aria-label="GPU 资源摘要">
          <dl className="history-gpu-kpis" aria-label="所选采样点 GPU 数值" role="group">
            {HISTORY_GPU_METRICS.map(({ key, label, unit }) => <div key={key}>
              <dt>{label}</dt><dd>{formatHistoryGpuValue(finiteNumber(inspectionPoint[key]), unit)}</dd>
              <small>已读取峰值 {formatHistoryGpuValue(summary[key].peak, unit)}</small>
            </div>)}
          </dl>
        </div> : null}
      </div> : null}
      <div className="history-gpu-metric-visibility" role="group" aria-label="GPU 指标显隐">
        {metrics.map(({ key, label }) => <label className="checkbox-row" key={key}>
          <input type="checkbox" checked={!hiddenMetrics.has(key)} onChange={(event) => toggleMetric(key, event.target.checked)} />
          {label}
        </label>)}
      </div>
      <div className="history-gpu-charts">
        {metrics.map(({ key, label }) => <MetricsChart key={`${device?.key || "total"}-${key}`} points={chartPoints}
        metric={key} label={label} timeAxis segmentedWindow total={history.total} hidden={hiddenMetrics.has(key)} />)}
      </div>
      {metrics.length > 0 && !visibleMetrics.length ? <p className="history-detail-empty">已隐藏所有 GPU 指标</p> : null}
    </div> : <p className="history-detail-empty">暂无 GPU 资源记录</p>}
  </section>;
}
