import { useMemo, useState } from "react";
import { MetricsChart } from "../../components/MetricsChart";
import { finiteNumber } from "../../components/trainingNumbers";
import { historyGpuChartPoints, historyGpuDevices } from "./historyGpuMetrics";

const METRICS = [
  { key: "vram_used_gb", label: "显存 (GB)" },
  { key: "gpu_util", label: "GPU 利用率 (%)" },
  { key: "gpu_temp", label: "GPU 温度 (°C)" },
] as const;

export function HistoryGpuMetrics({ points, total, whitelist = [] }: {
  points: Record<string, unknown>[]; total?: number; whitelist?: number[];
}) {
  const devices = useMemo(() => historyGpuDevices(points), [points]);
  const [selectedKey, setSelectedKey] = useState("");
  const device = devices.find((item) => item.key === selectedKey);
  const chartPoints = useMemo(() => historyGpuChartPoints(points, device?.key), [points, device?.key]);
  const metrics = METRICS.filter(({ key }) => chartPoints.some((point) => finiteNumber(point[key]) !== undefined));
  const scope = device
    ? whitelist.length ? whitelist.includes(device.index) ? "任务已选" : "未选用" : "参与状态未确认"
    : devices.length ? whitelist.length ? `任务已选 GPU ${whitelist.join("、")}` : "参与设备未确认" : "仅有汇总记录";

  return <section className="history-gpu-metrics" aria-label="GPU 资源历史">
    <div className="history-gpu-heading">
      <div><h2>GPU 资源</h2><p>{device ? `GPU ${device.index} · ${device.name} · ${scope}` : scope}</p></div>
      {devices.length ? <div className="history-gpu-choices" role="group" aria-label="GPU 资源范围">
        <button type="button" aria-pressed={!device} onClick={() => setSelectedKey("")}>汇总</button>
        {devices.map((item) => <button type="button" key={item.key} aria-pressed={device?.key === item.key}
          title={`${item.name} · ${item.key}`} onClick={() => setSelectedKey(item.key)}>
          GPU {item.index}{devices.some((other) => other !== item && other.index === item.index) ? ` · ${item.key.slice(-8)}` : ""}
        </button>)}
      </div> : null}
    </div>
    {metrics.length ? <div className="history-gpu-charts">
      {metrics.map(({ key, label }) => <MetricsChart key={`${device?.key || "total"}-${key}`} points={chartPoints}
        metric={key} label={label} timeAxis total={total} />)}
    </div> : <p className="history-detail-empty">暂无 GPU 资源记录</p>}
  </section>;
}
