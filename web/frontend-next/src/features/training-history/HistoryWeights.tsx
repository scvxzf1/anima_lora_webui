import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Copy, Download, Folder } from "lucide-react";
import { finiteNumber, formatLoss } from "../../components/trainingNumbers";
import { AssetPagination } from "./AssetPagination";
import { HistoryHotstart } from "./HistoryHotstart";
import { fetchHistoryWeights, historyAssetUrl, type HistoryMetricPoint, type HistoryWeight } from "./api";
import { checkpointLosses, commonWeightDirectory } from "./historyAssetPresentation";

export function HistoryWeights({ taskId, metrics }: { taskId: string; metrics: HistoryMetricPoint[] }) {
  const [offset, setOffset] = useState(0);
  const [sort, setSort] = useState("recent");
  const [copyStatus, setCopyStatus] = useState("");
  const query = useQuery({
    queryKey: ["history-weights", taskId, offset, sort],
    queryFn: ({ signal }) => fetchHistoryWeights(taskId, signal, 100, offset, sort),
  });
  const weights = query.data?.weights || [];
  const directory = commonWeightDirectory(weights);
  const losses = checkpointLosses(metrics);
  async function copyPath(path: string, label: string) {
    try {
      if (!navigator.clipboard?.writeText) throw new Error("Clipboard unavailable");
      await navigator.clipboard.writeText(path);
      setCopyStatus(`${label}已复制。`);
    } catch {
      setCopyStatus(`无法复制${label}，请手动复制：${path}`);
    }
  }
  return <section className="history-asset-section" aria-label="权重">
    <header className="history-asset-heading">
      <div><h2>权重 <small>Checkpoints</small></h2><p>Loss 仅匹配已读取训练指标中的相同步数，不使用邻近步数估算。</p></div>
      <label>排序 <select value={sort} onChange={(event) => { setSort(event.target.value); setOffset(0); setCopyStatus(""); }}>
        <option value="recent">最新文件优先</option><option value="name">文件名</option>
      </select></label>
    </header>
    {directory && <div className="history-weight-directory"><Folder size={16} aria-hidden="true" /><span>公共目录（本页）</span>
      <code title={directory}>{directory}</code><button type="button" aria-label="复制权重目录" title="复制权重目录" onClick={() => void copyPath(directory, "权重目录")}><Copy size={15} /></button>
    </div>}
    {query.isPending && <p role="status">正在读取权重</p>}
    {query.error && <p role="alert">{query.error.message} <button type="button" onClick={() => query.refetch()}>重试权重</button></p>}
    <div className="weight-list">
      {weights.map((weight) => <WeightRow key={weight.file} weight={weight} taskId={taskId} loss={losses.get(finiteNumber(weight.steps) ?? -1)} onCopy={copyPath} />)}
    </div>
    <p className="history-copy-status" role="status" aria-live="polite">{copyStatus}</p>
    {query.data?.weights.length === 0 && <p>{query.data.message || "本页暂无权重"}</p>}
    <AssetPagination label="权重" offset={offset} count={weights.length} total={query.data?.total ?? 0} size={100}
      next={query.data?.next_offset} pending={query.isFetching} onChange={(next) => { setOffset(next); setCopyStatus(""); }} />
  </section>;
}

function WeightRow({ weight, taskId, loss, onCopy }: {
  weight: HistoryWeight; taskId: string; loss?: number; onCopy: (path: string, label: string) => Promise<void>;
}) {
  const step = finiteNumber(weight.steps);
  const url = historyAssetUrl(taskId, weight.file, true);
  return <div>
    <div className="history-weight-file"><a href={url} download title={weight.abs_path || weight.file}>{weight.name}</a>
      <span>{step !== undefined ? `Step ${step} · ` : ""}{(weight.size_bytes / 1048576).toFixed(1)} MB · {weight.scope_label}</span>
    </div>
    <span className="history-weight-meta">Loss <strong>{loss !== undefined ? formatLoss(loss) : "未记录"}</strong></span>
    {weight.kind === "final" && <span className="history-final-model">FINAL MODEL</span>}
    <a className="history-weight-download" href={url} download aria-label={`下载 ${weight.name}`} title={`下载 ${weight.name}`}><Download aria-hidden="true" size={15} /></a>
    <button type="button" aria-label={`复制 ${weight.name} 的本地路径`} title={weight.abs_path || "本地绝对路径不可用"}
      disabled={!weight.abs_path} onClick={() => void onCopy(weight.abs_path || "", `${weight.name}的路径`)}><Copy aria-hidden="true" size={15} /></button>
    <HistoryHotstart weight={weight} />
  </div>;
}
