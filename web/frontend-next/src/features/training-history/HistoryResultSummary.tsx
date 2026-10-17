import { useQuery } from "@tanstack/react-query";
import { Link, useLocation, useSearchParams } from "react-router-dom";
import { QueryFeedback } from "../../components/QueryFeedback";
import { ResilientImage } from "../../components/ResilientImage";
import { finiteNumber } from "../../components/trainingNumbers";
import { fetchHistoryImages, fetchHistoryWeights, historyAssetUrl } from "./api";
import "./HistoryResultSummary.css";

export function HistoryResultSummary({ taskId, onViewChange }: { taskId: string; onViewChange?: (view: string) => void }) {
  const images = useQuery({
    queryKey: ["history-result-summary", taskId, "images"],
    queryFn: ({ signal }) => fetchHistoryImages(taskId, signal, 4), retry: false,
  });
  const weights = useQuery({
    queryKey: ["history-result-summary", taskId, "weights"],
    queryFn: ({ signal }) => fetchHistoryWeights(taskId, signal, 1), retry: false,
  });
  const [params] = useSearchParams();
  const location = useLocation();
  const next = new URLSearchParams(params);
  next.set("view", "artifacts");
  return <section className="history-overview-section" aria-label="训练产物摘要">
    <div className="toolbar"><h2>训练产物</h2>{onViewChange
      ? <button type="button" onClick={() => onViewChange("artifacts")}>查看全部产物</button>
      : <Link to={`?${next}`} state={location.state}>查看全部产物</Link>}</div>
    {(images.isPending || images.error) && <QueryFeedback query={images} label="样张摘要" hasData={Boolean(images.data)} />}
    {(weights.isPending || weights.error) && <QueryFeedback query={weights} label="权重摘要" hasData={Boolean(weights.data)} />}
    <dl className="history-info">
      <ResultRow taskId={taskId} label="样张" total={images.data?.total} items={images.data?.images} missing={images.data?.directory_exists === false} />
      <ResultRow taskId={taskId} label="权重" total={weights.data?.total} items={weights.data?.weights} missing={weights.data?.directory_exists === false} />
    </dl>
  </section>;
}

function ResultRow({ taskId, label, total, items, missing }: {
  taskId: string; label: string; total?: number; items?: { file?: string; name: string; kind?: string }[]; missing: boolean;
}) {
  if (!items) return null;
  const count = finiteNumber(total);
  return <div><dt>{label}</dt><dd>
    {missing ? "目录不存在" : count !== undefined ? `${count} 项` : `已发现 ${items.length} 项（总数未记录）`}
    {!missing && items[0] && (label === "样张" ? <span className="history-result-preview">
      {items.slice(0, 4).map((item) => item.file && <ResilientImage key={item.file} className="history-result-thumbnail"
        src={historyAssetUrl(taskId, item.file)} alt={`训练样张：${item.name}`} title={item.name} loading="lazy" />)}
    </span> : <span> · 最近：{items[0].name}{items[0].kind === "final" && <span className="history-final-model"> Final Model</span>}</span>)}
  </dd></div>;
}
