import { useQuery } from "@tanstack/react-query";
import { Link, useLocation, useSearchParams } from "react-router-dom";
import { QueryFeedback } from "../../components/QueryFeedback";
import { finiteNumber } from "../../components/trainingNumbers";
import { fetchHistoryImages, fetchHistoryWeights } from "./api";

export function HistoryResultSummary({ taskId }: { taskId: string }) {
  const images = useQuery({
    queryKey: ["history-result-summary", taskId, "images"],
    queryFn: ({ signal }) => fetchHistoryImages(taskId, signal, 1), retry: false,
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
    <div className="toolbar"><h2>训练产物</h2><Link to={`?${next}`} state={location.state}>查看全部产物</Link></div>
    {(images.isPending || images.error) && <QueryFeedback query={images} label="样张摘要" hasData={Boolean(images.data)} />}
    {(weights.isPending || weights.error) && <QueryFeedback query={weights} label="权重摘要" hasData={Boolean(weights.data)} />}
    <dl className="history-info">
      <ResultRow label="样张" total={images.data?.total} items={images.data?.images} missing={images.data?.directory_exists === false} />
      <ResultRow label="权重" total={weights.data?.total} items={weights.data?.weights} missing={weights.data?.directory_exists === false} />
    </dl>
  </section>;
}

function ResultRow({ label, total, items, missing }: {
  label: string; total?: number; items?: { name: string }[]; missing: boolean;
}) {
  if (!items) return null;
  const count = finiteNumber(total);
  return <div><dt>{label}</dt><dd>
    {missing ? "目录不存在" : count !== undefined ? `${count} 项` : `已发现 ${items.length} 项（总数未记录）`}
    {items[0] && <span> · 最近：{items[0].name}{label === "权重" && isNamedFinalModel(items[0].name) && <span className="history-final-model"> Final Model</span>}</span>}
  </dd></div>;
}

function isNamedFinalModel(name: string) {
  return /(?:^|[_\-.])(final|last)(?:[_\-.]|$)/i.test(name);
}
