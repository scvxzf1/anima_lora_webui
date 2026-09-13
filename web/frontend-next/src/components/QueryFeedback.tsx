import { RefreshCw } from "lucide-react";
import "./QueryFeedback.css";

type QueryState = {
  isPending: boolean;
  isFetching: boolean;
  error: Error | null;
  dataUpdatedAt: number;
  refetch: () => unknown;
};

export function QueryFeedback({ query, label, hasData }: {
  query: QueryState; label: string; hasData: boolean;
}) {
  return <div className="query-feedback" aria-label={`${label}读取状态`}>
    {query.error ? <p role="alert">{label}读取失败：{query.error.message}</p>
      : query.isPending ? <p role="status">正在读取{label}...</p> : null}
    {hasData && query.dataUpdatedAt > 0 ? <span>
      {query.error ? "保留上次数据 · " : ""}最近成功读取 <time dateTime={new Date(query.dataUpdatedAt).toISOString()}>
        {new Date(query.dataUpdatedAt).toLocaleTimeString()}
      </time>
    </span> : null}
    {query.error ? <button type="button" disabled={query.isFetching} onClick={() => query.refetch()}>
      <RefreshCw size={14} />{query.isFetching ? "重试中" : `重试${label}`}
    </button> : null}
  </div>;
}
