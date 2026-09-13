import { useQueries } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { CommandDialog } from "../../components/CommandDialog";
import { fetchHistoryTaskDetail, historyKeys } from "./api";
import { historyStateLabel, historySummary, snapshotFields } from "./historySummary";
import { formatLoss, formatStep } from "../../components/trainingNumbers";
import { ComparisonChart } from "./ComparisonChart";
import { ComparisonParameters } from "./ComparisonParameters";
import "./HistoryComparison.css";

export function HistoryComparison({
  ids,
  onClose,
}: {
  ids: string[];
  onClose: () => void;
}) {
  const queries = useQueries({
    queries: ids
      .slice(0, 4)
      .map((id) => ({
        queryKey: historyKeys.detail(id),
        queryFn: ({ signal }: { signal: AbortSignal }) =>
          fetchHistoryTaskDetail(id, signal),
      })),
  });
  const entries = queries.map((query, index) => ({ id: ids[index], name: query.data?.task?.name || ids[index], detail: query.data }));
  return (
    <CommandDialog title="历史任务对比" onClose={onClose}>
      <div className="history-comparison">
        {queries.map((query, index) => {
          const detail = query.data;
          const summary = historySummary(detail?.task, detail?.metrics);
          return (
            <section key={ids[index]}>
              <h3>
                <Link to={`/history/${encodeURIComponent(ids[index])}`}>
                  {detail?.task?.name || ids[index]}
                </Link>
              </h3>
              {query.isPending ? (
                <p role="status">正在读取记录</p>
              ) : query.error ? (
                <p role="alert">{query.error.message} <button type="button" onClick={() => query.refetch()}>重试读取</button></p>
              ) : (
                <>
                  <dl className="history-info">
                    <div>
                      <dt>状态</dt>
                      <dd>{historyStateLabel(detail?.task?.state)}</dd>
                    </div>
                    <div>
                      <dt>最后步数</dt>
                      <dd>{formatStep(summary.step)}</dd>
                    </div>
                    <div>
                      <dt>末次有效 Loss</dt>
                      <dd>{formatLoss(summary.loss)}</dd>
                    </div>
                    <div>
                      <dt>配置</dt>
                      <dd>{detail?.task?.history_source_config_file}</dd>
                    </div>
                  </dl>
                  {!detail?.config_toml?.trim() ? <p className="data-scope">未保存配置快照</p> : snapshotFields(detail).invalid ? <p role="alert">配置快照无法解析</p> : null}
                </>
              )}
            </section>
          );
        })}
      </div>
      <ComparisonChart entries={entries} />
      <ComparisonParameters entries={entries} />
    </CommandDialog>
  );
}
