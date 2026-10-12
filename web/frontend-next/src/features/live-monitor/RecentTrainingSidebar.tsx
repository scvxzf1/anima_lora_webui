import { useQuery } from "@tanstack/react-query";
import { X } from "lucide-react";
import { fetchHistoryTasks, historyKeys } from "../training-history/api";
import { historyStateLabel, historyTaskName } from "../training-history/historySummary";

export function RecentTrainingSidebar({ selectedTaskId, onSelect, onClose }: {
  selectedTaskId: string | null;
  onSelect: (taskId: string) => void;
  onClose: () => void;
}) {
  const query = useQuery({
    queryKey: [...historyKeys.list, "recent", 20],
    queryFn: ({ signal }) => fetchHistoryTasks(20, signal),
    retry: false,
  });
  const tasks = [...(query.data?.tasks || [])]
    .sort((a, b) => (b.started_at || 0) - (a.started_at || 0)).slice(0, 20);
  return <aside id="monitor-recent-training" className="monitor-recent-sidebar" aria-label="最近训练">
    <header>
      <div><h2>最近训练</h2><p>最近 20 条历史任务</p></div>
      <button type="button" aria-label="关闭最近训练" onClick={onClose}><X size={17} aria-hidden="true" /></button>
    </header>
    <div className="monitor-recent-list">
      {query.isPending && <p role="status">正在读取最近训练</p>}
      {query.error && <div role="alert"><p>无法读取最近训练：{query.error.message}</p>
        <button type="button" disabled={query.isFetching} onClick={() => void query.refetch()}>重新读取</button>
      </div>}
      {query.data && !tasks.length && <p role="status">暂无历史任务</p>}
      <ul>{tasks.map((task, index) => <li key={task.id || index}>
        <button type="button" className="monitor-recent-task" disabled={!task.id}
          aria-pressed={Boolean(task.id) && selectedTaskId === task.id} onClick={() => task.id && onSelect(task.id)}>
          <strong>{historyTaskName(task)}</strong>
          <span>{task.started_at_text || (task.started_at ? new Date(task.started_at * 1000).toLocaleString() : "未记录开始时间")}</span>
          <span className="monitor-recent-meta"><span>{historyStateLabel(task.state)}</span>
            <span>{task.job === "preprocess" ? "预处理" : "训练"}</span></span>
        </button>
      </li>)}</ul>
    </div>
  </aside>;
}
