import { useQuery } from "@tanstack/react-query";
import { Link, useParams, useLocation, useSearchParams } from "react-router-dom";
import { lazy, Suspense, useState } from "react";
import { RotateCcw } from "lucide-react";
import { fetchHistoryTaskDetail, historyKeys, type HistoryTaskDetail } from "./api";
import { historyStateLabel, historyTaskName } from "./historySummary";
import { finiteNumber } from "../../components/trainingNumbers";
import { HistorySnapshot } from "./HistorySnapshot";
import "./HistoryDetailPage.css";

const TrainingMetricsCharts = lazy(() => import("../../components/TrainingMetricsCharts").then((module) => ({ default: module.TrainingMetricsCharts })));
const HistoryGpuMetrics = lazy(() => import("./HistoryGpuMetrics").then((module) => ({ default: module.HistoryGpuMetrics })));
const HistoryLogs = lazy(() => import("./HistoryLogs").then((module) => ({ default: module.HistoryLogs })));
const HistoryResume = lazy(() => import("./HistoryResume").then((module) => ({ default: module.HistoryResume })));
const HistoryAssets = lazy(() => import("./HistoryAssets").then((module) => ({ default: module.HistoryAssets })));
const HistoryOverview = lazy(() => import("./HistoryOverview").then((module) => ({ default: module.HistoryOverview })));

const VIEWS = [["overview", "概览"], ["metrics", "指标"], ["artifacts", "产物"], ["logs", "日志"], ["config", "配置"]];

export function HistoryDetailPage() {
  const { taskId = "" } = useParams();
  const [params] = useSearchParams();
  const location = useLocation();
  const tab = VIEWS.some(([key]) => key === params.get("view")) ? params.get("view")! : "overview";
  const listSearch = params.get("from") || location.state?.listSearch || "";
  const [resumeOpen, setResumeOpen] = useState(false);
  const query = useQuery({
    queryKey: historyKeys.detail(taskId),
    queryFn: ({ signal }) => fetchHistoryTaskDetail(taskId, signal),
    enabled: Boolean(taskId),
  });
  const task = query.data?.task;
  return <div className="history-detail-shell">
    <main className="history-detail-page" data-view={tab}>
      <header className="history-detail-header">
        <div>
          <Link to={`/history${listSearch ? `?${listSearch}` : ""}`} className="history-back">返回历史</Link>
          <h1>{historyTaskName(task)}</h1>
          <p>{task?.job === "preprocess" ? "数据预处理" : task?.job === "training" ? "模型训练" : "历史任务"} · {task?.started_at_text || taskId}</p>
        </div>
        {task && <div className="history-detail-commands">
          <span className="history-detail-state" data-state={task.state}>{historyStateLabel(task.state)}</span>
          {task.job === "training" && <button type="button" disabled={Boolean(query.error)} onClick={() => setResumeOpen(true)}>
            <RotateCcw size={16} />检查点续训
          </button>}
        </div>}
      </header>
      <nav className="page-tabs" aria-label="历史详情视图">
        {VIEWS.map(([key, label]) => {
          const next = new URLSearchParams(params);
          next.set("view", key);
          return <Link key={key} to={`?${next}`} state={location.state} replace aria-current={tab === key ? "page" : undefined}>{label}</Link>;
        })}
      </nav>
      {query.error && <section className="history-detail-error" role="alert">
        <h2>无法读取历史任务</h2><p>{query.error.message}</p>
        {query.data && <p>以下为上次成功读取的记录。</p>}
        <button type="button" disabled={query.isFetching} onClick={() => void query.refetch()}>重新读取</button>
      </section>}
      {query.isPending && <p role="status">正在读取历史任务</p>}
      {query.data && <Suspense fallback={<p role="status">正在加载视图</p>}>
        <HistoryDetailContent key={taskId} detail={query.data} taskId={taskId} tab={tab} />
      </Suspense>}
    </main>
    {resumeOpen && <Suspense fallback={<p role="status">正在加载续训信息</p>}>
      <HistoryResume key={taskId} taskId={taskId} onClose={() => setResumeOpen(false)} />
    </Suspense>}
  </div>;
}

function HistoryDetailContent({ detail, taskId, tab }: { detail: HistoryTaskDetail; taskId: string; tab: string }) {
  switch (tab) {
    case "metrics": return <>
      {detail.task?.job === "training" ? <TrainingMetricsCharts points={detail.metrics || []} total={finiteNumber(detail.limits?.metrics_total)} />
        : <p className="data-scope">此任务不适用训练 Loss。</p>}
      <HistoryGpuMetrics points={detail.system || []} total={finiteNumber(detail.limits?.system_total)}
        whitelist={detail.task?.gpu_whitelist} />
    </>;
    case "artifacts": return <HistoryAssets taskId={taskId} />;
    case "logs": return <HistoryLogs taskId={taskId} running={detail.task?.state === "running"} />;
    case "config": return <HistorySnapshot detail={detail} taskId={taskId} />;
    default: return <HistoryOverview detail={detail} />;
  }
}
