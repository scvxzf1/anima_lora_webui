import { lazy, Suspense, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { History } from "lucide-react";
import { RecentTrainingSidebar } from "./RecentTrainingSidebar";
import "./MonitorHistoryWorkspace.css";

const HistoryDetailPage = lazy(() => import("../training-history/HistoryDetailPage").then((module) => ({ default: module.HistoryDetailPage })));

export function MonitorHistoryWorkspace({ children }: { children: (toggle: ReactNode) => ReactNode }) {
  const [open, setOpen] = useState(false);
  const [taskId, setTaskId] = useState<string | null>(null);
  const monitorRef = useRef<HTMLDivElement>(null);
  const detailRef = useRef<HTMLDivElement>(null);
  const scrollTop = useRef(0);
  const toggleRef = useRef<HTMLButtonElement>(null);
  const returnFocus = useRef(false);
  const workspaceRef = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const workspace = workspaceRef.current;
    if (!workspace) return;
    const resize = () => {
      const rect = workspace.getBoundingClientRect();
      const zoom = workspace.offsetWidth > 0 ? rect.width / workspace.offsetWidth : 1;
      if (zoom > 0) workspace.style.setProperty("--monitor-workspace-height", `${Math.max(0, window.innerHeight - rect.top) / zoom}px`);
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(workspace);
    if (workspace.parentElement) observer.observe(workspace.parentElement);
    window.addEventListener("resize", resize);
    return () => { observer.disconnect(); window.removeEventListener("resize", resize); };
  }, []);
  useLayoutEffect(() => {
    if (!taskId && monitorRef.current) {
      monitorRef.current.scrollTop = scrollTop.current;
      if (returnFocus.current) toggleRef.current?.focus({ preventScroll: true });
      returnFocus.current = false;
    }
    if (taskId && detailRef.current) {
      detailRef.current.scrollTop = 0;
      detailRef.current.focus({ preventScroll: true });
    }
  }, [taskId]);
  const back = () => {
    returnFocus.current = true;
    setTaskId(null);
  };
  const toggle = <button ref={toggleRef} type="button" className="monitor-recent-toggle" aria-expanded={open}
    aria-controls="monitor-recent-training" onClick={() => setOpen(!open)}>
    <History size={16} aria-hidden="true" />最近训练
  </button>;
  return <div ref={workspaceRef} className="monitor-shell monitor-history-workspace" data-recent-open={open}>
    {open && <RecentTrainingSidebar selectedTaskId={taskId} onClose={() => { back(); setOpen(false); }}
      onSelect={(id) => {
        if (!taskId) scrollTop.current = monitorRef.current?.scrollTop || 0;
        setTaskId(id);
      }} />}
    <div className="monitor-history-content">
      <div ref={monitorRef} className="monitor-current-view" hidden={Boolean(taskId)}>{children(toggle)}</div>
      {taskId && <div ref={detailRef} className="monitor-history-detail" tabIndex={-1} aria-label="历史任务详情">
        <Suspense fallback={<p role="status">正在加载历史详情</p>}>
          <HistoryDetailPage key={taskId} taskId={taskId} onBack={back} />
        </Suspense>
      </div>}
    </div>
  </div>;
}
