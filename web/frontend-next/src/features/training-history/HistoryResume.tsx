import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router-dom";
import { CommandDialog } from "../../components/CommandDialog";
import { fetchResumeOptions, resumeHistoryTask } from "./api";
import { resumeAvailability } from "./resumeAvailability";
import { ApiError } from "../../api/client";

export function HistoryResume({
  taskId,
  onClose,
}: {
  taskId: string;
  onClose: () => void;
}) {
  return <ResumeForm key={taskId} taskId={taskId} onClose={onClose} />;
}

type ResumeSubmission = { taskId: string; path: string; queue: boolean; appendSteps?: number };

function ResumeForm({ taskId, onClose }: { taskId: string; onClose: () => void }) {
  const query = useQuery({
    queryKey: ["history-resume", taskId],
    queryFn: ({ signal }) => fetchResumeOptions(taskId, signal),
    retry: false,
  });
  const [selected, setSelected] = useState("");
  const [steps, setSteps] = useState("");
  const [queue, setQueue] = useState(false);
  const [confirmation, setConfirmation] = useState("");
  const qc = useQueryClient();
  const checkpoint = selected
    ? query.data?.checkpoints.find((entry) => entry.path === selected)
    : query.data?.checkpoints.find((entry) => entry.path === query.data.default_checkpoint) || query.data?.checkpoints[0];
  const target = steps ? Number(steps) : undefined;
  const { targetValid, available, reason, appendSteps } = resumeAvailability(checkpoint, target);
  const confirmationKey = JSON.stringify([taskId, checkpoint?.path, steps, queue, query.dataUpdatedAt, query.data]);
  const confirmed = confirmation === confirmationKey;
  const resume = useMutation({
    mutationFn: (submission: ResumeSubmission) =>
      resumeHistoryTask(submission.taskId, submission.path, submission.queue, submission.appendSteps),
    retry: false,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["training-history"] });
      qc.invalidateQueries({ queryKey: ["training-queue"] });
    },
  });
  return (
    <CommandDialog
      title="从历史检查点续训"
      busy={resume.isPending}
      onClose={onClose}
    >
      <fieldset
        className="unframed-fieldset settings-grid"
        disabled={resume.isPending || resume.isSuccess}
      >
        <label className="full-width">
          <span>检查点</span>
          <select
            value={checkpoint?.path || ""}
            onChange={(e) => {
              setSelected(e.target.value);
              setConfirmation("");
            }}
          >
            {query.data?.checkpoints.map((entry) => (
              <option key={entry.path} value={entry.path}>
                {entry.name} · step {entry.step ?? "?"}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>训练目标总步数</span>
          <input
            type="number"
            min={Number(checkpoint?.step || 0) + 1}
            step={1}
            placeholder="保持历史目标"
            value={steps}
            onChange={(e) => {
              setSteps(e.target.value);
              setConfirmation("");
            }}
          />
        </label>
        <label>
          <span>执行方式</span>
          <select
            value={queue ? "queue" : "now"}
            onChange={(e) => {
              setQueue(e.target.value === "queue");
              setConfirmation("");
            }}
          >
            <option value="now">立即续训</option>
            <option value="queue">排队续训</option>
          </select>
        </label>
        <label className="checkbox-row full-width">
          <input
            type="checkbox"
            checked={confirmed}
            disabled={!available || query.isFetching || Boolean(query.error) || Boolean(resume.error)}
            onChange={(e) => setConfirmation(e.target.checked ? confirmationKey : "")}
          />
          <span>
            使用历史配置和训练状态创建新任务，保留原记录。排队续训会沿用当前调度状态。
          </span>
        </label>
      </fieldset>
      {checkpoint && <p className="data-scope">检查点 {checkpoint.step ?? "未记录"} 步 · 历史目标 {checkpoint.target_total_steps ?? "未估算"} 步{appendSteps !== undefined ? ` · 本次追加 ${appendSteps} 步` : checkpoint.remaining_steps !== undefined ? ` · 剩余 ${checkpoint.remaining_steps} 步` : ""}</p>}
      {!available && (
        <p role="status">
          {query.isPending ? "正在检查检查点" : checkpoint ? reason : query.data?.message || reason}
        </p>
      )}
      {query.error && (
        <p role="alert" className="form-error">
          {query.error.message} <button type="button" onClick={() => query.refetch()}>重试读取</button>
        </p>
      )}
      {resume.error && (
        <div role="alert" className="form-error">
          {resume.error.message}
          {resume.error instanceof ApiError && resume.error.status === 0 ? <p><Link to="/history">核对历史任务</Link> · <Link to="/queue">核对队列</Link></p> :
            <button type="button" disabled={query.isFetching} onClick={async () => {
              setConfirmation("");
              const refreshed = await query.refetch();
              if (!refreshed.error) resume.reset();
            }}>重新检查后重试</button>}
        </div>
      )}
      {resume.isSuccess ? (
        <div role="status">
          <p>{resume.data.message || "续训请求已提交"}</p>
          <Link to={resume.variables?.queue ? "/queue" : "/monitor"}>
            查看{resume.variables?.queue ? "队列" : "监控"}
          </Link>
        </div>
      ) : (
        <footer className="toolbar">
          <button type="button" disabled={resume.isPending} onClick={onClose}>
            取消
          </button>
          <button
            type="button"
            className="primary-command"
            disabled={
              !available ||
              !targetValid ||
              !confirmed ||
              resume.isPending ||
              query.isFetching || Boolean(query.error) ||
              Boolean(resume.error)
            }
            onClick={() => {
              if (checkpoint && confirmed && available && !query.isFetching && !query.error && !resume.error && !resume.isPending) {
                resume.mutate({ taskId, path: checkpoint.path, queue, appendSteps });
              }
            }}
          >
            确认续训
          </button>
        </footer>
      )}
    </CommandDialog>
  );
}
