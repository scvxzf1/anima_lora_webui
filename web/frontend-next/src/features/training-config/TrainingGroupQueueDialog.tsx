import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router-dom";
import { ApiError } from "../../api/client";
import type { TrainingConfigGroup } from "../../api/trainingContext";
import { CommandDialog } from "../../components/CommandDialog";
import {
  enqueueTrainingGroup,
  queueableTrainingFiles,
  type GroupQueueResponse,
} from "./trainingGroupQueue";

export function TrainingGroupQueueDialog({
  group,
  preset,
  gpuIds,
  deviceSummary,
  deviceIssue,
  onClose,
}: {
  group: TrainingConfigGroup;
  preset: string;
  gpuIds: string[];
  deviceSummary: string;
  deviceIssue: string;
  onClose: () => void;
}) {
  const files = queueableTrainingFiles(group);
  const [confirmed, setConfirmed] = useState(false);
  const qc = useQueryClient();
  const action = useMutation({
    mutationFn: () => enqueueTrainingGroup(group, files, preset, gpuIds),
    retry: false,
    onSuccess: () =>
      void qc.invalidateQueries({ queryKey: ["training-queue"] }),
    onError: (error) => {
      const payload =
        error instanceof ApiError
          ? (error.payload as GroupQueueResponse | null)
          : null;
      if (payload?.queued_count)
        void qc.invalidateQueries({ queryKey: ["training-queue"] });
    },
  });
  const payload =
    action.error instanceof ApiError
      ? (action.error.payload as GroupQueueResponse | null)
      : null;
  const failed = payload?.failures?.[0];
  const failedIndex = payload?.failed_index ?? failed?.index;
  const failedPath =
    failed?.config_file ||
    payload?.failed_item?.config_file ||
    (failedIndex !== undefined ? files[failedIndex]?.path : "");
  const unknownResult =
    action.error instanceof ApiError &&
    (action.error.status === 0 || action.error.status >= 500);
  const failureMessage =
    payload?.error ||
    failed?.error ||
    payload?.message ||
    action.error?.message;
  return (
    <CommandDialog
      title="批量加入训练队列"
      onClose={onClose}
      busy={action.isPending}
    >
      <dl className="command-summary">
        <dt>分组</dt>
        <dd>
          {group.label} · {files.length} 个配置
        </dd>
        <dt>运行预设</dt>
        <dd>{preset}</dd>
        <dt>训练设备</dt>
        <dd>{deviceSummary}</dd>
      </dl>
      <p>后端将逐项预检并冻结独立运行配置。队列保持暂停，原 TOML 不会修改。</p>
      <ul className="training-group-queue-files">
        {files.slice(0, 12).map((file) => (
          <li key={file.path}>
            <code>{file.path}</code>
          </li>
        ))}
      </ul>
      {files.length > 12 && <p>还有 {files.length - 12} 个配置</p>}
      {deviceIssue && (
        <p role="alert" className="form-error">
          {deviceIssue}
        </p>
      )}
      {action.error && (
        <p role="alert" className="form-error">
          {payload?.queued_count
            ? `已加入 ${payload.queued_count} 个配置；`
            : ""}
          {failedPath ? `${failedPath}：` : ""}
          {failureMessage}
          {unknownResult
            ? "；操作结果可能未知，请先核对训练队列，不要重复提交。"
            : ""}
        </p>
      )}
      {action.isSuccess ? (
        <div role="status">
          <p>{action.data.message || `已加入 ${files.length} 个配置`}</p>
          <Link to="/queue">查看训练队列</Link>
        </div>
      ) : (
        <>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={confirmed}
              disabled={action.isPending || Boolean(action.error)}
              onChange={(event) => setConfirmed(event.target.checked)}
            />
            <span>确认批量预检并将这些配置加入暂停队列</span>
          </label>
          <footer className="toolbar">
            <button type="button" onClick={onClose} disabled={action.isPending}>
              取消
            </button>
            <button
              type="button"
              className="primary-command"
              disabled={
                !confirmed ||
                !files.length ||
                !preset ||
                Boolean(deviceIssue) ||
                action.isPending ||
                Boolean(action.error)
              }
              onClick={() => action.mutate()}
            >
              {action.isPending ? "正在提交" : "确认加入队列"}
            </button>
          </footer>
        </>
      )}
    </CommandDialog>
  );
}
