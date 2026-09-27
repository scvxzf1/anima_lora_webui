import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router-dom";
import type { TrainingConfigFile } from "../../api/trainingContext";
import { CommandDialog } from "../../components/CommandDialog";
import { ApiError } from "../../api/client";
import {
  enqueueTraining,
  runTrainingPreflight,
  startTraining,
  type TrainingActionResponse,
} from "./api";
import { TrainingPreflightPanel } from "./TrainingPreflightPanel";

export function TrainingLaunchDialog({
  file,
  preset,
  mode,
  onClose,
  gpuIds,
  deviceSummary,
  deviceIssue,
}: {
  file: TrainingConfigFile;
  preset: string;
  mode: "start" | "queue";
  onClose: () => void;
  gpuIds: string[];
  deviceSummary: string;
  deviceIssue: string;
}) {
  const [confirmed, setConfirmed] = useState(false);
  const qc = useQueryClient();
  const preflight = useQuery({
    queryKey: ["training-launch-preflight", file.path, preset, gpuIds],
    queryFn: () => runTrainingPreflight(file, preset, gpuIds),
    retry: false,
    staleTime: 0,
    refetchOnWindowFocus: false,
    enabled: !deviceIssue,
  });
  const action = useMutation({
    mutationFn: () => {
      if (deviceIssue || !confirmed || !preflight.data?.ok || preflight.isFetching)
        throw new Error("请先完成预检并确认");
      const confirmation = {
        confirmed: true,
        confirm_preprocess: true,
      } as const;
      return mode === "start"
        ? startTraining(file, preset, confirmation, gpuIds)
        : enqueueTraining(file, preset, confirmation, gpuIds);
    },
    retry: false,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["training-queue"] });
      qc.invalidateQueries({ queryKey: ["live-monitor"] });
    },
  });
  const failedPreflight =
    action.error instanceof ApiError
      ? (action.error.payload as TrainingActionResponse)?.preflight
      : undefined;
  const startResultMayBeUnknown =
    mode === "start" &&
    (action.error instanceof TypeError ||
      (action.error instanceof ApiError &&
        (action.error.status === 0 || action.error.status >= 500)));
  return (
    <CommandDialog
      title={mode === "start" ? "确认启动训练" : "确认加入队列"}
      onClose={onClose}
      busy={action.isPending}
    >
      <dl className="command-summary">
        <dt>配置</dt>
        <dd>{file.path}</dd>
        <dt>运行预设</dt>
        <dd>{preset}</dd>
        <dt>训练设备</dt>
        <dd>{deviceSummary}</dd>
      </dl>
      <fieldset
        disabled={action.isPending || action.isSuccess}
        className="unframed-fieldset"
      >
        <legend>启动检查</legend>
        {deviceIssue && <p className="form-error" role="alert">{deviceIssue}</p>}
        <TrainingPreflightPanel
          result={failedPreflight || preflight.data}
          pending={preflight.isFetching}
          error={preflight.error?.message}
        />
        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={confirmed}
            onChange={(e) => setConfirmed(e.target.checked)}
          />
          <span>
            {mode === "start"
              ? "确认以上检查结果，允许必要的预处理完成后开始训练。"
              : "确认以上检查结果，冻结配置并暂停后续调度；稍后从队列恢复执行。"}
          </span>
        </label>
      </fieldset>
      {action.error && (
        <>
          <p className="form-error" role="alert">
            {action.error.message}
            {startResultMayBeUnknown
              ? "；启动结果可能未知，请先核对当前监控，不要重复提交。"
              : ""}
          </p>
          {mode === "start" && (
            <p>
              <Link to="/monitor">查看当前监控，核对启动结果</Link>
            </p>
          )}
        </>
      )}
      {action.isSuccess ? (
        <div role="status">
          <p>{action.data.message || "请求已完成"}</p>
          <Link to={mode === "start" ? "/monitor" : "/queue"}>
            查看{mode === "start" ? "当前监控" : "训练队列"}
          </Link>
        </div>
      ) : (
        <footer className="toolbar">
          <button type="button" onClick={onClose} disabled={action.isPending}>
            取消
          </button>
          <button
            type="button"
            className="primary-command"
            disabled={
              !confirmed ||
              Boolean(deviceIssue) ||
              !preflight.data?.ok ||
              preflight.isFetching ||
              action.isPending ||
              Boolean(action.error)
            }
            onClick={() => action.mutate()}
          >
            {action.isPending
              ? "正在提交"
              : mode === "start"
                ? "确认启动"
                : "确认入队"}
          </button>
        </footer>
      )}
    </CommandDialog>
  );
}
