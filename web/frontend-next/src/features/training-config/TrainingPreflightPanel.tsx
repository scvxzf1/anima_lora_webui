import type { TrainingPreflightResponse } from "./api";
import { taskLabel } from "../../api/modelCapabilities";

type Props = {
  result?: TrainingPreflightResponse;
  pending: boolean;
  error?: string;
  onLocate?: (key: string) => void;
};

export function TrainingPreflightPanel({
  result,
  pending,
  error,
  onLocate,
}: Props) {
  return (
    <section className="training-preflight-panel" aria-live="polite">
      <header>
        <div>
          <p className="eyebrow">PREFLIGHT</p>
          <h2>训练预检测</h2>
        </div>
        <span
          className="training-preflight-state"
          data-tone={result?.ok ? "success" : result ? "danger" : "neutral"}
        >
          {pending
            ? "检测中"
            : result?.ok
              ? "可以继续"
              : result
                ? "需要处理"
                : "尚未检测"}
        </span>
      </header>
      {error ? (
        <p className="training-command-error" role="alert">
          {error}
        </p>
      ) : null}
      {result ? (
        <>
          {result.training_task && <dl className="command-summary" aria-label="模型与数据集能力检查">
            <dt>当前模型</dt><dd>{result.training_task.model_name}</dd>
            <dt>训练能力</dt><dd>{result.training_task.supported_tasks.map(taskLabel).join("、")}</dd>
            <dt>当前任务</dt><dd>{taskLabel(result.training_task.configured_task)}</dd>
            <dt>数据集任务</dt><dd>{taskLabel(result.training_task.dataset_task)}</dd>
          </dl>}
          <div className="training-preflight-summary">
            <span>
              <strong>{result.summary.errors}</strong>错误
            </span>
            <span>
              <strong>{result.summary.warnings}</strong>警告
            </span>
            <span>
              <strong>{result.summary.checks}</strong>检查
            </span>
          </div>
          <ul className="training-preflight-checks">
            {result.checks.map((check, index) => (
              <li key={`${check.key}-${index}`} data-level={check.level}>
                <span>
                  {check.level === "ok"
                    ? "通过"
                    : check.level === "warning"
                      ? "警告"
                      : "错误"}
                </span>
                <div>
                  <strong>{check.message}</strong>
                  {check.path ? <code>{check.path}</code> : null}
                  {onLocate && check.level !== "ok" && check.key && (
                    <button type="button" onClick={() => onLocate(check.key)}>
                      定位 {check.key}
                    </button>
                  )}
                </div>
              </li>
            ))}
          </ul>
        </>
      ) : null}
    </section>
  );
}
