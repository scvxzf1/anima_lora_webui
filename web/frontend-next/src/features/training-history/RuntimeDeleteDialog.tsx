import { useEffect, useRef, useState } from "react";
import { CommandDialog } from "../../components/CommandDialog";
import {
  confirmRuntimeDelete,
  previewRuntimeDelete,
  type RuntimeDeletePreview,
  type RuntimeDeleteResult,
} from "./runtimeDeleteApi";
import "./RuntimeDeleteDialog.css";

export function RuntimeDeleteDialog({
  taskIds,
  onClose,
  onSuccess,
}: {
  taskIds: string[];
  onClose: () => void;
  onSuccess: (result: RuntimeDeleteResult) => void;
}) {
  const [preview, setPreview] = useState<RuntimeDeletePreview | null>(null);
  const [loading, setLoading] = useState(true);
  const [previewError, setPreviewError] = useState("");
  const [confirmed, setConfirmed] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [attempted, setAttempted] = useState(false);
  const [submitError, setSubmitError] = useState("");
  const [result, setResult] = useState<RuntimeDeleteResult | null>(null);
  const submitLock = useRef(false);

  useEffect(() => {
    let active = true;
    previewRuntimeDelete(taskIds)
      .then((value) => {
        if (active) setPreview(value);
      })
      .catch((error: unknown) => {
        if (active)
          setPreviewError(
            error instanceof Error ? error.message : String(error),
          );
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [taskIds]);

  async function submit() {
    if (!preview || preview.blocked.length || !confirmed || submitLock.current)
      return;
    submitLock.current = true;
    setAttempted(true);
    setSubmitting(true);
    setSubmitError("");
    try {
      const value = await confirmRuntimeDelete(taskIds, preview);
      setResult(value);
      onSuccess(value);
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : String(error));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <CommandDialog
      title="彻底删除历史与运行目录"
      onClose={onClose}
      busy={submitting}
    >
      <div className="runtime-delete-dialog">
        <p className="runtime-delete-warning">
          此操作会删除列出的历史记录及允许删除的运行目录，且无法撤销。权重等目录内容以服务端预览为准。
        </p>
        {loading && <p role="status">正在检查删除范围…</p>}
        {previewError && <p role="alert">预览失败：{previewError}</p>}
        {preview && (
          <>
            <section aria-label="将删除的历史记录">
              <h3>历史记录（{preview.task_count}）</h3>
              {preview.tasks.length ? (
                <ul>
                  {preview.tasks.map((task) => (
                    <li key={task.id}>
                      <strong>{task.name || task.id}</strong>
                      <small>
                        {task.id}
                        {task.state ? ` · ${task.state}` : ""}
                      </small>
                    </li>
                  ))}
                </ul>
              ) : (
                <p>没有可删除的历史记录。</p>
              )}
            </section>
            <section aria-label="运行目录删除范围">
              <h3>运行目录（{preview.runtime_dir_count}）</h3>
              {preview.runtime_dirs.length ? (
                <ul>
                  {preview.runtime_dirs.map((dir) => (
                    <li key={dir.path}>
                      <code>{dir.path}</code>
                      <small>
                        {dir.status === "missing" ? "已不存在" : "允许删除"}
                      </small>
                    </li>
                  ))}
                </ul>
              ) : (
                <p>没有关联运行目录。</p>
              )}
            </section>
            {preview.blocked.length > 0 && (
              <section className="runtime-delete-blocked" aria-label="阻止项">
                <h3>无法执行（{preview.blocked.length}）</h3>
                <ul>
                  {preview.blocked.map((item, index) => (
                    <li key={`${item.id || item.path || "blocked"}-${index}`}>
                      <strong>{item.id || item.path || "删除范围"}</strong>
                      <span>{item.reason}</span>
                    </li>
                  ))}
                </ul>
                <p>请先处理所有阻止项，再重新打开预览。</p>
              </section>
            )}
          </>
        )}
        {submitError && (
          <p role="alert">
            删除请求失败：{submitError}
            。请先刷新历史记录核对结果，不要直接重复提交。
          </p>
        )}
        {result && (
          <section className="runtime-delete-result" aria-label="删除结果">
            <h3>删除结果</h3>
            <p>
              已删除 {result.deleted_task_ids?.length ?? 0} 条历史记录，运行目录{" "}
              {result.deleted_runtime_dirs?.length ?? 0} 个。
            </p>
            {Object.entries(result.runtime_cleanup_errors || {}).length > 0 && (
              <>
                <h4>未能清理的运行目录</h4>
                <ul>
                  {Object.entries(result.runtime_cleanup_errors || {}).map(
                    ([path, error]) => (
                      <li key={path}>
                        <code>{path}</code>
                        <span>{error}</span>
                      </li>
                    ),
                  )}
                </ul>
              </>
            )}
            {Object.entries(result.cleanup_errors || {}).length > 0 && (
              <>
                <h4>未能清理的历史目录</h4>
                <ul>
                  {Object.entries(result.cleanup_errors || {}).map(([path, error]) => (
                    <li key={path}>
                      <code>{path}</code>
                      <span>{error}</span>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </section>
        )}
        {!result && (
          <label className="runtime-delete-confirm">
            <input
              type="checkbox"
              checked={confirmed}
              onChange={(event) => setConfirmed(event.target.checked)}
              disabled={
                !preview ||
                Boolean(preview.blocked.length) ||
                loading ||
                submitting
              }
            />
            <span>我已核对上述列表，并确认永久删除</span>
          </label>
        )}
        <div className="runtime-delete-actions">
          <button type="button" onClick={onClose} disabled={submitting}>
            {result ? "完成" : "取消"}
          </button>
          {!result && (
            <button
              type="button"
              className="history-danger"
              onClick={submit}
              disabled={
                !preview ||
                Boolean(preview.blocked.length) ||
                !confirmed ||
                loading ||
                submitting ||
                attempted
              }
            >
              {submitting ? "正在删除…" : "确认彻底删除"}
            </button>
          )}
        </div>
      </div>
    </CommandDialog>
  );
}
