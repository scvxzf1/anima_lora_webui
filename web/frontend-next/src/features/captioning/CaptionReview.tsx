import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Save,
  FileCheck2,
  Square,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { useState } from "react";
import { ApiError } from "../../api/client";
import { CaptionTranslation } from "./CaptionTranslation";
import { CaptionReviewContext } from "./CaptionReviewContext";
import { CaptionPreviewImage } from "./CaptionPreviewImage";
import { QueryFeedback } from "../../components/QueryFeedback";
import { ResilientImage } from "../../components/ResilientImage";
import { useUnsavedChangesGuard } from "../dataset-editor/useUnsavedChangesGuard";
import {
  captioningKeys,
  captionActive,
  captionImageUrl,
  fetchCaptionJob,
  fetchCaptionLogs,
  updateCaptionItem,
  commitCaptionJob,
  cancelCaptionJob,
  rerunCaptionJob,
} from "./api";

export function CaptionReview({ jobId }: { jobId: string }) {
  const qc = useQueryClient();
  const query = useQuery({
    queryKey: captioningKeys.job(jobId),
    queryFn: ({ signal }) => fetchCaptionJob(jobId, signal),
    retry: false,
    refetchInterval: (q) =>
      captionActive(q.state.data?.job.state) ? 2000 : false,
  });
  const logs = useQuery({
    queryKey: ["captioning", "logs", jobId],
    queryFn: ({ signal }) => fetchCaptionLogs(jobId, signal),
    retry: false,
    refetchInterval: 5000,
  });
  const [selectedId, setSelectedId] = useState("");
  const [checked, setChecked] = useState<string[]>([]);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [page, setPage] = useState(0);
  const [mobileView, setMobileView] = useState<"editor" | "items">("editor");
  const [needsReconcile, setNeedsReconcile] = useState(false);
  const dirty = Object.keys(drafts).length > 0;
  useUnsavedChangesGuard(
    dirty,
    "候选文本有未保存修改，离开会丢失这些修改。是否继续？",
  );
  const job = query.data?.job;
  const item =
    job?.items.find((entry) => entry.id === selectedId) || job?.items[0];
  const currentPage = Math.min(page, Math.max(0, Math.ceil((job?.items.length || 0) / 50) - 1));
  const pageItems = job?.items.slice(currentPage * 50, (currentPage + 1) * 50) || [];
  const selectedIds = checked.filter((id) => job?.items.some((item) => item.id === id));
  const writableIds = (job?.items || []).filter((entry) =>
    ["ready", "failed", "committed"].includes(entry.state) && entry.proposed_caption.trim(),
  ).map((entry) => entry.id);
  function locate(id: string) {
    const index = job?.items.findIndex((entry) => entry.id === id) ?? -1;
    if (index < 0) return;
    setSelectedId(id);
    setPage(Math.floor(index / 50));
    setMobileView("editor");
  }
  const running = captionActive(job?.state);
  const save = useMutation({
    mutationFn: ({ id, text }: { id: string; text: string }) =>
      updateCaptionItem(jobId, id, text),
    retry: false,
    onSuccess: (data, submitted) => {
      qc.setQueryData(captioningKeys.job(jobId), data);
      setDrafts((current) => {
        const next = { ...current };
        if (next[submitted.id] === submitted.text) delete next[submitted.id];
        return next;
      });
    },
  });
  const commit = useMutation({
    mutationFn: (ids: string[]) => commitCaptionJob(jobId, ids),
    retry: false,
    onSuccess: (data) => {
      qc.setQueryData(captioningKeys.job(jobId), { ok: true, job: data.job });
    },
  });
  const control = useMutation({
    mutationFn: (action: "cancel" | "rerun") =>
      action === "cancel"
        ? cancelCaptionJob(jobId)
        : rerunCaptionJob(jobId, job!.profile_id, selectedIds),
    retry: false,
    onSuccess: (data) => {
      qc.setQueryData(captioningKeys.job(jobId), data);
      qc.invalidateQueries({ queryKey: captioningKeys.jobs });
      setNeedsReconcile(false);
    },
    onError: async (error) => {
      if (error instanceof ApiError && error.status === 409) {
        await query.refetch();
      } else if (error instanceof ApiError && error.status === 0) {
        setNeedsReconcile(true);
      }
    },
  });
  const busy = save.isPending || commit.isPending || control.isPending || needsReconcile;
  async function reconcileUnknown() {
    const result = await query.refetch();
    if (result.isError || !result.data) return;
    setNeedsReconcile(false);
    control.reset();
    await qc.invalidateQueries({ queryKey: captioningKeys.jobs });
  }
  function write(ids: string[]) {
    if (
      !ids.length ||
      dirty ||
      query.isError ||
      !window.confirm(
        `将任务 ${jobId} 的 ${ids.length} 项已保存候选写入图片同名 TXT，包含其他页勾选项，已有文本可能被替换。确认写回吗？`,
      )
    )
      return;
    commit.mutate(ids);
  }
  if (!job)
    return (
      <QueryFeedback query={query} label="打标任务" hasData={false} />
    );
  return (
    <section>
      <header className="review-heading">
        <div>
          <h2>{job.profile_name}</h2>
          <p>
            {job.state} · {job.completed}/{job.total} · 失败 {job.failed}
          </p>
        </div>
        <div className="toolbar">
          <button
            type="button"
            disabled={!running || busy || needsReconcile}
            onClick={() => {
              if (window.confirm("取消此打标任务？已生成候选会保留。"))
                control.mutate("cancel");
            }}
          >
            <Square size={15} />
            取消任务
          </button>
          <button
            type="button"
            disabled={running || busy || dirty || query.isError || needsReconcile}
            onClick={() => {
              if (
                window.confirm(
                  "重新打标会重新使用此任务接入处理图片，并替换所选项的候选；外部接入可能产生费用。确认继续吗？",
                )
              )
                control.mutate("rerun");
            }}
          >
            <RefreshCw size={15} />
            重新打标
          </button>
        </div>
      </header>
      {needsReconcile && (
        <div className="toolbar">
          <p role="status">操作结果尚未确认。核对服务器任务状态后再继续。</p>
          <button
            type="button"
            disabled={query.isFetching}
            onClick={() => void reconcileUnknown()}
          >
            <RefreshCw size={15} />
            {query.isFetching ? "正在核对" : "核对任务状态"}
          </button>
        </div>
      )}
      <div className="toolbar">
        <button
          type="button"
          disabled={running || busy || dirty || query.isError || !selectedIds.length}
          onClick={() => write(selectedIds)}
        >
          <FileCheck2 size={16} />
          写回选中 ({selectedIds.length})
        </button>
        <button
          type="button"
          disabled={running || busy || dirty || query.isError || !writableIds.length}
          onClick={() => write(writableIds)}
        >
          写回全部候选 ({writableIds.length})
        </button>
      </div>
      <CaptionReviewContext items={job.items} current={item} page={currentPage} drafts={drafts} checked={selectedIds}
        onLocate={locate} mobileView={mobileView} onView={setMobileView} />
      <div className="caption-review-layout" data-mobile-view={mobileView}>
        <div className="review-items">
          <div className="toolbar">
            <button
              type="button"
              title="上一页"
              aria-label="上一页候选"
              disabled={!currentPage}
              onClick={() => setPage(currentPage - 1)}
            >
              <ChevronLeft size={16} />
            </button>
            <span>
              {currentPage + 1} / {Math.max(1, Math.ceil(job.items.length / 50))}
            </span>
            <button
              type="button"
              title="下一页"
              aria-label="下一页候选"
              disabled={(currentPage + 1) * 50 >= job.items.length}
              onClick={() => setPage(currentPage + 1)}
            >
              <ChevronRight size={16} />
            </button>
          </div>
          {pageItems.map((entry) => (
            <div
              key={entry.id}
              className="review-item"
              data-selected={entry.id === item?.id}
            >
              <input
                type="checkbox"
                aria-label={`选择 ${entry.name}`}
                checked={checked.includes(entry.id)}
                disabled={busy}
                onChange={(e) =>
                  setChecked((ids) =>
                    e.target.checked
                      ? [...ids, entry.id]
                      : ids.filter((id) => id !== entry.id),
                  )
                }
              />
              <button type="button" onClick={() => locate(entry.id)}>
                <ResilientImage className="caption-thumbnail"
                  src={captionImageUrl(entry.thumbnail_url || entry.url) || undefined}
                  loading="lazy"
                  alt=""
                />
                <span>
                  {entry.name}
                  <small>
                    {entry.state}
                    {drafts[entry.id] !== undefined ? " · 未保存" : ""}
                  </small>
                </span>
              </button>
            </div>
          ))}
        </div>
        {item && (
          <div className="caption-inspector">
            <CaptionPreviewImage key={item.id}
              src={captionImageUrl(item.url)}
              name={item.name}
            />
            <div className="caption-editor">
              <label>
                <span>原有标注</span>
                <textarea rows={4} value={item.caption} readOnly />
              </label>
              <label>
                <span>候选标注</span>
                <textarea
                  rows={7}
                  value={drafts[item.id] ?? item.proposed_caption}
                  disabled={running || busy}
                  onChange={(e) =>
                    setDrafts((current) => {
                      const next = { ...current };
                      if (e.target.value === item.proposed_caption)
                        delete next[item.id];
                      else next[item.id] = e.target.value;
                      return next;
                    })
                  }
                />
              </label>
              <button
                type="button"
                disabled={running || busy || query.isError || drafts[item.id] === undefined}
                onClick={() =>
                  save.mutate({ id: item.id, text: drafts[item.id] })
                }
              >
                <Save size={16} />
                保存候选
              </button>
              <CaptionTranslation
                key={item.id}
                value={drafts[item.id] ?? item.proposed_caption}
                disabled={running || busy}
                onApply={(text) =>
                  setDrafts((current) => ({ ...current, [item.id]: text }))
                }
              />
              {(item.error || item.commit_error) && (
                <p className="form-error" role="alert">
                  {item.error || item.commit_error}
                </p>
              )}
            </div>
          </div>
        )}
      </div>
      {(save.error || commit.error || (!needsReconcile && control.error) || query.error) && (
        <p className="form-error" role="alert">
          {
            (save.error || commit.error || (!needsReconcile && control.error) || query.error)
              ?.message
          }
        </p>
      )}
      {commit.data && (
        <div role="status">
          <p>
            写入 {commit.data.written} · 冲突 {commit.data.conflicts} · 跳过{" "}
            {commit.data.skipped}
          </p>
          {commit.data.errors.map((error, index) => (
            <p key={index} className="form-error">
              {error.file}: {error.error}
            </p>
          ))}
        </div>
      )}
      <details className="caption-logs">
        <summary>任务日志</summary>
        <QueryFeedback query={logs} label="打标日志" hasData={Boolean(logs.data)} />
        {logs.data && <pre>
          {logs.data?.lines
            .map((line) => `[${line.level}] ${line.message}`)
            .join("\n") || "暂无日志"}
        </pre>}
      </details>
    </section>
  );
}
