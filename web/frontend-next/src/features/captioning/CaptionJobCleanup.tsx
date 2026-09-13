import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Trash2 } from "lucide-react";
import { useState } from "react";
import { InlineConfirmDialog } from "../../components/InlineConfirmDialog";
import { captioningKeys, clearCaptionJobs, type CaptionJob } from "./api";

export function CaptionJobCleanup({
  jobs,
  onRemoved,
}: {
  jobs: CaptionJob[];
  onRemoved: (ids: string[]) => void;
}) {
  const qc = useQueryClient();
  const [pendingIds, setPendingIds] = useState<string[] | null>(null);
  const [notice, setNotice] = useState("");
  const eligible = jobs.filter((job) =>
    ["completed", "partial", "failed", "canceled"].includes(job.state),
  );
  const clear = useMutation({
    mutationFn: clearCaptionJobs,
    retry: false,
    onSuccess: async ({ removed, skipped }) => {
      onRemoved(removed);
      await Promise.all(
        removed.map((id) =>
          qc.cancelQueries({ queryKey: captioningKeys.job(id) }),
        ),
      );
      removed.forEach((id) =>
        qc.removeQueries({ queryKey: captioningKeys.job(id) }),
      );
      qc.invalidateQueries({ queryKey: captioningKeys.jobs });
      setNotice(
        `已清理 ${removed.length} 个任务${skipped.length ? `，${skipped.length} 个任务正在使用，已跳过` : ""}`,
      );
    },
  });
  return (
    <div className="caption-job-cleanup">
      <button
        type="button"
        title="清理已结束任务"
        aria-label="清理已结束任务"
        disabled={!eligible.length || clear.isPending}
        onClick={() => {
          setNotice("");
          clear.reset();
          setPendingIds(eligible.map((job) => job.id));
        }}
      >
        <Trash2 size={16} />
      </button>
      {pendingIds && (
        <InlineConfirmDialog
          title="清理已结束任务"
          danger
          confirmLabel="清理任务"
          message={`将清理 ${pendingIds.length} 个已结束任务及其候选标注，无法恢复。未写回的候选和编辑草稿将丢失；原始图片、已写回标注和模型不会删除。运行中或正在写回的任务会保留。`}
          onCancel={() => setPendingIds(null)}
          onConfirm={() => {
            clear.mutate(pendingIds);
            setPendingIds(null);
          }}
        />
      )}
      {clear.error && (
        <p role="alert" className="form-error">
          {clear.error.message}
        </p>
      )}
      {notice && <p role="status">{notice}</p>}
    </div>
  );
}
