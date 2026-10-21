import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { RefreshCw, RotateCcw, ShieldPlus, Trash2 } from "lucide-react";
import { useState } from "react";
import {
  createCaptionSnapshot,
  deleteCaptionSnapshot,
  fetchCaptionSnapshots,
  restoreCaptionSnapshot,
  snapshotKeys,
  type CaptionSnapshot,
} from "./snapshotApi";

type Props = {
  datasetFile?: string;
  datasetIndex?: number;
  source?: string;
  onRestored?: () => void;
};

function reasonLabel(reason: string): string {
  return reason === "commit" ? "写回前自动" : "手动备份";
}

function formatBytes(bytes: number): string {
  if (!bytes) return "0 B";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function SnapshotRow({
  snapshot,
  busy,
  onRestore,
  onDelete,
}: {
  snapshot: CaptionSnapshot;
  busy: boolean;
  onRestore: (id: string) => void;
  onDelete: (id: string) => void;
}) {
  const broken = Boolean(snapshot.broken);
  return (
    <li data-broken={broken ? "true" : undefined}>
      <span>
        <strong>{snapshot.created_at || snapshot.id}</strong>
        <small>
          {reasonLabel(snapshot.reason)} · {snapshot.file_count} 个文件 ·{" "}
          {formatBytes(snapshot.bytes)}
          {broken ? " · 文件损坏" : ""}
        </small>
      </span>
      <span className="toolbar">
        <button
          type="button"
          disabled={busy || broken}
          onClick={() => onRestore(snapshot.id)}
        >
          <RotateCcw size={14} />
          还原
        </button>
        <button type="button" disabled={busy} onClick={() => onDelete(snapshot.id)}>
          <Trash2 size={14} />
          删除
        </button>
      </span>
    </li>
  );
}

export function CaptionSnapshots({
  datasetFile = "",
  datasetIndex = 0,
  source = "source",
  onRestored,
}: Props) {
  const qc = useQueryClient();
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const key = snapshotKeys.list(datasetFile, datasetIndex);
  const query = useQuery({
    queryKey: key,
    queryFn: ({ signal }) => fetchCaptionSnapshots(datasetFile, datasetIndex, signal),
    retry: false,
    enabled: Boolean(datasetFile),
  });
  function reset() {
    setError("");
    setNotice("");
  }
  function fail(reason: unknown, fallback: string) {
    setNotice("");
    setError(reason instanceof Error ? reason.message : fallback);
  }
  const create = useMutation({
    mutationFn: () =>
      createCaptionSnapshot({ dataset_file: datasetFile, dataset_index: datasetIndex, source }),
    retry: false,
    onSuccess: () => {
      reset();
      setNotice("已创建还原点");
      void qc.invalidateQueries({ queryKey: key });
    },
    onError: (reason) => fail(reason, "创建还原点失败"),
  });
  const restore = useMutation({
    mutationFn: (id: string) => restoreCaptionSnapshot(id),
    retry: false,
    onSuccess: (data) => {
      reset();
      setNotice(`已还原 ${data.written} 个 caption，移除快照后新增的 ${data.removed} 个文件`);
      void qc.invalidateQueries({ queryKey: key });
      onRestored?.();
    },
    onError: (reason) => fail(reason, "还原失败"),
  });
  const remove = useMutation({
    mutationFn: (id: string) => deleteCaptionSnapshot(id),
    retry: false,
    onSuccess: () => {
      reset();
      setNotice("已删除还原点");
      void qc.invalidateQueries({ queryKey: key });
    },
    onError: (reason) => fail(reason, "删除还原点失败"),
  });
  if (!datasetFile) return null;
  const busy = create.isPending || restore.isPending || remove.isPending;
  const snapshots = query.data?.snapshots ?? [];
  return (
    <details className="caption-snapshots">
      <summary>
        <ShieldPlus size={15} />
        还原点（{snapshots.length}）
      </summary>
      <div className="caption-snapshots-body">
        <p className="caption-snapshots-hint">
          写回时会先尝试创建还原点：dataset 范围过大或目录不可用时可能没有还原点，
          届时审核页会明确提示。只备份与图片配对的 caption sidecar，还原不会影响图片。
        </p>
        <div className="toolbar">
          <button
            type="button"
            disabled={busy}
            onClick={() => {
              reset();
              create.mutate();
            }}
          >
            <ShieldPlus size={15} />
            立即备份
          </button>
          {query.isError && (
            <button
              type="button"
              disabled={query.isFetching}
              onClick={() => void query.refetch()}
            >
              <RefreshCw size={15} />
              {query.isFetching ? "正在重试" : "重试加载"}
            </button>
          )}
        </div>
        {error && (
          <p className="form-error" role="alert">
            {error}
          </p>
        )}
        {query.isError && (
          <p className="form-error" role="alert">
            还原点列表加载失败：
            {query.error instanceof Error ? query.error.message : "未知错误"}
          </p>
        )}
        {notice && !error && <p role="status">{notice}</p>}
        {query.isError ? null : snapshots.length === 0 ? (
          <p className="empty-state">
            {query.isPending ? "正在加载还原点…" : "暂无还原点"}
          </p>
        ) : (
          <ul className="caption-snapshot-list">
            {snapshots.map((snapshot) => (
              <SnapshotRow
                key={snapshot.id}
                snapshot={snapshot}
                busy={busy}
                onRestore={(id) => {
                  if (!window.confirm("还原会覆盖当前 caption 并移除快照后新增的 caption 文件，确认继续吗？")) return;
                  reset();
                  restore.mutate(id);
                }}
                onDelete={(id) => {
                  if (!window.confirm("删除这个还原点？")) return;
                  reset();
                  remove.mutate(id);
                }}
              />
            ))}
          </ul>
        )}
      </div>
    </details>
  );
}
