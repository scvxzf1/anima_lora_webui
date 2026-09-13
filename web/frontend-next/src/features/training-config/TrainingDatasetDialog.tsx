import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useQuery } from "@tanstack/react-query";
import { ExternalLink, RefreshCw, X } from "lucide-react";
import { Link } from "react-router-dom";
import { datasetLibraryQuery, datasetPresetQuery } from "../dataset-editor/api";
import { DatasetCover } from "../dataset-editor/DatasetCover";
import { useDialogLifecycle } from "../dataset-editor/useDialogLifecycle";
import {
  TrainingDatasetEditor,
  type DatasetEditorStatus,
} from "./TrainingDatasetEditor";
import "./TrainingDatasetDialog.css";

type Props = {
  value: string;
  onClose: () => void;
  onChoose: (file: string) => void;
};

export function TrainingDatasetDialog({ value, onClose, onChoose }: Props) {
  const library = useQuery(datasetLibraryQuery);
  const [selected, setSelected] = useState(value);
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState<DatasetEditorStatus>({
    dirty: false,
    busy: false,
  });
  const ref = useRef<HTMLElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const returnFocusRef = useRef(
    document.activeElement instanceof HTMLElement ? document.activeElement : null,
  );
  const detail = useQuery({
    ...datasetPresetQuery(selected),
    enabled: Boolean(selected),
    refetchOnWindowFocus: false,
  });
  const presets = library.data?.presets ?? [];
  const term = search.trim().toLocaleLowerCase();
  const groups = library.data?.groups ?? [];
  const grouped = new Set(
    groups.flatMap((group) => group.files.map((file) => file.path)),
  );
  const visible = [
    ...groups,
    {
      id: "other",
      label: "其他蓝图",
      files: presets.filter((file) => !grouped.has(file.path)),
    },
  ]
    .map((group) => ({
      ...group,
      files: group.files.filter((file) =>
        `${group.label} ${file.label || ""} ${file.path}`
          .toLocaleLowerCase()
          .includes(term),
      ),
    }))
    .filter((group) => group.files.length);

  function canLeave() {
    return (
      !status.busy &&
      (!status.dirty ||
        window.confirm("数据集参数有未保存修改，是否放弃这些修改？"))
    );
  }
  function close() {
    if (canLeave()) onClose();
  }
  useDialogLifecycle({
    dialogRef: ref,
    initialFocusRef: searchRef,
    returnFocusRef,
    onClose: close,
  });
  useEffect(() => {
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const guard = (event: BeforeUnloadEvent) => {
      if (status.dirty || status.busy) event.preventDefault();
    };
    window.addEventListener("beforeunload", guard);
    return () => {
      document.body.style.overflow = previous;
      window.removeEventListener("beforeunload", guard);
    };
  }, [status]);

  return createPortal(
    <div className="training-dataset-backdrop">
      <section
        ref={ref}
        role="dialog"
        aria-modal="true"
        aria-labelledby="training-dataset-title"
        className="training-dataset-dialog"
      >
        <header>
          <div>
            <h2 id="training-dataset-title">选择与配置数据集</h2>
            <span>{presets.length} 个蓝图</span>
          </div>
          <button
            type="button"
            className="icon-button"
            title="关闭数据集弹窗"
            aria-label="关闭数据集弹窗"
            disabled={status.busy}
            onClick={close}
          >
            <X size={18} />
          </button>
        </header>
        <div className="training-dataset-dialog-body">
          <aside aria-label="数据集蓝图库">
            <div className="training-dataset-library-search">
              <input
                ref={searchRef}
                aria-label="搜索数据集蓝图"
                type="search"
                placeholder="名称、分组或路径"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
              />
              <button
                type="button"
                className="icon-button"
                aria-label="刷新数据集蓝图"
                title="刷新数据集蓝图"
                disabled={library.isFetching || status.busy}
                onClick={() => void library.refetch()}
              >
                <RefreshCw size={16} />
              </button>
            </div>
            <div className="training-dataset-library-list">
              {library.isPending && <p role="status">正在读取蓝图库…</p>}
              {library.isError && <p role="alert">{library.error.message}</p>}
              {library.isSuccess && !visible.length && (
                <p role="status">
                  {presets.length ? "没有匹配的蓝图" : "暂无数据集蓝图"}
                </p>
              )}
              {visible.map((group) => (
                <section key={group.id}>
                  <h3>{group.label}</h3>
                  {group.files.map((file) => (
                    <button
                      type="button"
                      key={file.path}
                      className="training-dataset-library-item"
                      aria-pressed={selected === file.path}
                      disabled={status.busy}
                      onClick={() => {
                        if (file.path !== selected && canLeave()) {
                          setStatus({ dirty: false, busy: false });
                          setSelected(file.path);
                        }
                      }}
                    >
                      <DatasetCover file={file.path} />
                      <span>
                        <strong>
                          {file.label ||
                            file.filename ||
                            file.path.split("/").pop()}
                        </strong>
                        <small>
                          {file.summary?.dataset_count ?? "—"} 个子集
                        </small>
                        <code>{file.path}</code>
                      </span>
                    </button>
                  ))}
                </section>
              ))}
            </div>
            <Link
              to={`/datasets?${new URLSearchParams({ dataset: selected })}`}
              target="_blank"
              rel="noopener noreferrer"
            >
              <ExternalLink size={14} />
              完整蓝图管理
            </Link>
          </aside>
          <main className="training-dataset-dialog-detail">
            {!selected ? (
              <p className="training-dataset-empty">未选择数据集蓝图</p>
            ) : detail.isPending ? (
              <p role="status">正在读取子集参数…</p>
            ) : detail.isError ? (
              <div role="alert">
                <p>{detail.error.message}</p>
                <button type="button" onClick={() => void detail.refetch()}>
                  重试
                </button>
              </div>
            ) : (
              <TrainingDatasetEditor
                key={selected}
                preset={detail.data}
                onStatus={setStatus}
                onChoose={onChoose}
              />
            )}
          </main>
        </div>
        <footer>
          <span>
            {status.busy
              ? "正在保存蓝图…"
              : status.dirty
                ? "蓝图参数未保存"
                : "选择尚未应用到训练配置"}
          </span>
          <button type="button" disabled={status.busy} onClick={close}>
            取消
          </button>
        </footer>
      </section>
    </div>,
    document.body,
  );
}
