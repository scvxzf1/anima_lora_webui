import { useState } from "react";
import { createPortal } from "react-dom";
import { useQuery } from "@tanstack/react-query";
import { Folder, File, ArrowUp } from "lucide-react";
import { CommandDialog } from "../../components/CommandDialog";
import { loadModelFiles, modelFilesError } from "./modelFilesApi";

export function TrainingModelPathDialog({
  label,
  disabled,
  onClose,
  onSelect,
}: {
  label: string;
  disabled: boolean;
  onClose: () => void;
  onSelect: (path: string) => void;
}) {
  const [directory, setDirectory] = useState<string>();
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState("");
  const query = useQuery({
    queryKey: ["training-model-files", directory, offset],
    queryFn: ({ signal }) => loadModelFiles(directory, offset, signal),
    retry: false,
    staleTime: 0,
  });
  const navigate = (path: string) => {
    setDirectory(path);
    setOffset(0);
    setSelected("");
  };
  const page = query.data;
  return createPortal(
    <div
      className="training-model-browser"
      onClick={(event) => event.stopPropagation()}
    >
      <CommandDialog title={`选择${label}服务器路径`} onClose={onClose}>
        <p className="training-model-browser-note">
          选择服务器上的模型文件或 Diffusers 目录，只更新当前草稿。
        </p>
        {page && (
          <>
            <label className="training-model-root">
              模型位置
              <select
                aria-label="模型位置"
                value={
                  page.roots.find((root) => root.path === page.directory)
                    ?.path ?? ""
                }
                onChange={(event) => {
                  if (event.target.value) navigate(event.target.value);
                }}
              >
                <option value="" disabled>
                  选择模型根目录
                </option>
                {page.roots.map((root) => (
                  <option key={root.path} value={root.path}>
                    {root.name} · {root.path}
                  </option>
                ))}
              </select>
            </label>
            <div className="training-model-directory">
              <button
                type="button"
                aria-label="返回上级目录"
                disabled={!page.parent || query.isFetching}
                onClick={() => {
                  if (page.parent) navigate(page.parent);
                }}
              >
                <ArrowUp size={16} />
                上级
              </button>
              <code aria-label="当前目录">{page.directory}</code>
            </div>
          </>
        )}
        {query.isFetching && <p role="status">正在读取服务器目录…</p>}
        {query.error && (
          <div role="alert">
            <p>{modelFilesError(query.error)}</p>
            <button type="button" onClick={() => void query.refetch()}>
              重试读取目录
            </button>
          </div>
        )}
        {page && !query.error && (
          <>
            <div className="training-model-entries" aria-label="服务器模型文件">
              {!page.entries.length && <p>此目录没有可选模型文件或子目录。</p>}
              {page.entries.map((entry) => (
                <button
                  type="button"
                  key={`${entry.path}\0${entry.name}`}
                  disabled={query.isFetching}
                  className="training-model-entry"
                  title={entry.path}
                  aria-pressed={
                    entry.kind === "file" ? selected === entry.path : undefined
                  }
                  onClick={() =>
                    entry.kind === "directory"
                      ? navigate(entry.path)
                      : setSelected(entry.path)
                  }
                >
                  {entry.kind === "directory" ? (
                    <Folder size={18} aria-hidden="true" />
                  ) : (
                    <File size={18} aria-hidden="true" />
                  )}
                  <span>{entry.name}</span>
                  <small>
                    {entry.kind === "directory" ? "打开目录" : "模型文件"}
                  </small>
                </button>
              ))}
            </div>
            <div className="training-model-pagination">
              <span>
                共 {page.total} 项 · 第 {Math.floor(offset / 100) + 1} 页
                {page.truncated ? "（部分结果）" : ""}
              </span>
              <button
                type="button"
                disabled={offset === 0 || query.isFetching}
                onClick={() => {
                  setOffset(Math.max(0, offset - 100));
                  setSelected("");
                }}
              >
                上一页
              </button>
              <button
                type="button"
                disabled={page.next_offset === null || query.isFetching}
                onClick={() => {
                  if (page.next_offset !== null) {
                    setOffset(page.next_offset);
                    setSelected("");
                  }
                }}
              >
                下一页
              </button>
            </div>
            {selected && (
              <p className="training-model-selection">
                已选择：<code>{selected}</code>
              </p>
            )}
          </>
        )}
        <footer className="training-model-browser-footer">
          <button type="button" onClick={onClose}>
            取消
          </button>
          <button
            type="button"
            disabled={
              disabled ||
              !page?.directory ||
              query.isFetching ||
              Boolean(query.error)
            }
            onClick={() => {
              if (page?.directory) onSelect(page.directory);
            }}
          >
            选择当前目录
          </button>
          <button
            type="button"
            disabled={
              disabled || !selected || query.isFetching || Boolean(query.error)
            }
            onClick={() => onSelect(selected)}
          >
            使用所选文件
          </button>
        </footer>
      </CommandDialog>
    </div>,
    document.body,
  );
}
