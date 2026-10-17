import { useState } from "react";
import { createPortal } from "react-dom";
import { useQuery } from "@tanstack/react-query";
import { ArrowUp, Folder } from "lucide-react";
import { CommandDialog } from "../../components/CommandDialog";
import { fetchSettingsDirectories } from "./api";

export function SettingsDirectoryDialog({
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
  const query = useQuery({
    queryKey: ["settings-directories", directory, offset],
    queryFn: ({ signal }) => fetchSettingsDirectories(directory, offset, signal),
    retry: false,
    staleTime: 0,
  });
  const page = query.data;
  const navigate = (path: string) => {
    setDirectory(path);
    setOffset(0);
  };

  return createPortal(
    <div className="settings-directory-browser" onClick={(event) => event.stopPropagation()}>
      <CommandDialog title={`选择${label}`} onClose={onClose}>
        <p className="settings-directory-browser-note">
          仅显示服务器允许访问的目录；选择后只更新当前草稿。
        </p>
        {page && (
          <>
            <label className="settings-directory-root">
              目录位置
              <select
                aria-label="目录位置"
                value={page.roots.find((root) => root.path === page.directory)?.path ?? ""}
                onChange={(event) => event.target.value && navigate(event.target.value)}
              >
                <option value="" disabled>选择目录根</option>
                {page.roots.map((root) => (
                  <option key={root.path} value={root.path}>{root.name} · {root.path}</option>
                ))}
              </select>
            </label>
            <div className="settings-directory-current">
              <button
                type="button"
                aria-label="返回上级目录"
                disabled={!page.parent || query.isFetching}
                onClick={() => page.parent && navigate(page.parent)}
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
            <p>目录读取失败，请检查路径后重试。</p>
            <button type="button" onClick={() => void query.refetch()}>重试读取目录</button>
          </div>
        )}
        {page && !query.error && (
          <>
            <div className="settings-directory-entries" aria-label="服务器目录">
              {!page.entries.length && <p>此目录没有可进入的子目录。</p>}
              {page.entries.map((entry) => (
                <button
                  type="button"
                  key={entry.path}
                  className="settings-directory-entry"
                  disabled={query.isFetching}
                  onClick={() => navigate(entry.path)}
                >
                  <Folder size={17} aria-hidden="true" />
                  <span>{entry.name}</span>
                  <small>打开目录</small>
                </button>
              ))}
            </div>
            <div className="settings-directory-pagination">
              <span>共 {page.total} 项{page.truncated ? "（部分结果）" : ""}</span>
              <button
                type="button"
                disabled={offset === 0 || query.isFetching}
                onClick={() => setOffset(Math.max(0, offset - 100))}
              >上一页</button>
              <button
                type="button"
                disabled={page.next_offset === null || query.isFetching}
                onClick={() => page.next_offset !== null && setOffset(page.next_offset)}
              >下一页</button>
            </div>
          </>
        )}
        <footer className="settings-directory-footer">
          <button type="button" onClick={onClose}>取消</button>
          <button
            type="button"
            disabled={disabled || !page?.directory || query.isFetching || Boolean(query.error)}
            onClick={() => page?.directory && onSelect(page.directory)}
          >选择当前目录</button>
        </footer>
      </CommandDialog>
    </div>,
    document.body,
  );
}
