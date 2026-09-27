import { Copy, Download, Eraser } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { copyText } from "../features/dataset-editor/copyText";
import { downloadTextFile } from "../features/dataset-editor/downloadTextFile";

export function LogViewer({
  lines,
  filename = "training.log",
  total,
}: {
  lines: { id?: number; line?: string; type?: string }[];
  filename?: string;
  total?: number;
}) {
  const [search, setSearch] = useState("");
  const [autoScroll, setAutoScroll] = useState(true);
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [notice, setNotice] = useState("");
  const ref = useRef<HTMLPreElement>(null);
  const visible = lines
    .slice(-500)
    .filter(
      (line) =>
        !hidden.has(String(line.id ?? line.line)) &&
        (line.line || "").toLowerCase().includes(search.toLowerCase()),
    );
  const text = visible.map((line) => line.line || "").join("\n");
  useEffect(() => {
    if (autoScroll && ref.current)
      ref.current.scrollTop = ref.current.scrollHeight;
  }, [text, autoScroll]);
  return (
    <div className="log-viewer">
      <div className="toolbar">
        <input
          aria-label="搜索日志"
          placeholder="搜索日志"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={autoScroll}
            onChange={(e) => setAutoScroll(e.target.checked)}
          />
          自动滚屏
        </label>
        <button
          type="button"
          title="复制日志"
          aria-label="复制日志"
          onClick={() => {
            copyText(text)
              .then(() => setNotice("已复制"))
              .catch((error: Error) => setNotice(error.message));
          }}
        >
          <Copy size={16} />
        </button>
        <button
          type="button"
          title="下载当前日志"
          aria-label="下载当前日志"
          onClick={() => downloadTextFile(filename, text)}
        >
          <Download size={16} />
        </button>
        <button
          type="button"
          title="清空当前视图"
          aria-label="清空当前视图"
          onClick={() =>
            setHidden(
              new Set(
                lines.slice(-500).map((line) => String(line.id ?? line.line)),
              ),
            )
          }
        >
          <Eraser size={16} />
        </button>
      </div>
      <p className="data-scope">当前视图 {visible.length} 行 · 最近 {Math.min(500, lines.length)} / 已读取 {lines.length} 行{total !== undefined && total > lines.length ? ` / 共 ${total} 行（前段未读取）` : ""}</p>
      <pre
        ref={ref}
        role="log"
        aria-label="日志内容"
        aria-live="off"
        tabIndex={0}
      >
        {text || (lines.length ? "当前视图无匹配日志" : "暂无日志")}
      </pre>
      {notice && <p role="status">{notice}</p>}
    </div>
  );
}
