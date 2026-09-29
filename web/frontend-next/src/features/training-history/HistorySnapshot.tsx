import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, ChevronUp, Copy } from "lucide-react";
import type { HistoryTaskDetail } from "./api";
import { HistoryArtifacts } from "./HistoryArtifacts";
import "./HistorySnapshot.css";

export function HistorySnapshot({ detail, taskId }: { detail: HistoryTaskDetail; taskId: string }) {
  const content = detail.config_toml || "";
  const [search, setSearch] = useState("");
  const [current, setCurrent] = useState(0);
  const [copyStatus, setCopyStatus] = useState("");
  const preRef = useRef<HTMLPreElement>(null);
  const query = search.trim();
  const hits = useMemo(() => findHits(content, query), [content, query]);
  const active = hits.length ? current % hits.length : 0;

  useEffect(() => {
    if (!hits.length) return;
    const frame = requestAnimationFrame(() => preRef.current?.querySelector(".history-snapshot-hit-current")?.scrollIntoView?.({ block: "center" }));
    return () => cancelAnimationFrame(frame);
  }, [active, hits]);

  function moveMatch(direction: number) {
    if (!hits.length) return;
    setCurrent((index) => (index + direction + hits.length) % hits.length);
  }

  async function copyContent() {
    try {
      await navigator.clipboard.writeText(content);
      setCopyStatus("已复制");
    } catch {
      setCopyStatus("复制失败");
    }
  }

  return <section className="history-overview-section history-snapshot">
    <h2>配置快照</h2>
    {content ? <>
      <div className="history-snapshot-toolbar">
        <label className="history-snapshot-search">
          <span className="sr-only">搜索配置快照</span>
          <input type="search" value={search} aria-label="搜索配置快照" placeholder="搜索参数名或值"
            onChange={(event) => { setSearch(event.target.value); setCurrent(0); }}
            onKeyDown={(event) => {
              if (event.key !== "Enter" || !hits.length) return;
              event.preventDefault();
              moveMatch(event.shiftKey ? -1 : 1);
            }} />
        </label>
        <span className="history-snapshot-match" aria-live="polite">
          {query ? hits.length ? `${active + 1} / ${hits.length}` : "无匹配" : "未搜索"}
        </span>
        <div className="history-snapshot-nav">
          <button type="button" aria-label="上一个匹配项" title="上一个匹配项" disabled={!hits.length} onClick={() => moveMatch(-1)}><ChevronUp size={16} /></button>
          <button type="button" aria-label="下一个匹配项" title="下一个匹配项" disabled={!hits.length} onClick={() => moveMatch(1)}><ChevronDown size={16} /></button>
        </div>
        <button type="button" className="history-snapshot-copy" onClick={() => void copyContent()}><Copy size={15} />复制全部</button>
        <span className="history-snapshot-copy-status" role="status">{copyStatus}</span>
      </div>
      <pre className="history-detail-toml history-snapshot-code" aria-label="配置快照代码" ref={preRef}>
        <code>{renderLines(content, hits, active)}</code>
      </pre>
    </> : <p role="status">此任务未保存配置快照。</p>}
    <HistoryArtifacts taskId={taskId} />
  </section>;
}

type Hit = { start: number; end: number };

function findHits(text: string, query: string): Hit[] {
  if (!query) return [];
  const literal = query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return [...text.matchAll(new RegExp(literal, "giu"))].map((match) => ({
    start: match.index,
    end: match.index + match[0].length,
  }));
}

function renderLines(text: string, hits: Hit[], active: number) {
  let offset = 0;
  let hitCursor = 0;
  return text.split(/(?<=\n)/).map((line, lineIndex) => {
    const lineStart = offset;
    offset += line.length;
    const lineEnd = lineStart + line.length;
    const tokens = [...line.matchAll(/(^\s*\[\[?.+?\]\]?\s*$|^\s*[^=:#]+(?=\s*=)|(?<==\s*)[^#\n]+|#[^\n]*)/gm)];
    while (hitCursor < hits.length && hits[hitCursor].end <= lineStart) hitCursor += 1;
    const marks: Array<Hit & { hitIndex: number }> = [];
    for (let hitIndex = hitCursor; hitIndex < hits.length && hits[hitIndex].start < lineEnd; hitIndex += 1) {
      const start = Math.max(hits[hitIndex].start, lineStart);
      const end = Math.min(hits[hitIndex].end, lineEnd);
      if (end > start) marks.push({ start, end, hitIndex });
    }
    const boundaries = new Set([lineStart, lineEnd]);
    tokens.forEach((token) => { boundaries.add(lineStart + (token.index || 0)); boundaries.add(lineStart + (token.index || 0) + token[0].length); });
    marks.forEach((mark) => { boundaries.add(mark.start); boundaries.add(mark.end); });
    const points = [...boundaries].filter((point) => point >= lineStart && point <= lineEnd).sort((a, b) => a - b);
    const assignment = line.indexOf("=");
    return <span className="history-snapshot-line" key={lineIndex}>{points.slice(0, -1).map((start, pointIndex) => {
      const end = points[pointIndex + 1];
      const token = tokens.find((entry) => start >= lineStart + (entry.index || 0) && start < lineStart + (entry.index || 0) + entry[0].length);
      const hit = marks.find((entry) => start >= entry.start && start < entry.end);
      const value = text.slice(start, end);
      const tokenStart = lineStart + (token?.index || 0);
      const className = token ? token[0].trimStart().startsWith("#") ? "history-snapshot-comment"
        : token[0].trimStart().startsWith("[") ? "history-snapshot-section"
          : assignment >= 0 && tokenStart > lineStart + assignment ? "history-snapshot-value" : "history-snapshot-key" : undefined;
      const content = className ? <span className={className}>{value}</span> : value;
      return hit ? <mark className={hit.hitIndex === active ? "history-snapshot-hit-current" : ""} key={start}>{content}</mark> : <span key={start}>{content}</span>;
    })}</span>;
  });
}
