import { ArrowDownToLine, ArrowUpToLine, ChevronDown, ChevronUp, Download, RefreshCw, Search, CornerDownLeft } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { fetchLogPage, LOG_PAGE_SIZE, logText, logUrl, searchLogs, type LogMatch } from "./logApi";
import { useLogPages } from "./useLogPages";
import "./HistoryLogs.css";

const ROW_HEIGHT = 24;
const WINDOW_ROWS = 10000;

export function HistoryLogs({ taskId, running = false }: { taskId: string; running?: boolean }) {
  const viewport = useRef<HTMLDivElement>(null);
  const pendingScroll = useRef<number | null>(null);
  const [height, setHeight] = useState(600);
  const [top, setTop] = useState(0);
  const [base, setBase] = useState(0);
  const [follow, setFollow] = useState(true);
  const [revision, setRevision] = useState(0);
  const [search, setSearch] = useState("");
  const [match, setMatch] = useState<LogMatch | null>(null);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState("");
  const searchRequest = useRef<AbortController | null>(null);
  const [jumpValue, setJumpValue] = useState("1");
  const [jumpMode, setJumpMode] = useState("line");
  const metadata = useQuery({ queryKey: ["history-log-size", taskId, revision],
    queryFn: ({ signal }) => fetchLogPage(taskId, undefined, 1, signal), refetchInterval: running ? 3000 : false });
  const total = metadata.data?.total ?? 0;
  const first = Math.max(base, base + Math.floor(top / ROW_HEIGHT) - 8);
  const last = Math.min(total - 1, base + Math.ceil((top + height) / ROW_HEIGHT) + 8);
  const pages = useLogPages(taskId, first, last, total, revision);
  const atEnd = base + WINDOW_ROWS >= total && top + height >= (total - base) * ROW_HEIGHT - 2;
  const pageNumber = atEnd ? Math.max(1, Math.ceil(total / LOG_PAGE_SIZE))
    : Math.floor((base + Math.floor(top / ROW_HEIGHT)) / LOG_PAGE_SIZE) + 1;

  function jump(index: number, keepFollow = false) {
    const target = Math.max(0, Math.min(total - 1, index));
    const nextBase = Math.max(0, Math.min(Math.max(0, total - WINDOW_ROWS), target - WINDOW_ROWS / 2));
    setBase(nextBase);
    setTop((target - nextBase) * ROW_HEIGHT);
    pendingScroll.current = (target - nextBase) * ROW_HEIGHT;
    if (!keepFollow) setFollow(false);
  }
  useLayoutEffect(() => {
    if (pendingScroll.current !== null && viewport.current) {
      viewport.current.scrollTop = pendingScroll.current;
      setTop(viewport.current.scrollTop);
      pendingScroll.current = null;
    }
  });
  useEffect(() => {
    if (!viewport.current) return;
    const observer = new ResizeObserver(([entry]) => setHeight(entry.contentRect.height));
    observer.observe(viewport.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => { if (follow && total) jump(total - 1, true); }, [total, follow, height]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => () => searchRequest.current?.abort(), []);

  async function find(direction = "forward") {
    if (!search.trim()) return;
    searchRequest.current?.abort();
    const controller = new AbortController();
    searchRequest.current = controller;
    setSearching(true);
    setSearchError("");
    try {
      const visibleTop = Math.max(0, Math.min(total - 1, base + Math.floor(top / ROW_HEIGHT)));
      const cursor = match?.match_index == null ? visibleTop
        : match.match_index + (direction === "forward" ? 1 : -1);
      const result = await searchLogs(taskId, search, cursor, direction, controller.signal);
      if (controller.signal.aborted) return;
      setMatch(result);
      if (result.match_index !== null) jump(result.match_index);
    } catch (error) {
      if (!controller.signal.aborted) setSearchError((error as Error).message);
    } finally { if (!controller.signal.aborted) setSearching(false); }
  }
  const error = metadata.error?.message || pages.error || searchError;
  return <section className="history-log-workspace" aria-label="完整训练日志">
    <div className="history-log-toolbar">
      <form className="history-log-search" onSubmit={(event) => { event.preventDefault(); void find(); }}>
        <Search size={16} />
        <input aria-label="搜索全部日志" placeholder="搜索全部日志" maxLength={512} value={search} onChange={(event) => {
          searchRequest.current?.abort(); setSearching(false); setSearch(event.target.value); setMatch(null); setSearchError("");
        }} />
        <button type="submit" title="搜索全部日志" aria-label="执行全局搜索" disabled={!search.trim() || searching}><CornerDownLeft size={16} /></button>
      </form>
      <span className="history-log-match" role="status">{searching ? "搜索中…" : match ? `${match.match_ordinal} / ${match.matches_total} 匹配` : ""}</span>
      <button title="上一匹配" aria-label="上一匹配" disabled={!search.trim() || searching} onClick={() => void find("backward")}><ChevronUp size={16} /></button>
      <button title="下一匹配" aria-label="下一匹配" disabled={!search.trim() || searching} onClick={() => void find()}><ChevronDown size={16} /></button>
      <label className="checkbox-row"><input type="checkbox" checked={follow} onChange={(event) => setFollow(event.target.checked)} />跟随末尾</label>
      <button title="刷新日志" aria-label="刷新日志" onClick={() => setRevision((value) => value + 1)}><RefreshCw size={16} /></button>
      <a className="history-log-download" href={`${logUrl(taskId)}/download`} title="下载完整日志" aria-label="下载完整日志"><Download size={17} /></a>
    </div>
    {error && <div role="alert">{error}<button onClick={() => setRevision((value) => value + 1)}>重试</button></div>}
    <div className="history-log-viewport" ref={viewport} role="log" aria-live="off" tabIndex={0}
      onWheel={() => setFollow(false)} onPointerDown={() => setFollow(false)} onTouchStart={() => setFollow(false)} onKeyDown={(event) => { if (["ArrowUp", "PageUp", "Home"].includes(event.key)) setFollow(false); }}
      onScroll={(event) => {
        const element = event.currentTarget;
        const value = element.scrollTop;
        const row = base + Math.floor(value / ROW_HEIGHT);
        if ((value < 1200 && base > 0) || (value > (WINDOW_ROWS - 100) * ROW_HEIGHT && base + WINDOW_ROWS < total)) jump(row);
        else setTop(value);
      }}>
      <div className="history-log-spacer" style={{ height: Math.min(total - base, WINDOW_ROWS) * ROW_HEIGHT }}>
        <div className="history-log-rows" style={{ transform: `translateY(${(first - base) * ROW_HEIGHT}px)` }}>
          {Array.from({ length: Math.max(0, last - first + 1) }, (_, i) => {
            const index = first + i;
            const record = pages.recordAt(index);
            return <div className="history-log-row" key={index} data-match={index === match?.match_index} data-type={record?.type}>
              <span className="history-log-number">{index + 1}</span>
              <span>{record ? logText(record) : pages.loadedAt(index) ? "[日志记录无法解析]" : "正在读取…"}</span>
            </div>;
          })}
        </div>
      </div>
      {!total && <p className="history-log-empty">{metadata.isPending ? "正在读取日志…" : error ? "日志读取失败" : "暂无日志"}</p>}
    </div>
    <footer className="history-log-footer">
      <span>共 {total.toLocaleString()} 行 · 第 {total ? pageNumber : 0} / {Math.ceil(total / LOG_PAGE_SIZE)} 页</span>
      <div className="history-log-navigation">
        <button title="日志开头" aria-label="日志开头" disabled={!total} onClick={() => jump(0)}><ArrowUpToLine size={16} /></button>
        <button title="上一页" aria-label="上一页" disabled={pageNumber <= 1} onClick={() => jump((pageNumber - 2) * LOG_PAGE_SIZE)}><ChevronUp size={16} /></button>
        <form onSubmit={(event) => { event.preventDefault(); jump((Number(jumpValue) - 1) * (jumpMode === "page" ? LOG_PAGE_SIZE : 1)); }}>
          <select aria-label="跳转单位" value={jumpMode} onChange={(event) => setJumpMode(event.target.value)}><option value="line">行</option><option value="page">页</option></select>
          <input aria-label="跳转位置" type="number" min={1} max={Math.max(1, jumpMode === "page" ? Math.ceil(total / LOG_PAGE_SIZE) : total)} required value={jumpValue} onChange={(event) => setJumpValue(event.target.value)} />
          <button title="跳转" aria-label="跳转" disabled={!total}><CornerDownLeft size={16} /></button>
        </form>
        <button title="下一页" aria-label="下一页" disabled={pageNumber * LOG_PAGE_SIZE >= total} onClick={() => jump(pageNumber * LOG_PAGE_SIZE)}><ChevronDown size={16} /></button>
        <button title="日志末尾" aria-label="日志末尾" disabled={!total} onClick={() => jump(total - 1)}><ArrowDownToLine size={16} /></button>
      </div>
    </footer>
  </section>;
}
