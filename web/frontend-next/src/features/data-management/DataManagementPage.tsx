import { useMutation, useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowLeft,
  ChevronRight,
  File,
  Folder,
  HardDrive,
  LockKeyhole,
  RefreshCw,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Trash2,
} from "lucide-react";
import { useMemo, useRef, useState, type CSSProperties } from "react";
import { QueryFeedback } from "../../components/QueryFeedback";
import { CommandDialog } from "../../components/CommandDialog";
import { fetchDataEntries, fetchDataRoots, fetchDataRootsMeta, cleanupData, previewCleanup } from "./dataManagementApi";
import type {
  CleanupBlockedItem,
  DataEntry,
  DataRoot,
} from "./dataManagementTypes";
import type { RetentionScopeOption } from "./retentionTypes";
import { RetentionDrawer } from "./RetentionDrawer";
import { formatBytes, formatDate } from "./format";
import "../tools.css";
import "./DataManagementPage.css";

const isReadOnly = (access: string) => {
  const normalized = access.trim().toLowerCase().replace(/[-\s]/g, "_");
  return normalized === "read_only" || normalized === "readonly" || normalized === "ro"
    || normalized === "read" || normalized === "denied" || normalized === "forbidden"
    || normalized.includes("read_only") || normalized.includes("deny");
};

const accessLabel = (access: string) => isReadOnly(access) ? "仅查看" : "可清理";

const statusLabel = (entry: DataEntry) => {
  if (entry.cleanup_allowed) return "可清理";
  if (entry.status === "missing") return "已移除";
  if (entry.status === "unavailable") return "不可用";
  const reason = entry.cleanup_reason.toLocaleLowerCase();
  if (/busy|running|active|占用|任务/.test(reason)) return "任务占用";
  if (/unavailable|error|failed|不可用|失败/.test(reason)) return "不可用";
  return "只读";
};

const rootColors = ["#2878b5", "#d47732", "#39865b", "#a54862", "#8062a8", "#858d2f"];
const rootColor = (index: number) => rootColors[index % rootColors.length];
const normalizedRootPath = (path: string) => path.replace(/[\\/]+$/, "").replace(/\\/g, "/");
const rootIsNested = (root: DataRoot, roots: DataRoot[]) => roots.some((other) => {
  if (other.id === root.id) return false;
  const path = normalizedRootPath(root.path);
  const parent = normalizedRootPath(other.path);
  return path !== parent && path.startsWith(`${parent}/`);
});
const knownSize = (size: number | null | undefined) => typeof size === "number" && Number.isFinite(size) ? size : null;
const countLabel = (value: number | null | undefined) => typeof value === "number" && Number.isFinite(value) ? value.toLocaleString("zh-CN") : "—";

type EntryFilter = "all" | "cleanable" | "protected";
type EntrySort = "size-desc" | "time" | "name";

const relativeRunsPrefix = (outputPath: string, runsPath: string) => {
  const output = outputPath.replace(/[\\/]+$/, "").replace(/\\/g, "/");
  const runs = runsPath.replace(/[\\/]+$/, "").replace(/\\/g, "/");
  if (!output || !runs || !runs.startsWith(`${output}/`)) return "";
  return runs.slice(output.length + 1);
};

const pathParent = (path: string) => {
  const normalized = path.replace(/[\\/]$/, "");
  const index = Math.max(normalized.lastIndexOf("/"), normalized.lastIndexOf("\\"));
  return index < 0 ? "" : normalized.slice(0, index);
};

function blockedLabel(item: CleanupBlockedItem) {
  if (typeof item === "string") return item;
  return [item.name || item.relative_path || item.path || "清理项", item.reason].filter(Boolean).join("：");
}

function DataRootSection({
  root,
  search,
  selectedRootId,
  selectedItems,
  onToggle,
  filter,
  sort,
  color,
  scanning,
  scanError,
}: {
  root: DataRoot;
  search: string;
  selectedRootId: string | null;
  selectedItems: DataEntry[];
  onToggle: (root: DataRoot, entry: DataEntry, checked: boolean) => void;
  filter: EntryFilter;
  sort: EntrySort;
  color: string;
  scanning: boolean;
  scanError: string | null;
}) {
  const [currentPath, setCurrentPath] = useState("");
  const [expanded, setExpanded] = useState(!isReadOnly(root.access));
  const nested = useQuery({
    queryKey: ["data-management", "entries", root.id, currentPath],
    queryFn: ({ signal }) => fetchDataEntries(root.id, currentPath, signal),
    enabled: Boolean(currentPath),
    staleTime: Infinity,
    retry: false,
  });
  const entries = currentPath ? nested.data?.entries || [] : root.entries || [];
  const loadingEntries = currentPath ? nested.isPending : scanning;
  const query = search.trim().toLocaleLowerCase();
  const rootMatches = !query || `${root.label} ${root.path}`.toLocaleLowerCase().includes(query);
  const visibleEntries = useMemo(() => [...entries].filter((entry) => {
    if (filter === "cleanable" && !entry.cleanup_allowed) return false;
    if (filter === "protected" && entry.cleanup_allowed) return false;
    return rootMatches || `${entry.name} ${entry.relative_path}`.toLocaleLowerCase().includes(query);
  }).sort((left, right) => {
    if (sort === "name") return left.name.localeCompare(right.name, "zh-CN");
    if (sort === "time") return (right.modified_at || "").localeCompare(left.modified_at || "") || left.name.localeCompare(right.name, "zh-CN");
    return (knownSize(right.size_bytes) ?? -1) - (knownSize(left.size_bytes) ?? -1) || left.name.localeCompare(right.name, "zh-CN");
  }), [entries, filter, query, rootMatches, sort]);
  const readOnly = isReadOnly(root.access);
  const selectedPaths = new Set(selectedItems.map((entry) => entry.relative_path));
  const directorySize = entries.reduce((total, entry) => total + (knownSize(entry.size_bytes) ?? 0), 0);

  return <section className={`data-management-section${readOnly ? " is-readonly" : ""}`} style={{ "--data-root-color": color } as CSSProperties} aria-labelledby={`data-root-${root.id}`}>
    <button type="button" className="data-management-section-heading" aria-expanded={expanded} aria-controls={`data-root-content-${root.id}`} onClick={() => setExpanded((value) => !value)}>
      <div>
        <h2 id={`data-root-${root.id}`}>{root.label}</h2>
        <p className="data-management-root-path">{root.path}</p>
      </div>
      <div className="data-management-root-meta">
        <span className={readOnly ? "data-management-access-readonly" : "data-management-access-write"}>
          {readOnly ? <ShieldCheck size={14} aria-hidden="true" /> : <HardDrive size={14} aria-hidden="true" />} {accessLabel(root.access)}
        </span>
        {scanning ? <span>统计中…</span> : <>
          {knownSize(root.size_bytes) === null ? <span>大小未知</span> : <span>{formatBytes(root.size_bytes)}</span>}
          <span>{countLabel(root.file_count)} 个文件</span>
          <span>{countLabel(root.directory_count)} 个目录</span>
        </>}
      </div>
      <ChevronRight className="data-management-root-chevron" size={18} aria-hidden="true" />
    </button>
    {expanded && <div id={`data-root-content-${root.id}`} className="data-management-root-content">
    {currentPath && <div className="data-management-pathbar">
      <button type="button" onClick={() => setCurrentPath(pathParent(currentPath))} aria-label="返回上级目录">
        <ArrowLeft size={15} /> 返回上级
      </button>
      <ChevronRight size={14} aria-hidden="true" />
      <span className="data-management-entry-path">{currentPath}</span>
    </div>}
    {nested.error && <p role="alert" className="tool-error">目录读取失败：{nested.error.message}</p>}
    {scanError && <p role="alert" className="tool-error">目录统计失败：{scanError}</p>}
    {loadingEntries && <p role="status" className="data-management-empty">{currentPath ? "正在读取目录..." : "正在统计目录..."}</p>}
    {!loadingEntries && visibleEntries.length > 0 ? <ul className="data-management-entry-list" aria-label={`${root.label}条目`}>
      {visibleEntries.map((entry) => {
        const directory = entry.kind === "directory";
        const checked = selectedPaths.has(entry.relative_path);
        const disabled = !checked && selectedRootId !== null && selectedRootId !== root.id;
        const size = knownSize(entry.size_bytes);
        const percent = directorySize > 0 && size !== null ? Math.min(100, size / directorySize * 100) : 0;
        return <li className={`data-management-entry${entry.cleanup_allowed ? " is-cleanable" : " is-protected"}`} key={entry.relative_path}>
          <div className="data-management-entry-main">
            {entry.cleanup_allowed ? <label className="data-management-entry-check">
              <input type="checkbox" checked={checked} disabled={disabled} aria-label={`选择 ${entry.name}`} onChange={(event) => onToggle(root, entry, event.target.checked)} />
            </label> : <span className="data-management-entry-lock" title={entry.cleanup_reason || "只读保护"}><LockKeyhole size={15} aria-label="只读保护" /></span>}
            {directory ? <Folder size={17} aria-hidden="true" /> : <File size={17} aria-hidden="true" />}
            <div className="data-management-entry-label">
              {directory ? <button type="button" className="data-management-entry-open" onClick={() => setCurrentPath(entry.relative_path)}>
                <strong>{entry.name}</strong><ChevronRight size={14} aria-hidden="true" />
              </button> : <strong>{entry.name}</strong>}
              {query ? <span className="data-management-entry-path">{entry.relative_path}</span> : null}
            </div>
          </div>
          <div className="data-management-entry-meta">
            <span className="data-management-entry-size">{formatBytes(entry.size_bytes)}<span className="data-management-entry-size-track" aria-hidden="true"><span style={{ width: `${percent}%` }} /></span></span>
            <span className="data-management-entry-kind">{entry.kind === "directory" ? "目录" : entry.kind === "file" ? "文件" : "特殊文件"}</span>
            {!entry.cleanup_allowed ? <span className="data-management-entry-status is-protected" title={entry.cleanup_reason || "只读保护"}>{statusLabel(entry)}</span> : null}
            <span>{formatDate(entry.modified_at)}</span>
          </div>
        </li>;
      })}
    </ul> : !loadingEntries ? <p className="data-management-empty">{query ? "没有匹配的目录或条目" : "此目录暂无条目"}</p> : null}
    </div>}
  </section>;
}

export function DataManagementPage() {
  const queryClient = useQueryClient();
  const rootsMetaQuery = useQuery({
    queryKey: ["data-management", "roots"],
    queryFn: ({ signal }) => fetchDataRootsMeta(signal),
    staleTime: Infinity,
    retry: false,
  });
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<EntryFilter>("all");
  const [sort, setSort] = useState<EntrySort>("size-desc");
  const [selectedRootId, setSelectedRootId] = useState<string | null>(null);
  const [selectedItems, setSelectedItems] = useState<DataEntry[]>([]);
  const [plan, setPlan] = useState<Awaited<ReturnType<typeof previewCleanup>> | null>(null);
  const [cleanupSnapshot, setCleanupSnapshot] = useState<{ rootId: string; items: string[] } | null>(null);
  const previewGeneration = useRef(0);
  const [confirmed, setConfirmed] = useState(false);
  const [notice, setNotice] = useState("");
  const [resultError, setResultError] = useState("");
  const [retentionOpen, setRetentionOpen] = useState(false);
  const preview = useMutation({
    mutationFn: ({ rootId, items }: { rootId: string; items: string[]; generation: number }) => previewCleanup(rootId, items),
    retry: false,
    onSuccess: (response, variables) => {
      if (variables.generation !== previewGeneration.current) return;
      setPlan(response);
      setCleanupSnapshot({ rootId: variables.rootId, items: [...variables.items] });
      setConfirmed(false);
      setResultError("");
    },
    onError: (error, variables) => {
      if (variables.generation === previewGeneration.current)
        setResultError(`清理预览失败：${error.message}`);
    },
  });
  const cleanup = useMutation({
    mutationFn: ({ rootId, items, expectedItems, planRevision }: { rootId: string; items: string[]; expectedItems: string[]; planRevision: string }) => cleanupData(rootId, items, expectedItems, planRevision),
    retry: false,
    onSuccess: async (response) => {
      const details = [
        response.missing.length ? `缺少 ${response.missing.length} 项` : "",
        response.blocked.length ? `阻止 ${response.blocked.length} 项` : "",
        Object.keys(response.errors).length ? `失败 ${Object.keys(response.errors).length} 项` : "",
      ].filter(Boolean).join("，");
      setNotice(response.message || `已处理 ${response.deleted.length} 项${details ? `；${details}` : ""}`);
      setResultError(details ? `清理未完全完成：${details}` : "");
      previewGeneration.current += 1;
      setPlan(null);
      setCleanupSnapshot(null);
      setConfirmed(false);
      setSelectedItems([]);
      setSelectedRootId(null);
      await queryClient.invalidateQueries({ queryKey: ["data-management"] });
    },
    onError: (error) => setResultError(`清理请求失败：${error.message}`),
  });
  const refresh = useMutation({
    mutationFn: () => fetchDataRoots(undefined, true),
    retry: false,
    onSuccess: (response) => {
      queryClient.setQueryData(
        ["data-management", "roots"],
        (current: { ok: boolean; generated_at?: string; roots: DataRoot[] } | undefined) => (
          current ? { ...current, generated_at: response.generated_at } : current
        ),
      );
      for (const root of response.roots || []) {
        queryClient.setQueryData(["data-management", "root-scan", root.id], {
          ok: true,
          root_id: root.id,
          path: "",
          entries: root.entries || [],
          file_count: root.file_count,
          directory_count: root.directory_count,
          size_bytes: root.size_bytes,
          size_state: root.size_state,
          truncated: root.truncated,
        });
      }
      void queryClient.invalidateQueries({ queryKey: ["data-management", "entries"] });
      setPlan(null);
      setCleanupSnapshot(null);
      setConfirmed(false);
      setResultError("");
    },
    onError: (error) => {
      setNotice("");
      setResultError(`目录扫描失败：${error.message}`);
    },
  });
  const metaRoots = rootsMetaQuery.data?.roots || [];
  // Root cards paint from the metadata payload immediately; each root then
  // fills its own size through the entries endpoint, so a slow tree walk never
  // blocks first paint.
  const rootScans = useQueries({
    queries: metaRoots.map((root) => ({
      queryKey: ["data-management", "root-scan", root.id],
      queryFn: ({ signal }: { signal: AbortSignal }) => fetchDataEntries(root.id, "", signal),
      staleTime: Infinity,
      retry: false,
    })),
  });
  const roots: DataRoot[] = metaRoots.map((root, index) => {
    const scan = rootScans[index]?.data;
    if (!scan) return root;
    return {
      ...root,
      entries: scan.entries || [],
      file_count: scan.file_count ?? null,
      directory_count: scan.directory_count ?? null,
      size_bytes: scan.size_bytes ?? null,
      size_state: scan.size_state ?? "ready",
      truncated: scan.truncated,
    };
  });
  const selectedSize = selectedItems.every((entry) => typeof entry.size_bytes === "number")
    ? selectedItems.reduce((total, entry) => total + (entry.size_bytes as number), 0)
    : null;
  const countedRoots = roots.filter((root) => !rootIsNested(root, roots));
  const summary = {
    roots: roots.length,
    totalSize: countedRoots.reduce((total, root) => total + (knownSize(root.size_bytes) ?? 0), 0),
    cleanableSize: countedRoots.reduce((total, root) => total + (root.entries || []).reduce((entryTotal, entry) => entry.cleanup_allowed ? entryTotal + (knownSize(entry.size_bytes) ?? 0) : entryTotal, 0), 0),
    protectedSize: countedRoots.reduce((total, root) => isReadOnly(root.access) ? total + (knownSize(root.size_bytes) ?? 0) : total, 0),
    pending: countedRoots.filter((root) => root.size_state === "pending").length,
    unknown: countedRoots.filter((root) => root.size_state !== "pending" && knownSize(root.size_bytes) === null).length,
    countedRoots,
  };

  const outputRoot = roots.find((root) => root.id === "project-output");
  const runRoot = roots.find((root) => root.id === "training-runs");
  const runsPrefix = relativeRunsPrefix(outputRoot?.path ?? "", runRoot?.path ?? "");
  const selectedDirectories = selectedItems.filter((entry) => entry.kind === "directory");
  const retentionOptions: RetentionScopeOption[] = [];
  if (selectedRootId === "project-output" && selectedDirectories.length) {
    retentionOptions.push({
      id: "selected",
      label: `当前勾选的 ${selectedDirectories.length} 个目录`,
      rootId: "project-output",
      dirs: selectedDirectories.map((entry) => entry.relative_path),
    });
  } else if (selectedRootId === "training-runs" && selectedDirectories.length && runsPrefix) {
    retentionOptions.push({
      id: "selected",
      label: `当前勾选的 ${selectedDirectories.length} 个运行目录`,
      rootId: "project-output",
      dirs: selectedDirectories.map((entry) => `${runsPrefix}/${entry.relative_path}`),
    });
  }
  if (runsPrefix) {
    retentionOptions.push({
      id: "runs",
      label: `整个训练运行目录（${runsPrefix}，按每个实验独立计算）`,
      rootId: "project-output",
      dirs: [runsPrefix],
    });
  }

  function toggleEntry(root: DataRoot, entry: DataEntry, checked: boolean) {
    setNotice("");
    setResultError("");
    previewGeneration.current += 1;
    setPlan(null);
    setCleanupSnapshot(null);
    setConfirmed(false);
    if (checked) {
      if (selectedRootId && selectedRootId !== root.id) return;
      setSelectedRootId(root.id);
      setSelectedItems((current) => current.some((item) => item.relative_path === entry.relative_path) ? current : [...current, entry]);
    } else {
      setSelectedItems((current) => {
        const next = current.filter((item) => item.relative_path !== entry.relative_path);
        if (!next.length) setSelectedRootId(null);
        return next;
      });
    }
  }

  function openPreview() {
    if (!selectedRootId || !selectedItems.length || preview.isPending || retentionOpen) return;
    setNotice("");
    setResultError("");
    setRetentionOpen(false);
    const generation = ++previewGeneration.current;
    preview.mutate({ rootId: selectedRootId, items: selectedItems.map((entry) => entry.relative_path), generation });
  }

  function submitCleanup() {
    if (!plan || !cleanupSnapshot || !confirmed || plan.blocked.length || cleanup.isPending) return;
    cleanup.mutate({
      rootId: cleanupSnapshot.rootId,
      items: cleanupSnapshot.items,
      expectedItems: plan.items.map((item) => item.relative_path),
      planRevision: plan.plan_revision,
    });
  }

  return <main className="data-management-page">
    <header className="data-management-heading">
      <div><h1>数据管理</h1><p>查阅项目数据占用，仅清理明确的训练产物和缓存</p></div>
      <div className="data-management-heading-actions">
        <button type="button" className="data-management-retention-open" onClick={() => {
          previewGeneration.current += 1;
          setPlan(null);
          setCleanupSnapshot(null);
          setConfirmed(false);
          setNotice("");
          setResultError("");
          setRetentionOpen(true);
        }} disabled={!retentionOptions.length || preview.isPending || cleanup.isPending} title={retentionOptions.length ? "按检查点保留策略批量清理" : "先在下方勾选训练运行目录"}>
          <SlidersHorizontal size={16} />检查点与权重策略清理
        </button>
        <button type="button" onClick={() => {
          setNotice("");
          setResultError("");
          previewGeneration.current += 1;
          setPlan(null);
          setCleanupSnapshot(null);
          setConfirmed(false);
          refresh.mutate();
        }} disabled={refresh.isPending}>
          <RefreshCw size={16} />{refresh.isPending ? "扫描中" : "刷新"}
        </button>
      </div>
    </header>
    <QueryFeedback query={rootsMetaQuery} label="数据目录" hasData={Boolean(rootsMetaQuery.data)} />
    <div className="data-management-toolbar">
      <label className="data-management-search">
        <Search size={16} aria-hidden="true" />
        <span className="sr-only">搜索目录或条目</span>
        <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="搜索目录或条目名称" aria-label="搜索目录或条目" />
      </label>
      {rootsMetaQuery.data?.generated_at ? <span className="tool-muted">扫描于 {formatDate(rootsMetaQuery.data.generated_at)}</span> : null}
      <div className="data-management-view-controls" role="group" aria-label="条目筛选与排序">
        <div className="data-management-filter-group" role="group" aria-label="条目筛选">
          {(["all", "cleanable", "protected"] as const).map((value) => <button key={value} type="button" className={filter === value ? "is-active" : ""} onClick={() => setFilter(value)}>{value === "all" ? "全部" : value === "cleanable" ? "可清理" : "只读保护"}</button>)}
        </div>
        <label className="data-management-sort">排序
          <select value={sort} onChange={(event) => setSort(event.target.value as EntrySort)} aria-label="条目排序">
            <option value="size-desc">大小（从大到小）</option>
            <option value="time">修改时间</option>
            <option value="name">名称</option>
          </select>
        </label>
      </div>
    </div>
    {resultError && !plan ? <p className="data-management-result" role="alert">{resultError}</p> : null}
    {notice ? <p className="data-management-notice" role="status">{notice}</p> : null}
    <div className="data-management-summary" aria-label="数据管理摘要">
      <div className="data-management-storage-overview">
        <div className="data-management-storage-bar" role="img" aria-label={`已扫描总容量 ${formatBytes(summary.totalSize)}`}>
          {summary.countedRoots.map((root, index) => <span key={root.id} className="data-management-storage-segment" style={{ width: `${summary.totalSize > 0 ? (knownSize(root.size_bytes) ?? 0) / summary.totalSize * 100 : 0}%`, backgroundColor: rootColor(index) }} title={`${root.label}：${formatBytes(root.size_bytes)}`} />)}
        </div>
        <div className="data-management-storage-legend">
          {summary.countedRoots.map((root, index) => <span key={root.id}><i style={{ backgroundColor: rootColor(index) }} />{root.label} · {formatBytes(root.size_bytes)}</span>)}
        </div>
      </div>
      <div className="data-management-metric"><strong>{formatBytes(summary.totalSize)}</strong><span>已扫描总容量{summary.pending ? "（统计中…）" : summary.unknown ? "（" + summary.unknown + " 项未统计）" : ""}</span></div>
      <div className="data-management-metric"><strong>{formatBytes(summary.cleanableSize)}</strong><span>可清理容量</span></div>
      <div className="data-management-metric"><strong>{formatBytes(summary.protectedSize)}</strong><span>只读保护容量</span></div>
    </div>
    {roots.map((root, index) => <DataRootSection key={root.id} root={root} search={search} filter={filter} sort={sort} color={rootColor(index)} selectedRootId={selectedRootId} selectedItems={selectedRootId === root.id ? selectedItems : []} onToggle={toggleEntry} scanning={Boolean(rootScans[index]?.isPending)} scanError={rootScans[index]?.error?.message ?? null} />)}
    {rootsMetaQuery.data && !roots.length ? <p className="data-management-empty">没有可查看的数据目录。</p> : null}
    {selectedItems.length > 0 && selectedRootId ? <div className="data-management-selection-bar data-management-floating-bar" aria-label="清理选择">
      <p><strong>已选择 {selectedItems.length} 项</strong> · 预计释放 {formatBytes(selectedSize)}</p>
      <button type="button" onClick={() => {
        previewGeneration.current += 1;
        setSelectedItems([]);
        setSelectedRootId(null);
        setPlan(null);
        setCleanupSnapshot(null);
        setConfirmed(false);
      }}>取消选择</button>
      <button type="button" onClick={() => {
        previewGeneration.current += 1;
        setPlan(null);
        setCleanupSnapshot(null);
        setConfirmed(false);
        setNotice("");
        setResultError("");
        setRetentionOpen(true);
      }} disabled={!retentionOptions.length || preview.isPending || cleanup.isPending}>
        <SlidersHorizontal size={16} />检查点策略清理
      </button>
      <button type="button" className="data-management-danger" onClick={openPreview} disabled={preview.isPending}>
        <Trash2 size={16} />{preview.isPending ? "正在生成计划" : "预览并清理"}
      </button>
    </div> : null}
    {plan && <CommandDialog title="核对清理计划" onClose={() => {
      if (!cleanup.isPending) {
        setPlan(null);
        setCleanupSnapshot(null);
        setConfirmed(false);
      }
    }} busy={cleanup.isPending}>
      <div className="data-management-plan">
        <p className="data-management-plan-intro">以下条目将从所选数据目录永久删除。计划生成后目录状态若有变化，提交会被拒绝。</p>
        <div className="data-management-plan-summary"><strong>{plan.item_count} 项</strong><span>{formatBytes(plan.total_size_bytes)}</span><span>状态已锁定，提交前会再次核对</span></div>
        <ul className="data-management-plan-items" aria-label="待删除条目">
          {plan.items.map((item) => <li key={item.relative_path}><strong>{item.name}</strong><span className="data-management-plan-path">{item.path || item.relative_path}</span><small>{item.kind === "directory" ? "目录" : "文件"} · {formatBytes(item.size_bytes)} · {formatDate(item.modified_at)}</small></li>)}
        </ul>
        {plan.blocked.length > 0 && <><h3>阻止项（无法清理）</h3><ul className="data-management-plan-blocked" aria-label="清理阻止项">{plan.blocked.map((item, index) => <li key={`${blockedLabel(item)}-${index}`}>{blockedLabel(item)}</li>)}</ul></>}
        {resultError && <p className="data-management-plan-error" role="alert">{resultError}</p>}
        <label className="data-management-plan-confirm"><input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} disabled={cleanup.isPending || plan.blocked.length > 0} /><span>我已核对上述列表，并确认永久删除</span></label>
        <div className="data-management-plan-actions"><button type="button" onClick={() => {
          setPlan(null);
          setCleanupSnapshot(null);
          setConfirmed(false);
        }} disabled={cleanup.isPending}>取消</button><button type="button" className="data-management-danger" onClick={submitCleanup} disabled={!confirmed || Boolean(plan.blocked.length) || cleanup.isPending}><Trash2 size={16} />{cleanup.isPending ? "正在清理" : "确认永久删除"}</button></div>
      </div>
    </CommandDialog>}
    {retentionOpen ? <RetentionDrawer options={retentionOptions} onClose={() => setRetentionOpen(false)} onCompleted={(message) => {
      setNotice(message);
      setResultError("");
      setSelectedItems([]);
      setSelectedRootId(null);
      void queryClient.invalidateQueries({ queryKey: ["data-management"] });
    }} /> : null}
  </main>;
}
