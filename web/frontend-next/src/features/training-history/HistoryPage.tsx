import {
  useIsMutating,
  useInfiniteQuery,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Layers, List } from "lucide-react";
import {
  batchUpdateHistoryTasks,
  fetchHistoryTasks,
  fetchHistoryCollections,
  historyKeys,
  type HistoryTaskSummary,
} from "./api";
import { HistoryCollections } from "./HistoryCollections";
import { HistoryConfigGroups } from "./HistoryConfigGroups";
import { HistoryComparison } from "./HistoryComparison";
import { HistoryTimeline } from "./HistoryTimeline";
import {
  orderedHistoryTasks,
  historyConfigKey,
  historyStacks,
} from "./historyOrder";
import { HistoryDrag } from "./HistoryDrag";
import { HistoryTaskCard } from "./HistoryTaskCard";
import { HistoryTaskStack } from "./HistoryTaskStack";
import { readStackPages, useHistoryAnchor } from "./historyNavigation";
import { renderedHistoryIds } from "./historySelectionScope";
import { historyReturnSearch, useHistoryRestore } from "./useHistoryRestore";
import "./HistoryPage.css";

function filterTasks(
  tasks: HistoryTaskSummary[],
  state: string,
  archived: string,
  advanced: Record<string, string>,
) {
  return tasks.filter((task) => {
    if (state !== "all" && task.state !== state) return false;
    if (archived === "active" && task.archived) return false;
    if (archived === "archived" && !task.archived) return false;
    if (advanced.base && task.model_family !== advanced.base) return false;
    if (advanced.variant && (task.training_variant || task.variant) !== advanced.variant) return false;
    if (advanced.source && (task.history_source_config_file || "") !== advanced.source) return false;
    if (advanced.precision && (task.precision_preference || "") !== advanced.precision) return false;
    if (advanced.preprocess_precision && (task.preprocess_precision || "") !== advanced.preprocess_precision) return false;
    if (advanced.swap && (task.block_swap_precision || "") !== advanced.swap) return false;
    if (advanced.compute && (task.base_compute || "") !== advanced.compute) return false;
    return true;
  });
}

export function HistoryPage() {
  const queryClient = useQueryClient();
  const [params, setParams] = useSearchParams();
  const search = params.get("q") || "";
  const state = params.get("state") || "all";
  const archived = params.get("archived") || "active";
  const collection = params.get("collection") ?? "all";
  const configGroup = params.get("config") || "";
  const advanced = {
    base: params.get("base") || "",
    variant: params.get("variant") || "",
    source: params.get("source") || "",
    precision: params.get("precision") || "",
    preprocess_precision: params.get("preprocess_precision") || "",
    swap: params.get("swap") || "",
    compute: params.get("compute") || "",
  };
  const stacked = params.get("layout") !== "list";
  const busy = useIsMutating({ mutationKey: ["training-history"] }) > 0;
  const [comparison, setComparison] = useState<string[]>([]);
  const [timeline, setTimeline] = useState<string[]>([]);
  const collections = useQuery({
    queryKey: historyKeys.collections,
    queryFn: ({ signal }) => fetchHistoryCollections(signal),
  });
  const page = Math.max(0, Number(params.get("page")) || 0);
  const limit = Math.max(200, Math.min(100000, Math.floor(Number(params.get("limit")) || 200)));
  const commandLock = useRef(false);
  function filter(key: string, value: string) {
    setSelected([]);
    setParams(
      (current) => {
        current.set(key, value);
        if (key === "collection") current.delete("config");
        current.delete("page");
        current.delete("depth");
        current.delete("anchor");
        return current;
      },
      { replace: true },
    );
  }
  const [selected, setSelected] = useState<string[]>([]);
  const [notice, setNotice] = useState("");

  const query = useInfiniteQuery({
    queryKey: [...historyKeys.list, limit, search],
    queryFn: ({ signal, pageParam }) => fetchHistoryTasks(limit, signal, search, pageParam),
    initialPageParam: "" as string | number,
    getNextPageParam: (last) => last.next_cursor ?? undefined,
    retry: false,
  });
  const tasks = query.data?.pages.flatMap((pageData) => pageData.tasks || []) || [];
  const restore = useHistoryRestore(query, params.toString());
  const listSearch = historyReturnSearch(params.toString(), query.data?.pages.length || 1);
  const missingAnchor = useHistoryAnchor(params.toString(), Boolean(query.data) && !query.isFetching && !query.error && !restore.restoring);
  const visible = useMemo(
    () =>
      orderedHistoryTasks(
        filterTasks(tasks, state, archived, advanced).filter(
          (task) =>
            (collection === "all" || (task.group || "") === collection) &&
            (!configGroup || historyConfigKey(task) === configGroup),
        ),
        collections.data,
      ),
    [tasks, search, state, archived, collection, configGroup, advanced.base, advanced.variant, advanced.source, advanced.precision, advanced.preprocess_precision, advanced.swap, advanced.compute, collections.data],
  );

  const options = (read: (task: HistoryTaskSummary) => string | undefined) =>
    [...new Set(tasks.map(read).filter(Boolean) as string[])].sort();
  const advancedFields = [
    ["base", "基座模型", (task: HistoryTaskSummary) => task.model_family],
    ["variant", "训练变体", (task: HistoryTaskSummary) => task.training_variant || task.variant],
    ["source", "来源", (task: HistoryTaskSummary) => task.history_source_config_file],
    ["precision", "精度倾向", (task: HistoryTaskSummary) => task.precision_preference],
    ["preprocess_precision", "预处理精度", (task: HistoryTaskSummary) => task.preprocess_precision],
    ["swap", "块交换精度", (task: HistoryTaskSummary) => task.block_swap_precision],
    ["compute", "底模计算路径", (task: HistoryTaskSummary) => task.base_compute],
  ] as const;

  const stacks = useMemo(() => historyStacks(visible), [visible]);
  const pageSize = stacked ? 10 : 100;
  const itemCount = stacked ? stacks.length : visible.length;
  const currentPage = Math.min(
    page,
    Math.max(0, Math.ceil(itemCount / pageSize) - 1),
  );
  const renderedIds = renderedHistoryIds(visible, stacks, stacked, currentPage, readStackPages(params.get("stacks")));
  const hiddenSelected = selected.filter((id) => !renderedIds.has(id)).length;
  const activeFilters = [search && `搜索：${search}`, state !== "all" && `状态：${state}`, archived !== "active" && `归档：${archived}`, collection !== "all" && `集合：${collection || "未分类"}`, configGroup && `配置组：${configGroup}`, ...advancedFields.map(([key, label]) => advanced[key] && `${label}：${advanced[key]}`)].filter(Boolean);

  function clearFilters() {
    setSelected([]);
    setParams((current) => {
      const next = new URLSearchParams();
      if (current.has("layout")) next.set("layout", current.get("layout")!);
      return next;
    }, { replace: true });
  }

  const batch = useMutation({
    mutationKey: historyKeys.list,
    mutationFn: batchUpdateHistoryTasks,
    onSuccess: async (result) => {
      setSelected([]);
      setNotice(result.message || "历史任务已更新。");
      await queryClient.invalidateQueries({ queryKey: historyKeys.list });
    },
    onSettled: () => {
      commandLock.current = false;
    },
    retry: false,
  });

  function toggleSelected(taskId: string) {
    setSelected((current) =>
      current.includes(taskId)
        ? current.filter((id) => id !== taskId)
        : [...current, taskId],
    );
  }

  function runBatch(action: "archive" | "unarchive" | "delete") {
    if (!selected.length || commandLock.current || busy) return;
    const message =
      action === "delete"
        ? `确定彻底删除已选 ${selected.length} 条历史记录吗？该操作不会删除运行目录和权重。`
        : `确定${action === "archive" ? "归档" : "取消归档"}已选 ${selected.length} 条历史记录吗？`;
    if (!window.confirm(message)) return;
    commandLock.current = true;
    batch.mutate({ action, task_ids: selected });
  }

  function refreshHistory() {
    const hadBatchError = Boolean(batch.error);
    void query.refetch().then((result) => {
      if (!result.isSuccess || !result.data) return;
      const loadedIds = new Set(
        result.data.pages.flatMap((pageData) => pageData.tasks || [])
          .map((task) => String(task.id || "")),
      );
      setSelected((current) => current.filter((id) => loadedIds.has(id)));
      if (hadBatchError) {
        batch.reset();
        setNotice("已刷新历史记录，当前列表已核对。");
      }
    });
  }

  const counts = useMemo(
    () => ({
      total: tasks.length,
      training: tasks.filter((task) => task.job === "training").length,
      preprocess: tasks.filter((task) => task.job === "preprocess").length,
      error: tasks.filter((task) => task.state === "error").length,
      archived: tasks.filter((task) => task.archived).length,
    }),
    [tasks],
  );

  return (
    <div className="history-shell">
      <main className="history-page">
        <header className="history-header">
          <div>
            <p className="eyebrow">HISTORY FORGE</p>
            <h1>历史任务</h1>
          </div>
          <button type="button" disabled={query.isFetching || busy} onClick={refreshHistory}>
            {query.isFetching ? "刷新中" : "刷新"}
          </button>
        </header>

        <section className="history-stats" aria-label="历史任务统计">
          <span>
            <strong>{counts.total}</strong>已加载
          </span>
          <span>
            <strong>{counts.training}</strong>训练
          </span>
          <span>
            <strong>{counts.preprocess}</strong>预处理
          </span>
          <span>
            <strong>{counts.error}</strong>异常
          </span>
          <span>
            <strong>{counts.archived}</strong>归档
          </span>
        </section>

        <div className="history-toolbar">
          <label className="history-search">
            <span>搜索历史记录</span>
            <input
              type="search"
              placeholder="任务 / 配置 / 目录"
              value={search}
              onChange={(event) => filter("q", event.target.value)}
            />
          </label>
          <label>
            <span>状态</span>
            <select
              value={state}
              onChange={(event) => filter("state", event.target.value)}
            >
              <option value="all">全部</option>
              <option value="idle">完成</option>
              <option value="running">运行中</option>
              <option value="error">异常</option>
              <option value="interrupted">已中断</option>
            </select>
          </label>
          <details className="history-advanced-filter">
            <summary>高级筛选</summary>
            <div className="history-advanced-grid">
              {advancedFields.map(([key, label, read]) => (
                <label key={key}>
                  <span>{label}</span>
                  <select
                    value={advanced[key]}
                    onChange={(event) => filter(key, event.target.value)}
                  >
                    <option value="">全部</option>
                    {options(read).map((value) => (
                      <option key={value} value={value}>{value}</option>
                    ))}
                  </select>
                </label>
              ))}
            </div>
          </details>
          <label>
            <span>归档</span>
            <select
              value={archived}
              onChange={(event) => filter("archived", event.target.value)}
            >
              <option value="active">未归档</option>
              <option value="all">全部</option>
              <option value="archived">已归档</option>
            </select>
          </label>
          <div
            className="history-layout-switch"
            role="group"
            aria-label="历史展示方式"
          >
            <button
              type="button"
              aria-label="按配置堆叠"
              title="按配置堆叠"
              aria-pressed={stacked}
              onClick={() => filter("layout", "stack")}
            >
              <Layers size={16} />
            </button>
            <button
              type="button"
              aria-label="平铺任务"
              title="平铺任务"
              aria-pressed={!stacked}
              onClick={() => filter("layout", "list")}
            >
              <List size={16} />
            </button>
          </div>
          <div className="history-status-legend" aria-label="状态颜色说明">
            <span><i data-status="idle" />完成</span>
            <span><i data-status="error" />异常</span>
            <span><i data-status="running" />运行中</span>
            <span><i data-status="interrupted" />中断</span>
          </div>
          {selected.length ? (
            <div className="history-bulk-bar">
              <strong>已选 {selected.length} 项</strong>
              <span>其中 {hiddenSelected} 项不在当前展开页</span>
              <button type="button" disabled={busy} onClick={() => setSelected([])}>清除选择</button>
              <button
                type="button"
                disabled={selected.length < 2 || selected.length > 4 || busy}
                onClick={() => setComparison([...selected])}
              >
                对比记录 (2-4)
              </button>
              <button type="button" disabled={selected.length < 2 || busy} onClick={() => setTimeline([...selected])}>
                合并查看
              </button>
              <button
                type="button"
                disabled={busy}
                onClick={() => runBatch("archive")}
              >
                归档已选
              </button>
              <button
                type="button"
                disabled={busy}
                onClick={() => runBatch("unarchive")}
              >
                取消归档
              </button>
              <button
                type="button"
                disabled={busy}
                onClick={() => {
                  const group = window.prompt(
                    "移动到集合（留空为未分类）；同配置关联的全部历史记录会一起调整。",
                  );
                  if (group === null || commandLock.current || busy) return;
                  commandLock.current = true;
                  batch.mutate({
                    action: "set_group",
                    task_ids: selected,
                    group,
                  });
                }}
              >
                移动到集合
              </button>
              <button
                type="button"
                disabled={busy}
                className="history-danger"
                onClick={() => runBatch("delete")}
              >
                彻底删除
              </button>
            </div>
          ) : null}
        </div>

        <div className="history-filter-scope">
          <p className="data-scope">服务端搜索{search.trim() ? `“${search.trim()}”` : "全部历史"} · 已读取 {tasks.length}{query.data?.pages[0]?.total !== undefined ? ` / ${query.data.pages[0].total}` : ""} 条 · 状态、集合及高级筛选仅作用于已读取记录。</p>
          {activeFilters.length > 0 && <div className="toolbar"><span>{activeFilters.join(" · ")}</span><button type="button" onClick={clearFilters}>清除筛选</button></div>}
        </div>
        {missingAnchor && <p className="history-notice" role="status">原任务已不在当前列表位置，可能已删除、移动或被筛选隐藏。<button type="button" onClick={() => setParams((current) => { const next = new URLSearchParams(current); next.delete("anchor"); return next; }, { replace: true })}>保留当前列表</button></p>}

        {batch.error ? (
          <p className="history-error" role="alert">
            {batch.error.message}
          </p>
        ) : null}
        {query.error && (
          <p className="history-error" role="alert">
            {query.error.message}
            {!restore.restoring && <button type="button" disabled={query.isFetching} onClick={() => query.refetch()}>重试读取历史</button>}
          </p>
        )}
        {notice ? (
          <p className="history-notice" role="status">
            {notice}
          </p>
        ) : null}

        <HistoryDrag
          tasks={tasks}
          settings={collections.error ? undefined : collections.data}
          selected={selected}
          onMoved={() => setSelected([])}
        >
          <div className="model-workspace">
            <div>
              <HistoryCollections
                tasks={tasks}
                complete={
                  Boolean(query.data) &&
                  !query.isFetching &&
                  !query.error &&
                  !search.trim() &&
                  !query.hasNextPage
                }
                current={collection}
                onSelect={(value) => filter("collection", value)}
              />
              <HistoryConfigGroups
                tasks={tasks}
                collection={collection}
                selected={configGroup}
                onSelect={(value) =>
                  setParams((current) => {
                    if (value) current.set("config", value);
                    else current.delete("config");
                    current.delete("page");
                    current.delete("depth");
                    current.delete("anchor");
                    return current;
                  }, { replace: true })
                }
              />
            </div>
            <section className="history-list" aria-label="历史任务列表">
              {restore.restoring ? <div className="history-restore" role="status">
                <p>正在恢复列表位置：已读取 {restore.loaded} / {restore.target} 批</p>
                {restore.paused && !query.error ? <button type="button" onClick={restore.continueRestore}>继续恢复位置</button> : null}
                {query.error ? <button type="button" disabled={query.isFetching} onClick={() => query.fetchNextPage()}>重试恢复位置</button> : null}
              </div> : visible.length ? (
                stacked ? (
                  stacks
                    .slice(currentPage * pageSize, (currentPage + 1) * pageSize)
                    .map((stack) => (
                      <HistoryTaskStack
                        key={stack.id}
                        stack={stack}
                        selected={selected}
                        busy={busy}
                        listSearch={listSearch}
                        onToggle={toggleSelected}
                        onToggleGroup={(ids) =>
                          setSelected((current) =>
                            ids.every((id) => current.includes(id))
                              ? current.filter((id) => !ids.includes(id))
                              : [...new Set([...current, ...ids])],
                          )
                        }
                      />
                    ))
                ) : (
                  visible
                    .slice(currentPage * pageSize, (currentPage + 1) * pageSize)
                    .map((task) => (
                      <HistoryTaskCard
                        key={task.id}
                        task={task}
                        selected={selected}
                        busy={busy}
                        listSearch={listSearch}
                        onToggle={toggleSelected}
                      />
                    ))
                )
              ) : (
                <p className="history-empty">
                  {query.isFetching
                    ? "正在读取历史任务…"
                    : query.error ? "历史记录读取失败。"
                      : query.hasNextPage ? "已加载结果中没有符合筛选条件的记录，仍有结果未加载。"
                        : "没有匹配的历史记录。"}
                </p>
              )}
              <div className="toolbar">
                {/* Commit pagination before another click can reuse a transition's old search params. */}
                <span>
                  {visible.length} 条
                  {stacked ? ` · ${stacks.length} 个配置组` : ""} · 第{" "}
                  {currentPage + 1} 页
                </span>
                <button
                  type="button"
                  disabled={restore.restoring || currentPage === 0}
                  onClick={() =>
                    setParams((current) => {
                      const next = new URLSearchParams(current);
                      next.set("page", String(Math.max(0, (Number(current.get("page")) || 0) - 1)));
                      next.delete("anchor");
                      return next;
                    }, { flushSync: true })
                  }
                >
                  上一页
                </button>
                <button
                  type="button"
                  disabled={restore.restoring || (currentPage + 1) * pageSize >= itemCount}
                  onClick={() =>
                    setParams((current) => {
                      const next = new URLSearchParams(current);
                      next.set("page", String(Math.min(Math.ceil(itemCount / pageSize) - 1, (Number(current.get("page")) || 0) + 1)));
                      next.delete("anchor");
                      return next;
                    }, { flushSync: true })
                  }
                >
                  下一页
                </button>
                <button
                  type="button"
                  disabled={restore.restoring || query.isFetching || !query.hasNextPage}
                  onClick={() => query.fetchNextPage()}
                >
                  载入更多记录
                </button>
              </div>
            </section>
          </div>
        </HistoryDrag>
      </main>
      {comparison.length > 0 && (
        <HistoryComparison ids={comparison} onClose={() => setComparison([])} />
      )}
      {timeline.length > 0 && <HistoryTimeline taskIds={timeline} onClose={() => setTimeline([])} />}
    </div>
  );
}
