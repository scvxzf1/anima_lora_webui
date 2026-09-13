import {
  useIsMutating,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { useRef, useState } from "react";
import { CommandDialog } from "../../components/CommandDialog";
import { HistoryDragItem } from "./HistoryDrag";
import { ArrowDown, ArrowUp, FolderPlus, Pencil, Trash2 } from "lucide-react";
import {
  batchUpdateHistoryTasks,
  fetchHistoryCollections,
  historyKeys,
  saveHistoryCollections,
  type HistoryTaskSummary,
} from "./api";

export function HistoryCollections({
  tasks,
  current,
  onSelect,
  complete,
}: {
  tasks: HistoryTaskSummary[];
  current: string;
  onSelect: (value: string) => void;
  complete: boolean;
}) {
  const [dialog, setDialog] = useState<"new" | { rename: string } | null>(null);
  const [draftName, setDraftName] = useState("");
  const nameInputRef = useRef<HTMLInputElement>(null);
  const qc = useQueryClient();
  const query = useQuery({
    queryKey: historyKeys.collections,
    queryFn: ({ signal }) => fetchHistoryCollections(signal),
  });
  const settings = query.data || {
    collection_order: [],
    config_group_order: {},
  };
  const names = [
    ...new Set([
      ...settings.collection_order,
      ...tasks.map((task) => task.group || "").filter(Boolean),
    ]),
  ];
  const mutation = useMutation({
    mutationKey: historyKeys.collections,
    mutationFn: (operation: () => Promise<unknown>) => operation(),
    retry: false,
    onSettled: async () => {
      await qc.invalidateQueries({ queryKey: historyKeys.collections });
      await qc.invalidateQueries({ queryKey: historyKeys.list });
    },
  });
  const pending = useIsMutating({ mutationKey: ["training-history"] }) > 0;
  const busy = query.isPending || Boolean(query.error) || pending;
  function change(name: string, next: string | null) {
    if (!complete || busy) return;
    if (next && (next.length > 48 || names.includes(next))) {
      window.alert("集合名称不能重复，且不能超过 48 个字符。");
      return;
    }
    const ids = tasks
      .filter((task) => task.group === name && task.id)
      .map((task) => task.id!);
    if (
      ids.length &&
      !window.confirm(
        `变更集合 ${name} 及其中 ${ids.length} 条记录？后端会同时调整这些配置关联的全部历史记录；文件和产物保留。`,
      )
    )
      return;
    mutation.mutate(async () => {
      if (ids.length)
        await batchUpdateHistoryTasks({
          action: "set_group",
          group: next || "",
          task_ids: ids,
        });
      const order = settings.collection_order.filter((entry) => entry !== name);
      const configOrder = { ...settings.config_group_order };
      const previous = configOrder[name];
      delete configOrder[name];
      if (next) {
        order.push(next);
        if (previous) configOrder[next] = previous;
      }
      await saveHistoryCollections({
        collection_order: [...new Set(order)],
        config_group_order: configOrder,
      });
      onSelect(next || "all");
    });
  }
  function submitName() {
    const next = draftName.trim();
    if (!next || next.length > 48 || names.includes(next)) return;
    if (dialog === "new")
      mutation.mutate(() =>
        saveHistoryCollections({
          ...settings,
          collection_order: [...new Set([...names, next])],
        }),
      );
    else if (dialog?.rename) change(dialog.rename, next);
    setDialog(null);
    setDraftName("");
  }
  return (
    <aside className="object-library">
      <header>
        <h2>任务集合</h2>
        <button
          type="button"
          disabled={busy}
          title="新建集合"
          aria-label="新建集合"
          onClick={() => {
            setDraftName("");
            setDialog("new");
          }}
        >
          <FolderPlus size={16} />
        </button>
      </header>
      <button
        className="object-row"
        type="button"
        data-selected={current === "all"}
        onClick={() => onSelect("all")}
      >
        全部集合
      </button>
      <HistoryDragItem
        id="collection:"
        data={{ kind: "collection", collection: "", label: "未分类" }}
        handle={false}
        disabled={busy}
      >
        <button
          className="object-row"
          type="button"
          data-selected={current === ""}
          onClick={() => onSelect("")}
        >
          未分类
        </button>
      </HistoryDragItem>
      {names.map((name) => (
        <div key={name}>
          <HistoryDragItem
            id={`collection:${name}`}
            data={{ kind: "collection", collection: name, label: name }}
            disabled={busy}
          >
            <button
              type="button"
              className="object-row"
              data-selected={current === name}
              onClick={() => onSelect(name)}
            >
              <span>{name}</span>
              <small>
                {tasks.filter((task) => task.group === name).length} 条已加载
              </small>
            </button>
          </HistoryDragItem>
          {current === name && (
            <div className="toolbar">
              <button
                type="button"
                disabled={busy || !complete}
                title="重命名集合"
                aria-label="重命名集合"
                onClick={() => {
                  setDraftName(name);
                  setDialog({ rename: name });
                }}
              >
                <Pencil size={16} />
              </button>
              <button
                type="button"
                disabled={busy || !complete}
                title="移除集合"
                aria-label="移除集合"
                onClick={() => {
                  if (window.confirm(`移除集合 ${name}，保留历史任务和文件？`))
                    change(name, null);
                }}
              >
                <Trash2 size={16} />
              </button>
              {[-1, 1].map((direction) => (
                <button
                  key={direction}
                  type="button"
                  disabled={
                    busy ||
                    names.indexOf(name) + direction < 0 ||
                    names.indexOf(name) + direction >= names.length
                  }
                  title={direction < 0 ? "上移集合" : "下移集合"}
                  aria-label={direction < 0 ? "上移集合" : "下移集合"}
                  onClick={() => {
                    const order = [...names];
                    const index = order.indexOf(name);
                    [order[index], order[index + direction]] = [
                      order[index + direction],
                      order[index],
                    ];
                    mutation.mutate(() =>
                      saveHistoryCollections({
                        ...settings,
                        collection_order: order,
                      }),
                    );
                  }}
                >
                  {direction < 0 ? (
                    <ArrowUp size={16} />
                  ) : (
                    <ArrowDown size={16} />
                  )}
                </button>
              ))}
            </div>
          )}
        </div>
      ))}
      {!complete && (
        <p className="muted">
          仍有未加载记录。加载完整列表后可重命名或移除集合。
        </p>
      )}
      {(query.error || mutation.error) && (
        <p role="alert" className="form-error">
          {(query.error || mutation.error)?.message}
        </p>
      )}
      {dialog && (
        <CommandDialog
          title={dialog === "new" ? "新建集合" : "重命名集合"}
          onClose={() => !mutation.isPending && setDialog(null)}
          busy={mutation.isPending}
          initialFocusRef={nameInputRef}
        >
          <form
            className="command-form"
            onSubmit={(event) => {
              event.preventDefault();
              submitName();
            }}
          >
            <label>
              集合名称
              <input
                ref={nameInputRef}
                value={draftName}
                maxLength={48}
                onChange={(event) => setDraftName(event.target.value)}
              />
            </label>
            {draftName.trim() && names.includes(draftName.trim()) && (
              <p className="form-error" role="alert">
                集合名称不能重复。
              </p>
            )}
            <div className="toolbar">
              <button
                type="button"
                onClick={() => setDialog(null)}
                disabled={mutation.isPending}
              >
                取消
              </button>
              <button
                type="submit"
                className="primary"
                disabled={
                  !draftName.trim() ||
                  names.includes(draftName.trim()) ||
                  mutation.isPending
                }
              >
                保存
              </button>
            </div>
          </form>
        </CommandDialog>
      )}
    </aside>
  );
}
