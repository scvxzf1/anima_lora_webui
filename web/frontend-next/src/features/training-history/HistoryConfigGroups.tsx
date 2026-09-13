import {
  useIsMutating,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { HistoryDragItem } from "./HistoryDrag";
import {
  historyConfigKey,
  historyConfigLabel,
  historyConfigOrder,
} from "./historyOrder";
import { ArrowDown, ArrowUp } from "lucide-react";
import {
  fetchHistoryCollections,
  historyKeys,
  saveHistoryCollections,
  type HistoryTaskSummary,
} from "./api";

export function HistoryConfigGroups({
  tasks,
  collection,
  selected,
  onSelect,
}: {
  tasks: HistoryTaskSummary[];
  collection: string;
  selected: string;
  onSelect: (key: string) => void;
}) {
  const qc = useQueryClient();
  const busy = useIsMutating({ mutationKey: ["training-history"] }) > 0;
  const query = useQuery({
    queryKey: historyKeys.collections,
    queryFn: ({ signal }) => fetchHistoryCollections(signal),
  });
  const save = useMutation({
    mutationKey: historyKeys.collections,
    mutationFn: saveHistoryCollections,
    retry: false,
    onSuccess: (data) => {
      qc.setQueryData(historyKeys.collections, data);
    },
  });
  if (collection === "all") return null;
  const entries = new Map(
    tasks
      .filter((task) => (task.group || "") === collection)
      .map((task) => [historyConfigKey(task), historyConfigLabel(task)]),
  );
  const order = historyConfigOrder(
    tasks,
    collection,
    query.data || { collection_order: [], config_group_order: {} },
  );
  return (
    <div className="history-config-groups">
      <h3>配置分组</h3>
      <button
        type="button"
        className="object-row"
        data-selected={!selected}
        onClick={() => onSelect("")}
      >
        全部配置
      </button>
      {order
        .filter((key) => entries.has(key))
        .map((key) => (
          <div key={key}>
            <HistoryDragItem
              id={`config-nav:${JSON.stringify([collection, key])}`}
              disabled={busy || !query.data}
              data={{
                kind: "config",
                collection,
                key,
                region: "nav",
                label: entries.get(key)!,
                taskIds: tasks
                  .filter(
                    (task) =>
                      (task.group || "") === collection &&
                      historyConfigKey(task) === key,
                  )
                  .flatMap((task) => (task.id ? [task.id] : [])),
              }}
            >
              <button
                type="button"
                className="object-row"
                data-selected={selected === key}
                onClick={() => onSelect(key)}
              >
                <span>{entries.get(key)}</span>
              </button>
            </HistoryDragItem>
            {selected === key && (
              <div className="toolbar">
                {[-1, 1].map((direction) => (
                  <button
                    key={direction}
                    type="button"
                    title={direction < 0 ? "上移配置分组" : "下移配置分组"}
                    aria-label={direction < 0 ? "上移配置分组" : "下移配置分组"}
                    disabled={
                      !query.data ||
                      busy ||
                      order.indexOf(key) + direction < 0 ||
                      order.indexOf(key) + direction >= order.length
                    }
                    onClick={() => {
                      const next = [...order];
                      const index = next.indexOf(key);
                      [next[index], next[index + direction]] = [
                        next[index + direction],
                        next[index],
                      ];
                      save.mutate({
                        ...query.data!,
                        config_group_order: {
                          ...query.data!.config_group_order,
                          [collection]: next,
                        },
                      });
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
      {save.error && (
        <p role="alert" className="form-error">
          {save.error.message}
        </p>
      )}
    </div>
  );
}
