import { useIsMutating, useQuery } from "@tanstack/react-query";
import { HistoryDragItem } from "./HistoryDrag";
import {
  historyConfigKey,
  historyConfigLabel,
  historyConfigOrder,
} from "./historyOrder";
import { fetchHistoryCollections, historyKeys, type HistoryTaskSummary } from "./api";

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
  const busy = useIsMutating({ mutationKey: ["training-history"] }) > 0;
  const query = useQuery({
    queryKey: historyKeys.collections,
    queryFn: ({ signal }) => fetchHistoryCollections(signal),
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
              rowActivator
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
                <span title={entries.get(key)}>{entries.get(key)}</span>
              </button>
            </HistoryDragItem>
          </div>
        ))}
    </div>
  );
}
