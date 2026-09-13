import type { HistoryCollections, HistoryTaskSummary } from "./api";

export function historyConfigKey(task: HistoryTaskSummary) {
  const legacy = [task.methods_subdir, task.variant, task.preset];
  return (
    task.history_group_key ||
    task.history_source_config_file ||
    (legacy.some(Boolean)
      ? legacy.join(":")
      : `task:${task.id || task.run_dir || task.output_dir || JSON.stringify(task)}`)
  );
}

export function historyConfigLabel(task: HistoryTaskSummary) {
  return (
    task.history_group_label ||
    task.history_source_config_file ||
    task.variant ||
    task.name ||
    task.id ||
    "未命名配置"
  );
}

export function historyStacks(tasks: HistoryTaskSummary[]) {
  const groups = new Map<
    string,
    {
      id: string;
      key: string;
      collection: string;
      label: string;
      tasks: HistoryTaskSummary[];
    }
  >();
  for (const task of tasks) {
    const collection = task.group || "";
    const key = historyConfigKey(task);
    const id = JSON.stringify([collection, key]);
    if (!groups.has(id))
      groups.set(id, {
        id,
        key,
        collection,
        label: historyConfigLabel(task),
        tasks: [],
      });
    groups.get(id)!.tasks.push(task);
  }
  return [...groups.values()];
}

export function historyCollectionNames(
  tasks: HistoryTaskSummary[],
  settings: HistoryCollections,
) {
  return [
    ...new Set([
      ...settings.collection_order,
      ...tasks.map((task) => task.group || "").filter(Boolean),
    ]),
  ];
}

export function historyConfigOrder(
  tasks: HistoryTaskSummary[],
  collection: string,
  settings: HistoryCollections,
) {
  return [
    ...new Set([
      ...(settings.config_group_order[collection] || []),
      ...tasks
        .filter((task) => (task.group || "") === collection)
        .map(historyConfigKey),
    ]),
  ];
}

export function moveHistoryOrder(order: string[], from: string, to: string) {
  const a = order.indexOf(from);
  const b = order.indexOf(to);
  if (a < 0 || b < 0 || a === b) return order;
  const next = [...order];
  next.splice(a, 1);
  next.splice(b, 0, from);
  return next;
}

export function historyDragSelection(source: string[], selected: string[]) {
  return [
    ...new Set(source.some((id) => selected.includes(id)) ? selected : source),
  ];
}

export function orderedHistoryTasks(
  tasks: HistoryTaskSummary[],
  settings?: HistoryCollections,
) {
  if (!settings) return tasks;
  const rank = (order: string[], value: string) => {
    const index = order.indexOf(value);
    return index === -1 ? order.length : index;
  };
  return [...tasks].sort((a, b) => {
    const groupA = a.group || "";
    const groupB = b.group || "";
    if (groupA !== groupB)
      return (
        rank(settings.collection_order, groupA) -
        rank(settings.collection_order, groupB)
      );
    const order = settings.config_group_order[groupA] || [];
    return rank(order, historyConfigKey(a)) - rank(order, historyConfigKey(b));
  });
}
