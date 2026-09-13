import type { HistoryTaskSummary } from "./api";
import type { historyStacks } from "./historyOrder";

export function renderedHistoryIds(tasks: HistoryTaskSummary[], stacks: ReturnType<typeof historyStacks>, stacked: boolean, page: number, stackPages: Record<string, number>) {
  const rendered = stacked
    ? stacks.slice(page * 10, (page + 1) * 10).flatMap((stack) => {
      if (!Object.hasOwn(stackPages, stack.id)) return [];
      const offset = Math.min(stackPages[stack.id], Math.max(0, Math.ceil(stack.tasks.length / 10) - 1)) * 10;
      return stack.tasks.slice(offset, offset + 10);
    })
    : tasks.slice(page * 100, (page + 1) * 100);
  return new Set(rendered.flatMap((task) => task.id ? [task.id] : []));
}
