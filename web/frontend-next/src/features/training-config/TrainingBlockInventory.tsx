import type { TrainingDraft } from "./trainingForm";
import { useTrainingBlockInventory, type TrainingBlockLookup } from "./useTrainingBlockInventory";

export function TrainingBlockInventory({ draft }: { draft: TrainingDraft }) {
  const family = String(draft.model_family || "anima");
  const path = String(draft.pretrained_model_name_or_path || "").trim();
  if (!path) return <TrainingBlockInventoryHelp />;
  return <ModelBlockLookup family={family} path={path} />;
}

function ModelBlockLookup({ family, path }: { family: string; path: string }) {
  const lookup = useTrainingBlockInventory(family, path);
  return <TrainingBlockInventoryHelp lookup={lookup} />;
}

export function TrainingBlockInventoryHelp({ lookup }: { lookup?: TrainingBlockLookup }) {
  if (!lookup) return <small className="training-field-help">填写基础模型路径后读取实际 block 数。</small>;
  const { ready, query, inventory } = lookup;
  return <small className="training-field-help" aria-live="polite" data-testid="training-block-inventory">
    {!ready || query.isFetching ? "正在读取主模型 block 数…"
      : query.isError ? `无法读取主模型 block 数：${query.error.message}`
      : inventory ? `当前主模型：${inventory.block_count} 个 block · 可交换上限 ${inventory.max_swap}（保留 2 块驻留）`
      : "正在读取主模型 block 数…"}
    {ready && <button type="button" disabled={query.isFetching} aria-label="刷新主模型 block 数"
      onClick={(event) => { event.preventDefault(); event.stopPropagation(); void query.refetch(); }}>刷新</button>}
  </small>;
}
