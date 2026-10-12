import { useState } from "react";
import { createPortal } from "react-dom";
import { CommandDialog } from "../../components/CommandDialog";
import type { TrainingDraft, TrainingFieldSpec } from "./trainingForm";
import { useTrainingBlockInventory, type TrainingBlockLookup } from "./useTrainingBlockInventory";

type Props = {
  draft: TrainingDraft;
  field: TrainingFieldSpec;
  disabled: boolean;
  onChange: (key: string, value: string | number | boolean) => void;
};

export function TrainingBlockSwapControl(props: Props) {
  const family = String(props.draft.model_family || "anima");
  const path = String(props.draft.pretrained_model_name_or_path || "").trim();
  return path ? <ModelBlockSwapControl {...props} family={family} path={path} />
    : <BlockSwapInput {...props} />;
}

function ModelBlockSwapControl(props: Props & { family: string; path: string }) {
  const lookup = useTrainingBlockInventory(props.family, props.path);
  return <BlockSwapInput key={`${props.family}\n${props.path}`} {...props} lookup={lookup} />;
}

function BlockSwapInput({ draft, field, disabled, onChange, lookup }: Props & { lookup?: TrainingBlockLookup }) {
  const current = String(draft[field.key] ?? "");
  const [editing, setEditing] = useState<string | null>(null);
  const [rejected, setRejected] = useState<{ value: string; max: number } | null>(null);
  const max = lookup?.inventory?.max_swap;
  const commit = () => {
    if (editing === null) return;
    if (editing !== "" && max !== undefined && Number(editing) > max) {
      setRejected({ value: editing, max });
      setEditing(null);
      return;
    }
    onChange(field.key, editing === "" ? "" : Number(editing));
    setEditing(null);
  };
  return <>
    <input id={`training-control-${field.key}`} type="number" aria-label={field.label}
      value={editing ?? current} min={field.min} max={max} step={field.step} disabled={disabled}
      onChange={(event) => setEditing(event.target.value)} onBlur={commit}
      onKeyDown={(event) => {
        if (event.key === "Enter") { event.preventDefault(); event.currentTarget.blur(); }
      }} />
    {rejected && createPortal(<CommandDialog title="Block swap 数量超过上限" onClose={() => setRejected(null)}>
      <p>当前主模型最多可交换 {rejected.max} 个 block。输入的 {rejected.value} 超过上限，已恢复修改前的数值 {current || "（未设置）"}。</p>
      <footer><button type="button" onClick={(event) => { event.preventDefault(); event.stopPropagation(); setRejected(null); }}>知道了</button></footer>
    </CommandDialog>, document.body)}
  </>;
}
