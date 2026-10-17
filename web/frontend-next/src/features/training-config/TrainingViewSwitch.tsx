import type { TrainingEditorView } from "./useTrainingEditorView";
export function TrainingViewSwitch({
  view,
  onChange,
}: {
  view: TrainingEditorView;
  onChange: (view: TrainingEditorView) => void;
}) {
  return (
    <label className="training-view-switch">
      <span className="sr-only">配置布局</span>
      <select
        aria-label="配置布局"
        value={view}
        onChange={(event) => onChange(event.target.value as TrainingEditorView)}
      >
        <option value="groups">分组视图</option>
        <option value="stream">积木流视图</option>
      </select>
    </label>
  );
}
