import type { TrainingEditorView } from "./useTrainingEditorView";
export function TrainingViewSwitch({
  view,
  onChange,
}: {
  view: TrainingEditorView;
  onChange: (view: TrainingEditorView) => void;
}) {
  return (
    <div className="training-view-switch" role="group" aria-label="配置布局">
      {(
        [
          ["groups", "分组视图"],
          ["stream", "积木流视图"],
        ] as const
      ).map(([value, title]) => (
        <button
          key={value}
          type="button"
          aria-pressed={view === value}
          onMouseDown={(event) => event.preventDefault()}
          onClick={() => onChange(value)}
        >
          {title}
        </button>
      ))}
    </div>
  );
}
