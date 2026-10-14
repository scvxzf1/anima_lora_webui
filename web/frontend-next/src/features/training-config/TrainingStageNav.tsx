import { GROUPS } from "./useTrainingWorkspace";
import type { TrainingFieldSpec } from "./trainingForm";
import type { TrainingEditorView } from "./useTrainingEditorView";
export function TrainingStageNav({
  view,
  activeStage,
  onSelect,
  filtered,
  fields,
}: {
  view: TrainingEditorView;
  activeStage: string;
  onSelect: (stage: string) => void;
  filtered: boolean;
  fields: TrainingFieldSpec[];
}) {
  const stream = view === "stream";
  return (
    <nav
      className="training-stage-nav"
      role={stream ? undefined : "tablist"}
      aria-label={stream ? "快速定位配置阶段" : "配置阶段"}
    >
      {GROUPS.map(([group, title], index) => (
        <button
          type="button"
          key={group}
          role={stream ? undefined : "tab"}
          id={`training-tab-${group}`}
          aria-current={
            stream && activeStage === group ? "location" : undefined
          }
          aria-selected={stream ? undefined : activeStage === group}
          aria-controls={
            stream || activeStage === group
              ? `training-stage-${group}`
              : undefined
          }
          tabIndex={stream || activeStage === group ? 0 : -1}
          data-active={activeStage === group}
          onClick={() => onSelect(group)}
          onKeyDown={(event) => {
            if (stream) return;
            const target = {
              ArrowRight: (index + 1) % GROUPS.length,
              ArrowLeft: (index + GROUPS.length - 1) % GROUPS.length,
              Home: 0,
              End: GROUPS.length - 1,
            }[event.key];
            if (target === undefined) return;
            event.preventDefault();
            onSelect(GROUPS[target][0]);
            document
              .getElementById(`training-tab-${GROUPS[target][0]}`)
              ?.focus();
          }}
        >
          {title}
          {filtered && (
            <small>
              {" "}
              ({fields.filter((field) => field.group === group).length})
            </small>
          )}
        </button>
      ))}
    </nav>
  );
}
