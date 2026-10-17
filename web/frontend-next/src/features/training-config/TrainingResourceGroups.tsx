import { useState, type ComponentProps } from "react";
import { ChevronDown, ChevronRight, SlidersHorizontal } from "lucide-react";
import { TrainingFieldEditor } from "./TrainingFieldEditor";
import { groupStageFields, stageSummary, stageGroupDefaultOpen } from "./stageGroups";
import "./TrainingResourceGroups.css";

type Expansion = Record<string, Record<string, boolean>>;
type Props = ComponentProps<typeof TrainingFieldEditor> & {
  search: string; view: string; expandedGroups?: Expansion;
  onExpandedGroupsChange?: (value: Expansion) => void;
};

export function TrainingResourceGroups(props: Props) {
  return <TrainingStageFields {...props} stage="resources" />;
}

export function TrainingStageFields({ stage, search, view, expandedGroups, onExpandedGroupsChange, ...editor }: Props & { stage: string }) {
  const [localExpanded, setLocalExpanded] = useState<Expansion>({});
  const expanded = expandedGroups ?? localExpanded;
  const setExpanded = onExpandedGroupsChange ?? setLocalExpanded;
  const filtered = Boolean(search.trim()) || view === "changed";
  const filter = `${stage}\0${search}\0${view}`;
  const values = expanded[filter] ?? {};
  return (
    <div className="training-resource-groups">
      {groupStageFields(stage, editor.fields).map(({ id, title, fields }) => {
        const open = values[id] ?? (filtered || stageGroupDefaultOpen(stage, id));
        const ownFieldCount = fields.filter((field) => editor.ownKeys.has(field.key)).length;
        const summary = stageSummary(stage, id, editor.draft);
        const Icon = open ? ChevronDown : ChevronRight;
        return (
          <section className="training-resource-group" data-stage={stage} data-group={id} key={id}>
            <h3>
              <button
                type="button"
                id={`resource-heading-${stage}-${id}`}
                aria-expanded={open}
                aria-controls={`resource-fields-${stage}-${id}`}
                onClick={() => setExpanded({ ...expanded, [filter]: { ...values, [id]: !open } })}
              >
                <Icon size={16} aria-hidden="true" />
                <span className="resource-title">
                  <SlidersHorizontal size={14} aria-hidden="true" />
                  <span>{title}</span>
                </span>
                <span className="resource-meta">
                  {summary && <span className="resource-summary-badge">{summary}</span>}
                  <span className="resource-field-count resource-chip">{fields.length} 项</span>
                  {ownFieldCount > 0 && (
                    <span className="resource-own-count resource-chip">当前文件 {ownFieldCount} 项</span>
                  )}
                </span>
              </button>
            </h3>
            <div id={`resource-fields-${stage}-${id}`} hidden={!open} role="region" aria-labelledby={`resource-heading-${stage}-${id}`}>
              <TrainingFieldEditor {...editor} fields={fields} />
            </div>
          </section>
        );
      })}
    </div>
  );
}
