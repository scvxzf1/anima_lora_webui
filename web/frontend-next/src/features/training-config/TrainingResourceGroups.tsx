import { useState, type ComponentProps } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { TrainingFieldEditor } from "./TrainingFieldEditor";
import { groupStageFields, stageSummary, stageGroupDefaultOpen } from "./stageGroups";
import "./TrainingResourceGroups.css";

type Props = ComponentProps<typeof TrainingFieldEditor> & { search: string; view: string };

export function TrainingResourceGroups(props: Props) {
  return <TrainingStageFields {...props} stage="resources" />;
}

export function TrainingStageFields({ stage, search, view, ...editor }: Props & { stage: string }) {
  const [expanded, setExpanded] = useState<{ filter: string; values: Record<string, boolean> }>({ filter: "", values: {} });
  const filtered = Boolean(search.trim()) || view === "changed";
  const filter = `${stage}\0${search}\0${view}`;
  const values = expanded.filter === filter ? expanded.values : {};
  return (
    <div className="training-resource-groups">
      {groupStageFields(stage, editor.fields).map(({ id, title, fields }) => {
        const open = values[id] ?? (filtered || stageGroupDefaultOpen(stage, id));
        const Icon = open ? ChevronDown : ChevronRight;
        return (
          <section className="training-resource-group" key={id}>
            <h3>
              <button
                type="button"
                id={`resource-heading-${id}`}
                aria-expanded={open}
                aria-controls={`resource-fields-${id}`}
                onClick={() => setExpanded({ filter, values: { ...values, [id]: !open } })}
              >
                <Icon size={16} aria-hidden="true" />
                <span>{title}</span>
                <small>{stageSummary(stage, id, editor.draft)}</small>
                <span className="resource-field-count">{fields.length} 项</span>
              </button>
            </h3>
            <div id={`resource-fields-${id}`} hidden={!open} role="region" aria-labelledby={`resource-heading-${id}`}>
              <TrainingFieldEditor {...editor} fields={fields} />
            </div>
          </section>
        );
      })}
    </div>
  );
}
