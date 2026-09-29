import { sameTrainingValue, type TrainingDraft, type TrainingFieldSpec } from "./trainingForm";
import { FilePenLine, HelpCircle, Undo2 } from "lucide-react";
import { useState } from "react";
import { TrainingDatasetField } from "./TrainingDatasetField";
import { availableFieldOptions, fieldAvailability } from "./fieldCatalog";
import { FIELD_HELP_SUMMARY_ZH } from "./domain/field-help-summary.js";
import { FORM_UI_DEFAULTS } from "./domain/defaults.js";
import { TrainingFieldHelpDialog } from "./TrainingFieldHelpDialog";

type Props = {
  fields: TrainingFieldSpec[];
  draft: TrainingDraft;
  baseline?: TrainingDraft;
  ownKeys: Set<string>;
  disabled: boolean;
  onChange: (key: string, value: string | number | boolean) => void;
  method?: string;
  onEditPrompts?: () => void;
};

export function TrainingFieldEditor({
  fields,
  draft,
  baseline = draft,
  ownKeys,
  disabled,
  onChange,
  method = "lora",
  onEditPrompts,
}: Props) {
  const [helpField, setHelpField] = useState<TrainingFieldSpec | null>(null);
  const modelFamily = String(draft.model_family || "anima");
  return (
    <>
    <div className="training-edit-fields">
      {fields.map((field) => {
        const dirty = !sameTrainingValue(draft[field.key], baseline[field.key], field.kind);
        if (field.key === "dataset_config") return <TrainingDatasetField key={field.key} value={String(draft.dataset_config ?? "")} dirty={dirty} disabled={disabled} own={ownKeys.has(field.key)} onHelp={() => setHelpField(field)} onChange={(value) => onChange(field.key, value)} onUndo={() => onChange(field.key, baseline.dataset_config ?? "")} />;
        const availability = fieldAvailability(field.key, draft, method);
        const options = availableFieldOptions(
          field,
          String(draft.model_family || "anima"),
        );
        const controlDisabled = disabled || !availability.enabled;
        return (
          <label
            key={field.key}
            id={`training-field-${field.key}`}
            tabIndex={-1}
            data-available={availability.enabled}
            data-visible={availability.visible}
            data-availability-code={availability.code || undefined}
            onClick={(event) => {
              // Keep the large field row clickable for text inputs, but do not let
              // an incidental click in the checkbox row toggle its value.
              if (field.kind === "boolean" && !(event.target instanceof HTMLInputElement)) {
                event.preventDefault();
              }
            }}
          >
            <span className="training-field-label">
              <span>{field.label}</span>
              <code className="training-field-tag">{field.key}</code>
              {dirty && <span className="training-field-dirty" role="status">已修改</span>}
              <span className="training-field-actions">
                  <button type="button" className="training-field-help-trigger" title={`查看${field.label}帮助`} aria-label={`查看${field.label}帮助`} onClick={(event) => { event.preventDefault(); event.stopPropagation(); setHelpField(field); }}><HelpCircle size={16} aria-hidden="true" /></button>
                  {dirty && <button type="button" className="training-field-undo" aria-label={`撤销${field.label}修改`} title={`撤销 ${field.label} 修改`} disabled={disabled} onClick={(event) => { event.preventDefault(); event.stopPropagation(); onChange(field.key, baseline[field.key] ?? ""); }}><Undo2 size={15} aria-hidden="true" /></button>}
                  {field.key === "sample_prompts" && onEditPrompts && (
                    <button type="button" title="编辑样张提示词" aria-label="编辑样张提示词" disabled={controlDisabled} onClick={(event) => { event.preventDefault(); onEditPrompts(); }}><FilePenLine size={16} /></button>
                  )}
                </span>
            </span>
            {field.kind === "json" ? (
              <textarea
                aria-label={field.label}
                rows={5}
                value={String(draft[field.key] ?? "")}
                disabled={controlDisabled}
                onChange={(event) => onChange(field.key, event.target.value)}
              />
            ) : field.kind === "boolean" ? (
              <input
                type="checkbox"
                aria-label={field.label}
                checked={Boolean(draft[field.key])}
                disabled={controlDisabled}
                onChange={(event) => onChange(field.key, event.target.checked)}
              />
            ) : field.kind === "select" ? (
              <select
                aria-label={field.label}
                value={String(draft[field.key] ?? "")}
                disabled={controlDisabled}
                onChange={(event) => onChange(field.key, event.target.value)}
              >
                {!options?.includes(String(draft[field.key] ?? "")) && (
                  <option value={String(draft[field.key] ?? "")}>
                    {String(draft[field.key] ?? "") || "未设置"}
                  </option>
                )}
                {options?.map((option) => (
                  <option key={option} value={option}>
                    {field.optionLabels?.[option] || option}
                  </option>
                ))}
              </select>
            ) : (
              <input
                type={field.kind}
                aria-label={field.label}
                value={String(draft[field.key] ?? "")}
                min={field.min}
                max={field.max}
                step={field.step}
                disabled={controlDisabled}
                onChange={(event) =>
                  onChange(
                    field.key,
                    field.kind === "number" && event.target.value !== ""
                      ? Number(event.target.value)
                      : event.target.value,
                  )
                }
              />
            )}
            <small data-source={ownKeys.has(field.key) ? "file" : "merged"}>
              {ownKeys.has(field.key) ? "当前文件" : "继承/预设"}
            </small>
            {(field.help || FIELD_HELP_SUMMARY_ZH[field.key]) && (
              <small className="training-field-help">
                {field.help || FIELD_HELP_SUMMARY_ZH[field.key]}
              </small>
            )}
            {!availability.enabled && (
              <small className="field-availability-reason">
                {availability.reason}
              </small>
            )}
          </label>
        );
      })}
    </div>
    {helpField && <TrainingFieldHelpDialog label={helpField.label} fieldKey={helpField.key} value={draft[helpField.key]} defaultValue={helpField.defaultValue ?? (FORM_UI_DEFAULTS as Record<string, unknown>)[helpField.key]} modelFamily={modelFamily} unavailableReason={fieldAvailability(helpField.key, draft, method).enabled ? undefined : fieldAvailability(helpField.key, draft, method).reason} onClose={() => setHelpField(null)} />}
    </>
  );
}
