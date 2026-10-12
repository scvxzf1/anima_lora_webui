import { useId, useRef } from "react";
import { createPortal } from "react-dom";
import { X } from "lucide-react";
import { resolveFieldHelp } from "./domain/resolveFieldHelp";
import { useDialogLifecycle } from "../dataset-editor/useDialogLifecycle";
import { sameTrainingValue, type TrainingDraft, type TrainingFieldKind } from "./trainingForm";
import { TrainingBlockInventory } from "./TrainingBlockInventory";
import { TrainingFieldHelpContent, displayHelpValue } from "./TrainingFieldHelpContent";
import "./TrainingFieldHelpDialog.css";

type Props = {
  label: string;
  fieldKey: string;
  value: unknown;
  defaultValue?: unknown;
  kind?: TrainingFieldKind;
  options?: string[];
  optionLabels?: Record<string, string>;
  modelFamily: string;
  draft: TrainingDraft;
  unavailableReason?: string;
  onClose: () => void;
};

export function TrainingFieldHelpDialog({
  label, fieldKey, value, defaultValue, kind, options, optionLabels, modelFamily, draft, unavailableReason, onClose,
}: Props) {
  const help = resolveFieldHelp(fieldKey);
  const differsFromDefault = defaultValue !== undefined && !sameTrainingValue(
    value as TrainingDraft[string], defaultValue as TrainingDraft[string], kind,
  );
  const dialogRef = useRef<HTMLElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const returnFocusRef = useRef(document.activeElement instanceof HTMLElement ? document.activeElement : null);
  const titleId = useId();
  useDialogLifecycle({ dialogRef, initialFocusRef: closeRef, returnFocusRef, onClose });

  return createPortal(
    <div className="training-field-help-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <section ref={dialogRef} className="training-field-help-dialog" role="dialog" aria-modal="true" aria-labelledby={titleId}>
        <header>
          <div><h2 id={titleId}>{label}</h2><code>{fieldKey}</code></div>
          <button ref={closeRef} type="button" className="icon-button" aria-label="关闭字段说明" onClick={onClose}><X size={18} aria-hidden="true" /></button>
        </header>
        <div className="training-field-help-context">
          <dl><div><dt>当前值</dt><dd>{displayHelpValue(value, optionLabels)}</dd></div></dl>
          {differsFromDefault && <span className="training-field-help-default">与默认值不同 · 默认 {displayHelpValue(defaultValue, optionLabels)}</span>}
          {fieldKey !== "model_family" && <span className="training-field-help-family" aria-label={`模型族 ${modelFamily || "未指定"}`}>{modelFamily || "未指定"}</span>}
        </div>
        {unavailableReason && <dl className="training-field-help-unavailable"><dt>不可用原因</dt><dd>{unavailableReason}</dd></dl>}
        {help ? (
          <div className="training-field-help-content">
            <TrainingFieldHelpContent help={help} value={value} options={options} optionLabels={optionLabels} />
            {fieldKey === "blocks_to_swap" && <section className="training-field-help-section">
              <h3>实际模型结构</h3>
              <TrainingBlockInventory draft={draft} />
            </section>}
          </div>
        ) : (
          <div className="training-field-help-content">
            <p className="training-field-help-summary">此字段暂无详细帮助内容。请先在高级配置或项目文档中核对它的作用，再决定是否修改。</p>
          </div>
        )}
      </section>
    </div>,
    document.body,
  );
}
