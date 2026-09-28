import { useId, useRef } from "react";
import { createPortal } from "react-dom";
import { X } from "lucide-react";
import { FIELD_HELP_ZH, type FieldHelp } from "./domain/field-help.js";
import { useDialogLifecycle } from "../dataset-editor/useDialogLifecycle";
import "./TrainingFieldHelpDialog.css";

type Props = {
  label: string;
  fieldKey: string;
  value: unknown;
  defaultValue?: unknown;
  modelFamily: string;
  unavailableReason?: string;
  onClose: () => void;
};

function displayValue(value: unknown) {
  if (value === undefined || value === null || value === "") return "未设置";
  if (typeof value === "boolean") return value ? "是" : "否";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function HelpSection({ title, content }: { title: string; content?: string | string[] }) {
  if (!content || (Array.isArray(content) && content.length === 0)) return null;
  const items = Array.isArray(content) ? content : [content];
  return (
    <section className="training-field-help-section">
      <h3>{title}</h3>
      {items.length === 1 ? <p>{items[0]}</p> : <ul>{items.map((item, index) => <li key={`${index}-${item}`}>{item}</li>)}</ul>}
    </section>
  );
}

export function TrainingFieldHelpDialog({
  label, fieldKey, value, defaultValue, modelFamily, unavailableReason, onClose,
}: Props) {
  const help: FieldHelp | undefined = FIELD_HELP_ZH[fieldKey];
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
          <button ref={closeRef} type="button" className="icon-button" title="关闭字段说明" aria-label="关闭字段说明" onClick={onClose}><X size={18} aria-hidden="true" /></button>
        </header>
        <dl className="training-field-help-context">
          <div><dt>当前值</dt><dd>{displayValue(value)}</dd></div>
          {defaultValue !== undefined && <div><dt>页面默认值</dt><dd>{displayValue(defaultValue)}</dd></div>}
          <div><dt>模型族</dt><dd>{modelFamily || "未指定"}</dd></div>
          {unavailableReason && <div className="training-field-help-unavailable"><dt>不可用原因</dt><dd>{unavailableReason}</dd></div>}
        </dl>
        {help ? (
          <div className="training-field-help-content">
            <HelpSection title="摘要" content={help.summary} />
            <HelpSection title="新手建议" content={help.recommend} />
            <HelpSection title="为什么通常这样设" content={help.fill} />
            <HelpSection title="好处" content={help.benefit} />
            <HelpSection title="代价" content={help.cost} />
            <HelpSection title="风险" content={help.risk} />
            <HelpSection title="补充说明" content={help.ps ?? "暂无补充说明。"} />
          </div>
        ) : (
          <div className="training-field-help-content">
            <HelpSection title="字段说明" content="此字段暂无详细帮助内容。请保持当前值，除非你已了解该参数的作用；可在高级配置或项目文档中核对后再修改。" />
          </div>
        )}
      </section>
    </div>,
    document.body,
  );
}
