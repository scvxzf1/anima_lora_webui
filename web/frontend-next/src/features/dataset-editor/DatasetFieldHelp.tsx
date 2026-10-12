import { HelpCircle, X } from 'lucide-react';
import { useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

import { DATASET_FIELD_HELP, type DatasetFieldHelpKey, type DatasetHelpContent } from './field-help';
import { useDatasetFieldHelp } from './useDatasetFieldHelp';
import './DatasetFieldHelp.css';

type Props = {
  field: DatasetFieldHelpKey;
  label?: string;
};

export function DatasetFieldHelp({ field, label }: Props) {
  const [open, setOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const popoverRef = useRef<HTMLElement>(null);
  const contentId = useId();
  const help: DatasetHelpContent = DATASET_FIELD_HELP[field];
  const accessibleLabel = label || field;
  const position = useDatasetFieldHelp(open, triggerRef, popoverRef, setOpen);
  const portalRoot = triggerRef.current?.closest('[role="dialog"]') ?? document.body;

  return (
    <span className="dataset-field-help">
      <button
        ref={triggerRef}
        type="button"
        className="dataset-field-help-trigger"
        aria-label={`查看${accessibleLabel}帮助`}
        aria-expanded={open}
        aria-controls={open ? contentId : undefined}
        title={`查看${accessibleLabel}帮助`}
        onClick={(event) => {
          event.preventDefault();
          event.stopPropagation();
          setOpen((current) => !current);
        }}
      >
        <HelpCircle size={15} aria-hidden="true" />
      </button>
      {open ? createPortal(
        <section ref={popoverRef} id={contentId} className="dataset-field-help-popover" role="note" aria-label={`${accessibleLabel}说明`} style={position}>
          <header>
            <strong>{accessibleLabel}</strong>
            <button type="button" className="dataset-field-help-close" aria-label="关闭字段帮助" onClick={() => {
              setOpen(false);
              triggerRef.current?.focus();
            }}><X size={16} aria-hidden="true" /></button>
          </header>
          <p>{help.summary}</p>
          <div className="dataset-field-help-recommendation"><strong>建议</strong><p>{help.recommendation}</p></div>
          {help.options?.length ? <table>
            <thead><tr><th>选项</th><th>什么时候选</th></tr></thead>
            <tbody>{help.options.map((option) => <tr key={option.value}>
              <td><strong>{option.label}</strong><code>{option.value}</code></td><td>{option.when}</td>
            </tr>)}</tbody>
          </table> : null}
          {help.note ? <div className="dataset-field-help-note"><strong>注意</strong><p>{help.note}</p></div> : null}
          {help.advanced?.length ? <details>
            <summary>进阶</summary>
            {help.advanced.map((paragraph) => <p key={paragraph}>{paragraph}</p>)}
          </details> : null}
        </section>, portalRoot,
      ) : null}
    </span>
  );
}
