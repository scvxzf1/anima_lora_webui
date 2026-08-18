import { X } from 'lucide-react';
import { useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

import type { DatasetLibraryGroup } from './types';
import { useDialogLifecycle } from './useDialogLifecycle';

type Props = {
  action: 'rename' | 'delete';
  group: DatasetLibraryGroup;
  busy: boolean;
  error?: string;
  onCancel: () => void;
  onRename: (label: string) => void;
  onDelete: () => void;
};

export function DatasetGroupDialog({ action, group, busy, error, onCancel, onRename, onDelete }: Props) {
  const [label, setLabel] = useState(group.label);
  const inputRef = useRef<HTMLInputElement>(null);
  const dialogRef = useRef<HTMLElement>(null);
  const titleId = useId();
  useDialogLifecycle({
    dialogRef,
    onClose: onCancel,
    initialFocusRef: action === 'rename' ? inputRef : undefined,
    selectInitialFocus: action === 'rename',
  });

  return createPortal(
    <div
      className="dataset-dialog-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onCancel();
      }}
    >
      <section
        ref={dialogRef}
        className="dataset-dialog"
        role={action === 'delete' ? 'alertdialog' : 'dialog'}
        aria-modal="true"
        aria-labelledby={titleId}
      >
        <header>
          <h3 id={titleId}>
            {action === 'rename' ? '重命名数据集分组' : '删除数据集分组'}
          </h3>
          <button type="button" className="icon-command" onClick={onCancel} aria-label="关闭">
            <X size={17} aria-hidden="true" />
          </button>
        </header>
        {action === 'rename' ? (
          <form
            onSubmit={(event) => {
              event.preventDefault();
              const nextLabel = label.trim();
              if (nextLabel && nextLabel !== group.label) onRename(nextLabel);
            }}
          >
            <label>
              <span>分组名称</span>
              <input ref={inputRef} value={label} onChange={(event) => setLabel(event.target.value)} required />
            </label>
            <footer>
              <button type="button" onClick={onCancel}>取消</button>
              <button type="submit" className="primary-command" disabled={busy}>保存名称</button>
            </footer>
          </form>
        ) : (
          <>
            <p>将删除分组“{group.label}”的组织信息。</p>
            <p>TOML 预设文件不会被删除，它们会保留在其他可见分组中。</p>
            <footer>
              <button type="button" onClick={onCancel}>取消</button>
              <button type="button" className="danger-command" onClick={onDelete} disabled={busy}>
                删除分组
              </button>
            </footer>
          </>
        )}
        {error ? <p className="dataset-command-error" role="alert">{error}</p> : null}
      </section>
    </div>,
    document.body,
  );
}
