import { X } from 'lucide-react';
import { useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

import { useDialogLifecycle } from './useDialogLifecycle';

type Props = {
  sourceName: string;
  initialName: string;
  busy: boolean;
  error?: string;
  onCancel: () => void;
  onConfirm: (name: string) => void;
};

export function DatasetImportDialog({ sourceName, initialName, busy, error, onCancel, onConfirm }: Props) {
  const [name, setName] = useState(initialName);
  const inputRef = useRef<HTMLInputElement>(null);
  const dialogRef = useRef<HTMLElement>(null);
  const titleId = useId();
  useDialogLifecycle({
    dialogRef,
    onClose: onCancel,
    initialFocusRef: inputRef,
    selectInitialFocus: true,
  });

  return createPortal(
    <div
      className="dataset-dialog-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onCancel();
      }}
    >
      <section ref={dialogRef} className="dataset-dialog" role="dialog" aria-modal="true" aria-labelledby={titleId}>
        <header>
          <h3 id={titleId}>导入数据集预设</h3>
          <button type="button" className="icon-command" onClick={onCancel} aria-label="关闭">
            <X size={17} aria-hidden="true" />
          </button>
        </header>
        <p>来源文件：{sourceName}</p>
        <p>默认不覆盖已有预设，名称冲突时需换一个名称。</p>
        {error ? <p className="dataset-command-error" role="alert">{error}</p> : null}
        <form
          onSubmit={(event) => {
            event.preventDefault();
            const nextName = name.trim();
            if (nextName) onConfirm(nextName);
          }}
        >
          <label>
            <span>预设名称</span>
            <input ref={inputRef} value={name} onChange={(event) => setName(event.target.value)} required />
          </label>
          <footer>
            <button type="button" onClick={onCancel}>取消</button>
            <button type="submit" className="primary-command" disabled={busy}>导入预设</button>
          </footer>
        </form>
      </section>
    </div>,
    document.body,
  );
}
