import { X } from 'lucide-react';
import { createPortal } from 'react-dom';
import { useId, useRef, useState } from 'react';

import { useDialogLifecycle } from '../dataset-editor/useDialogLifecycle';

type Props = {
  initialName: string;
  busy: boolean;
  error?: string;
  onCancel: () => void;
  onConfirm: (name: string) => void;
};

export function TrainingSaveAsDialog({ initialName, busy, error, onCancel, onConfirm }: Props) {
  const [name, setName] = useState(initialName);
  const dialogRef = useRef<HTMLElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const titleId = useId();
  useDialogLifecycle({ dialogRef, onClose: onCancel, initialFocusRef: inputRef, selectInitialFocus: true });

  return createPortal(
    <div className="dataset-dialog-backdrop" role="presentation" onMouseDown={(event) => {
      if (!busy && event.target === event.currentTarget) onCancel();
    }}>
      <section ref={dialogRef} className="dataset-dialog" role="dialog" aria-modal="true" aria-labelledby={titleId}>
        <header>
          <h3 id={titleId}>另存训练配置</h3>
          <button type="button" className="icon-command" onClick={onCancel} disabled={busy} aria-label="关闭">
            <X size={17} aria-hidden="true" />
          </button>
        </header>
        <p>创建 `configs/imported/` 下的新 TOML，不覆盖已有文件。</p>
        <form onSubmit={(event) => {
          event.preventDefault();
          const next = name.trim();
          if (next) onConfirm(next);
        }}>
          <label>
            <span>配置名称</span>
            <input ref={inputRef} value={name} onChange={(event) => setName(event.target.value)} required />
          </label>
          {error ? <p className="training-command-error" role="alert">{error}</p> : null}
          <footer>
            <button type="button" onClick={onCancel} disabled={busy}>取消</button>
            <button type="submit" className="primary-command" disabled={busy}>{busy ? '另存中' : '确认另存'}</button>
          </footer>
        </form>
      </section>
    </div>,
    document.body,
  );
}
