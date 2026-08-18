import { X } from 'lucide-react';
import { useId, useRef } from 'react';
import { createPortal } from 'react-dom';

import { useDialogLifecycle } from './useDialogLifecycle';

type Props = {
  file: string;
  onCancel: () => void;
  onConfirm: () => void;
};

export function DatasetDeleteDialog({ file, onCancel, onConfirm }: Props) {
  const dialogRef = useRef<HTMLElement>(null);
  const titleId = useId();
  useDialogLifecycle({ dialogRef, onClose: onCancel });

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
        role="alertdialog"
        aria-modal="true"
        aria-labelledby={titleId}
      >
        <header>
          <h3 id={titleId}>删除数据集预设</h3>
          <button type="button" className="icon-command" onClick={onCancel} aria-label="关闭删除确认">
            <X size={17} aria-hidden="true" />
          </button>
        </header>
        <p>{file}</p>
        <p>只删除 TOML 预设，不删除图片、缩放图或缓存目录。</p>
        <footer>
          <button type="button" onClick={onCancel}>取消</button>
          <button type="button" className="danger-command" onClick={onConfirm}>删除预设</button>
        </footer>
      </section>
    </div>,
    document.body,
  );
}
