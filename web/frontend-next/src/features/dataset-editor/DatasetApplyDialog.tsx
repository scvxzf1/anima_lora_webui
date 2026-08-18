import { X } from 'lucide-react';
import { createPortal } from 'react-dom';
import { useId, useRef } from 'react';

import type { TrainingConfigFile } from '../../api/trainingContext';
import { useDialogLifecycle } from './useDialogLifecycle';

type Props = {
  datasetFile: string;
  trainFile: TrainingConfigFile;
  busy: boolean;
  error?: string;
  onCancel: () => void;
  onConfirm: () => void;
};

export function DatasetApplyDialog({
  datasetFile,
  trainFile,
  busy,
  error,
  onCancel,
  onConfirm,
}: Props) {
  const dialogRef = useRef<HTMLElement>(null);
  const confirmRef = useRef<HTMLButtonElement>(null);
  const titleId = useId();
  useDialogLifecycle({
    dialogRef,
    onClose: onCancel,
    initialFocusRef: confirmRef,
  });

  return createPortal(
    <div
      className="dataset-dialog-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (!busy && event.target === event.currentTarget) onCancel();
      }}
    >
      <section
        ref={dialogRef}
        className="dataset-dialog dataset-apply-dialog"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby={titleId}
      >
        <header>
          <h3 id={titleId}>应用数据集到训练配置</h3>
          <button type="button" className="icon-command" onClick={onCancel} disabled={busy} aria-label="关闭">
            <X size={17} aria-hidden="true" />
          </button>
        </header>
        <p>这会更新目标训练 TOML 中的数据集引用和兼容字段，不会启动训练。</p>
        <dl className="dataset-apply-summary">
          <div><dt>数据集预设</dt><dd>{datasetFile}</dd></div>
          <div><dt>训练配置</dt><dd>{trainFile.label || trainFile.filename || trainFile.path}</dd></div>
          <div><dt>写入路径</dt><dd>{trainFile.path}</dd></div>
        </dl>
        <p className="dataset-apply-warning">只会应用磁盘中已保存的数据集版本；当前未保存草稿不会被写入。</p>
        {error ? <p className="dataset-command-error" role="alert">{error}</p> : null}
        <footer>
          <button type="button" onClick={onCancel} disabled={busy}>取消</button>
          <button ref={confirmRef} type="button" className="primary-command" onClick={onConfirm} disabled={busy}>
            {busy ? '应用中' : '确认应用'}
          </button>
        </footer>
      </section>
    </div>,
    document.body,
  );
}
