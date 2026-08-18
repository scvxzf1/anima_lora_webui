import { X } from 'lucide-react';
import { useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

import { useDialogLifecycle } from './useDialogLifecycle';

export type DatasetNameAction = 'new' | 'save-as' | 'copy' | 'rename';

type Props = {
  action: DatasetNameAction;
  initialValue: string;
  onCancel: () => void;
  onConfirm: (name: string) => void;
};

const dialogCopy: Record<DatasetNameAction, { title: string; description: string; confirm: string }> = {
  new: {
    title: '新建数据集预设',
    description: '先建立未保存草稿，保存后写入 configs/datasets/。',
    confirm: '创建草稿',
  },
  'save-as': {
    title: '另存数据集预设',
    description: '使用当前编辑器内容创建新的 TOML 预设。',
    confirm: '另存预设',
  },
  copy: {
    title: '复制数据集预设',
    description: '复制当前内容并切换到新预设。',
    confirm: '复制预设',
  },
  rename: {
    title: '重命名数据集预设',
    description: '先保存新 TOML，再删除旧 TOML；图片和缓存目录不受影响。',
    confirm: '确认重命名',
  },
};

export function DatasetNameDialog({ action, initialValue, onCancel, onConfirm }: Props) {
  const [name, setName] = useState(initialValue);
  const inputRef = useRef<HTMLInputElement>(null);
  const dialogRef = useRef<HTMLElement>(null);
  const titleId = useId();
  const copy = dialogCopy[action];
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
          <h3 id={titleId}>{copy.title}</h3>
          <button type="button" className="icon-command" onClick={onCancel} aria-label="关闭">
            <X size={17} aria-hidden="true" />
          </button>
        </header>
        <p>{copy.description}</p>
        <form
          onSubmit={(event) => {
            event.preventDefault();
            event.stopPropagation();
            const nextName = name.trim();
            if (nextName) onConfirm(nextName);
          }}
        >
          <label>
            <span>预设名称</span>
            <input
              ref={inputRef}
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="dataset_name"
              required
            />
          </label>
          <footer>
            <button type="button" onClick={onCancel}>取消</button>
            <button type="submit" className="primary-command">{copy.confirm}</button>
          </footer>
        </form>
      </section>
    </div>,
    document.body,
  );
}
