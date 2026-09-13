import { useRef } from 'react';
import { CommandDialog } from '../../components/CommandDialog';

export function DatasetDiscardDialog({ action, busy, onCancel, onConfirm }: {
  action: string;
  busy: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const cancelRef = useRef<HTMLButtonElement>(null);
  return (
    <CommandDialog title={busy ? '正在保存数据集' : '放弃未保存修改？'}
      onClose={onCancel} initialFocusRef={cancelRef}>
      <p>{busy ? '保存完成前不能离开当前数据集。' : `当前数据集有未保存修改。${action}将丢弃这些修改。`}</p>
      <footer className="toolbar">
        <button ref={cancelRef} type="button" onClick={onCancel}>继续编辑</button>
        <button type="button" className="danger-command" disabled={busy} onClick={onConfirm}>放弃修改并继续</button>
      </footer>
    </CommandDialog>
  );
}
