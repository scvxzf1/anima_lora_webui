import { useRef } from "react";
import { CommandDialog } from "./CommandDialog";

export function InlineConfirmDialog({
  title = "请确认操作",
  message,
  confirmLabel = "确定",
  danger = false,
  onConfirm,
  onCancel,
}: {
  title?: string;
  message: string;
  confirmLabel?: string;
  danger?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const initialFocusRef = useRef<HTMLButtonElement>(null);
  return (
    <CommandDialog
      title={title}
      onClose={onCancel}
      initialFocusRef={initialFocusRef}
    >
      <div className="inline-confirm-body">
        <p>{message}</p>
        <div className="toolbar inline-confirm-actions">
          <button type="button" onClick={onCancel}>
            取消
          </button>
          <button
            ref={initialFocusRef}
            type="button"
            className={danger ? "history-danger" : "button-primary"}
            onClick={onConfirm}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </CommandDialog>
  );
}
