import { useRef, useState } from "react";
import { CommandDialog } from "./CommandDialog";

export function InlineConfirmDialog({
  title = "请确认操作",
  message,
  confirmLabel = "确定",
  danger = false,
  busy = false,
  onConfirm,
  onCancel,
}: {
  title?: string;
  message: string;
  confirmLabel?: string;
  danger?: boolean;
  busy?: boolean;
  onConfirm: () => void | Promise<unknown>;
  onCancel: () => void;
}) {
  const cancelFocusRef = useRef<HTMLButtonElement>(null);
  const submittingRef = useRef(false);
  const [submitting, setSubmitting] = useState(false);
  const isBusy = busy || submitting;

  const confirm = () => {
    if (isBusy || submittingRef.current) return;
    submittingRef.current = true;
    try {
      const result = onConfirm();
      if (result && typeof result.then === "function") {
        setSubmitting(true);
        void result.then(() => {
          submittingRef.current = false;
          setSubmitting(false);
        }, () => {
          submittingRef.current = false;
          setSubmitting(false);
        });
      }
    } catch (error) {
      submittingRef.current = false;
      throw error;
    }
  };

  return (
    <CommandDialog
      title={title}
      onClose={() => {
        if (!isBusy) onCancel();
      }}
      busy={isBusy}
      initialFocusRef={cancelFocusRef}
    >
      <div className="inline-confirm-body">
        <p>{message}</p>
        <div className="toolbar inline-confirm-actions">
          <button ref={cancelFocusRef} type="button" onClick={onCancel} disabled={isBusy}>
            取消
          </button>
          <button
            type="button"
            className={danger ? "history-danger" : "button-primary"}
            onClick={confirm}
            disabled={isBusy}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </CommandDialog>
  );
}
