import { X } from "lucide-react";
import { useId, useRef, type ReactNode, type RefObject } from "react";
import { useDialogLifecycle } from "../features/dataset-editor/useDialogLifecycle";

export function CommandDialog({
  title,
  children,
  onClose,
  busy = false,
  initialFocusRef,
  size = "default",
}: {
  title: string;
  children: ReactNode;
  onClose: () => void;
  busy?: boolean;
  initialFocusRef?: RefObject<HTMLElement | null>;
  size?: "default" | "wide";
}) {
  const ref = useRef<HTMLDivElement>(null);
  const returnFocusRef = useRef(
    document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null,
  );
  const id = useId();
  const descriptionId = `${id}-description`;
  useDialogLifecycle({
    dialogRef: ref,
    initialFocusRef,
    returnFocusRef,
    onClose: () => {
      if (!busy) onClose();
    },
  });
  return (
    <div className="command-backdrop">
      <div
        className={size === "wide" ? "command-dialog is-wide" : "command-dialog"}
        ref={ref}
        role="dialog"
        aria-modal="true"
        aria-labelledby={id}
        aria-describedby={descriptionId}
      >
        <header>
          <h2 id={id}>{title}</h2>
          <span id={descriptionId} className="sr-only">弹窗内容。按 Escape 关闭。</span>
          <button
            type="button"
            className="icon-button"
            title="关闭"
            aria-label="关闭"
            onClick={() => {
              if (!busy) onClose();
            }}
            disabled={busy}
          >
            <X size={18} />
          </button>
        </header>
        {children}
      </div>
    </div>
  );
}
