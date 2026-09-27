import { X } from "lucide-react";
import { useId, useRef, type ReactNode, type RefObject } from "react";
import { useDialogLifecycle } from "../features/dataset-editor/useDialogLifecycle";

export function CommandDialog({
  title,
  children,
  onClose,
  busy = false,
  initialFocusRef,
}: {
  title: string;
  children: ReactNode;
  onClose: () => void;
  busy?: boolean;
  initialFocusRef?: RefObject<HTMLElement | null>;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const returnFocusRef = useRef(
    document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null,
  );
  const id = useId();
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
        className="command-dialog"
        ref={ref}
        role="dialog"
        aria-modal="true"
        aria-labelledby={id}
      >
        <header>
          <h2 id={id}>{title}</h2>
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
