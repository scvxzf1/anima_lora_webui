import { useEffect, useRef, type RefObject } from "react";

import { trapDialogFocus } from "./trapDialogFocus";

type Options = {
  dialogRef: RefObject<HTMLElement | null>;
  onClose: () => void;
  initialFocusRef?: RefObject<HTMLElement | null>;
  selectInitialFocus?: boolean;
  returnFocusRef?: RefObject<HTMLElement | null>;
};

export function useDialogLifecycle({
  dialogRef,
  onClose,
  initialFocusRef,
  selectInitialFocus = false,
  returnFocusRef,
}: Options) {
  const onCloseRef = useRef(onClose);
  const lifecycle = useRef(0);

  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);

  useEffect(() => {
    const generation = ++lifecycle.current;
    const returnFocus =
      returnFocusRef?.current ??
      (document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null);
    const initialFocus =
      initialFocusRef?.current ??
      dialogRef.current?.querySelector<HTMLElement>(
        "button, input, select, textarea",
      );
    initialFocus?.focus();
    if (selectInitialFocus && initialFocus instanceof HTMLInputElement)
      initialFocus.select();

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onCloseRef.current();
        return;
      }
      trapDialogFocus(event, dialogRef.current);
    };

    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      queueMicrotask(() => {
        // StrictMode can recreate this effect before deferred restoration runs.
        if (generation !== lifecycle.current) return;
        const activeDialog = document.activeElement?.closest('[role="dialog"], [role="alertdialog"]');
        if (activeDialog?.isConnected) return;
        if (returnFocus?.isConnected) returnFocus.focus();
      });
    };
  }, [dialogRef, initialFocusRef, selectInitialFocus, returnFocusRef]);
}
