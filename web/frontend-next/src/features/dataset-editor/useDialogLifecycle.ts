import { useEffect, useRef, type RefObject } from 'react';

import { trapDialogFocus } from './trapDialogFocus';

type Options = {
  dialogRef: RefObject<HTMLElement | null>;
  onClose: () => void;
  initialFocusRef?: RefObject<HTMLElement | null>;
  selectInitialFocus?: boolean;
};

export function useDialogLifecycle({
  dialogRef,
  onClose,
  initialFocusRef,
  selectInitialFocus = false,
}: Options) {
  const onCloseRef = useRef(onClose);

  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);

  useEffect(() => {
    const returnFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const initialFocus = initialFocusRef?.current
      ?? dialogRef.current?.querySelector<HTMLElement>('button, input, select, textarea');
    initialFocus?.focus();
    if (selectInitialFocus && initialFocus instanceof HTMLInputElement) initialFocus.select();

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        onCloseRef.current();
        return;
      }
      trapDialogFocus(event, dialogRef.current);
    };

    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('keydown', onKeyDown);
      if (returnFocus?.isConnected) returnFocus.focus();
    };
  }, [dialogRef, initialFocusRef, selectInitialFocus]);
}
