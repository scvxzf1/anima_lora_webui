import { useCallback, useEffect, useRef, useState } from 'react';
import { useBeforeUnload, useBlocker } from 'react-router-dom';

export function useDatasetDiscardGuard(dirty: boolean, busy: boolean) {
  const blocker = useBlocker(dirty || busy);
  const [action, setAction] = useState<string | null>(null);
  const pending = useRef<((accepted: boolean) => void) | null>(null);
  useBeforeUnload(useCallback((event) => {
    if (dirty || busy) event.preventDefault();
  }, [dirty, busy]));
  useEffect(() => () => { pending.current?.(false); }, []);

  function confirmDiscard(nextAction: string): Promise<boolean> {
    if (busy || pending.current || blocker.state === 'blocked') return Promise.resolve(false);
    if (!dirty) return Promise.resolve(true);
    setAction(nextAction);
    return new Promise(resolve => { pending.current = resolve; });
  }

  function finish(accepted: boolean) {
    if (accepted && busy) return;
    const resolve = pending.current;
    pending.current = null;
    setAction(null);
    resolve?.(accepted);
    if (blocker.state === 'blocked') {
      if (accepted) blocker.proceed();
      else blocker.reset();
    }
  }

  return {
    confirmDiscard,
    discardDialog: action || blocker.state === 'blocked' ? {
      action: action || '离开页面',
      busy,
      onCancel: () => finish(false),
      onConfirm: () => finish(true),
    } : null,
  };
}
