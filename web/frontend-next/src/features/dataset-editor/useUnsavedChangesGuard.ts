import { useCallback, useEffect } from 'react';
import { useBeforeUnload, useBlocker } from 'react-router-dom';

export function useUnsavedChangesGuard(active: boolean, message: string) {
  const blocker = useBlocker(active);

  useBeforeUnload(useCallback((event) => {
    if (active) event.preventDefault();
  }, [active]));

  useEffect(() => {
    if (blocker.state !== 'blocked') return;
    if (window.confirm(message)) blocker.proceed();
    else blocker.reset();
  }, [blocker, message]);
}
