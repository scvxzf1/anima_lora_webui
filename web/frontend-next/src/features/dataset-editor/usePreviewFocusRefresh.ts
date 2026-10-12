import { useEffect, useRef } from 'react';

// Query v5 observes visibility changes, but does not observe window focus.
// Keep both return signals local to this preview and coalesce their burst.
export function usePreviewFocusRefresh(refresh: () => Promise<unknown>) {
  const refreshRef = useRef(refresh);
  refreshRef.current = refresh;
  useEffect(() => {
    let pending = false;
    let disposed = false;
    let lastRefresh = -Infinity;
    const onReturn = () => {
      if (document.visibilityState === 'hidden' || pending || Date.now() - lastRefresh < 1000) return;
      pending = true;
      lastRefresh = Date.now();
      void Promise.resolve().then(() => {
        if (!disposed && document.visibilityState !== 'hidden') return refreshRef.current();
      }).catch(() => {}).finally(() => { pending = false; });
    };
    window.addEventListener('focus', onReturn);
    document.addEventListener('visibilitychange', onReturn);
    return () => {
      disposed = true;
      window.removeEventListener('focus', onReturn);
      document.removeEventListener('visibilitychange', onReturn);
    };
  }, []);
}
