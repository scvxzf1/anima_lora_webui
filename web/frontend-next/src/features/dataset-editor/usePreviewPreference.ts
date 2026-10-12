import { useState } from 'react';

export function usePreviewPreference(key: string, fallback: boolean) {
  const [value, setValue] = useState(() => {
    try {
      const stored = window.localStorage.getItem(key);
      return stored === 'true' ? true : stored === 'false' ? false : fallback;
    } catch {
      return fallback;
    }
  });

  function update(next: boolean) {
    setValue(next);
    try {
      window.localStorage.setItem(key, String(next));
    } catch {}
  }

  return [value, update] as const;
}
