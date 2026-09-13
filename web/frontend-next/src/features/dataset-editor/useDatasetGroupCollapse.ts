import { useEffect, useState } from 'react';

export const DATASET_COLLAPSE_KEY = 'dragon-next:dataset-groups:collapsed:v1';

function readCollapsed(): string[] {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(DATASET_COLLAPSE_KEY) || '[]');
    return Array.isArray(value) ? value.filter((id): id is string => typeof id === 'string') : [];
  } catch { return []; }
}

export function useDatasetGroupCollapse(dragging: boolean, searchActive: boolean, hoveredGroupId: string | null = null) {
  const [collapsed, setCollapsed] = useState(readCollapsed);

  useEffect(() => {
    const sync = (event: StorageEvent) => {
      if (event.key === DATASET_COLLAPSE_KEY || event.key === null) setCollapsed(readCollapsed());
    };
    window.addEventListener('storage', sync);
    return () => window.removeEventListener('storage', sync);
  }, []);


  function toggle(id: string) {
    if (dragging || searchActive) return;
    const next = collapsed.includes(id) ? collapsed.filter(value => value !== id) : [...collapsed, id];
    setCollapsed(next);
    try { localStorage.setItem(DATASET_COLLAPSE_KEY, JSON.stringify(next)); } catch { /* Session state still works. */ }
  }

  return {
    toggle,
    isCollapsed: (id: string) => !searchActive && collapsed.includes(id),
    isTemporary: (_id: string) => false,
  };
}
