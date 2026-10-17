import { Ellipsis } from 'lucide-react';
import { useEffect, useRef, type ReactNode } from 'react';

/** A disclosure, not a menu: actions retain their normal Tab navigation. */
export function DatasetActionDisclosure({ label, children, expanded = false, compact = false }: {
  label: string;
  children: ReactNode;
  expanded?: boolean;
  compact?: boolean;
}) {
  const ref = useRef<HTMLDetailsElement>(null);
  useEffect(() => {
    const dismiss = (event: PointerEvent) => {
      if (ref.current && !ref.current.contains(event.target as Node)) ref.current.open = false;
    };
    document.addEventListener('pointerdown', dismiss);
    return () => document.removeEventListener('pointerdown', dismiss);
  }, []);

  if (expanded) return <div className="dataset-expanded-actions">{children}</div>;
  return (
    <details ref={ref} className="dataset-action-disclosure" onKeyDown={(event) => {
      if (event.key === 'Escape' && ref.current) {
        event.stopPropagation();
        ref.current.open = false;
        ref.current.querySelector('summary')?.focus();
      }
    }}>
      <summary role="button" aria-label={label} title={label}>
        <Ellipsis size={16} aria-hidden="true" />{!compact && <span>更多操作</span>}
      </summary>
      <div className="dataset-action-panel">{children}</div>
    </details>
  );
}
