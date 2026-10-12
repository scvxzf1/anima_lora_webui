import { useLayoutEffect, useState, type RefObject } from 'react';

function placePopover(trigger: HTMLElement, popover: HTMLElement) {
  const margin = 12;
  const gap = 8;
  const anchor = trigger.getBoundingClientRect();
  const width = popover.getBoundingClientRect().width;
  const below = window.innerHeight - anchor.bottom - margin - gap;
  const above = anchor.top - margin - gap;
  const availableHeight = Math.max(0, window.innerHeight - margin * 2);
  const maxHeight = Math.min(availableHeight, Math.max(120, below, above));
  const height = Math.min(popover.scrollHeight, maxHeight);
  const top = height <= below || below >= above ? anchor.bottom + gap : anchor.top - gap - height;
  return {
    top: Math.max(margin, Math.min(top, window.innerHeight - margin - height)),
    left: Math.max(margin, Math.min(anchor.left, window.innerWidth - margin - width)),
    maxHeight,
  };
}

export function useDatasetFieldHelp(
  open: boolean,
  triggerRef: RefObject<HTMLButtonElement | null>,
  popoverRef: RefObject<HTMLElement | null>,
  setOpen: (open: boolean) => void,
) {
  const [position, setPosition] = useState({ top: 12, left: 12, maxHeight: 500 });

  useLayoutEffect(() => {
    if (!open) return;
    const trigger = triggerRef.current;
    const popover = popoverRef.current;
    if (!trigger || !popover) return;
    const update = (event?: Event) => {
      if (event?.target instanceof Node && popover.contains(event.target)) return;
      setPosition(placePopover(trigger, popover));
    };
    const closeOutside = (event: PointerEvent) => {
      if (event.target instanceof Node && !trigger.contains(event.target) && !popover.contains(event.target)) {
        setOpen(false);
      }
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      event.preventDefault();
      event.stopPropagation();
      setOpen(false);
      trigger.focus();
    };
    update();
    const observer = new ResizeObserver(() => update());
    observer.observe(popover);
    window.addEventListener('resize', update);
    window.addEventListener('scroll', update, true);
    document.addEventListener('pointerdown', closeOutside, true);
    // Capture Escape before an enclosing stage dialog handles it.
    document.addEventListener('keydown', closeOnEscape, true);
    return () => {
      observer.disconnect();
      window.removeEventListener('resize', update);
      window.removeEventListener('scroll', update, true);
      document.removeEventListener('pointerdown', closeOutside, true);
      document.removeEventListener('keydown', closeOnEscape, true);
    };
  }, [open, popoverRef, setOpen, triggerRef]);

  return position;
}
