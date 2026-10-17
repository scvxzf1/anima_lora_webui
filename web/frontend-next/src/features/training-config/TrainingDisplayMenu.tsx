import { useEffect, useId, useRef, useState, type KeyboardEvent } from "react";
import { Check, ChevronDown, LayoutGrid } from "lucide-react";
import { useTrainingFieldHelpDisplay } from "./TrainingFieldHelpPreference";
import type { TrainingEditorView } from "./useTrainingEditorView";
import "./TrainingDisplayMenu.css";

const options = [
  ["hidden", "完全隐藏"],
  ["single", "单行显示"],
  ["full", "完整显示"],
] as const;

const layoutOptions = [
  ["groups", "分组"],
  ["stream", "积木流"],
] as const satisfies ReadonlyArray<readonly [TrainingEditorView, string]>;

function useDisplayMenu() {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const menuId = useId();
  const titleId = useId();
  const close = () => {
    setOpen(false);
    triggerRef.current?.focus();
  };
  useEffect(() => {
    if (!open) return;
    menuRef.current
      ?.querySelector<HTMLButtonElement>('[aria-checked="true"]')
      ?.focus();
    const dismissOutside = (event: Event) => {
      if (
        event.target instanceof Node &&
        !rootRef.current?.contains(event.target)
      )
        setOpen(false);
    };
    document.addEventListener("pointerdown", dismissOutside);
    document.addEventListener("focusin", dismissOutside);
    return () => {
      document.removeEventListener("pointerdown", dismissOutside);
      document.removeEventListener("focusin", dismissOutside);
    };
  }, [open]);
  const onMenuKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      close();
      return;
    }
    const buttons = Array.from(
      menuRef.current?.querySelectorAll<HTMLButtonElement>("button") ?? [],
    );
    const current = buttons.findIndex(
      (button) => button === document.activeElement,
    );
    const next =
      event.key === "ArrowDown"
        ? (current + 1) % buttons.length
        : event.key === "ArrowUp"
          ? (current - 1 + buttons.length) % buttons.length
          : event.key === "Home"
            ? 0
            : event.key === "End"
              ? buttons.length - 1
              : null;
    if (next !== null) {
      event.preventDefault();
      buttons[next]?.focus();
    }
  };
  return { open, setOpen, rootRef, triggerRef, menuRef, menuId, titleId, close, onMenuKeyDown };
}

export function TrainingDisplayMenu({
  view,
  onViewChange,
}: {
  view?: TrainingEditorView;
  onViewChange?: (view: TrainingEditorView) => void;
}) {
  const { display, setDisplay } = useTrainingFieldHelpDisplay();
  const { open, setOpen, rootRef, triggerRef, menuRef, menuId, titleId, close, onMenuKeyDown } = useDisplayMenu();
  return (
    <div className="training-display-menu" ref={rootRef}>
      <button
        type="button"
        className="compact-tool"
        ref={triggerRef}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        title="视图与说明设置"
        aria-label="视图与说明设置"
        onClick={() => setOpen(!open)}
        onKeyDown={(event) => {
          if (event.key === "ArrowDown" || event.key === "ArrowUp") {
            event.preventDefault();
            setOpen(true);
          }
        }}
      >
        <LayoutGrid size={16} />
        <span>
          {view
            ? view === "stream"
              ? "积木流"
              : "分组"
            : "说明显示"}
        </span>
        <ChevronDown size={14} />
      </button>
      {open && (
        <div
          className="training-display-menu-panel"
          id={menuId}
          ref={menuRef}
          role="menu"
          aria-labelledby={titleId}
          onKeyDown={onMenuKeyDown}
        >
          <p id={titleId}>视图与说明</p>
          {view && onViewChange ? (
            <>
              <h3>布局模式</h3>
              {layoutOptions.map(([value, label]) => (
                <button
                  key={value}
                  type="button"
                  role="menuitemradio"
                  aria-checked={view === value}
                  tabIndex={-1}
                  onClick={() => {
                    onViewChange(value);
                    close();
                  }}
                >
                  <span>{label}</span>
                  {view === value && <Check size={15} aria-hidden="true" />}
                </button>
              ))}
            </>
          ) : null}
          <h3>简要说明</h3>
          {options.map(([value, label]) => (
            <button
              key={value}
              type="button"
              role="menuitemradio"
              aria-checked={display === value}
              tabIndex={-1}
              onClick={() => {
                setDisplay(value);
                close();
              }}
            >
              <span>{label}</span>
              {display === value && <Check size={15} aria-hidden="true" />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
