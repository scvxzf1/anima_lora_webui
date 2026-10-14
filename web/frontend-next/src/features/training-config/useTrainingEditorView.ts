import { useLayoutEffect, useRef, useState } from "react";

export type TrainingEditorView = "groups" | "stream";
export const TRAINING_EDITOR_VIEW_KEY = "dragon-next.training-editor-view.v1";
export function readTrainingEditorView(): TrainingEditorView {
  try {
    return localStorage.getItem(TRAINING_EDITOR_VIEW_KEY) === "stream"
      ? "stream"
      : "groups";
  } catch {
    return "groups";
  }
}
export function trainingScrollRoot(): HTMLElement | null {
  return document.querySelector(
    window.matchMedia?.("(max-width: 800px)").matches
      ? ".training-config-page"
      : ".training-edit-groups",
  );
}
export function useTrainingEditorView(
  activeStage: string,
  setActiveStage: (stage: string) => void,
) {
  const [view, setView] = useState<TrainingEditorView>(readTrainingEditorView);
  const anchor = useRef<{
    id: string;
    focused?: HTMLElement;
    offset: number;
  } | null>(null);
  const changeView = (next: TrainingEditorView) => {
    if (next === view) return;
    const root = trainingScrollRoot();
    const focused = document.activeElement as HTMLElement | null;
    const editing = focused?.closest<HTMLElement>('[id^="training-field-"]');
    const top = root?.getBoundingClientRect().top ?? 0;
    const visible = Array.from(
      document.querySelectorAll<HTMLElement>('[id^="training-field-"]'),
    ).find(
      (field) =>
        field.getClientRects().length &&
        field.getBoundingClientRect().bottom > top + 60,
    );
    const field = editing ?? visible;
    const stage = field?.closest<HTMLElement>("[data-training-stage]")?.dataset
      .trainingStage;
    if (stage) setActiveStage(stage);
    anchor.current = field
      ? {
          id: field.id,
          focused: editing ? (focused ?? undefined) : undefined,
          offset: field.getBoundingClientRect().top - top,
        }
      : null;
    // Commit blur-only controls before any stage can unmount.
    focused?.blur();
    setView(next);
    try {
      localStorage.setItem(TRAINING_EDITOR_VIEW_KEY, next);
    } catch {
      /* Session preference still works. */
    }
  };
  useLayoutEffect(() => {
    const pending = anchor.current;
    anchor.current = null;
    if (!pending) return;
    const field = document.getElementById(pending.id);
    const root = trainingScrollRoot();
    if (field && root)
      root.scrollTop +=
        field.getBoundingClientRect().top -
        root.getBoundingClientRect().top -
        pending.offset;
    if (pending.focused?.isConnected)
      pending.focused.focus({ preventScroll: true });
  }, [view, activeStage]);
  return { view, changeView };
}
