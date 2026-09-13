import { useEffect, useRef, useState } from "react";
import { useUnsavedChangesGuard } from "../features/dataset-editor/useUnsavedChangesGuard";

export function useServerDraft<T>(incoming: T | undefined, extraDirty = false) {
  const editor = useServerDraftState(incoming, extraDirty);
  useUnsavedChangesGuard(
    editor.dirty,
    "当前页面有未保存修改，离开会丢失这些修改。是否继续？",
  );
  return editor;
}

export function useServerDraftState<T>(
  incoming: T | undefined,
  extraDirty = false,
) {
  const [baseline, setBaseline] = useState<T>();
  const [draft, setDraft] = useState<T>();
  const consumed = useRef<T | undefined>(undefined);
  const dirty =
    JSON.stringify(draft) !== JSON.stringify(baseline) || extraDirty;

  useEffect(() => {
    if (incoming !== undefined && incoming !== consumed.current && !dirty) {
      consumed.current = incoming;
      setBaseline(structuredClone(incoming));
      setDraft(structuredClone(incoming));
    }
  }, [incoming, dirty]);

  function accept(saved: T, submitted: T) {
    // A successful save supersedes the query snapshot visible at submission.
    consumed.current = incoming;
    setBaseline(structuredClone(saved));
    setDraft((current) =>
      JSON.stringify(current) === JSON.stringify(submitted)
        ? structuredClone(saved)
        : current,
    );
  }
  return {
    draft,
    setDraft,
    baseline,
    dirty,
    accept,
    reset: () => setDraft(structuredClone(baseline)),
  };
}
