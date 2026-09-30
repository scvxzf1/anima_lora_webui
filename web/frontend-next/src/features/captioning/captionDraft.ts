import { useCallback, useState } from "react";

export type CaptionDraftScope = {
  dataset: string;
  subset: number;
  source: string;
};

export type CaptionDraft = {
  selected: string[];
  profileId: string | null;
  prompt: string;
  systemPrompt: string;
  promptPresetId: string;
};

const drafts = new Map<string, CaptionDraft>();
export function clearCaptionDrafts() { drafts.clear(); }
const maxDraftScopes = 64;
const emptyDraft = (): CaptionDraft => ({
  selected: [],
  profileId: null,
  prompt: "",
  systemPrompt: "",
  promptPresetId: "",
});

function scopeKey(scope: CaptionDraftScope) {
  return JSON.stringify([scope.dataset, scope.subset, scope.source]);
}

function saveDraft(key: string, draft: CaptionDraft) {
  drafts.delete(key);
  drafts.set(key, { ...draft, selected: [...draft.selected] });
  while (drafts.size > maxDraftScopes) drafts.delete(drafts.keys().next().value!);
}

export function useCaptionDraft(scope: CaptionDraftScope) {
  const key = scopeKey(scope);
  const load = (scopeKey: string) => ({
    ...emptyDraft(),
    ...drafts.get(scopeKey),
    selected: [...(drafts.get(scopeKey)?.selected || [])],
  });
  const [state, setState] = useState(() => ({ key, draft: load(key) }));
  if (state.key !== key) setState({ key, draft: load(key) });
  const draft = state.key === key ? state.draft : load(key);

  const update = useCallback((change: Partial<CaptionDraft> | ((current: CaptionDraft) => Partial<CaptionDraft>)) => {
    setState((previous) => {
      const current = previous.key === key ? previous.draft : load(key);
      const next = { ...current, ...(typeof change === "function" ? change(current) : change) };
      saveDraft(key, next);
      return { key, draft: next };
    });
  }, [key]);

  return { draft, update };
}
