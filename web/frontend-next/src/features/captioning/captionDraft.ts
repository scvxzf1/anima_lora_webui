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
  onExisting: string;
  triggerWord: string;
};

export const CAPTION_WRITE_POLICIES = [
  { value: "overwrite", label: "覆盖已有 caption" },
  { value: "skip", label: "跳过已有 caption" },
  { value: "append", label: "追加到已有 caption" },
] as const;

export type CaptionJobPayload = {
  dataset_file: string;
  dataset_index: number;
  source: string;
  profile_id: string;
  user_prompt: string;
  system_prompt: string;
  on_existing: string;
  trigger_word: string;
  items: { file: string }[];
};

export function buildCaptionJobPayload(args: {
  datasetFile: string;
  datasetIndex: number;
  source: string;
  profileId: string;
  prompt: string;
  systemPrompt: string;
  onExisting: string;
  triggerWord: string;
  items: string[];
}): CaptionJobPayload {
  return {
    dataset_file: args.datasetFile,
    dataset_index: args.datasetIndex,
    source: args.source,
    profile_id: args.profileId,
    user_prompt: args.prompt,
    system_prompt: args.systemPrompt,
    on_existing: args.onExisting || "overwrite",
    trigger_word: args.triggerWord.trim(),
    items: args.items.map((file) => ({ file })),
  };
}

const drafts = new Map<string, CaptionDraft>();
export function clearCaptionDrafts() { drafts.clear(); }
const maxDraftScopes = 64;
const emptyDraft = (): CaptionDraft => ({
  selected: [],
  profileId: null,
  prompt: "",
  systemPrompt: "",
  promptPresetId: "",
  onExisting: "overwrite",
  triggerWord: "",
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
