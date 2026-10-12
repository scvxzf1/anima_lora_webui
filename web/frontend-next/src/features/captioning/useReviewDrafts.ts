import { useState, type SetStateAction } from "react";
import { boundReviewHistory } from "./reviewHistory";
import type { CaptionItem } from "./api";
import { applyDeltas, candidateText, type CaptionDelta, type Drafts } from "./captionBatch";
export function useReviewDrafts(items: CaptionItem[]) {
  const [drafts, rawSet] = useState<Drafts>({});
  const [past, setPast] = useState<CaptionDelta[][]>([]);
  const [future, setFuture] = useState<CaptionDelta[][]>([]);
  const [expected, setExpected] = useState<Drafts>({});
  function setDrafts(action: SetStateAction<Drafts>) {
    const next = typeof action === "function" ? action(drafts) : action;
    const deltas = items.flatMap((item) => {
      const before = candidateText(item, drafts), after = candidateText(item, next);
      return before === after ? [] : [{ id: item.id, before, after }];
    });
    if (!deltas.length) return;
    setExpected((current) => ({ ...Object.fromEntries(deltas.map((d) => [d.id, items.find((i) => i.id === d.id)!.proposed_caption])), ...current }));
    rawSet(next); setPast(boundReviewHistory([...past, deltas])); setFuture([]);
  }
  function move(undo: boolean) {
    const source = undo ? past : future, delta = source.at(-1);
    if (!delta) return;
    rawSet(applyDeltas(items, drafts, delta, undo));
    if (undo) { setPast(past.slice(0, -1)); setFuture([...future, delta]); }
    else { setFuture(future.slice(0, -1)); setPast(boundReviewHistory([...past, delta])); }
  }
  function acknowledge(submitted: Drafts) {
    rawSet((current) => Object.fromEntries(Object.entries(current).filter(([id, text]) => submitted[id] !== text)));
    setExpected((current) => Object.fromEntries(Object.entries(current).filter(([id]) => submitted[id] === undefined)));
    setPast([]); setFuture([]);
  }
  function rebase() { setExpected(Object.fromEntries(items.map((item) => [item.id, item.proposed_caption]))); setPast([]); setFuture([]); }
  return { rebase, drafts, setDrafts, expected, acknowledge, undo: () => move(true), redo: () => move(false), canUndo: !!past.length, canRedo: !!future.length };
}
