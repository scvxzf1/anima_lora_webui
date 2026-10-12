import type { CaptionItem } from "./api";
import { joinCaptionTags, splitCaptionTags } from "./captionTags";
export type Drafts = Record<string, string>;
export type BatchOperation = "append" | "delete" | "replace" | "front" | "dedupe";
export type CaptionDelta = { id: string; before: string; after: string };
export function candidateText(item: CaptionItem, drafts: Drafts) { return drafts[item.id] ?? item.proposed_caption; }
export function tagCounts(items: CaptionItem[], drafts: Drafts) {
  const counts = new Map<string, number>();
  for (const item of items) for (const tag of new Set(splitCaptionTags(candidateText(item, drafts)))) counts.set(tag, (counts.get(tag) || 0) + 1);
  return [...counts].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
}
export function filterCandidates(items: CaptionItem[], drafts: Drafts, tag: string) {
  return tag ? items.filter((item) => splitCaptionTags(candidateText(item, drafts)).includes(tag)) : items;
}
export function batchPreview(items: CaptionItem[], drafts: Drafts, ids: string[], operation: BatchOperation, source: string, target: string): CaptionDelta[] {
  const wanted = new Set(ids), from = [...new Set(splitCaptionTags(source))], to = splitCaptionTags(target);
  return items.filter((item) => wanted.has(item.id)).flatMap((item) => {
    const before = candidateText(item, drafts), tags = splitCaptionTags(before);
    let next = tags;
    if (operation === "append") next = [...tags, ...from.filter((tag) => !tags.includes(tag))];
    if (operation === "delete") next = tags.filter((tag) => !from.includes(tag));
    if (operation === "replace") next = tags.flatMap((tag) => from.includes(tag) ? to : [tag]);
    if (operation === "front") next = [...new Set(from), ...tags.filter((tag) => !from.includes(tag))];
    if (operation === "dedupe") next = [...new Set(tags)];
    // Preserve untouched source bytes; parsing alone never rewrites text.
    if (next.length === tags.length && next.every((tag, index) => tag === tags[index])) return [];
    const after = joinCaptionTags(next);
    return after.trim() && after !== before ? [{ id: item.id, before, after }] : [];
  });
}
export function applyDeltas(items: CaptionItem[], drafts: Drafts, deltas: CaptionDelta[], undo = false) {
  const next = { ...drafts }, bases = new Map(items.map((item) => [item.id, item.proposed_caption]));
  for (const delta of deltas) {
    const text = undo ? delta.before : delta.after;
    if (text === bases.get(delta.id)) delete next[delta.id]; else next[delta.id] = text;
  }
  return next;
}

export function deltasAreCurrent(items: CaptionItem[], drafts: Drafts, deltas: CaptionDelta[]) {
  const current = new Map(items.map((item) => [item.id, candidateText(item, drafts)]));
  return deltas.length > 0 && deltas.every((delta) => current.get(delta.id) === delta.before);
}
