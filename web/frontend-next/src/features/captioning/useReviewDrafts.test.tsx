import { act, renderHook } from "@testing-library/react";
import { expect, it } from "vitest";
import { useReviewDrafts } from "./useReviewDrafts";
import { boundReviewHistory } from "./reviewHistory";
import type { CaptionItem } from "./api";
const items = [{ id: "1", proposed_caption: "base" }] as CaptionItem[];
it("bounds history by UTF-8 bytes without discarding drafts", () => {
  expect(boundReviewHistory([[{id:"1", before:"é", after:"界"}]], 4)).toEqual([]);
  const { result } = renderHook(() => useReviewDrafts(items));
  act(() => result.current.setDrafts({ "1": "界".repeat(800000) }));
  expect(result.current.drafts["1"].length).toBe(800000);
  expect(result.current.canUndo).toBe(false);
});
it("preserves prior draft through undo/redo and retains original server baseline", () => {
  const { result, rerender } = renderHook(({ source }) => useReviewDrafts(source), { initialProps: { source: items } });
  act(() => result.current.setDrafts({ "1": "first" }));
  act(() => result.current.setDrafts({ "1": "second" }));
  act(() => result.current.undo()); expect(result.current.drafts).toEqual({ "1": "first" });
  act(() => result.current.redo()); expect(result.current.drafts).toEqual({ "1": "second" });
  rerender({ source: [{ ...items[0], proposed_caption: "server" }] });
  expect(result.current.expected).toEqual({ "1": "base" });
  act(() => result.current.rebase());
  expect(result.current.drafts).toEqual({ "1": "second" });
  expect(result.current.expected).toEqual({ "1": "server" });
  expect(result.current.canUndo).toBe(false);
});
