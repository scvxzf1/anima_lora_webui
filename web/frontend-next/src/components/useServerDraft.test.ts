import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useServerDraftState } from "./useServerDraft";

describe("server drafts", () => {
  it("does not restore stale query data after accepting a save", () => {
    const incoming = { value: "original" };
    const { result, rerender } = renderHook(
      ({ value }) => useServerDraftState(value),
      { initialProps: { value: incoming } },
    );
    act(() => result.current.setDraft({ value: "submitted" }));
    act(() =>
      result.current.accept({ value: "normalized" }, { value: "submitted" }),
    );
    expect(result.current.draft).toEqual({ value: "normalized" });
    expect(result.current.dirty).toBe(false);
    rerender({ value: { value: "fresh" } });
    expect(result.current.draft).toEqual({ value: "fresh" });
  });
  it("clones nested inputs and keeps refreshes away from dirty edits", () => {
    const initial = { nested: { name: "server" } };
    const { result, rerender } = renderHook(
      ({ incoming }) => useServerDraftState(incoming),
      { initialProps: { incoming: initial } },
    );
    expect(result.current.draft).not.toBe(initial);
    expect(result.current.draft?.nested).not.toBe(initial.nested);
    act(() => result.current.setDraft({ nested: { name: "edited" } }));
    rerender({ incoming: { nested: { name: "refreshed" } } });
    expect(result.current.draft?.nested.name).toBe("edited");
    expect(result.current.baseline?.nested.name).toBe("server");
  });
  it("advances only the submitted baseline when newer edits exist", () => {
    const incoming = { value: "original" };
    const { result } = renderHook(() => useServerDraftState(incoming));
    act(() => result.current.setDraft({ value: "newer" }));
    act(() =>
      result.current.accept({ value: "saved" }, { value: "submitted" }),
    );
    expect(result.current.baseline).toEqual({ value: "saved" });
    expect(result.current.draft).toEqual({ value: "newer" });
    expect(result.current.dirty).toBe(true);
  });
});
