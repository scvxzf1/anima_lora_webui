import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { create } from "zustand";

type HistorySelection = { selected: string[]; setSelected: (ids: string[] | ((current: string[]) => string[])) => void };
const useHistorySelection = create<HistorySelection>()((set) => ({
  selected: [],
  setSelected: (value) => set(({ selected }) => ({ selected: typeof value === "function" ? value(selected) : value })),
}));

export function useHistorySelected() {
  const state = useHistorySelection();
  return { selected: state.selected, setSelected: state.setSelected };
}

export function resetHistorySelection() {
  useHistorySelection.getState().setSelected([]);
}

const useHistoryAnchors = create<{ anchors: Record<string, string> }>(() => ({ anchors: {} }));

export function rememberHistoryTask(search: string, id: string) {
  useHistoryAnchors.setState(({ anchors }) => ({
    anchors: Object.fromEntries([...Object.entries(anchors).filter(([key]) => key !== search), [search, id]].slice(-30)),
  }));
}

export function useHistoryAnchor(search: string, ready: boolean) {
  const restored = useRef("");
  const [missing, setMissing] = useState({ search: "", id: "" });
  useEffect(() => {
    const id = new URLSearchParams(search).get("anchor") || useHistoryAnchors.getState().anchors[search];
    if (!ready || !id || restored.current === search) return;
    const frame = requestAnimationFrame(() => {
      const row = Array.from(document.querySelectorAll<HTMLElement>("[data-history-task]")).find((node) => node.dataset.historyTask === id);
      if (!row) { setMissing({ search, id }); return; }
      setMissing({ search, id: "" });
      row.scrollIntoView?.({ block: "center" });
      restored.current = search;
      const anchors = { ...useHistoryAnchors.getState().anchors };
      delete anchors[search];
      useHistoryAnchors.setState({ anchors });
    });
    return () => cancelAnimationFrame(frame);
  }, [search, ready]);
  return ready && missing.search === search ? missing.id : "";
}

export function readStackPages(raw: string | null): Record<string, number> {
  try {
    const value: unknown = JSON.parse(raw || "{}");
    if (!value || typeof value !== "object" || Array.isArray(value)) return {};
    return Object.fromEntries(Object.entries(value).filter(([, page]) => Number.isSafeInteger(page) && page >= 0));
  } catch { return {}; }
}

export function useHistoryStackState(id: string) {
  const [params, setParams] = useSearchParams();
  const pages = readStackPages(params.get("stacks"));
  const open = Object.hasOwn(pages, id);
  const page = pages[id] ?? 0;
  const update = (value: number | undefined) => setParams((current) => {
    const next = new URLSearchParams(current);
    next.delete("anchor");
    const values = readStackPages(current.get("stacks"));
    if (value === undefined) delete values[id];
    else values[id] = value;
    if (Object.keys(values).length) next.set("stacks", JSON.stringify(values));
    else next.delete("stacks");
    return next;
  }, { replace: true });
  return { open, page, toggle: () => update(open ? undefined : 0), setPage: (value: number) => update(value) };
}
