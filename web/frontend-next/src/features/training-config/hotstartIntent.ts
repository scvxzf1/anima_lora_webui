import { create } from "zustand";
import type { HotstartTarget } from "../training-history/historyHotstartApi";

export type HotstartIntent = HotstartTarget & { path: string; id: number };

type HotstartIntentState = {
  intent: HotstartIntent | null;
  offer: (value: Omit<HotstartIntent, "id">) => void;
  clear: (id: number) => void;
};

let nextId = 0;
export const useHotstartIntent = create<HotstartIntentState>((set) => ({
  intent: null,
  offer: (value) => set({ intent: { ...value, id: ++nextId } }),
  clear: (id) => set((state) => state.intent?.id === id ? { intent: null } : state),
}));
