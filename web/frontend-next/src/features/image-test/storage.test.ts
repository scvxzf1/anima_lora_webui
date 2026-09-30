import { afterEach, expect, it } from "vitest";
import { IMAGE_TEST_STORAGE_KEY, readImageTestState } from "./storage";

afterEach(() => localStorage.clear());

it("falls back safely for malformed, unsupported, and invalid storage", () => {
  localStorage.setItem(IMAGE_TEST_STORAGE_KEY, "{");
  expect(readImageTestState().history_range).toBe("7");

  localStorage.setItem(IMAGE_TEST_STORAGE_KEY, JSON.stringify({
    version: 2,
    history_range: "yesterday",
    draft: { sampler: "unknown", attn_mode: "not-a-backend", gpu_index: "-1", prompt: "restored" },
    dirty_fields: ["prompt", "not_a_field"],
  }));
  const state = readImageTestState();
  expect(state.history_range).toBe("7");
  expect(state.draft.prompt).toBe("restored");
  expect(state.draft.sampler).toBe("euler");
  expect(state.draft.attn_mode).toBe("flash");
  expect(state.draft.gpu_index).toBe("");
});
