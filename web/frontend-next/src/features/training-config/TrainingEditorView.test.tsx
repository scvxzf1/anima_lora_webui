import { useState } from "react";
import {
  act,
  cleanup,
  fireEvent,
  render,
  renderHook,
  screen,
} from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import {
  readTrainingEditorView,
  TRAINING_EDITOR_VIEW_KEY,
  useTrainingEditorView,
} from "./useTrainingEditorView";
import { TrainingViewSwitch } from "./TrainingViewSwitch";
import { TrainingStageFields } from "./TrainingResourceGroups";
import type { TrainingFieldSpec } from "./trainingForm";
vi.mock("./TrainingFieldEditor", () => ({
  TrainingFieldEditor: ({ draft }: { draft: Record<string, string> }) => (
    <span>{draft.output_dir}</span>
  ),
}));
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  localStorage.clear();
});
it("defaults to grouped view for unknown and inaccessible preferences", () => {
  localStorage.setItem(TRAINING_EDITOR_VIEW_KEY, "future-layout");
  expect(readTrainingEditorView()).toBe("groups");
  localStorage.setItem(TRAINING_EDITOR_VIEW_KEY, "stream");
  expect(readTrainingEditorView()).toBe("stream");
  vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
    throw new Error("blocked");
  });
  expect(readTrainingEditorView()).toBe("groups");
});
it("keeps the switch usable when browser storage is blocked", () => {
  vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
    throw new Error("blocked");
  });
  const { result } = renderHook(() => useTrainingEditorView("input", vi.fn()));
  act(() => result.current.changeView("stream"));
  expect(result.current.view).toBe("stream");
});
function Harness() {
  const [stage, setStage] = useState("training");
  const layout = useTrainingEditorView(stage, setStage);
  const [expanded, setExpanded] = useState<
    Record<string, Record<string, boolean>>
  >({});
  const [search, setSearch] = useState("");
  const [draft, setDraft] = useState({ output_dir: "unsaved-output" });
  const fields: TrainingFieldSpec[] = [
    { key: "output_dir", label: "输出目录", kind: "text", group: "training" },
  ];
  return (
    <>
      <TrainingViewSwitch view={layout.view} onChange={layout.changeView} />
      <button onClick={() => setSearch(search ? "" : "output_dir")}>
        筛选
      </button>
      <button onClick={() => setDraft({ output_dir: "edited-output" })}>
        编辑
      </button>
      <TrainingStageFields
        stage={stage}
        fields={fields}
        draft={draft}
        ownKeys={new Set()}
        disabled={false}
        onChange={vi.fn()}
        search={search}
        view="all"
        expandedGroups={expanded}
        onExpandedGroupsChange={setExpanded}
      />
    </>
  );
}
it("preserves edited draft and per-filter fold state through layout changes", () => {
  render(<Harness />);
  const fold = screen.getByRole("button", { name: /保存与续训点/ });
  fireEvent.click(fold);
  const folded = fold.getAttribute("aria-expanded");
  fireEvent.click(screen.getByRole("button", { name: "编辑" }));
  fireEvent.click(screen.getByRole("button", { name: "积木流视图" }));
  expect(fold).toHaveAttribute("aria-expanded", folded);
  expect(screen.getByText("edited-output")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "筛选" }));
  fireEvent.click(screen.getByRole("button", { name: "分组视图" }));
  fireEvent.click(screen.getByRole("button", { name: "筛选" }));
  expect(screen.getByRole("button", { name: /保存与续训点/ })).toHaveAttribute(
    "aria-expanded",
    folded,
  );
  expect(localStorage.getItem(TRAINING_EDITOR_VIEW_KEY)).toBe("groups");
});
