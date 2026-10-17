import { useState } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { expect, it } from "vitest";
import { TrainingFieldHelpProvider } from "./TrainingFieldHelpPreference";
import { TrainingDisplayMenu } from "./TrainingDisplayMenu";
import type { TrainingEditorView } from "./useTrainingEditorView";

function Harness() {
  const [view, setView] = useState<TrainingEditorView>("groups");
  return (
    <TrainingFieldHelpProvider>
      <TrainingDisplayMenu view={view} onViewChange={setView} />
    </TrainingFieldHelpProvider>
  );
}

it("combines layout and description density while showing the active layout", () => {
  render(<Harness />);
  const trigger = screen.getByRole("button", { name: "视图与说明设置" });
  expect(trigger).toHaveTextContent("分组视图");

  fireEvent.click(trigger);
  expect(screen.getByRole("heading", { name: "布局模式" })).toBeInTheDocument();
  expect(screen.getByRole("heading", { name: "简要说明" })).toBeInTheDocument();
  fireEvent.click(screen.getByRole("menuitemradio", { name: "积木流视图" }));

  expect(screen.getByRole("button", { name: "视图与说明设置" })).toHaveTextContent("积木流视图");
  expect(screen.queryByRole("menu")).not.toBeInTheDocument();

  fireEvent.click(screen.getByRole("button", { name: "视图与说明设置" }));
  fireEvent.click(screen.getByRole("menuitemradio", { name: "完整显示" }));
  expect(localStorage.getItem("dragon-next.training-field-help-display.v1")).toBe("full");
});
