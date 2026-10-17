import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { TrainingFieldEditor } from "./TrainingFieldEditor";
import type { TrainingDraft, TrainingFieldSpec } from "./trainingForm";

const draft: TrainingDraft = {
  optimizer_type: "AdamW",
  learning_rate: 0.0001,
  optimizer_args: "weight_decay=0.01",
  lr_scheduler: "constant",
};

function renderFields(fields: TrainingFieldSpec[]) {
  return render(
    <TrainingFieldEditor
      fields={fields}
      draft={draft}
      ownKeys={new Set()}
      disabled={false}
      onChange={vi.fn()}
    />,
  );
}

const field = (key: string, kind: TrainingFieldSpec["kind"]): TrainingFieldSpec => ({
  key,
  label: key,
  group: "training",
  kind,
});

describe("TrainingFieldEditor automatic card layout markers", () => {
  afterEach(() => cleanup());
  it("marks two compact optimizer controls for a balanced row", () => {
    renderFields([field("optimizer_type", "select"), field("learning_rate", "number")]);
    expect(document.querySelector('[data-field-count="2"]')).toBeInTheDocument();
    expect(screen.getByLabelText("optimizer_type").closest(".training-field-card")).toHaveAttribute("data-auto-span", "single");
    expect(screen.getByLabelText("learning_rate").closest(".training-field-card")).toHaveAttribute("data-auto-span", "single");
  });

  it("marks optimizer args wide while short scheduler cards remain dense candidates", () => {
    renderFields([field("optimizer_args", "text"), field("lr_scheduler", "select"), field("learning_rate", "number")]);
    expect(screen.getByLabelText("optimizer_args").closest(".training-field-card")).toHaveAttribute("data-auto-span", "wide");
    expect(screen.getByLabelText("lr_scheduler").closest(".training-field-card")).toHaveAttribute("data-auto-span", "single");
  });

  it("keeps a lone text field wide without a width cap", () => {
    renderFields([field("optimizer_args", "text")]);
    expect(document.querySelector('[data-field-count="1"]')).toBeInTheDocument();
    expect(screen.getByLabelText("optimizer_args").closest(".training-field-card")).toHaveAttribute("data-auto-span", "wide");
  });
});
