import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { afterEach, expect, it } from "vitest";
import { TrainingFieldEditor } from "./TrainingFieldEditor";
import {
  draftFromMerged,
  trainingPatchValues,
  type TrainingDraft,
} from "./trainingForm";
import { fieldsForConfig } from "./fieldCatalog";

afterEach(cleanup);

function CachePolicyForm({ family = "qwen_image_2_1" }: { family?: string }) {
  const source = { model_family: family };
  const allFields = fieldsForConfig(source);
  const fields = allFields.filter(
    (field) => field.key === "qwen_text_encoder_cache_policy",
  );
  const baseline = draftFromMerged(source, allFields);
  const [draft, setDraft] = useState<TrainingDraft>(baseline);
  const patch = trainingPatchValues(draft, baseline, allFields, source);

  return (
    <>
      <TrainingFieldEditor
        fields={fields}
        draft={draft}
        ownKeys={new Set()}
        disabled={false}
        onChange={(key, value) => setDraft((current) => ({ ...current, [key]: value }))}
      />
      <output aria-label="保存补丁">{JSON.stringify(patch)}</output>
    </>
  );
}

it.each([
  ["cpu_offload", "GPU 计算＋CPU 卸载"],
  ["gpu", "全部驻留 GPU"],
  ["cpu", "仅 CPU"],
])("applies %s through the real select and saves the changed value", async (value, label) => {
  const user = userEvent.setup();
  render(<CachePolicyForm />);

  const select = screen.getByRole("combobox", { name: "文本编码器缓存策略" });
  expect(select).toHaveValue("auto");
  expect(screen.getByRole("option", { name: "自动（推荐）" })).toBeInTheDocument();
  expect(screen.getByRole("option", { name: label })).toBeInTheDocument();

  await user.selectOptions(select, value);

  expect(select).toHaveValue(value);
  expect(screen.getByRole("status", { name: "保存补丁" })).toHaveTextContent(
    JSON.stringify({ qwen_text_encoder_cache_policy: value }),
  );
});

it("does not save an unchanged auto default", () => {
  render(<CachePolicyForm />);
  expect(screen.getByRole("status", { name: "保存补丁" })).toHaveTextContent("{}");
});

it("disables this cache policy outside the Qwen model family", () => {
  render(<CachePolicyForm family="anima" />);
  expect(screen.getByRole("combobox", { name: "文本编码器缓存策略" })).toBeDisabled();
});
