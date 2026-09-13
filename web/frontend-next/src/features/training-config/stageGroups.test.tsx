import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { fieldsForConfig } from "./fieldCatalog";
import { groupStageFields } from "./stageGroups";
import { TrainingStageFields } from "./TrainingResourceGroups";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";

afterEach(cleanup);
const fields = fieldsForConfig({ output_dir: "output", caption_extension: ".txt" });
for (const stage of ["input", "method", "training"]) {
  it(`groups all ${stage} fields once and exposes filtered matches`, async () => {
    const selected = fields.filter((field) => field.group === stage);
    const groups = groupStageFields(stage, selected);
    expect(groups.flatMap((group) => group.fields).map((field) => field.key).sort())
      .toEqual(selected.map((field) => field.key).sort());
    const onChange = vi.fn();
    const props = { stage, fields: selected, draft: {}, ownKeys: new Set<string>(), disabled: false, onChange, search: "", view: "all" };
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    client.setQueryData(["datasets", "library"], { presets: [], groups: [] });
    const view = render(<TrainingStageFields {...props} />, { wrapper: ({ children }) =>
      <QueryClientProvider client={client}><MemoryRouter>{children}</MemoryRouter></QueryClientProvider> });
    expect(screen.getAllByRole("heading", { level: 3 })).toHaveLength(groups.length);
    const closed = screen.getAllByRole("button").find((button) => button.getAttribute("aria-expanded") === "false")!;
    await userEvent.click(closed);
    expect(closed).toHaveAttribute("aria-expanded", "true");
    expect(onChange).not.toHaveBeenCalled();
    const target = groups.at(-1)!.fields[0];
    view.rerender(<TrainingStageFields {...props} search={target.key} fields={[target]} />);
    expect(screen.getAllByRole("heading", { level: 3 })).toHaveLength(1);
    expect(screen.getByRole("button")).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByLabelText(target.label)).toBeVisible();
  });
}

it("places supplemental fields with their related settings", () => {
  const preprocessing = fields.filter((field) => field.key === "max_data_loader_n_workers");
  expect(preprocessing[0].group).toBe("input");
  expect(groupStageFields("input", preprocessing)[0].title).toBe("预处理与数据加载");
  expect(groupStageFields("input", fields.filter((field) => field.group === "input"))
    .find((group) => group.id === "captions")!.fields.map((field) => field.key)).toContain("caption_extension");
  expect(groupStageFields("training", fields.filter((field) => field.group === "training"))
    .find((group) => group.id === "output")!.fields.map((field) => field.key)).toEqual(expect.arrayContaining(["output_dir", "save_every_n_steps"]));
});
