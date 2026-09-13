import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { fieldsForConfig } from "./fieldCatalog";
import { groupResourceFields, resourceSummary } from "./resourceGroups";
import { TrainingResourceGroups } from "./TrainingResourceGroups";

afterEach(cleanup);
const draft = { mixed_precision: "bf16", auto_block_swap: false, blocks_to_swap: 20, torch_compile: false };
const fields = fieldsForConfig(draft).filter((field) => field.group === "resources");

it("groups every resource exactly once, with AUTO before manual swap", () => {
  const groups = groupResourceFields(fields);
  expect(groups.flatMap((group) => group.fields).map((field) => field.key).sort())
    .toEqual(fields.map((field) => field.key).sort());
  const swap = groups.find((group) => group.id === "residency")!.fields.map((field) => field.key);
  expect(swap[0]).toBe("auto_block_swap");
  expect(swap.indexOf("auto_block_swap_mode")).toBeLessThan(swap.indexOf("blocks_to_swap"));
  expect(groups.find((group) => group.id === "compute")!.fields[0].key).toBe("mixed_precision");
  expect(groups.find((group) => group.id === "diagnostics")!.fields.map((field) => field.key)).toContain("debug_finite_checks");
  expect(groupResourceFields([{ key: "future", label: "Future", kind: "text", group: "resources" }])[0].id).toBe("other");
});

it("shows summaries, expands groups without edits, and reveals search matches", async () => {
  const onChange = vi.fn();
  const props = { fields, draft, onChange, ownKeys: new Set<string>(), disabled: false, view: "all", search: "" };
  const view = render(<TrainingResourceGroups {...props} />);
  const user = userEvent.setup();
  const swap = screen.getByRole("button", { name: /块交换与内存卸载/ });
  expect(swap).toHaveAttribute("aria-expanded", "false");
  expect(swap).toHaveTextContent("手动 · 20 块");
  await user.click(swap);
  expect(swap).toHaveAttribute("aria-expanded", "true");
  expect(within(screen.getByRole("region", { name: /块交换/ })).getByLabelText("AUTO 块交换（实验）")).toBeVisible();
  expect(onChange).not.toHaveBeenCalled();
  await user.click(swap);
  view.rerender(<TrainingResourceGroups {...props} search="auto_block_swap" fields={fields.filter((field) => field.key === "auto_block_swap")} />);
  expect(screen.getByRole("button", { name: /块交换/ })).toHaveAttribute("aria-expanded", "true");
  expect(screen.getByLabelText("AUTO 块交换（实验）")).toBeVisible();
  expect(screen.queryByRole("button", { name: /精度与计算/ })).not.toBeInTheDocument();
  expect(resourceSummary("residency", { auto_block_swap: true, auto_block_swap_mode: "dynamic" })).toBe("AUTO · dynamic");
});
