import { act, cleanup, render, renderHook, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { TrainingFieldEditor } from "./TrainingFieldEditor";
import { inheritedDraft, useTrainingInheritedDefaults } from "./useTrainingInheritedDefaults";
import { invalidateTrainingQueries } from "./useTrainingWorkspace";
import type { TrainingFieldSpec } from "./trainingForm";

const field: TrainingFieldSpec = { key: "network_dim", label: "LoRA rank", kind: "number", group: "method" };
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it("separates restore-inheritance from undo-baseline and hides unknown defaults", async () => {
  const onChange = vi.fn();
  const props = { fields: [field], draft: { network_dim: 32 }, baseline: { network_dim: 16 }, ownKeys: new Set(["network_dim"]), disabled: false, onChange };
  const view = render(<TrainingFieldEditor {...props} defaults={{ network_dim: 8 }} />);
  expect(screen.getByText("已覆写(未保存)")).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "恢复LoRA rank继承值" }));
  expect(onChange).toHaveBeenLastCalledWith("network_dim", 8);
  await userEvent.click(screen.getByRole("button", { name: "撤销LoRA rank修改" }));
  expect(onChange).toHaveBeenLastCalledWith("network_dim", 16);
  view.rerender(<TrainingFieldEditor {...props} defaults={{}} />);
  expect(screen.queryByText(/已覆写/)).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "恢复LoRA rank继承值" })).not.toBeInTheDocument();
});

it("does not promote UI fallback values to known inherited values", () => {
  const fields: TrainingFieldSpec[] = [field, { key: "lokr_use_einsum", label: "Route", kind: "boolean", group: "method" }];
  expect(inheritedDraft({}, fields)).toEqual({});
  expect(inheritedDraft({ network_args: ["lokr_use_einsum=true"] }, fields)).toEqual({ lokr_use_einsum: true });
  expect(inheritedDraft({ network_dim: 16 }, fields)).toEqual({ network_dim: 16 });
});

it("recognizes precision and adapter inheritance from their persisted keys", () => {
  const fields: TrainingFieldSpec[] = [
    { key: "precision_preference", label: "Precision", kind: "select", group: "resources" },
    { key: "lora_adapter_kind", label: "Adapter", kind: "select", group: "method" },
  ];
  expect(inheritedDraft({}, fields)).toEqual({});
  expect(inheritedDraft({ mixed_precision: "fp16", use_lokr: true }, fields)).toEqual({ precision_preference: "fp16", lora_adapter_kind: "lokr" });
});

it("refreshes inherited values when saving a method changes its defaults", async () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const file = { path: "configs/gui-methods/lora.toml", method: "lora", methods_subdir: "gui-methods" };
  let savedRank = 16;
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ network_dim: savedRank }))));
  const hook = renderHook(() => useTrainingInheritedDefaults(file, "default", [field]), {
    wrapper: ({ children }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>,
  });
  await waitFor(() => expect(hook.result.current?.network_dim).toBe(16));
  savedRank = 32;
  await act(async () => { await invalidateTrainingQueries(client, file.path, "default"); });
  await waitFor(() => expect(hook.result.current?.network_dim).toBe(32));
  client.clear();
});
