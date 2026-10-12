import { useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { jsonResponse } from "../../test/renderInApp";
import { TrainingFieldEditor } from "./TrainingFieldEditor";
import { TRAINING_FIELDS, type TrainingDraft } from "./trainingForm";

const field = TRAINING_FIELDS.find((item) => item.key === "blocks_to_swap")!;
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

function mount(initial: TrainingDraft = {}, onChange = vi.fn()) {
  function Harness() {
    const [draft, setDraft] = useState<TrainingDraft>({ model_family: "krea2_raw", pretrained_model_name_or_path: "first.safetensors", blocks_to_swap: 8, ...initial });
    return <>
      <TrainingFieldEditor fields={[field]} draft={draft} ownKeys={new Set()} disabled={false}
        onChange={(key, value) => { onChange(key, value); setDraft((current) => ({ ...current, [key]: value })); }} />
      <button onClick={() => setDraft((current) => ({ ...current, model_family: "z_image", pretrained_model_name_or_path: "second.safetensors" }))}>换模型</button>
    </>;
  }
  render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><Harness /></QueryClientProvider>);
  return onChange;
}

it("keeps the inventory out of the field row while enforcing HTML max and committing legal values", async () => {
  const fetch = vi.fn(async () => jsonResponse({ block_count: 36, max_swap: 34 }));
  vi.stubGlobal("fetch", fetch);
  const onChange = mount();
  const input = screen.getByRole("spinbutton", { name: field.label });
  await waitFor(() => expect(input).toHaveAttribute("max", "34"));
  expect(screen.queryByTestId("training-block-inventory")).not.toBeInTheDocument();
  expect(input.nextElementSibling).toHaveAttribute("data-source", "merged");
  const user = userEvent.setup();
  await user.clear(input);
  await user.type(input, "34");
  expect(onChange).not.toHaveBeenCalled();
  await user.keyboard("{Enter}");
  expect(onChange).toHaveBeenCalledWith("blocks_to_swap", 34);
  expect(fetch).toHaveBeenCalledTimes(1);
});

it("rejects an over-limit edit, restores the last accepted value and uses CommandDialog", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ block_count: 36, max_swap: 34 })));
  const onChange = mount();
  const input = screen.getByRole("spinbutton", { name: field.label });
  await waitFor(() => expect(input).toHaveAttribute("max", "34"));
  fireEvent.change(input, { target: { value: "20" } });
  fireEvent.blur(input);
  fireEvent.change(input, { target: { value: "35" } });
  expect(input).toHaveValue(35);
  fireEvent.blur(input);
  expect(input).toHaveValue(20);
  expect(onChange).toHaveBeenCalledTimes(1);
  expect(screen.getByRole("dialog", { name: "Block swap 数量超过上限" })).toHaveTextContent("已恢复修改前的数值 20");
  await userEvent.setup().click(screen.getByRole("button", { name: "知道了" }));
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
});

it("removes the old model limit immediately and retains existing invalid configuration", async () => {
  const fetch = vi.fn(async (url: RequestInfo | URL) => String(url).includes("second")
    ? jsonResponse({ error: "读取失败" }, 400)
    : jsonResponse({ block_count: 36, max_swap: 34 }));
  vi.stubGlobal("fetch", fetch);
  const onChange = mount({ blocks_to_swap: 40 });
  await waitFor(() => expect(screen.getByRole("spinbutton")).toHaveAttribute("max", "34"));
  expect(screen.getByRole("spinbutton")).toHaveValue(40);
  expect(onChange).not.toHaveBeenCalled();
  await userEvent.setup().click(screen.getByRole("button", { name: "换模型" }));
  expect(screen.getByRole("spinbutton")).not.toHaveAttribute("max");
  expect(screen.queryByText(/可交换上限 34/)).not.toBeInTheDocument();
  await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
  expect(screen.getByRole("spinbutton")).not.toHaveAttribute("max");
});

it("shows the shared live inventory and refresh only in the help dialog", async () => {
  const fetch = vi.fn(async () => jsonResponse({ block_count: 32, max_swap: 30 }));
  vi.stubGlobal("fetch", fetch);
  const onChange = mount({ model_family: "qwen_image_2_1", pretrained_model_name_or_path: "qwen.safetensors" });
  const input = screen.getByRole("spinbutton", { name: field.label });
  await waitFor(() => expect(input).toHaveAttribute("max", "30"));
  const user = userEvent.setup();
  await user.click(screen.getByRole("button", { name: `查看${field.label}帮助` }));
  const dialog = await screen.findByRole("dialog", { name: field.label });
  expect(within(dialog).getByRole("heading", { name: "实际模型结构" })).toBeInTheDocument();
  expect(within(dialog).getByTestId("training-block-inventory")).toHaveTextContent("32 个 block · 可交换上限 30");
  expect(String(fetch.mock.calls[0])).toContain("qwen.safetensors");
  expect(fetch).toHaveBeenCalledTimes(1);
  fetch.mockImplementation(async () => jsonResponse({ block_count: 36, max_swap: 34 }));
  await user.click(within(dialog).getByRole("button", { name: "刷新主模型 block 数" }));
  await waitFor(() => expect(input).toHaveAttribute("max", "34"));
  expect(within(dialog).getByTestId("training-block-inventory")).toHaveTextContent("36 个 block · 可交换上限 34");
  expect(fetch).toHaveBeenCalledTimes(2);
  expect(onChange).not.toHaveBeenCalled();
  await user.click(within(dialog).getByRole("button", { name: "关闭字段说明" }));
  expect(screen.queryByTestId("training-block-inventory")).not.toBeInTheDocument();
  expect(input).toHaveAttribute("max", "34");
});
