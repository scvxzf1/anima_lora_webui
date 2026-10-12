import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { TrainingFieldEditor } from "./TrainingFieldEditor";
import { fieldsForConfig } from "./fieldCatalog";
import { FIELD_HELP_SUMMARY_ZH } from "./domain/field-help-summary.js";
import { TrainingFieldHelpDialog } from "./TrainingFieldHelpDialog";

afterEach(cleanup);

it("shows the live state, advice and choices without legacy headings", async () => {
  const user = userEvent.setup();
  const field = fieldsForConfig({ qwen_text_encoder_cache_policy: "auto" }).find((item) => item.key === "qwen_text_encoder_cache_policy")!;
  const view = render(<TrainingFieldEditor fields={[field]} draft={{ qwen_text_encoder_cache_policy: "auto", model_family: "krea2_raw" }} ownKeys={new Set()} disabled={false} onChange={vi.fn()} />);
  const trigger = screen.getByRole("button", { name: "查看文本编码器缓存策略帮助" });
  await user.selectOptions(screen.getByLabelText("文本编码器缓存策略"), "cpu");
  view.rerender(<TrainingFieldEditor fields={[field]} draft={{ qwen_text_encoder_cache_policy: "cpu", model_family: "krea2_raw" }} ownKeys={new Set()} disabled={false} onChange={vi.fn()} />);
  await user.click(trigger);

  expect(await screen.findByRole("dialog")).toBeInTheDocument();
  expect(screen.getByText("当前值").parentElement).toHaveTextContent("仅 CPU");
  expect(screen.getByText(/与默认值不同/)).toHaveTextContent("自动（推荐）");
  expect(screen.getByLabelText("模型族 krea2_raw")).toBeInTheDocument();
  for (const heading of ["建议", "选项怎么选", "注意"]) {
    expect(screen.getByRole("heading", { name: heading })).toBeInTheDocument();
  }
  expect(screen.getByRole("table")).toHaveTextContent("cpu_offload");
  for (const heading of ["设置说明", "作用", "好处", "代价", "风险"]) {
    expect(screen.queryByRole("heading", { name: heading })).not.toBeInTheDocument();
  }
  expect(screen.queryByText("暂无补充说明。")).not.toBeInTheDocument();
});

it("uses catalog defaults for non-core training fields", async () => {
  const user = userEvent.setup();
  const field = fieldsForConfig({ lokr_grouped_delta_backend: "eager" })
    .find((item) => item.key === "lokr_grouped_delta_backend")!;
  render(<TrainingFieldEditor fields={[field]} draft={{ model_family: "anima", lora_adapter_kind: "lokr", lokr_grouped_delta_backend: "eager" }} ownKeys={new Set()} disabled={false} onChange={vi.fn()} />);
  await user.click(screen.getByRole("button", { name: "查看LoKr 分组 Delta 后端帮助" }));
  await screen.findByRole("dialog");
  expect(screen.getByText(/与默认值不同/)).toHaveTextContent("triton");
});

it("keeps the summary inline and loads the full help chapters when opened", async () => {
  const user = userEvent.setup();
  const field = fieldsForConfig({ network_dim: 16 }).find((item) => item.key === "network_dim")!;
  render(<TrainingFieldEditor fields={[field]} draft={{ model_family: "anima", network_dim: 16 }} ownKeys={new Set()} disabled={false} onChange={vi.fn()} />);

  expect(screen.getByText(FIELD_HELP_SUMMARY_ZH.network_dim)).toBeInTheDocument();
  expect(screen.queryByRole("heading", { name: "作用" })).not.toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "查看LoRA rank帮助" }));
  const dialog = await screen.findByRole("dialog");
  expect(within(dialog).getByText(FIELD_HELP_SUMMARY_ZH.network_dim)).toBeInTheDocument();
  expect(within(dialog).getByRole("heading", { name: "建议" })).toBeInTheDocument();
});

it("shows reviewed option labels, keeps advanced details collapsed, and only shows a different default", async () => {
  const user = userEvent.setup();
  const field = fieldsForConfig({ sample_vae_decode: "tiled" }).find((item) => item.key === "sample_vae_decode")!;
  const view = render(<TrainingFieldEditor fields={[field]} draft={{ sample_vae_decode: "tiled", model_family: "krea2_raw" }} ownKeys={new Set()} disabled={false} onChange={vi.fn()} />);
  await user.click(screen.getByRole("button", { name: "查看预览 VAE 解码帮助" }));
  const dialog = await screen.findByRole("dialog");
  expect(within(dialog).getByRole("table")).toHaveTextContent("分块 GPU 解码（推荐）");
  expect(within(dialog).queryByText(/与默认值不同/)).not.toBeInTheDocument();
  const details = dialog.querySelector("details")!;
  expect(details.open).toBe(false);
  await user.tab();
  expect(within(dialog).getByText("进阶")).toHaveFocus();
  await user.tab();
  expect(within(dialog).getByRole("button", { name: "关闭字段说明" })).toHaveFocus();
  await user.click(within(dialog).getByText("进阶"));
  expect(details.open).toBe(true);
  view.rerender(<TrainingFieldEditor fields={[field]} draft={{ sample_vae_decode: "cpu", model_family: "krea2_raw" }} ownKeys={new Set()} disabled={false} onChange={vi.fn()} />);
  expect(within(dialog).getByText(/与默认值不同/)).toHaveTextContent("分块 GPU 解码（推荐）");
});

it("does not repeat the model family and preserves unknown option values", () => {
  render(<TrainingFieldHelpDialog label="模型族" fieldKey="model_family" value="future_family" options={["anima", "krea2_raw"]} modelFamily="future_family" draft={{ model_family: "future_family" }} onClose={vi.fn()} />);
  expect(screen.getByText("future_family")).toBeInTheDocument();
  expect(screen.queryByLabelText("模型族 future_family")).not.toBeInTheDocument();
  const table = screen.getByRole("table");
  expect(within(table).queryByText("z_image")).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /恢复/ })).not.toBeInTheDocument();
});

it("explains why a disabled field is unavailable and closes with focus restoration", async () => {
  const user = userEvent.setup();
  const field = fieldsForConfig({ max_train_steps: 1600 }).find((item) => item.key === "max_train_steps")!;
  render(<TrainingFieldEditor fields={[field]} draft={{ model_family: "krea2_raw", max_train_epochs: 2, max_train_steps: 1600 }} ownKeys={new Set()} disabled={false} onChange={vi.fn()} />);
  const trigger = screen.getByRole("button", { name: "查看最大训练步数帮助" });
  await user.click(trigger);
  expect(screen.getByText("不可用原因").parentElement).toHaveTextContent("max_train_epochs 已设置");
  await user.click(screen.getByRole("button", { name: "关闭字段说明" }));
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(trigger).toHaveFocus();
});

it("shows a clear fallback for unknown fields and closes on Escape or backdrop click", async () => {
  const user = userEvent.setup();
  const field = { key: "future_field", label: "未来字段", kind: "text" as const, group: "training" as const };
  const view = render(<TrainingFieldEditor fields={[field]} draft={{ future_field: "safe <text>" }} ownKeys={new Set()} disabled={false} onChange={vi.fn()} />);
  await user.click(screen.getByRole("button", { name: "查看未来字段帮助" }));
  expect(screen.getByText(/此字段暂无详细帮助内容/)).toBeInTheDocument();
  expect(screen.getByText("safe <text>")).toBeInTheDocument();
  await user.keyboard("{Escape}");
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "查看未来字段帮助" }));
  await user.click(document.querySelector(".training-field-help-backdrop")!);
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  view.unmount();
});
