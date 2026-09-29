import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { TrainingFieldEditor } from "./TrainingFieldEditor";
import { fieldsForConfig } from "./fieldCatalog";
import { FIELD_HELP_SUMMARY_ZH } from "./domain/field-help-summary.js";

afterEach(cleanup);

it("shows every help chapter with the live value, default and model family", async () => {
  const user = userEvent.setup();
  const field = fieldsForConfig({ qwen_text_encoder_cache_policy: "auto" }).find((item) => item.key === "qwen_text_encoder_cache_policy")!;
  const view = render(<TrainingFieldEditor fields={[field]} draft={{ qwen_text_encoder_cache_policy: "auto", model_family: "krea2_raw" }} ownKeys={new Set()} disabled={false} onChange={vi.fn()} />);
  const trigger = screen.getByRole("button", { name: "查看文本编码器缓存策略帮助" });
  await user.selectOptions(screen.getByLabelText("文本编码器缓存策略"), "cpu");
  view.rerender(<TrainingFieldEditor fields={[field]} draft={{ qwen_text_encoder_cache_policy: "cpu", model_family: "krea2_raw" }} ownKeys={new Set()} disabled={false} onChange={vi.fn()} />);
  await user.click(trigger);

  expect(await screen.findByRole("dialog")).toBeInTheDocument();
  expect(screen.getByText("当前值").parentElement).toHaveTextContent("cpu");
  expect(screen.getByText("页面默认值").parentElement).toHaveTextContent("auto");
  expect(screen.getByText("模型族").parentElement).toHaveTextContent("krea2_raw");
  for (const heading of ["摘要", "新手建议", "为什么通常这样设", "好处", "代价", "风险", "补充说明"]) {
    expect(screen.getByRole("heading", { name: heading })).toBeInTheDocument();
  }
  expect(screen.getAllByText(/仅影响 Qwen3-VL/).length).toBeGreaterThan(0);
  expect(screen.getByRole("heading", { name: "新手建议" }).nextElementSibling).toHaveTextContent("一般保持自动");
  expect(screen.getByRole("heading", { name: "为什么通常这样设" }).nextElementSibling).toHaveTextContent("cpu_offload");
});

it("uses catalog defaults for non-core training fields", async () => {
  const user = userEvent.setup();
  const field = fieldsForConfig({ lokr_grouped_delta_backend: "eager" })
    .find((item) => item.key === "lokr_grouped_delta_backend")!;
  render(<TrainingFieldEditor fields={[field]} draft={{ model_family: "anima", lora_adapter_kind: "lokr", lokr_grouped_delta_backend: "eager" }} ownKeys={new Set()} disabled={false} onChange={vi.fn()} />);
  await user.click(screen.getByRole("button", { name: "查看LoKr 分组 Delta 后端帮助" }));
  await screen.findByRole("dialog");
  expect(screen.getByText("页面默认值").parentElement).toHaveTextContent("triton");
});

it("keeps the summary inline and loads the full help chapters when opened", async () => {
  const user = userEvent.setup();
  const field = fieldsForConfig({ network_dim: 16 }).find((item) => item.key === "network_dim")!;
  render(<TrainingFieldEditor fields={[field]} draft={{ model_family: "anima", network_dim: 16 }} ownKeys={new Set()} disabled={false} onChange={vi.fn()} />);

  expect(screen.getByText(FIELD_HELP_SUMMARY_ZH.network_dim)).toBeInTheDocument();
  expect(screen.queryByRole("heading", { name: "好处" })).not.toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "查看LoRA rank帮助" }));
  const dialog = await screen.findByRole("dialog");
  expect(within(dialog).getByText(FIELD_HELP_SUMMARY_ZH.network_dim)).toBeInTheDocument();
  expect(within(dialog).getByRole("heading", { name: "好处" })).toBeInTheDocument();
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
  expect(screen.getByText("此字段暂无详细帮助内容。请保持当前值，除非你已了解该参数的作用；可在高级配置或项目文档中核对后再修改。")).toBeInTheDocument();
  expect(screen.getByText("safe <text>")).toBeInTheDocument();
  await user.keyboard("{Escape}");
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "查看未来字段帮助" }));
  await user.click(document.querySelector(".training-field-help-backdrop")!);
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  view.unmount();
});
