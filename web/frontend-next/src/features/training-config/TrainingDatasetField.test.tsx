import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, expect, it, vi } from "vitest";
import { TrainingDatasetField } from "./TrainingDatasetField";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});
const file = "configs/datasets/portrait.toml";
function setup({ readonly = false, fail = false, disabled = false } = {}) {
  const item = { path: file, label: "人像蓝图", summary: { dataset_count: 2 } };
  const preset = {
    ok: true,
    file,
    name: "portrait",
    content: "",
    readonly,
    defaults: { resolution: 1024, custom: "keep" },
    datasets: [
      {
        source_dir: "images/a",
        image_dir: "cache/a",
        num_repeats: 2,
        settings: { keep_tokens: 1 },
      },
      {
        source_dir: "images/b",
        image_dir: "cache/b",
        num_repeats: 3,
        is_reg: true,
      },
    ],
    summary: { dataset_count: 2 },
    stage_schedule_enabled: false,
    stage_schedule: [],
  };
  const writes: Record<string, unknown>[] = [];
  const fetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), "http://localhost");
    let payload: unknown = {};
    if (init?.method === "PUT") {
      const body = JSON.parse(String(init.body));
      writes.push(body);
      return new Response(
        JSON.stringify(fail ? { error: "保存失败" } : { ...preset, ...body }),
        {
          status: fail ? 500 : 200,
          headers: { "Content-Type": "application/json" },
        },
      );
    }
    if (url.pathname.endsWith("/dataset-presets"))
      payload = {
        ok: true,
        presets: [item],
        groups: [{ id: "a", label: "人像", files: [item] }],
      };
    if (url.pathname.endsWith("/read")) payload = preset;
    if (url.pathname.endsWith("/images"))
      payload = {
        total: url.searchParams.get("dataset_index") === "0" ? 24 : 12,
        count: 0,
        images: [],
      };
    return new Response(JSON.stringify(payload), {
      headers: { "Content-Type": "application/json" },
    });
  });
  vi.stubGlobal("fetch", fetch);
  const onChange = vi.fn();
  render(
    <QueryClientProvider
      client={
        new QueryClient({
          defaultOptions: {
            queries: { retry: false },
            mutations: { retry: false },
          },
        })
      }
    >
      <MemoryRouter>
        <TrainingDatasetField
          value={file}
          disabled={disabled}
          own
          onChange={onChange}
        />
      </MemoryRouter>
    </QueryClientProvider>,
  );
  return { onChange, fetch, writes, user: userEvent.setup() };
}

it("loads only after opening and applies a selection only on confirmation", async () => {
  const { user, fetch, writes, onChange } = setup();
  expect(fetch).not.toHaveBeenCalled();
  await user.click(screen.getByRole("button", { name: "选择与配置数据集" }));
  expect(await screen.findByLabelText("子集 2 重复次数")).toHaveValue(3);
  expect(await screen.findByText("24 张")).toBeInTheDocument();
  expect(await screen.findByText("12 张")).toBeInTheDocument();
  expect(onChange).not.toHaveBeenCalled();
  await user.click(screen.getByRole("button", { name: "使用此数据集" }));
  expect(onChange).toHaveBeenCalledWith(file);
  expect(writes).toEqual([]);
});

it("preserves all subsets and unknown settings when saving repeats", async () => {
  const { user, writes, onChange } = setup();
  vi.spyOn(window, "confirm").mockReturnValue(true);
  await user.click(screen.getByRole("button", { name: "选择与配置数据集" }));
  const repeat = await screen.findByLabelText("子集 2 重复次数");
  await user.clear(repeat);
  await user.type(repeat, "5");
  expect(screen.getByRole("button", { name: "使用此数据集" })).toBeDisabled();
  await user.click(screen.getByRole("button", { name: "保存蓝图参数" }));
  await screen.findByText("蓝图参数已保存");
  expect(writes[0]).toMatchObject({
    file,
    overwrite: true,
    defaults: { custom: "keep" },
    datasets: [
      { num_repeats: 2, settings: { keep_tokens: 1 } },
      { num_repeats: 5, is_reg: true },
    ],
  });
  expect(onChange).not.toHaveBeenCalled();
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "使用此数据集" })).toBeEnabled(),
  );
});

it("retains drafts after save failure and guards closing", async () => {
  const { user, onChange } = setup({ fail: true });
  const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
  await user.click(screen.getByRole("button", { name: "选择与配置数据集" }));
  const repeat = await screen.findByLabelText("子集 1 重复次数");
  await user.clear(repeat);
  await user.type(repeat, "7");
  await user.click(screen.getByRole("button", { name: "保存蓝图参数" }));
  await screen.findByText("保存失败");
  expect(repeat).toHaveValue(7);
  confirm.mockReturnValue(false);
  await user.keyboard("{Escape}");
  expect(screen.getByRole("dialog")).toBeInTheDocument();
  expect(onChange).not.toHaveBeenCalled();
});

it("supports adding and removing subsets without writing until save", async () => {
  const { user, writes } = setup();
  vi.spyOn(window, "confirm").mockReturnValue(true);
  await user.click(screen.getByRole("button", { name: "选择与配置数据集" }));
  await screen.findByLabelText("子集 2 重复次数");
  await user.click(screen.getByRole("button", { name: "添加子集" }));
  expect(screen.getByRole("button", { name: "保存蓝图参数" })).toBeDisabled();
  await user.type(screen.getByLabelText("子集 3 原始图片目录"), "images/c");
  expect(screen.getByRole("button", { name: "保存蓝图参数" })).toBeEnabled();
  await user.click(screen.getByRole("button", { name: "移除子集 3" }));
  expect(screen.queryByLabelText("子集 3 重复次数")).not.toBeInTheDocument();
  expect(writes).toEqual([]);
});

it("allows selecting readonly blueprints but prevents mutation", async () => {
  const { user } = setup({ readonly: true });
  await user.click(screen.getByRole("button", { name: "选择与配置数据集" }));
  expect(await screen.findByLabelText("子集 1 重复次数")).toBeDisabled();
  expect(screen.getByRole("button", { name: "添加子集" })).toBeDisabled();
  expect(screen.getByRole("button", { name: "使用此数据集" })).toBeEnabled();
});

it("disables opening and reference edits when the training form is busy", () => {
  setup({ disabled: true });
  expect(
    screen.getByRole("button", { name: "选择与配置数据集" }),
  ).toBeDisabled();
  expect(screen.getByLabelText("数据集配置")).toBeDisabled();
});
