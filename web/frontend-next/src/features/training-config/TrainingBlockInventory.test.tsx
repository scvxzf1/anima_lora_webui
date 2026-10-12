import { cleanup, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { jsonResponse, renderInApp } from "../../test/renderInApp";
import { TrainingBlockInventory } from "./TrainingBlockInventory";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it("reads the draft model and refreshes without writing configuration", async () => {
  const fetch = vi.fn(async () => jsonResponse({ block_count: 36, max_swap: 34 }));
  vi.stubGlobal("fetch", fetch);
  renderInApp(<TrainingBlockInventory draft={{ model_family: "krea2_raw", pretrained_model_name_or_path: "models/draft.safetensors", blocks_to_swap: 8 }} />);
  expect(await screen.findByText(/36 个 block · 可交换上限 34/)).toBeInTheDocument();
  expect(String(fetch.mock.calls[0])).toContain("models%2Fdraft.safetensors");
  await userEvent.setup().click(screen.getByRole("button", { name: "刷新主模型 block 数" }));
  expect(fetch).toHaveBeenCalledTimes(2);
});

it("keeps a missing model and header failure visible", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ error: "主模型路径不存在" }, 400)));
  renderInApp(<TrainingBlockInventory draft={{ pretrained_model_name_or_path: "missing.safetensors" }} />);
  expect(await screen.findByText(/无法读取主模型 block 数：主模型路径不存在/)).toBeInTheDocument();
});
