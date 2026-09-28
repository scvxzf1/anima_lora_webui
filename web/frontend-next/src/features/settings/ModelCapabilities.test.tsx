import { cleanup, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { jsonResponse, renderInApp } from "../../test/renderInApp";
import { capabilityLabels, editCapabilityIssue, findModelCapability } from "../../api/modelCapabilities";
import { ModelConfigPage } from "./ModelConfigPage";

const items = [
  { name: "anima", display_name: "Anima", aliases: ["anima"], supported_tasks: ["t2i"] },
  { name: "qwen_image_2_1", display_name: "Qwen Image 2.1", aliases: ["qwen21"], supported_tasks: ["t2i", "edit"], plain_lora_only: true },
];
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe("backend-derived model capabilities", () => {
  it("normalizes aliases and fails closed for old or missing catalogs", () => {
    expect(findModelCapability(items, "qwen21")).toBe(items[1]);
    expect(editCapabilityIssue(items[1])).toBeNull();
    expect(editCapabilityIssue(items[0])).toContain("Anima");
    expect(editCapabilityIssue(undefined)).toContain("不可用");
    expect(capabilityLabels({ name: "old", display_name: "Old", aliases: [] })).toEqual(["能力未知"]);
  });

  it("refreshes read-only labels when family changes and filters model rows", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, _options?: RequestInit) => {
      if (String(input).includes("model-families")) return jsonResponse({ items });
      return jsonResponse({ revision: "r1", default_id: "a", items: [
        { id: "a", name: "Model A", model_family: "anima", pretrained_model_name_or_path: "dit", qwen3: "te", vae: "vae" },
        { id: "b", name: "Model B", model_family: "qwen_image_2_1", pretrained_model_name_or_path: "dit", qwen3: "te", vae: "vae" },
      ] });
    });
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    renderInApp(<ModelConfigPage />);
    const tags = await screen.findByLabelText("模型训练能力");
    expect(within(tags).getByText("普通文生图")).toBeInTheDocument();
    expect(within(tags).queryByText("编辑训练")).not.toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("模型族"), "qwen_image_2_1");
    expect(within(tags).getByText("编辑训练")).toBeInTheDocument();
    expect(within(tags).getByText("仅支持 LoRA")).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("模型族"), "anima");
    await user.selectOptions(screen.getByLabelText("筛选训练能力"), "edit");
    expect(screen.queryByRole("button", { name: /Model A/ })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Model B/ })).toBeInTheDocument();
    expect(fetchMock.mock.calls.every(([, options]) => !options?.method || options.method === "GET")).toBe(true);
  });

  it("keeps model rows visible when capability filtering cannot be verified", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      if (String(input).includes("model-families")) throw new Error("offline");
      return jsonResponse({ revision: "r1", default_id: "a", items: [
        { id: "a", name: "Model A", model_family: "anima", pretrained_model_name_or_path: "dit", qwen3: "te", vae: "vae" },
        { id: "b", name: "Model B", model_family: "qwen_image_2_1", pretrained_model_name_or_path: "dit", qwen3: "te", vae: "vae" },
      ] });
    });
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    renderInApp(<ModelConfigPage />);
    await screen.findByRole("button", { name: /Model A/ });
    await user.selectOptions(screen.getByLabelText("筛选训练能力"), "edit");
    expect(await screen.findByText("模型能力读取失败，暂不应用能力筛选，已保留全部模型配置。")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Model A/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Model B/ })).toBeInTheDocument();
  });
});
