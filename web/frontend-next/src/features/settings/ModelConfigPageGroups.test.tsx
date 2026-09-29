import { cleanup, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { jsonResponse, renderInApp } from "../../test/renderInApp";
import { ModelConfigPage } from "./ModelConfigPage";

const items = ["a", "b", "c", "d", "e"].map((id) => ({
  id,
  name: `Model ${id.toUpperCase()}`,
  model_family: "anima",
  pretrained_model_name_or_path: `dit-${id}`,
  qwen3: `text-${id}`,
  vae: `vae-${id}`,
}));

const groupedConfig = {
  revision: "revision-17",
  default_id: "a",
  groups: [
    { id: "keep-first", label: "保留一", item_ids: ["a", "b"] },
    { id: "remove", label: "删除目标", item_ids: ["c", "d"] },
    { id: "keep-last", label: "保留二", item_ids: ["e"] },
  ],
  items,
};

function installFetch(config: typeof groupedConfig) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.includes("model-families")) return jsonResponse({ items: [] });
    if (url.includes("/api/settings/model-configs") && init?.method === "PUT") {
      return jsonResponse(JSON.parse(String(init.body)));
    }
    return jsonResponse(config);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("model config group deletion", () => {
  it("leaves the draft unchanged when deletion is cancelled and does not save", async () => {
    const fetchMock = installFetch(groupedConfig);
    vi.stubGlobal("confirm", vi.fn(() => false));
    const user = userEvent.setup();
    renderInApp(<ModelConfigPage />);

    await screen.findByRole("button", { name: /Model A/ });
    await user.click(screen.getByRole("button", { name: "删除分组 删除目标" }));

    expect(screen.getByRole("button", { name: "删除目标" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "保存模型配置" })).toBeDisabled();
    expect(fetchMock.mock.calls.some(([, init]) => init?.method === "PUT")).toBe(false);
  });

  it("moves deleted members to the first remaining group in order and saves the full revisioned payload", async () => {
    const fetchMock = installFetch(groupedConfig);
    vi.stubGlobal("confirm", vi.fn(() => true));
    const user = userEvent.setup();
    renderInApp(<ModelConfigPage />);

    await screen.findByRole("button", { name: /Model A/ });
    await user.click(screen.getByRole("button", { name: "删除分组 删除目标" }));
    expect(screen.queryByRole("button", { name: "删除目标" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "保存模型配置" }));

    const put = fetchMock.mock.calls.find(([, init]) => init?.method === "PUT");
    expect(put).toBeDefined();
    const payload = JSON.parse(String(put?.[1]?.body));
    expect(payload.revision).toBe("revision-17");
    expect(payload.groups).toEqual([
      { id: "keep-first", label: "保留一", item_ids: ["a", "b", "c", "d"] },
      { id: "keep-last", label: "保留二", item_ids: ["e"] },
    ]);
    const members = payload.groups.flatMap((group: { item_ids: string[] }) => group.item_ids);
    expect(members).toEqual(["a", "b", "c", "d", "e"]);
    expect(new Set(members).size).toBe(members.length);
    expect(payload.items.map((item: { id: string }) => item.id)).toEqual(["a", "b", "c", "d", "e"]);
  });

  it("disables deletion when only one group remains", async () => {
    const onlyGroup = {
      ...groupedConfig,
      groups: [{ id: "only", label: "唯一分组", item_ids: ["a", "b", "c", "d", "e"] }],
    };
    installFetch(onlyGroup);
    const user = userEvent.setup();
    renderInApp(<ModelConfigPage />);

    const deleteButton = await screen.findByRole("button", { name: "删除分组 唯一分组" });
    expect(deleteButton).toBeDisabled();
    await user.click(deleteButton);
    expect(screen.getByRole("button", { name: "唯一分组" })).toBeInTheDocument();
  });
});
