import { cleanup, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { jsonResponse, renderInApp } from "../../test/renderInApp";
import { CaptionProviders } from "./CaptionProviders";
import type { CaptionProfile, Profiles } from "./api";

const profile = (overrides: Partial<CaptionProfile> = {}): CaptionProfile => ({
  id: "p1",
  name: "Studio",
  provider: "openai_compatible",
  kind: "remote",
  available: true,
  status: "ready",
  config: { base_url: "https://caption.example/v1", model: "vision-1" },
  api_key_hint: "已配置 API Key",
  api_key_configured: true,
  ...overrides,
});

const library = (profiles = [profile()], active = "p1"): Profiles => ({
  profiles,
  active_profile_id: active,
  provider_types: [
    { id: "openai_compatible", label: "OpenAI 兼容", kind: "remote" },
    { id: "wd14", label: "WD14", kind: "local" },
  ],
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

it("creates, edits and activates a profile through the rendered controls", async () => {
  const calls: { path: string; method: string; body?: unknown }[] = [];
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = new URL(String(input), window.location.origin).pathname;
    const method = init?.method || "GET";
    const body = init?.body ? JSON.parse(String(init.body)) : undefined;
    calls.push({ path, method, body });
    return jsonResponse(library([profile({ name: String(body?.name || "Studio") })], "p1"));
  });
  vi.stubGlobal("fetch", fetchMock);
  renderInApp(<CaptionProviders library={library([profile()], "other")} />);
  const user = userEvent.setup();

  await user.click(screen.getByRole("button", { name: "新建接入" }));
  const create = screen.getByRole("dialog", { name: "新建接入预设" });
  await user.type(within(create).getByLabelText("名称"), "New profile");
  await user.type(within(create).getByLabelText("API 地址"), "https://new.example/v1");
  await user.type(within(create).getByLabelText("模型名称"), "model-x");
  await user.click(within(create).getByRole("button", { name: "保存接入" }));
  await screen.findByRole("heading", { name: /接入预设/ });
  expect(calls[0]).toMatchObject({
    path: "/api/captioning/profiles",
    method: "POST",
    body: {
      name: "New profile",
      provider: "openai_compatible",
      config: { base_url: "https://new.example/v1", model: "model-x" },
    },
  });

  await user.click(screen.getByRole("button", { name: "编辑" }));
  const edit = screen.getByRole("dialog", { name: "编辑接入预设" });
  const name = within(edit).getByLabelText("名称");
  await user.clear(name);
  await user.type(name, "Studio edited");
  await user.click(within(edit).getByRole("button", { name: "保存接入" }));
  await screen.findByRole("heading", { name: /接入预设/ });
  expect(calls[1]).toMatchObject({
    path: "/api/captioning/profiles/p1",
    method: "PUT",
    body: { name: "Studio edited" },
  });

  await user.click(screen.getByRole("button", { name: "设为当前" }));
  expect(await screen.findByRole("status")).toHaveTextContent("接入预设已更新");
  expect(calls[2]).toMatchObject({ path: "/api/captioning/profiles/p1/activate", method: "POST" });
});

it("confirms deletion and does not send a request when declined", async () => {
  const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => jsonResponse(library([], "")));
  vi.stubGlobal("fetch", fetchMock);
  vi.spyOn(window, "confirm").mockReturnValue(false);
  renderInApp(<CaptionProviders library={library([profile()], "other")} />);
  const user = userEvent.setup();

  await user.click(screen.getByRole("button", { name: "删除" }));
  expect(window.confirm).toHaveBeenCalledWith("删除接入预设“Studio”及其凭据？");
  expect(fetchMock).not.toHaveBeenCalled();

  vi.mocked(window.confirm).mockReturnValue(true);
  await user.click(screen.getByRole("button", { name: "删除" }));
  await screen.findByRole("status");
  expect(fetchMock).toHaveBeenCalledTimes(1);
  expect(fetchMock.mock.calls[0][0]).toBe("/api/captioning/profiles/p1");
  expect(fetchMock.mock.calls[0][1]).toMatchObject({ method: "DELETE" });
});

it("keeps the edited draft after a failed save and never echoes or resubmits the stored secret", async () => {
  const serverSecret = "server-only-secret-must-not-leak";
  const submissions: unknown[] = [];
  const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
    const body = init?.body ? JSON.parse(String(init.body)) : undefined;
    submissions.push(body);
    return submissions.length === 1
      ? jsonResponse({ error: "保存暂不可用" }, 503)
      : jsonResponse(library([profile({ name: "Renamed" })]));
  });
  vi.stubGlobal("fetch", fetchMock);
  renderInApp(<CaptionProviders library={library([profile({ config: {
    base_url: "https://caption.example/v1", model: "vision-1", api_key: serverSecret,
  } })])} />);
  const user = userEvent.setup();

  await user.click(screen.getByRole("button", { name: "编辑" }));
  const dialog = screen.getByRole("dialog", { name: "编辑接入预设" });
  expect(within(dialog).getByLabelText("API Key （已配置）")).toHaveValue("");
  expect(dialog).not.toHaveTextContent(serverSecret);
  const name = within(dialog).getByLabelText("名称");
  await user.clear(name);
  await user.type(name, "Renamed");
  await user.click(within(dialog).getByRole("button", { name: "保存接入" }));
  expect(await within(dialog).findByRole("alert")).toHaveTextContent("保存暂不可用");
  expect(name).toHaveValue("Renamed");
  expect(JSON.stringify(submissions[0])).not.toContain(serverSecret);
  expect(submissions[0]).not.toHaveProperty("api_key");

  await user.click(within(dialog).getByRole("button", { name: "保存接入" }));
  await screen.findByRole("heading", { name: /接入预设/ });
  expect(JSON.stringify(submissions[1])).not.toContain(serverSecret);
  expect(submissions[1]).not.toHaveProperty("api_key");
});
