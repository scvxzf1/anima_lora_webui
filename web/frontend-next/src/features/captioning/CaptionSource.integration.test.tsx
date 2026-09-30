import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter, RouterProvider } from "react-router-dom";
import { afterEach, expect, it, vi } from "vitest";
import { CaptioningPage } from "./CaptioningPage";
import { clearCaptionDrafts } from "./captionDraft";

const datasetA = "configs/datasets/draft-a.toml";
const datasetB = "configs/datasets/draft-b.toml";
const profiles = {
  active_profile_id: "profile-a",
  profiles: [
    { id: "profile-a", name: "Studio A", kind: "external", provider: "fixture", status: "可用", available: true, config: {}, api_key_hint: "", api_key_configured: false },
    { id: "profile-b", name: "Studio B", kind: "external", provider: "fixture", status: "可用", available: true, config: {}, api_key_hint: "", api_key_configured: false },
  ],
  provider_types: [],
};

function installApi(profileData = profiles) {
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = new URL(String(input), window.location.origin);
    if (url.pathname === "/api/captioning/profiles") return Response.json(profileData);
    if (url.pathname === "/api/captioning/jobs") return Response.json({ jobs: [] });
    if (url.pathname === "/api/captioning/model-assets") return Response.json({ assets: [], downloads: [] });
    if (url.pathname === "/api/captioning/tag-dictionary") return Response.json({ installed: false, state: "missing", source_name: "", entry_count: 0, download_size: 0 });
    if (url.pathname === "/api/captioning/prompt-presets") return Response.json({ presets: [{ id: "fixture-prompt", name: "Fixture prompt", system_prompt: "system base", user_prompt: "user base", builtin: true }] });
    if (url.pathname === "/api/config/dataset-presets") return Response.json({ ok: true, presets: [datasetA, datasetB].map((path) => ({ path, filename: path, label: path })), groups: [] });
    if (url.pathname === "/api/config/dataset-presets/read") return Response.json({
      ok: true,
      file: url.searchParams.get("file"),
      name: "fixture",
      content: "",
      datasets: [{ source_dir: "images/one", image_dir: "cache/one" }, { source_dir: "images/two", image_dir: "cache/two" }],
      defaults: {},
      readonly: false,
      summary: {},
    });
    if (url.pathname === "/api/config/dataset-presets/images") {
      const source = url.searchParams.get("source");
      const dataset = url.searchParams.get("file");
      const subset = url.searchParams.get("dataset_index");
      const file = `${dataset === datasetA ? "a" : "b"}-${subset}-${source}.png`;
      return Response.json({ ok: true, total: 1, images: [{ file, name: file, url: "/api/config/dataset-presets/image?fixture=1" }] });
    }
    return Response.json({ presets: [] });
  }));
}

function renderPage(start = `/captioning?dataset=${encodeURIComponent(datasetA)}&subset=0`) {
  const router = createMemoryRouter([{ path: "/captioning/*", element: <CaptioningPage /> }], { initialEntries: [start] });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);
  return { router, client };
}

function imageCheckbox(name: string): HTMLInputElement {
  const checkbox = screen.getByRole("img", { name }).parentElement?.querySelector<HTMLInputElement>("input[type=checkbox]");
  if (!checkbox) throw new Error(`Missing selection checkbox for ${name}`);
  return checkbox;
}

afterEach(() => {
  cleanup();
  clearCaptionDrafts();
  vi.unstubAllGlobals();
});

it("restores real source draft after visiting providers and prompts", async () => {
  installApi();
  const user = userEvent.setup();
  const { router } = renderPage();

  await user.selectOptions(await screen.findByRole("combobox", { name: "目录类型" }), "training");
  await user.selectOptions(screen.getByRole("combobox", { name: "接入预设" }), "profile-b");
  await user.selectOptions(screen.getByRole("combobox", { name: "提示词预设" }), "fixture-prompt");
  await user.clear(screen.getByRole("textbox", { name: "用户提示词" }));
  await user.type(screen.getByRole("textbox", { name: "用户提示词" }), "draft user prompt");
  await user.clear(screen.getByRole("textbox", { name: "系统提示词" }));
  await user.type(screen.getByRole("textbox", { name: "系统提示词" }), "draft system prompt");
  await user.click(screen.getByRole("button", { name: "扫描图片" }));
  await screen.findByRole("img", { name: "a-0-training.png" });
  const imageChoice = imageCheckbox("a-0-training.png");
  await user.click(imageChoice);

  await user.click(screen.getByRole("link", { name: "接入预设" }));
  await screen.findByRole("heading", { name: "Studio B" });
  await user.click(screen.getByRole("link", { name: "提示词预设" }));
  await screen.findByRole("heading", { name: "提示词预设" });
  await user.click(screen.getByRole("link", { name: "本地资源" }));
  await screen.findByRole("heading", { name: "本地资源" });
  await user.click(screen.getByRole("link", { name: "图片与任务" }));

  expect(await screen.findByRole("combobox", { name: "目录类型" })).toHaveValue("training");
  expect(screen.getByRole("combobox", { name: "接入预设" })).toHaveValue("profile-b");
  expect(screen.getByRole("combobox", { name: "提示词预设" })).toHaveValue("");
  expect(screen.getByRole("textbox", { name: "用户提示词" })).toHaveValue("draft user prompt");
  expect(screen.getByRole("textbox", { name: "系统提示词" })).toHaveValue("draft system prompt");
  await screen.findByRole("img", { name: "a-0-training.png" });
  expect(imageCheckbox("a-0-training.png")).toBeChecked();
});

it("keeps dataset and subset image selections isolated", async () => {
  installApi();
  const user = userEvent.setup();
  renderPage();
  await user.click(await screen.findByRole("button", { name: "扫描图片" }));
  await screen.findByRole("img", { name: "a-0-source.png" });
  await user.click(imageCheckbox("a-0-source.png"));

  await user.selectOptions(screen.getByRole("combobox", { name: "图片组" }), "1");
  await user.click(await screen.findByRole("button", { name: "扫描图片" }));
  await screen.findByRole("img", { name: "a-1-source.png" });
  expect(imageCheckbox("a-1-source.png")).not.toBeChecked();

  await user.selectOptions(screen.getByRole("combobox", { name: "数据集预设" }), datasetB);
  await user.click(await screen.findByRole("button", { name: "扫描图片" }));
  await screen.findByRole("img", { name: "b-0-source.png" });
  expect(imageCheckbox("b-0-source.png")).not.toBeChecked();
  await waitFor(() => expect(screen.getByRole("button", { name: "开始打标 (0)" })).toBeDisabled());
  await user.selectOptions(screen.getByRole("combobox", { name: "数据集预设" }), datasetA);
  await screen.findByRole("img", { name: "a-0-source.png" });
  expect(imageCheckbox("a-0-source.png")).toBeChecked();
});

it("restores an unchanged prompt preset with its selected identifier", async () => {
  installApi();
  const user = userEvent.setup();
  renderPage();
  await screen.findByRole("option", { name: "Fixture prompt" });
  await user.selectOptions(await screen.findByRole("combobox", { name: "提示词预设" }), "fixture-prompt");
  await user.click(screen.getByRole("link", { name: "接入预设" }));
  await screen.findByRole("heading", { name: "接入预设" });
  await user.click(screen.getByRole("link", { name: "图片与任务" }));
  expect(await screen.findByRole("combobox", { name: "提示词预设" })).toHaveValue("fixture-prompt");
});

it("fails closed when a restored provider profile is no longer available", async () => {
  installApi();
  const user = userEvent.setup();
  const { client } = renderPage();
  await user.selectOptions(await screen.findByRole("combobox", { name: "目录类型" }), "training");
  await user.selectOptions(screen.getByRole("combobox", { name: "接入预设" }), "profile-b");
  client.setQueryData(["captioning", "profiles"], {
    ...profiles,
    active_profile_id: "profile-a",
    profiles: [profiles.profiles[0]],
  });

  await waitFor(() => expect(screen.getByRole("combobox", { name: "接入预设" })).toHaveValue(""));
  expect(screen.getByRole("button", { name: /开始打标/ })).toBeDisabled();
});

it("switches draft isolation when SPA query scope changes and restores the original scope", async () => {
  installApi();
  const user = userEvent.setup();
  const { router } = renderPage();
  await user.selectOptions(await screen.findByRole("combobox", { name: "目录类型" }), "training");
  await user.clear(screen.getByRole("textbox", { name: "用户提示词" }));
  await user.type(screen.getByRole("textbox", { name: "用户提示词" }), "scope A prompt");
  await user.click(screen.getByRole("button", { name: "扫描图片" }));
  await screen.findByRole("img", { name: "a-0-training.png" });
  await user.click(imageCheckbox("a-0-training.png"));

  await router.navigate(`/captioning?dataset=${encodeURIComponent(datasetB)}&subset=0&source=training`);
  await waitFor(() => expect(screen.getByRole("textbox", { name: "用户提示词" })).toHaveValue(""));
  await user.type(screen.getByRole("textbox", { name: "用户提示词" }), "scope B prompt");

  await router.navigate(`/captioning?dataset=${encodeURIComponent(datasetA)}&subset=0&source=training`);
  await waitFor(() => expect(screen.getByRole("textbox", { name: "用户提示词" })).toHaveValue("scope A prompt"));
  await screen.findByRole("img", { name: "a-0-training.png" });
  expect(imageCheckbox("a-0-training.png")).toBeChecked();
});
