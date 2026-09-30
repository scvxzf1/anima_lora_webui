import { cleanup, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { jsonResponse, renderInApp } from "../../test/renderInApp";
import { CaptionProfileEditor } from "./CaptionProfileEditor";
import type { CaptionProfile, ProviderType } from "./api";

const profile: CaptionProfile = {
  id: "local-1", name: "Local", provider: "cltagger", kind: "local", available: true,
  status: "ready", config: { asset_id: "fixture", device: "cpu" }, api_key_hint: "", api_key_configured: false,
};
const types: ProviderType[] = [{ id: "cltagger", label: "CLTagger", kind: "local" }];

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

it("saves local category label toggles as provider config", async () => {
  let submitted: Record<string, unknown> | undefined;
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), window.location.origin);
    if (url.pathname === "/api/captioning/profiles/local-1" && init?.method === "PUT") {
      submitted = JSON.parse(String(init.body));
      return jsonResponse({ profiles: [profile], active_profile_id: "local-1", provider_types: types });
    }
    return jsonResponse({ gpus: [], stale: false });
  }));
  renderInApp(<CaptionProfileEditor profile={profile} types={types} onClose={vi.fn()} />);
  const user = userEvent.setup();

  for (const label of ["作品标签", "画师标签", "元数据标签", "模型标签", "评级标签", "质量标签"]) {
    expect(screen.getByRole("checkbox", { name: label })).toBeInTheDocument();
  }
  await user.click(screen.getByRole("checkbox", { name: "作品标签" }));
  for (const label of ["画师标签", "元数据标签", "模型标签", "评级标签", "质量标签"])
    await user.click(screen.getByRole("checkbox", { name: label }));
  await user.click(screen.getByRole("button", { name: "保存接入" }));

  expect(submitted).toMatchObject({
    provider: "cltagger",
    config: {
      asset_id: "fixture", device: "cpu", add_copyright_tag: false,
      add_artist_tag: true, add_meta_tag: true, add_model_tag: true,
      add_rating_tag: true, add_quality_tag: true,
    },
  });
});
