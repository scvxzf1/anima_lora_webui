import { cleanup, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { jsonResponse, renderInApp } from "../../test/renderInApp";
import { CaptionAssets } from "./CaptionAssets";

const assets = (download?: Record<string, unknown>) => ({
  assets: [{
    id: "wd14",
    label: "WD14",
    repo_id: "fixture/wd14",
    license: "Apache 2.0",
    state: download ? "downloading" : "missing",
    installed: false,
    total_size: 1000000,
    requires_auth: false,
    auth_configured: true,
    auth_hint: "",
    ...(download ? { download } : {}),
  }],
  downloads: download ? [download] : [],
});

const dictionary = {
  installed: false,
  state: "missing",
  source_name: "Fixture dictionary",
  entry_count: 0,
  download_size: 100000,
};

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

it("starts a model download and cancels an active download", async () => {
  const calls: { path: string; method: string }[] = [];
  let reads = 0;
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), window.location.origin);
    const method = init?.method || "GET";
    calls.push({ path: url.pathname, method });
    if (url.pathname === "/api/captioning/model-assets") {
      reads += 1;
      return jsonResponse(reads === 1 ? assets() : assets({
        id: "download-1", asset_id: "wd14", state: "running", bytes_downloaded: 10,
        total_bytes: 1000,
      }));
    }
    if (url.pathname === "/api/captioning/tag-dictionary") return jsonResponse(dictionary);
    if (url.pathname === "/api/captioning/model-assets/wd14/download") {
      return jsonResponse({ download: { id: "download-1", asset_id: "wd14", state: "queued", bytes_downloaded: 0, total_bytes: 1000 } });
    }
    if (url.pathname === "/api/captioning/downloads/download-1/cancel") {
      return jsonResponse({ download: { id: "download-1", asset_id: "wd14", state: "cancelled", bytes_downloaded: 10, total_bytes: 1000 } });
    }
    return jsonResponse({});
  }));
  vi.spyOn(window, "confirm").mockReturnValue(true);
  renderInApp(<CaptionAssets />);
  const user = userEvent.setup();

  await user.click(await screen.findByRole("button", { name: "下载模型" }));
  expect(calls).toContainEqual({ path: "/api/captioning/model-assets/wd14/download", method: "POST" });
  await user.click(await screen.findByRole("button", { name: "取消下载" }));
  expect(calls).toContainEqual({ path: "/api/captioning/downloads/download-1/cancel", method: "POST" });
});

it("shows a download command error without retrying or hiding the action", async () => {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = new URL(String(input), window.location.origin).pathname;
    if (path === "/api/captioning/model-assets") return jsonResponse(assets());
    if (path === "/api/captioning/tag-dictionary") return jsonResponse(dictionary);
    if (path.endsWith("/download") && init?.method === "POST")
      return jsonResponse({ error: "模型下载失败" }, 503);
    return jsonResponse({});
  });
  vi.stubGlobal("fetch", fetchMock);
  vi.spyOn(window, "confirm").mockReturnValue(true);
  renderInApp(<CaptionAssets />);
  const user = userEvent.setup();

  await user.click(await screen.findByRole("button", { name: "下载模型" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("模型下载失败");
  expect(fetchMock).toHaveBeenCalledTimes(3);
  expect(screen.getByRole("button", { name: "下载模型" })).toBeEnabled();
});
