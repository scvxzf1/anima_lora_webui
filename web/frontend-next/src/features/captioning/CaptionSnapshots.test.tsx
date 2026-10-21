import { cleanup, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { jsonResponse, renderInApp } from "../../test/renderInApp";
import { CaptionSnapshots } from "./CaptionSnapshots";

const snapshot = {
  id: "20261021-101530-a1b2c3",
  created_at: "2026-10-21T10:15:30+08:00",
  created_at_ms: 1761006530000,
  reason: "commit",
  label: "",
  dataset_file: "configs/datasets/example.toml",
  dataset_index: 0,
  source: "source",
  scope_dir: "/tmp/dataset",
  file_count: 2,
  bytes: 2048,
};

function stubFetch(calls: string[]) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    calls.push(`${init?.method || "GET"} ${url}`);
    if (url.includes("/restore")) return jsonResponse({ ok: true, id: snapshot.id, written: 2, removed: 1, skipped: [] });
    if (init?.method === "DELETE") return jsonResponse({ ok: true, deleted: snapshot.id });
    if (init?.method === "POST") return jsonResponse({ ok: true, snapshot });
    return jsonResponse({ ok: true, snapshots: [snapshot] });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("caption restore points", () => {
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("lists restore points for the dataset scope and restores after confirmation", async () => {
    const calls: string[] = [];
    stubFetch(calls);
    const restored = vi.fn();
    renderInApp(
      <CaptionSnapshots
        datasetFile="configs/datasets/example.toml"
        datasetIndex={0}
        onRestored={restored}
      />,
    );
    const user = userEvent.setup();
    await user.click(await screen.findByText(/还原点（1）/));
    expect(await screen.findByText(/写回前自动 · 2 个文件/)).toBeTruthy();
    expect(calls[0]).toContain("dataset_file=configs%2Fdatasets%2Fexample.toml");

    vi.spyOn(window, "confirm").mockReturnValue(false);
    await user.click(screen.getByRole("button", { name: "还原" }));
    expect(calls.some((entry) => entry.includes("/restore"))).toBe(false);

    vi.spyOn(window, "confirm").mockReturnValue(true);
    await user.click(screen.getByRole("button", { name: "还原" }));
    expect(await screen.findByRole("status")).toHaveTextContent("已还原 2 个 caption");
    expect(calls.some((entry) => entry === `POST /api/captioning/caption-snapshots/${snapshot.id}/restore`)).toBe(true);
    expect(restored).toHaveBeenCalled();
  });

  it("creates and deletes restore points through the aliased API", async () => {
    const calls: string[] = [];
    stubFetch(calls);
    renderInApp(
      <CaptionSnapshots datasetFile="configs/datasets/example.toml" datasetIndex={1} />,
    );
    const user = userEvent.setup();
    await user.click(await screen.findByText(/还原点（1）/));
    await user.click(screen.getByRole("button", { name: "立即备份" }));
    expect(await screen.findByRole("status")).toHaveTextContent("已创建还原点");

    vi.spyOn(window, "confirm").mockReturnValue(true);
    await user.click(screen.getByRole("button", { name: "删除" }));
    expect(await screen.findByRole("status")).toHaveTextContent("已删除还原点");
    expect(calls.some((entry) => entry === `DELETE /api/captioning/caption-snapshots/${snapshot.id}`)).toBe(true);
  });

  it("surfaces a failed snapshot list and offers an explicit retry", async () => {
    let attempts = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
        if (init?.method === "POST") return jsonResponse({ ok: true, snapshot });
        attempts += 1;
        if (attempts === 1)
          return jsonResponse({ ok: false, error: "打标服务尚未初始化" }, 503);
        return jsonResponse({ ok: true, snapshots: [snapshot] });
      }),
    );
    renderInApp(
      <CaptionSnapshots datasetFile="configs/datasets/example.toml" datasetIndex={0} />,
    );
    const user = userEvent.setup();
    await user.click(await screen.findByText(/还原点（0）/));
    expect(await screen.findByText(/还原点列表加载失败/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "立即备份" })).not.toBeDisabled();

    await user.click(screen.getByRole("button", { name: "重试加载" }));
    expect(await screen.findByText(/还原点（1）/)).toBeInTheDocument();
    expect(screen.queryByText(/还原点列表加载失败/)).toBeNull();
  });

  it("does not promise an unconditional restore point for every write-back", async () => {
    const calls: string[] = [];
    stubFetch(calls);
    renderInApp(
      <CaptionSnapshots datasetFile="configs/datasets/example.toml" datasetIndex={0} />,
    );
    const user = userEvent.setup();
    await user.click(await screen.findByText(/还原点（1）/));
    expect(await screen.findByText(/可能没有还原点/)).toBeInTheDocument();
  });

  it("renders nothing without a dataset scope", () => {
    stubFetch([]);
    const { container } = renderInApp(<CaptionSnapshots />);
    expect(container.querySelector(".caption-snapshots")).toBeNull();
  });
});
