import { cleanup, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { jsonResponse, renderInApp } from "../../test/renderInApp";
import { CaptionReview } from "./CaptionReview";
import { captionImageUrl } from "./api";

describe("caption review write boundaries", () => {
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });
  it("saves candidates without committing TXT and exposes commit conflicts", async () => {
    const item = {
      id: "i1",
      name: "sample.png",
      file: "sample.png",
      state: "ready",
      caption: "original",
      proposed_caption: "candidate",
      url: "/api/config/dataset-presets/image?image=sample.png",
    };
    const job = {
      id: "j1",
      state: "done",
      profile_name: "Fixture",
      total: 1,
      completed: 1,
      failed: 0,
      items: [item],
    };
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        if (String(input).includes("/logs?"))
          return jsonResponse({ lines: [] });
        if (init?.method === "PATCH")
          item.proposed_caption = JSON.parse(
            String(init.body),
          ).proposed_caption;
        if (String(input).endsWith("/commit"))
          return jsonResponse({
            written: 0,
            conflicts: 1,
            skipped: 0,
            errors: [{ file: "sample.png", error: "TXT 已被其他程序修改" }],
            job,
          });
        return jsonResponse({ ok: true, job });
      },
    );
    vi.stubGlobal("fetch", fetchMock);
    vi.spyOn(window, "confirm").mockReturnValue(true);
    renderInApp(<CaptionReview jobId="j1" />);
    const user = userEvent.setup();
    const caption = await screen.findByLabelText("候选标注");
    await user.clear(caption);
    await user.type(caption, "edited");
    await user.click(screen.getByRole("button", { name: "保存候选" }));
    expect(
      fetchMock.mock.calls.some(([input]) => String(input).endsWith("/commit")),
    ).toBe(false);
    expect(
      fetchMock.mock.calls.find(([, init]) => init?.method === "PATCH")?.[1]
        ?.body,
    ).toBe('{"proposed_caption":"edited"}');
    await user.click(screen.getByRole("button", { name: "写回全部候选 (1)" }));
    expect(
      await screen.findByText("sample.png: TXT 已被其他程序修改"),
    ).toBeInTheDocument();
  });
  it("rejects external and script image URLs", () => {
    expect(captionImageUrl("https://example.com/image.png")).toBe("");
    expect(captionImageUrl("javascript:alert(1)")).toBe("");
    expect(
      captionImageUrl("/api/config/dataset-presets/image?image=a.png"),
    ).toContain("image=a.png");
  });
});
