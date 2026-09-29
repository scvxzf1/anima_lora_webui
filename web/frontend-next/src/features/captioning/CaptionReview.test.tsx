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
      proposed_caption: "1girl, blue_hair\nsolo",
      url: "/api/config/dataset-presets/image?image=sample.png",
    };
    const job = {
      id: "j1",
      state: "done",
      profile_name: "Fixture",
      settings: { provider: "cltagger" },
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
    await screen.findByLabelText("候选标注");
    await user.click(screen.getByRole("button", { name: "标签" }));
    await screen.findByRole("button", { name: "上移 solo" });
    await user.click(screen.getByRole("button", { name: "上移 solo" }));
    await user.click(screen.getByRole("button", { name: "上移 solo" }));
    expect(
      fetchMock.mock.calls.some(([, init]) => init?.method === "PATCH"),
    ).toBe(false);
    await user.click(screen.getByRole("button", { name: "保存候选" }));
    expect(
      fetchMock.mock.calls.filter(([, init]) => init?.method === "PATCH").at(-1)?.[1]
        ?.body,
    ).toBe('{"proposed_caption":"solo, 1girl, blue_hair"}');
    await user.click(screen.getByRole("button", { name: "原始文本" }));
    const caption = screen.getByLabelText("候选标注");
    await user.clear(caption);
    await user.type(caption, "solo, edited");
    await user.click(screen.getByRole("button", { name: "保存候选" }));
    expect(
      fetchMock.mock.calls.some(([input]) => String(input).endsWith("/commit")),
    ).toBe(false);
    expect(
      fetchMock.mock.calls.filter(([, init]) => init?.method === "PATCH").at(-1)?.[1]
        ?.body,
    ).toBe('{"proposed_caption":"solo, edited"}');
    await user.click(screen.getByRole("button", { name: "写回全部候选 (1)" }));
    expect(
      await screen.findByText("sample.png: TXT 已被其他程序修改"),
    ).toBeInTheDocument();
  });
  it("shows the provider-aware running state from the job snapshot", async () => {
    const job = {
      id: "j1",
      state: "running",
      profile_name: "Fixture",
      settings: { provider: "cltagger" },
      total: 1,
      completed: 0,
      failed: 0,
      items: [],
    };
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) =>
        String(input).includes("/logs?")
          ? jsonResponse({ lines: [] })
          : jsonResponse({ ok: true, job }),
      ),
    );
    renderInApp(<CaptionReview jobId="j1" />);
    expect(await screen.findByText("正在本地打标 · 0/1 · 失败 0")).toBeInTheDocument();
  });
  it("commits tag drafts on blur before Save and keeps empty rows until explicit deletion", async () => {
    const item = { id: "i1", name: "one.png", file: "one.png", state: "ready", caption: "", proposed_caption: "red hair, red hair", url: "" };
    const job = { id: "j1", state: "done", profile_name: "Fixture", profile_id: "p1", settings: { provider: "wd14" }, total: 1, completed: 1, failed: 0, items: [item] };
    const calls: string[] = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).includes("/logs?")) return jsonResponse({ lines: [] });
      if (init?.method === "PATCH") {
        calls.push(JSON.parse(String(init.body)).proposed_caption);
        item.proposed_caption = calls.at(-1)!;
      }
      return jsonResponse({ ok: true, job });
    }));
    renderInApp(<CaptionReview jobId="j1" />);
    const user = userEvent.setup();
    await screen.findByLabelText("候选标注");
    await user.click(screen.getByRole("button", { name: "标签" }));
    const duplicateRows = screen.getAllByRole("textbox", { name: "编辑标签 red hair" });
    await user.click(duplicateRows[0]);
    await user.clear(duplicateRows[0]);
    await user.type(duplicateRows[0], "long multi word tag");
    expect(duplicateRows[0]).toHaveFocus();
    expect(duplicateRows[0]).toHaveValue("long multi word tag");
    await user.click(screen.getByRole("button", { name: "保存候选" }));
    expect(calls).toEqual(["long multi word tag, red hair"]);
  });
  it("keeps provider snapshots that are external, unknown, or missing raw-only", async () => {
    for (const settings of [{ provider: "openai_compatible" }, { provider: "future" }, undefined]) {
      cleanup();
      const job = { id: "j1", state: "done", profile_name: "Fixture", profile_id: "p1", settings, total: 1, completed: 1, failed: 0,
        items: [{ id: "i1", name: "one.png", file: "one.png", state: "ready", caption: "", proposed_caption: "A sentence, with commas\nand prose.", url: "" }] };
      vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => String(input).includes("/logs?") ? jsonResponse({ lines: [] }) : jsonResponse({ ok: true, job })));
      renderInApp(<CaptionReview jobId="j1" />);
      const raw = await screen.findByLabelText("候选标注");
      expect(raw).toHaveValue("A sentence, with commas\nand prose.");
      expect(screen.queryByRole("button", { name: "标签" })).not.toBeInTheDocument();
    }
  });
  it("rejects external and script image URLs", () => {
    expect(captionImageUrl("https://example.com/image.png")).toBe("");
    expect(captionImageUrl("javascript:alert(1)")).toBe("");
    expect(
      captionImageUrl("/api/config/dataset-presets/image?image=a.png"),
    ).toContain("image=a.png");
  });
});
