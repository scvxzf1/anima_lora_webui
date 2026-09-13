import { cleanup, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { renderInApp, jsonResponse } from "../../test/renderInApp";
import { CaptionTranslation } from "./CaptionTranslation";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});
describe("caption translation", () => {
  it("queries the local dictionary and only applies after an explicit command", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        jsonResponse({ translations: ["红色", "蓝色"], matched: 2, total: 2 }),
      );
    vi.stubGlobal("fetch", fetchMock);
    const apply = vi.fn();
    renderInApp(
      <CaptionTranslation value="red, blue" disabled={false} onApply={apply} />,
    );
    const user = userEvent.setup();
    await user.click(screen.getByText("标签翻译"));
    await user.click(screen.getByRole("button", { name: "查询词典" }));
    expect(await screen.findByText("红色, 蓝色")).toBeInTheDocument();
    expect(apply).not.toHaveBeenCalled();
    expect(fetchMock.mock.calls[0][0]).toBe("/api/captioning/translate-tags");
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      tags: ["red", "blue"],
      target_language: "zh",
    });
    await user.click(screen.getByRole("button", { name: "采用翻译候选" }));
    expect(apply).toHaveBeenCalledWith("红色, 蓝色");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
