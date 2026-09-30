import { cleanup, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { jsonResponse, renderInApp } from "../../test/renderInApp";
import { CaptionPrompts } from "./CaptionPrompts";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

it("opens a builtin prompt as a copy without writing until explicit save", async () => {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const path = new URL(String(input), window.location.origin).pathname;
    if (path === "/api/captioning/prompt-presets")
      return jsonResponse({ presets: [{ id: "builtin", name: "内置描述", builtin: true, system_prompt: "system", user_prompt: "describe" }] });
    return jsonResponse({ presets: [] });
  });
  vi.stubGlobal("fetch", fetchMock);
  vi.spyOn(window, "confirm").mockReturnValue(true);
  renderInApp(<CaptionPrompts />);
  const user = userEvent.setup();

  await user.click(await screen.findByRole("button", { name: "复制编辑" }));
  const dialog = screen.getByRole("dialog", { name: "提示词预设" });
  expect(screen.getByLabelText("名称")).toHaveValue("内置描述 副本");
  expect(fetchMock).toHaveBeenCalledTimes(1);
  await user.type(screen.getByLabelText("用户提示词"), "追加内容");
  expect(fetchMock).toHaveBeenCalledTimes(1);
  await user.click(screen.getByRole("button", { name: "取消" }));
  expect(dialog).not.toBeInTheDocument();
  expect(fetchMock).toHaveBeenCalledTimes(1);
});
