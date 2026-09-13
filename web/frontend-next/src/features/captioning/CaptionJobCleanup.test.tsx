import { cleanup, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { renderInApp, jsonResponse } from "../../test/renderInApp";
import { CaptionJobCleanup } from "./CaptionJobCleanup";
import type { CaptionJob } from "./api";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});
const jobs = [
  { id: "done", state: "completed" },
  { id: "running", state: "running" },
] as CaptionJob[];

it("confirms cleanup and submits only the displayed finished IDs", async () => {
  const fetchMock = vi
    .fn()
    .mockResolvedValue(jsonResponse({ removed: ["done"], skipped: [] }));
  vi.stubGlobal("fetch", fetchMock);
  const onRemoved = vi.fn();
  renderInApp(<CaptionJobCleanup jobs={jobs} onRemoved={onRemoved} />);
  const user = userEvent.setup();
  await user.click(screen.getByRole("button", { name: "清理已结束任务" }));
  expect(screen.getByText(/未写回的候选和编辑草稿将丢失/)).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: /^取消$/ }));
  expect(fetchMock).not.toHaveBeenCalled();
  await user.click(screen.getByRole("button", { name: "清理已结束任务" }));
  await user.click(
    screen.getByRole("button", { name: /^清理任务$/ }),
  );
  expect(await screen.findByRole("status")).toHaveTextContent(
    "已清理 1 个任务",
  );
  expect(fetchMock.mock.calls[0][0]).toBe("/api/captioning/jobs/cleanup");
  expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
    job_ids: ["done"],
  });
  expect(onRemoved).toHaveBeenCalledWith(["done"]);
});

it("disables cleanup for running-only tasks", () => {
  renderInApp(<CaptionJobCleanup jobs={[jobs[1]]} onRemoved={vi.fn()} />);
  expect(screen.getByRole("button", { name: "清理已结束任务" })).toBeDisabled();
});
