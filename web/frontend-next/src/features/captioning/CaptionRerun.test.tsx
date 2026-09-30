import { cleanup, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { jsonResponse, renderInApp } from "../../test/renderInApp";
import { CaptionReview } from "./CaptionReview";

const makeJob = (state = "done") => ({
  id: "job-1",
  state,
  profile_name: "External captions",
  profile_id: "profile-remote",
  settings: { provider: "openai_compatible" },
  dataset_file: "configs/datasets/test.toml",
  dataset_index: 0,
  total: 2,
  completed: state === "running" || state === "queued" ? 0 : 2,
  failed: 0,
  items: [
    { id: "item-1", name: "one.png", file: "one.png", url: "", state: "ready", caption: "", proposed_caption: "one" },
    { id: "item-2", name: "two.png", file: "two.png", url: "", state: "ready", caption: "", proposed_caption: "two" },
  ],
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

it("requires explicit billing confirmation and reruns only selected items on the original profile", async () => {
  const job = makeJob();
  const writes: { path: string; method: string; body?: unknown }[] = [];
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = new URL(String(input), window.location.origin).pathname;
    const method = init?.method || "GET";
    const body = init?.body ? JSON.parse(String(init.body)) : undefined;
    writes.push({ path, method, body });
    if (path.endsWith("/logs")) return jsonResponse({ lines: [] });
    if (path.endsWith("/rerun")) return jsonResponse({ ok: true, job: makeJob("queued") });
    return jsonResponse({ ok: true, job });
  });
  vi.stubGlobal("fetch", fetchMock);
  const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
  renderInApp(<CaptionReview jobId="job-1" />);
  const user = userEvent.setup();
  const rerun = await screen.findByRole("button", { name: "重新打标" });
  await user.click(screen.getByRole("checkbox", { name: "选择 one.png" }));
  expect(rerun).toBeEnabled();
  await user.click(rerun);
  expect(confirm).toHaveBeenCalledWith("重新打标会重新使用此任务接入处理图片，并替换所选项的候选；外部接入可能产生费用。确认继续吗？");
  expect(writes.filter((write) => write.path.endsWith("/rerun"))).toHaveLength(0);

  confirm.mockReturnValue(true);
  await user.click(rerun);
  await screen.findByText("待处理 · 0/2 · 失败 0");
  expect(writes.filter((write) => write.path.endsWith("/rerun"))).toEqual([{
    path: "/api/captioning/jobs/job-1/rerun",
    method: "POST",
    body: { profile_id: "profile-remote", item_ids: ["item-1"] },
  }]);
  expect(writes.some((write) => write.method === "PATCH" || write.path.endsWith("/commit"))).toBe(false);
});

it("blocks rerun while the job is running, edits are dirty, or a request failed", async () => {
  const runningFetch = vi.fn(async (input: RequestInfo | URL) =>
    String(input).includes("/logs?")
      ? jsonResponse({ lines: [] })
      : jsonResponse({ ok: true, job: makeJob("running") }),
  );
  vi.stubGlobal("fetch", runningFetch);
  const runningView = renderInApp(<CaptionReview jobId="job-1" />);
  expect(await screen.findByRole("button", { name: "重新打标" })).toBeDisabled();
  runningView.unmount();

  const job = makeJob();
  let rerunCalls = 0;
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input);
    if (path.includes("/logs?")) return jsonResponse({ lines: [] });
    if (path.endsWith("/rerun")) {
      rerunCalls += 1;
      return jsonResponse({ error: "接入暂不可用" }, 503);
    }
    return jsonResponse({ ok: true, job });
  });
  vi.stubGlobal("fetch", fetchMock);
  const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
  renderInApp(<CaptionReview jobId="job-1" />);
  const user = userEvent.setup();
  const rerun = await screen.findByRole("button", { name: "重新打标" });
  await user.click(screen.getByRole("checkbox", { name: "选择 one.png" }));
  const candidate = screen.getByLabelText("候选标注");
  await user.clear(candidate);
  await user.type(candidate, "edited candidate");
  expect(rerun).toBeDisabled();
  expect(rerunCalls).toBe(0);

  await user.clear(candidate);
  await user.type(candidate, "one");
  expect(rerun).toBeEnabled();
  await user.click(rerun);
  await vi.waitFor(() => {
    expect(screen.getAllByRole("alert").some((node) =>
      node.textContent?.includes("接入暂不可用"),
    )).toBe(true);
  });
  expect(rerun).toBeEnabled();
  expect(confirm).toHaveBeenCalledTimes(1);
  expect(rerunCalls).toBe(1);
  expect(fetchMock.mock.calls.some((call) => call[1]?.method === "PATCH")).toBe(false);
});
