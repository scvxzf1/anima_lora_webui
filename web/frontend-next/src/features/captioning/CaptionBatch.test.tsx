import { cleanup, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { renderInApp, jsonResponse } from "../../test/renderInApp";
import { CaptionReview } from "./CaptionReview";
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });
it("previews cross-page changes without mutation, applies local drafts and saves one batch PATCH", async () => {
  const job = { id: "j1", state: "completed", profile_name: "Batch", settings: { provider: "wd14" }, total: 65, completed: 65, failed: 0,
    items: Array.from({ length: 65 }, (_, i) => ({ id: String(i), name: `${i}.png`, caption: "", proposed_caption: "red, red", state: "ready", url: "" })) };
  const writes: { path: string; body: { items: { id: string; proposed_caption: string; expected_caption: string }[] } }[] = [];
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    if (String(input).includes("/logs?")) return jsonResponse({ lines: [] });
    if (init?.method === "PATCH") {
      const body = JSON.parse(String(init.body)); writes.push({ path: String(input), body });
      for (const entry of body.items) job.items.find((i) => i.id === entry.id)!.proposed_caption = entry.proposed_caption;
    }
    return jsonResponse({ ok: true, job });
  }));
  renderInApp(<CaptionReview jobId="j1" />); const user = userEvent.setup();
  await screen.findByText("red (65)");
  await user.selectOptions(screen.getByLabelText("操作范围"), "all");
  await user.selectOptions(screen.getByLabelText("批量操作"), "dedupe");
  await user.click(screen.getByRole("button", { name: "预览差异 (65)" }));
  expect(writes).toHaveLength(0);
  await user.click(screen.getByRole("button", { name: "应用到草稿" }));
  expect(writes).toHaveLength(0);
  await user.click(screen.getByRole("button", { name: "撤销" }));
  expect(screen.getByRole("button", { name: "保存全部草稿 (0)" })).toBeDisabled();
  await user.click(screen.getByRole("button", { name: "重做" }));
  await user.click(screen.getByRole("button", { name: "保存全部草稿 (65)" }));
  expect(writes).toHaveLength(1); expect(writes[0].path).toBe("/api/captioning/jobs/j1/items");
  expect(writes[0].body.items).toHaveLength(65); expect(writes[0].body.items[64].expected_caption).toBe("red, red");
});

for (const failure of [409, 0]) it(`reconciles ${failure} without retry and shows server versus retained draft`, async () => {
  let text = "red, red", calls = 0;
  const snapshot = () => ({ ok: true, job: { id: "j1", state: "completed", profile_name: "Batch", settings: { provider: "wd14" }, total: 1, completed: 1, failed: 0,
    items: [{ id: "1", name: "one.png", caption: "", proposed_caption: text, state: "ready", url: "" }] } });
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    if (String(input).includes("/logs?")) return jsonResponse({ lines: [] });
    if (init?.method === "PATCH") { calls++; text = "server"; if (!failure) throw new TypeError("lost response"); return jsonResponse({error:"conflict"}, failure); }
    return jsonResponse(snapshot());
  }));
  renderInApp(<CaptionReview jobId="j1" />); const user = userEvent.setup();
  await screen.findByText("red (1)");
  expect(screen.getByRole("button", { name: "重新打标" })).toBeEnabled();
  await user.selectOptions(screen.getByLabelText("操作范围"), "all");
  await user.selectOptions(screen.getByLabelText("批量操作"), "dedupe");
  await user.click(screen.getByRole("button", { name: "预览差异 (1)" }));
  await user.click(screen.getByRole("button", { name: "应用到草稿" }));
  await user.click(screen.getByRole("button", { name: "保存全部草稿 (1)" }));
  await user.click(await screen.findByRole("button", { name: "核对任务状态" }));
  await screen.findByText("服务器：server");
  expect(screen.getByText("本地草稿：red")).toBeVisible();
  expect(screen.getByRole("button", { name: "保存全部草稿 (1)" })).toBeDisabled();
  await user.click(screen.getByRole("button", { name: "保留草稿并采用服务器基线" }));
  expect(screen.getByRole("button", { name: "保存全部草稿 (1)" })).toBeEnabled();
  expect(calls).toBe(1);
});

it("isolates drafts and undo history when the task changes", async () => {
  const React = await import("react");
  function Tasks() { const [id, setId] = React.useState("j1"); return <><button onClick={() => setId("j2")}>切换测试任务</button><CaptionReview jobId={id} /></>; }
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    if (String(input).includes("/logs?")) return jsonResponse({lines:[]});
    return jsonResponse({ok:true, job:{id:String(input).split("/").at(-1), state:"completed", profile_name:"Batch", settings:{provider:"wd14"}, total:1, completed:1, failed:0,
      items:[{id:"1", name:"one.png", caption:"", proposed_caption:"red, red", state:"ready", url:""}]}});
  }));
  renderInApp(<Tasks />); const user = userEvent.setup();
  await screen.findByText("red (1)");
  await user.selectOptions(screen.getByLabelText("操作范围"), "all");
  await user.selectOptions(screen.getByLabelText("批量操作"), "dedupe");
  await user.click(screen.getByRole("button", {name:"预览差异 (1)"}));
  await user.click(screen.getByRole("button", {name:"应用到草稿"}));
  expect(screen.getByRole("button", {name:"撤销"})).toBeEnabled();
  await user.click(screen.getByRole("button", {name:"切换测试任务"}));
  await screen.findByText("red (1)");
  expect(screen.getByRole("button", {name:"保存全部草稿 (0)"})).toBeDisabled();
  expect(screen.getByRole("button", {name:"撤销"})).toBeDisabled();
});
