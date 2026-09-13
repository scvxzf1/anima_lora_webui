import { act, cleanup, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { jsonResponse, renderInApp } from "../../test/renderInApp";
import { HistoryResume } from "./HistoryResume";
import type { ResumeCheckpoint } from "./api";

const checkpoint: ResumeCheckpoint = {
  path: "output/task/checkpoint-state", name: "checkpoint-state", step: 42,
  target_total_steps: 100, remaining_steps: 58, state_complete: true,
  state_integrity: { ok: true, optimizer: true, scheduler: true }, resume_available: true,
};
const options = (entry: ResumeCheckpoint = checkpoint) => ({ checkpoints: [entry], default_checkpoint: entry.path, message: "" });
const submit = () => screen.getByRole("button", { name: "确认续训" });

function setup(initial = options()) {
  let payload = initial;
  let failure = "";
  const writes: { url: string; body: unknown }[] = [];
  vi.stubGlobal("fetch", vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
    if (init?.method === "POST") {
      writes.push({ url: String(url), body: JSON.parse(String(init.body)) });
      if (failure === "network") throw new TypeError("disconnected");
      return failure ? jsonResponse({ error: failure }, 409) : jsonResponse({ ok: true, message: "已提交新任务" });
    }
    return jsonResponse(payload);
  }));
  const app = renderInApp(<HistoryResume taskId="task-1" onClose={vi.fn()} />);
  return { ...app, writes, setPayload: (value: typeof initial) => { payload = value; }, fail: (value: string) => { failure = value; } };
}

describe("resume confirmation contract", () => {
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it.each([false, true])("submits the confirmed checkpoint once (queue=%s)", async (queue) => {
    const app = setup();
    await screen.findByRole("option", { name: "checkpoint-state · step 42" });
    expect(submit()).toBeDisabled();
    const user = userEvent.setup();
    if (queue) await user.selectOptions(screen.getByLabelText("执行方式"), "queue");
    await user.click(screen.getByRole("checkbox"));
    await user.dblClick(submit());
    await screen.findByText("已提交新任务");
    expect(app.writes).toEqual([{ url: queue ? "/api/training/queue/resume" : "/api/training/resume", body: { task_id: "task-1", checkpoint: checkpoint.path } }]);
    expect(screen.getByRole("link", { name: queue ? "查看队列" : "查看监控" })).toHaveAttribute("href", queue ? "/queue" : "/monitor");
  });

  it("keeps missing and incomplete states unavailable", async () => {
    const app = setup({ checkpoints: [], default_checkpoint: "", message: "无检查点" });
    await screen.findByText("无检查点");
    expect(submit()).toBeDisabled();
    app.setPayload(options({ ...checkpoint, state_complete: false, state_integrity: { ok: false, missing: ["optimizer.bin"] }, resume_available: false, unavailable_reason: "缺少 optimizer.bin" }));
    await act(async () => { await app.client.refetchQueries({ queryKey: ["history-resume"] }); });
    await screen.findByText("缺少 optimizer.bin");
    await userEvent.setup().type(screen.getByLabelText("训练目标总步数"), "200");
    expect(submit()).toBeDisabled();
    expect(app.writes).toHaveLength(0);
  });

  it("allows a complete reached-target checkpoint only after a larger target and fresh confirmation", async () => {
    const app = setup(options({ ...checkpoint, target_total_steps: 42, remaining_steps: 0, resume_available: false }));
    await screen.findByText("历史目标已达到；提高总步数后可继续");
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("训练目标总步数"), "200");
    expect(submit()).toBeDisabled();
    await user.click(screen.getByRole("checkbox"));
    await user.click(submit());
    await screen.findByText("已提交新任务");
    expect(app.writes[0].body).toEqual({ task_id: "task-1", checkpoint: checkpoint.path, duration_overrides: { max_train_steps: 158 } });
  });

  it("invalidates confirmation when the background checkpoint snapshot changes", async () => {
    const app = setup();
    await screen.findByRole("option", { name: "checkpoint-state · step 42" });
    const user = userEvent.setup();
    await user.click(screen.getByRole("checkbox"));
    expect(submit()).toBeEnabled();
    app.setPayload(options({ ...checkpoint, path: "output/task/new-state", name: "new-state", step: 60 }));
    await act(async () => { await app.client.refetchQueries({ queryKey: ["history-resume"] }); });
    await screen.findByRole("option", { name: "new-state · step 60" });
    expect(screen.getByRole("checkbox")).not.toBeChecked();
    expect(submit()).toBeDisabled();
    await user.click(screen.getByRole("checkbox"));
    await user.click(submit());
    await screen.findByText("已提交新任务");
    expect(app.writes[0].body).toEqual({ task_id: "task-1", checkpoint: "output/task/new-state" });
  });

  it("requires recheck and reconfirmation after a rejected submission", async () => {
    const app = setup(); app.fail("检查点已变更");
    await screen.findByRole("option", { name: "checkpoint-state · step 42" });
    const user = userEvent.setup();
    await user.click(screen.getByRole("checkbox")); await user.click(submit());
    await screen.findByText("检查点已变更");
    expect(submit()).toBeDisabled();
    app.fail("");
    await user.click(screen.getByRole("button", { name: "重新检查后重试" }));
    await waitFor(() => expect(screen.queryByText("检查点已变更")).not.toBeInTheDocument());
    expect(submit()).toBeDisabled();
    expect(screen.getByRole("checkbox")).not.toBeChecked();
    await user.click(screen.getByRole("checkbox")); await user.click(submit());
    await screen.findByText("已提交新任务");
    expect(app.writes).toHaveLength(2);
  });

  it("does not offer blind resubmission after an unknown network result", async () => {
    const app = setup(); app.fail("network");
    await screen.findByRole("option", { name: "checkpoint-state · step 42" });
    const user = userEvent.setup();
    await user.click(screen.getByRole("checkbox")); await user.click(submit());
    await screen.findByRole("link", { name: "核对历史任务" });
    expect(screen.getByRole("link", { name: "核对队列" })).toHaveAttribute("href", "/queue");
    expect(screen.queryByRole("button", { name: "重新检查后重试" })).not.toBeInTheDocument();
    expect(submit()).toBeDisabled();
    expect(app.writes).toHaveLength(1);
  });
});
