import { cleanup, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { jsonResponse, renderInApp } from "../../test/renderInApp";
import { RuntimeDeleteDialog } from "./RuntimeDeleteDialog";

const preview = {
  ok: true,
  dry_run: true,
  task_count: 2,
  runtime_dir_count: 1,
  tasks: [
    { id: "task-1", name: "Train one", state: "idle" },
    { id: "task-2", name: "Train two", state: "error" },
  ],
  runtime_dirs: [{ path: "output/runs/run-1", status: "ready" }],
  blocked: [],
};

function setup(
  options: {
    dryRun?: unknown;
    final?: unknown;
    dryStatus?: number;
    finalStatus?: number;
  } = {},
) {
  const writes: { url: string; body: Record<string, unknown> }[] = [];
  const onClose = vi.fn();
  const onSuccess = vi.fn();
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      writes.push({ url: String(url), body });
      if (body.dry_run)
        return jsonResponse(
          options.dryRun ?? preview,
          options.dryStatus ?? 200,
        );
      return jsonResponse(
        options.final ?? {
          ok: true,
          dry_run: false,
          deleted_task_ids: ["task-1", "task-2"],
          deleted_runtime_dirs: ["output/runs/run-1"],
          runtime_cleanup_errors: {},
        },
        options.finalStatus ?? 200,
      );
    }),
  );
  const app = renderInApp(
    <RuntimeDeleteDialog
      taskIds={["task-1"]}
      onClose={onClose}
      onSuccess={onSuccess}
    />,
  );
  return { ...app, writes, onClose, onSuccess };
}

describe("RuntimeDeleteDialog", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("previews without submitting a delete and requires explicit confirmation", async () => {
    const app = setup();
    await screen.findByText("Train one");
    expect(app.writes).toEqual([
      {
        url: "/api/training/history/batch",
        body: {
          action: "delete",
          task_ids: ["task-1"],
          delete_runtime_dirs: true,
          dry_run: true,
        },
      },
    ]);
    const submit = screen.getByRole("button", { name: "确认彻底删除" });
    expect(submit).toBeDisabled();
    await userEvent.setup().click(screen.getByRole("checkbox"));
    expect(submit).toBeEnabled();
    expect(app.writes).toHaveLength(1);
  });

  it("does not enable execution when the backend reports any blocker", async () => {
    setup({
      dryRun: {
        ...preview,
        blocked: [
          { path: "/outside/run", reason: "运行目录不在 WebUI 输出根目录内" },
        ],
      },
    });
    await screen.findByText("运行目录不在 WebUI 输出根目录内");
    expect(screen.getByRole("button", { name: "确认彻底删除" })).toBeDisabled();
    expect(screen.getByRole("checkbox")).toBeDisabled();
  });

  it("sends the confirmed mutation once and reports cleanup errors individually", async () => {
    const app = setup({
      final: {
        ok: true,
        dry_run: false,
        deleted_task_ids: ["task-1"],
        deleted_runtime_dirs: ["output/runs/one"],
        runtime_cleanup_errors: { "output/runs/two": "permission denied" },
        preview,
      },
    });
    await screen.findByText("Train one");
    const user = userEvent.setup();
    await user.click(screen.getByRole("checkbox"));
    await user.click(screen.getByRole("button", { name: "确认彻底删除" }));
    await screen.findByText("permission denied");
    expect(app.writes).toHaveLength(2);
    expect(app.writes[1].body).toEqual({
      action: "delete",
      task_ids: ["task-1"],
      delete_runtime_dirs: true,
      confirmed: true,
    });
    expect(app.onSuccess).toHaveBeenCalledTimes(1);
    expect(app.onSuccess).toHaveBeenCalledWith(
      expect.objectContaining({ deleted_task_ids: ["task-1"] }),
    );
    expect(
      screen.getByText(/已删除 1 条历史记录，运行目录 1 个/),
    ).toBeInTheDocument();
  });

  it("surfaces preview errors and never offers a delete request", async () => {
    setup({ dryRun: { error: "服务暂不可用" }, dryStatus: 503 });
    await screen.findByText(/预览失败：服务暂不可用/);
    expect(screen.getByRole("button", { name: "确认彻底删除" })).toBeDisabled();
  });

  it("does not retry a failed confirmed mutation", async () => {
    const app = setup({ final: { error: "运行目录已变更" }, finalStatus: 409 });
    await screen.findByText("Train one");
    const user = userEvent.setup();
    await user.click(screen.getByRole("checkbox"));
    await user.click(screen.getByRole("button", { name: "确认彻底删除" }));
    await screen.findByText(/删除请求失败：运行目录已变更/);
    await waitFor(() => expect(app.writes).toHaveLength(2));
    expect(
      app.writes.filter((entry) => entry.body.confirmed === true),
    ).toHaveLength(1);
    expect(screen.getByRole("button", { name: "确认彻底删除" })).toBeDisabled();
  });
});
