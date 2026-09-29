import { cleanup, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { createMemoryRouter, RouterProvider } from "react-router-dom";
import { render } from "@testing-library/react";
import { useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderInApp, jsonResponse } from "../../test/renderInApp";
import type { TrainingConfigFile } from "../../api/trainingContext";
import { TrainingConfigLibrary } from "./TrainingConfigLibrary";
import { TrainingDevices } from "./TrainingDevices";
import { TrainingLaunchDialog } from "./TrainingLaunchDialog";
import { useTrainingDevices } from "./useTrainingDevices";
import { DEVICE_STORAGE_KEY } from "./trainingDevices";
import type { useTrainingWorkspace } from "./useTrainingWorkspace";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  localStorage.clear();
});

it("shows both full paths when configuration filenames collide", async () => {
  const onSelect = vi.fn();
  const paths = [
    "configs/imported/portrait/train.toml",
    "configs/gui-methods/portrait/train.toml",
  ];
  renderInApp(
    <TrainingConfigLibrary
      expanded
      files={paths.map((path) => ({
        path,
        filename: "train.toml",
        label: "train.toml",
        methods_subdir: path.split("/")[1],
      }))}
      onSelect={onSelect}
      onCreate={() => {}}
    />,
  );

  const user = userEvent.setup();
  await user.click(screen.getByRole("button", { name: /imported/ }));
  await user.click(screen.getByRole("button", { name: /gui-methods/ }));
  expect(screen.getAllByText("train.toml")).toHaveLength(2);
  for (const path of paths) expect(screen.getByText(path)).toBeInTheDocument();
  await user.click(screen.getByTitle(paths[1]));
  expect(onSelect).toHaveBeenLastCalledWith(paths[1]);
});

it("uses the production error boundary and recovery route after startup failure", async () => {
  const { router: productionRouter } = await import("../../app/router");
  const routes = productionRouter.routes.map((root) => {
    if (root.index) return root;
    return { ...root, children: root.children?.map((route) => {
      if (route.index) return route;
      if (route.path === "/datasets") return { ...route, loader: () => { throw new Error("dataset route failed"); } };
      if (route.path === "/training") return { ...route, element: <main>Recovered training workspace</main> };
      return route;
    }) };
  });
  const router = createMemoryRouter(routes, { initialEntries: ["/datasets"] });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({})));
  vi.stubGlobal("matchMedia", () => ({ matches: false, addEventListener: vi.fn(), removeEventListener: vi.fn() }));
  render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);

  expect(await screen.findByRole("alert")).toHaveTextContent("工作区未能加载");
  expect(screen.getByRole("alert")).toHaveTextContent("dataset route failed");
  await userEvent.click(screen.getByRole("link", { name: "训练配置" }));
  expect(await screen.findByText("Recovered training workspace")).toBeInTheDocument();
  expect(router.state.location.pathname).toBe("/training");
  router.dispose();
  productionRouter.dispose();
  client.clear();
});

const file: TrainingConfigFile = {
  path: "configs/imported/fixture.toml",
  method: "lora",
  methods_subdir: "imported",
};

function LaunchHarness({ mode }: { mode: "start" | "queue" }) {
  const device = useTrainingDevices();
  const [launch, setLaunch] = useState(false);
  const state = {
    deviceState: device,
    busy: false,
    draft: {},
    setDraft: vi.fn(),
    preflight: { reset: vi.fn() },
  } as unknown as ReturnType<typeof useTrainingWorkspace>;
  return (
    <>
      <TrainingDevices state={state} />
      <button
        type="button"
        disabled={Boolean(device.issue)}
        onClick={() => setLaunch(true)}
      >
        打开启动确认
      </button>
      {launch && (
        <TrainingLaunchDialog
          file={file}
          preset="default"
          mode={mode}
          gpuIds={device.gpuIds}
          deviceSummary={device.summary}
          deviceIssue={device.issue}
          onClose={() => setLaunch(false)}
        />
      )}
    </>
  );
}

it.each(["start", "queue"] as const)(
  "sends every explicitly selected GPU to preflight and %s",
  async (mode) => {
    localStorage.setItem(
      DEVICE_STORAGE_KEY,
      JSON.stringify({ mode: "ddp", devices: [] }),
    );
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url === "/api/training/gpus") {
        return jsonResponse({
          gpus: [
            { index: 0, name: "GPU A", memory_total_gb: 12 },
            { index: 1, name: "GPU B", memory_total_gb: 12 },
          ],
        });
      }
      if (url === "/api/training/preflight") {
        return jsonResponse({
          ok: true,
          variant: "lora",
          preset: "default",
          methods_subdir: "imported",
          summary: { errors: 0, warnings: 0, checks: 1 },
          checks: [{ level: "ok", key: "model", message: "检查通过" }],
          errors: [],
          warnings: [],
        });
      }
      if (url === `/api/training/${mode}` || url === "/api/training/queue") {
        return jsonResponse({ ok: true, message: "请求完成" });
      }
      throw new Error(`unexpected request ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    renderInApp(<LaunchHarness mode={mode} />);

    await screen.findByRole("checkbox", { name: "GPU 0 · GPU A" });
    await user.click(screen.getByRole("checkbox", { name: "GPU 0 · GPU A" }));
    await user.click(screen.getByRole("checkbox", { name: "GPU 1 · GPU B" }));
    await user.click(screen.getByRole("button", { name: "打开启动确认" }));
    const dialog = screen.getByRole("dialog");
    await within(dialog).findByText("检查通过");
    await user.click(within(dialog).getByRole("checkbox"));
    await user.click(
      within(dialog).getByRole("button", {
        name: mode === "start" ? "确认启动" : "确认入队",
      }),
    );
    await screen.findByRole("status");

    const calls = fetchMock.mock.calls as unknown as [
      string,
      RequestInit | undefined,
    ][];
    const preflight = calls.find(([url]) => url === "/api/training/preflight");
    const action = calls.find(([url]) => url === `/api/training/${mode}`);
    expect(preflight).toBeDefined();
    expect(JSON.parse(String(preflight?.[1]?.body)).gpu_whitelist).toEqual([
      "0",
      "1",
    ]);
    expect(action).toBeDefined();
    expect(JSON.parse(String(action?.[1]?.body)).gpu_whitelist).toEqual([
      "0",
      "1",
    ]);
  },
);
