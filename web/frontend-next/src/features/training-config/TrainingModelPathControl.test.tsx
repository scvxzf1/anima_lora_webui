import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { TrainingModelPathControl } from "./TrainingModelPathControl";
import type { ModelFilesPage } from "./modelFilesApi";

const page: ModelFilesPage = {
  ok: true,
  roots: [{ name: "项目模型", path: "/srv/models" }],
  directory: "/srv/models",
  parent: null,
  offset: 0,
  next_offset: null,
  total: 2,
  entries: [
    { name: "Diffusers", path: "/srv/models/Diffusers", kind: "directory" },
    {
      name: "weights.safetensors",
      path: "/srv/models/weights.safetensors",
      kind: "file",
    },
  ],
};
const clients: QueryClient[] = [];
afterEach(() => {
  cleanup();
  clients.forEach((client) => client.clear());
  clients.length = 0;
  vi.unstubAllGlobals();
});
function setup(payload: ModelFilesPage = page) {
  const fetchMock = vi
    .fn()
    .mockResolvedValue(new Response(JSON.stringify(payload), { status: 200 }));
  vi.stubGlobal("fetch", fetchMock);
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  clients.push(client);
  const onChange = vi.fn();
  render(
    <QueryClientProvider client={client}>
      <TrainingModelPathControl
        fieldKey="vae"
        label="VAE"
        value="unchanged"
        disabled={false}
        onChange={onChange}
      />
    </QueryClientProvider>,
  );
  return { fetchMock, onChange };
}
async function open() {
  fireEvent.click(screen.getByRole("button", { name: "浏览VAE服务器路径" }));
  await screen.findByRole("button", { name: /weights.safetensors/ });
}
it("loads only when opened and only applies a confirmed server file to the draft", async () => {
  const { fetchMock, onChange } = setup();
  expect(fetchMock).not.toHaveBeenCalled();
  await open();
  expect(fetchMock.mock.calls[0][0]).toBe(
    "/api/settings/model-files?offset=0&limit=100",
  );
  fireEvent.click(screen.getByRole("button", { name: /weights.safetensors/ }));
  expect(onChange).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "使用所选文件" }));
  expect(onChange).toHaveBeenCalledExactlyOnceWith(
    "/srv/models/weights.safetensors",
  );
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
});
it("navigates directories without changing the draft and permits selecting a Diffusers directory", async () => {
  const { fetchMock, onChange } = setup();
  await open();
  fetchMock.mockResolvedValueOnce(
    new Response(
      JSON.stringify({
        ...page,
        directory: "/srv/models/Diffusers",
        parent: "/srv/models",
        entries: [],
        total: 0,
      }),
      { status: 200 },
    ),
  );
  fireEvent.click(screen.getByRole("button", { name: /Diffusers/ }));
  await screen.findByText("此目录没有可选模型文件或子目录。");
  expect(fetchMock.mock.calls[1][0]).toContain(
    "directory=%2Fsrv%2Fmodels%2FDiffusers",
  );
  expect(onChange).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "选择当前目录" }));
  expect(onChange).toHaveBeenCalledExactlyOnceWith("/srv/models/Diffusers");
});
it("Escape cancels selection and restores focus to the browse button", async () => {
  const { onChange } = setup();
  const browse = screen.getByRole("button", { name: "浏览VAE服务器路径" });
  browse.focus();
  await open();
  fireEvent.click(screen.getByRole("button", { name: /weights.safetensors/ }));
  fireEvent.keyDown(document, { key: "Escape" });
  await waitFor(() => expect(browse).toHaveFocus());
  expect(onChange).not.toHaveBeenCalled();
});
it("explains old backend support and supports retry", async () => {
  const { fetchMock } = setup();
  fetchMock.mockResolvedValueOnce(
    new Response(JSON.stringify({ error: "Not found" }), { status: 404 }),
  );
  fireEvent.click(screen.getByRole("button", { name: "浏览VAE服务器路径" }));
  await screen.findByText("目录浏览暂不可用，可手动填写路径或从模型库选择。");
  expect(screen.getByRole("button", { name: "选择当前目录" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "重试读取目录" }));
  await screen.findByRole("button", { name: /weights.safetensors/ });
});
it("paginates using the server cursor and clears prior file selection", async () => {
  const { fetchMock } = setup({ ...page, next_offset: 100, total: 101 });
  await open();
  fireEvent.click(screen.getByRole("button", { name: /weights.safetensors/ }));
  fetchMock.mockResolvedValueOnce(
    new Response(
      JSON.stringify({ ...page, offset: 100, total: 101, entries: [] }),
      { status: 200 },
    ),
  );
  fireEvent.click(screen.getByRole("button", { name: "下一页" }));
  await screen.findByText("此目录没有可选模型文件或子目录。");
  expect(fetchMock.mock.calls[1][0]).toBe(
    "/api/settings/model-files?offset=100&limit=100",
  );
  expect(screen.getByRole("button", { name: "使用所选文件" })).toBeDisabled();
});
