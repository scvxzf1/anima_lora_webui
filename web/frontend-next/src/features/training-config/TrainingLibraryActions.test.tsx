import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { TrainingLibraryActions } from "./TrainingLibraryActions";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const file = { path: "configs/imported/a.toml", filename: "a.toml" };
const groups = [
  {
    id: "imported",
    label: "导入配置",
    files: [file, { path: "configs/imported/b.toml" }],
  },
  { id: "custom", label: "测试分组", files: [] },
  { id: "system", label: "系统配置", readonly: true, files: [] },
];

function show(disabled = false) {
  return render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      <TrainingLibraryActions
        scope="file"
        groups={groups}
        targetGroup={groups[0]}
        file={file}
        disabled={disabled}
        onRenamed={vi.fn()}
      />
    </QueryClientProvider>,
  );
}

it("moves and orders the targeted file with boundary guards", async () => {
  const fetchMock = vi.fn(
    async () => new Response(JSON.stringify({ ok: true })),
  );
  vi.stubGlobal("fetch", fetchMock);
  const user = userEvent.setup();
  show();
  expect(
    screen.getByRole("button", { name: "上移配置 a.toml" }),
  ).toBeDisabled();
  expect(screen.getByRole("option", { name: "系统配置" })).toBeDisabled();
  await user.click(screen.getByRole("button", { name: "下移配置 a.toml" }));
  await waitFor(() =>
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/config/file-groups/reorder-file",
      expect.objectContaining({
        body: JSON.stringify({
          file: file.path,
          group: "imported",
          direction: "down",
        }),
      }),
    ),
  );
  await waitFor(() => expect(screen.getByRole("combobox")).toBeEnabled());
  await user.selectOptions(screen.getByRole("combobox"), "custom");
  await waitFor(() =>
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/config/file-groups/move-file",
      expect.objectContaining({
        body: JSON.stringify({ file: file.path, group: "custom" }),
      }),
    ),
  );
});

it("disables file management while the workspace is dirty or busy", () => {
  show(true);
  expect(screen.getByRole("combobox")).toBeDisabled();
  for (const button of screen.getAllByRole("button"))
    expect(button).toBeDisabled();
});
