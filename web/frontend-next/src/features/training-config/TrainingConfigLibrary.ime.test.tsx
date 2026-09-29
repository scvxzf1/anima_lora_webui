import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import { TrainingConfigLibrary } from "./TrainingConfigLibrary";

afterEach(cleanup);

it("keeps a Chinese IME search focused and filters after composition ends", () => {
  const file = {
    path: "configs/imported/人像训练.toml",
    label: "人像训练",
    filename: "人像训练.toml",
    methods_subdir: "imported",
  };
  render(
    <QueryClientProvider client={new QueryClient()}>
      <TrainingConfigLibrary
        expanded
        files={[file]}
        selectedPath="configs/imported/人像训练.toml"
        preset="default"
        gpuIds={["0"]}
        deviceSummary="GPU 0"
        deviceIssue=""
        onSelect={() => {}}
        onCreate={() => {}}
      />
    </QueryClientProvider>,
  );

  const input = screen.getByRole("textbox", { name: "搜索配置" });
  input.focus();
  fireEvent.compositionStart(input);
  fireEvent.change(input, { target: { value: "人像" } });

  expect(input).toHaveFocus();
  expect(input).toHaveValue("人像");
  expect(screen.getByText(file.path)).toBeInTheDocument();

  fireEvent.compositionEnd(input, { data: "人像" });

  expect(input).toHaveFocus();
  expect(screen.getByText(file.path)).toBeInTheDocument();
});
