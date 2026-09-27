import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { jsonResponse, renderInApp } from "../../test/renderInApp";
import { EnvironmentPage } from "./EnvironmentPage";

afterEach(() => vi.unstubAllGlobals());

it("shows failed checks as a report and refreshes independently", async () => {
  const fetcher = vi.fn(async () => jsonResponse({ ok: false, platform: { system: "Linux" }, summary: { checks: 1, errors: 1, warnings: 0 }, checks: [{ key: "python", level: "error", group: "运行环境", message: "Python 不可用" }] }));
  vi.stubGlobal("fetch", fetcher);
  renderInApp(<EnvironmentPage />);
  expect(await screen.findByText(/Python 不可用/)).toBeInTheDocument();
  expect(screen.getByText("运行环境")).toBeInTheDocument();
  await userEvent.setup().click(screen.getByRole("button", { name: "重新检测" }));
  await waitFor(() => expect(fetcher).toHaveBeenCalledTimes(2));
});
