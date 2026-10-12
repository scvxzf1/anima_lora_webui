import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { jsonResponse } from "../../test/renderInApp";
import { TrainingLaunchDialog } from "./TrainingLaunchDialog";
import type { TrainingPreflightResponse } from "./api";

const file = {
  path: "configs/imported/risk.toml",
  method: "lora",
  methods_subdir: "imported",
};
function riskReport(message = "此组合仍有数值风险"): TrainingPreflightResponse {
  const error = {
    level: "error",
    key: "compile",
    message,
    overridable: true,
    risk_code: "mixed_nf4_compile",
  };
  return {
    ok: false,
    can_override: true,
    variant: "lora",
    preset: "default",
    methods_subdir: "imported",
    summary: { errors: 1, warnings: 0, checks: 1 },
    checks: [error],
    errors: [error],
    warnings: [],
  };
}
function mount(mode: "start" | "queue" = "start") {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const props = {
    file,
    preset: "default",
    mode,
    gpuIds: ["0"],
    deviceSummary: "GPU 0",
    deviceIssue: "",
    onClose: vi.fn(),
  };
  const tree = (updates = {}) => (
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <TrainingLaunchDialog {...props} {...updates} />
      </MemoryRouter>
    </QueryClientProvider>
  );
  const view = render(tree());
  return {
    client,
    rerender: (updates: object) => view.rerender(tree(updates)),
  };
}
const riskLabel = "我已了解以上风险，仍要继续";
async function confirmBoth() {
  const user = userEvent.setup();
  await user.click(screen.getByRole("checkbox", { name: /确认以上检查结果/ }));
  await user.click(screen.getByRole("checkbox", { name: riskLabel }));
  return user;
}

describe("explicit launch risk consent", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });
  it.each(["start", "queue"] as const)(
    "requires both confirmations for %s and sends one-time risk consent",
    async (mode) => {
      const fetchMock = vi.fn(
        async (url: RequestInfo | URL, _init?: RequestInit) =>
          jsonResponse(
            String(url).endsWith("preflight")
              ? riskReport()
              : { ok: true, message: "已完成" },
          ),
      );
      vi.stubGlobal("fetch", fetchMock);
      mount(mode);
      await screen.findByText("此组合仍有数值风险");
      expect(
        screen.getByText("风险", { selector: "li span" }),
      ).toBeInTheDocument();
      const button = screen.getByRole("button", {
        name: mode === "start" ? "无视风险继续启动" : "无视风险继续入队",
      });
      expect(button).toBeDisabled();
      const user = userEvent.setup();
      await user.click(screen.getByRole("checkbox", { name: riskLabel }));
      expect(button).toBeDisabled();
      await user.click(
        screen.getByRole("checkbox", { name: /确认以上检查结果/ }),
      );
      await user.dblClick(button);
      await screen.findByRole("status");
      const calls = fetchMock.mock.calls.filter(
        ([url]) => String(url) === `/api/training/${mode}`,
      );
      expect(calls).toHaveLength(1);
      expect(JSON.parse(String(calls[0][1]?.body))).toMatchObject({
        confirmed: true,
        ignore_risks: true,
      });
    },
  );
  it.each(["legacy", "hard", "unknown", "inconsistent"])(
    "does not allow %s reports to be bypassed",
    async (kind) => {
      const report = riskReport();
      if (kind === "legacy") delete report.can_override;
      if (kind === "hard") report.errors[0].overridable = false;
      if (kind === "unknown") report.errors[0].risk_code = "future_risk";
      if (kind === "inconsistent") report.summary.errors = 2;
      vi.stubGlobal(
        "fetch",
        vi.fn(async () => jsonResponse(report)),
      );
      mount();
      await screen.findByText("此组合仍有数值风险");
      expect(
        screen.queryByRole("checkbox", { name: riskLabel }),
      ).not.toBeInTheDocument();
      await userEvent.setup().click(screen.getByRole("checkbox"));
      expect(screen.getByRole("button", { name: "确认启动" })).toBeDisabled();
    },
  );
  it("replaces server-rejected preflight and requires fresh consent before retrying", async () => {
    let attempts = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: RequestInfo | URL) => {
        if (String(url).endsWith("preflight"))
          return jsonResponse(riskReport());
        attempts++;
        return attempts === 1
          ? jsonResponse(
              {
                ok: false,
                error: "预检已变化",
                preflight: riskReport("新报告风险"),
              },
              409,
            )
          : jsonResponse({ ok: true });
      }),
    );
    mount();
    await screen.findByText("此组合仍有数值风险");
    const user = await confirmBoth();
    await user.click(screen.getByRole("button", { name: "无视风险继续启动" }));
    await screen.findByText("新报告风险");
    screen
      .getAllByRole("checkbox")
      .forEach((checkbox) => expect(checkbox).not.toBeChecked());
    expect(
      screen.getByRole("button", { name: "无视风险继续启动" }),
    ).toBeDisabled();
    await confirmBoth();
    await user.click(screen.getByRole("button", { name: "无视风险继续启动" }));
    await screen.findByRole("status");
    expect(attempts).toBe(2);
  });
  it.each(["file", "preset", "gpuIds", "report"])(
    "clears consent when %s changes",
    async (kind) => {
      vi.stubGlobal(
        "fetch",
        vi.fn(async () => jsonResponse(riskReport())),
      );
      const view = mount();
      await screen.findByText("此组合仍有数值风险");
      await confirmBoth();
      if (kind === "report")
        view.client.setQueryData(
          [
            "training-launch-preflight",
            file.path,
            "default",
            ["0"],
            file.method,
            file.methods_subdir,
          ],
          riskReport("风险报告已更新"),
        );
      else
        view.rerender(
          kind === "file"
            ? { file: { ...file, path: "configs/imported/other.toml" } }
            : kind === "preset"
              ? { preset: "other" }
              : { gpuIds: ["1"] },
        );
      await screen.findByRole("checkbox", { name: riskLabel });
      await waitFor(() =>
        screen
          .getAllByRole("checkbox")
          .forEach((checkbox) => expect(checkbox).not.toBeChecked()),
      );
      expect(
        screen.getByRole("button", { name: "无视风险继续启动" }),
      ).toBeDisabled();
    },
  );
  it.each([
    ["start", 0],
    ["start", 503],
    ["queue", 0],
    ["queue", 503],
  ] as const)(
    "does not retry a result-unknown %s error %s even with a fresh report",
    async (mode, status) => {
      let attempts = 0;
      vi.stubGlobal(
        "fetch",
        vi.fn(async (url: RequestInfo | URL) => {
          if (String(url).endsWith("preflight"))
            return jsonResponse(riskReport());
          attempts++;
          if (status === 0) throw new TypeError("network failed");
          return jsonResponse(
            {
              ok: false,
              error: "服务暂不可用",
              preflight: riskReport("新的风险"),
            },
            status,
          );
        }),
      );
      mount(mode);
      await screen.findByText("此组合仍有数值风险");
      const user = await confirmBoth();
      await user.click(
        screen.getByRole("button", {
          name: mode === "start" ? "无视风险继续启动" : "无视风险继续入队",
        }),
      );
      await screen.findByText(/结果可能未知/);
      await confirmBoth();
      const button = screen.getByRole("button", {
        name: mode === "start" ? "无视风险继续启动" : "无视风险继续入队",
      });
      expect(button).toBeDisabled();
      await user.click(button);
      expect(attempts).toBe(1);
    },
  );
});
