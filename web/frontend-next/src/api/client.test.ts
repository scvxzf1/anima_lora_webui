import { afterEach, describe, expect, it, vi } from "vitest";
import { apiRequest } from "./client";

afterEach(() => vi.unstubAllGlobals());
describe("HTTP client", () => {
  it("marks a lost mutation response body as uncertain", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue({
          text: () => Promise.reject(new TypeError("Stream interrupted")),
        }),
    );
    await expect(
      apiRequest("/api/example", { method: "PUT" }),
    ).rejects.toMatchObject({
      status: 0,
      message: expect.stringContaining("尚未确认"),
    });
  });
  it("returns a successful JSON envelope", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response(JSON.stringify({ ok: true, value: 3 }), { status: 200 }),
        ),
    );
    await expect(
      apiRequest<{ value: number }>("/api/example"),
    ).resolves.toMatchObject({ value: 3 });
  });
  it("rejects ok=false business responses even with HTTP 200", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ ok: false, error: "invalid config" }), {
          status: 200,
        }),
      ),
    );
    await expect(apiRequest("/api/example")).rejects.toMatchObject({
      name: "ApiError",
      message: "invalid config",
      status: 200,
      payload: { ok: false, error: "invalid config" },
    });
  });
  it("merges structured headers without dropping JSON content type", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('{"ok":true}'));
    vi.stubGlobal("fetch", fetchMock);
    await apiRequest("/api/example", {
      headers: new Headers({ "X-Test": "yes" }),
    });
    const headers = fetchMock.mock.calls[0][1].headers as Headers;
    expect(headers.get("content-type")).toBe("application/json");
    expect(headers.get("x-test")).toBe("yes");
  });
  it("preserves supplied content types and structured HTTP errors", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response('{"ok":false,"error":"conflict"}', { status: 409 }),
      );
    vi.stubGlobal("fetch", fetchMock);
    await expect(
      apiRequest("/api/example", { headers: [["Content-Type", "text/plain"]] }),
    ).rejects.toMatchObject({ status: 409, message: "conflict" });
    expect(fetchMock.mock.calls[0][1].headers.get("content-type")).toBe(
      "text/plain",
    );
  });
  it("reports uncertain mutations without retrying", async () => {
    const fetchMock = vi
      .fn()
      .mockRejectedValue(new TypeError("Failed to fetch"));
    vi.stubGlobal("fetch", fetchMock);
    await expect(
      apiRequest("/api/example", { method: "POST" }),
    ).rejects.toMatchObject({
      status: 0,
      message: expect.stringContaining("尚未确认"),
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
