import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useWebSocket } from "./useWebSocket";

class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  onopen: ((event: Event) => unknown) | null = null;
  onclose: ((event: CloseEvent) => unknown) | null = null;
  onerror: ((event: Event) => unknown) | null = null;
  onmessage: ((event: MessageEvent) => unknown) | null = null;
  close = vi.fn();

  constructor(readonly url: string) {
    FakeWebSocket.instances.push(this);
  }

  open() {
    this.onopen?.(new Event("open"));
  }

  disconnect() {
    this.onclose?.(new CloseEvent("close"));
  }

  fail() {
    this.onerror?.(new Event("error"));
  }
}

describe("useWebSocket reconnect behavior", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    FakeWebSocket.instances = [];
    vi.stubGlobal("WebSocket", FakeWebSocket);
  });

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("backs off across consecutive closes and resets after a successful connection", async () => {
    const onOpen = vi.fn();
    const { result, unmount } = renderHook(() =>
      useWebSocket("/ws/training", vi.fn(), true, onOpen),
    );
    expect(result.current.status).toBe("connecting");
    expect(FakeWebSocket.instances).toHaveLength(1);

    act(() => FakeWebSocket.instances[0].disconnect());
    expect(result.current.status).toBe("closed");
    await act(() => vi.advanceTimersByTimeAsync(999));
    expect(FakeWebSocket.instances).toHaveLength(1);
    await act(() => vi.advanceTimersByTimeAsync(1));
    expect(FakeWebSocket.instances).toHaveLength(2);
    expect(result.current.status).toBe("connecting");

    act(() => FakeWebSocket.instances[1].disconnect());
    await act(() => vi.advanceTimersByTimeAsync(1999));
    expect(FakeWebSocket.instances).toHaveLength(2);
    await act(() => vi.advanceTimersByTimeAsync(1));
    expect(FakeWebSocket.instances).toHaveLength(3);

    act(() => FakeWebSocket.instances[2].fail());
    expect(result.current.error).toBe("实时连接异常");
    expect(result.current.status).toBe("connecting");
    act(() => FakeWebSocket.instances[2].disconnect());
    await act(() => vi.advanceTimersByTimeAsync(3999));
    expect(FakeWebSocket.instances).toHaveLength(3);
    await act(() => vi.advanceTimersByTimeAsync(1));
    expect(FakeWebSocket.instances).toHaveLength(4);
    expect(result.current.error).toBe("");

    act(() => FakeWebSocket.instances[3].open());
    expect(result.current.status).toBe("open");
    expect(onOpen).toHaveBeenCalledTimes(1);
    act(() => FakeWebSocket.instances[3].disconnect());
    await act(() => vi.advanceTimersByTimeAsync(999));
    expect(FakeWebSocket.instances).toHaveLength(4);
    await act(() => vi.advanceTimersByTimeAsync(1));
    expect(FakeWebSocket.instances).toHaveLength(5);

    unmount();
    expect(FakeWebSocket.instances[4].close).toHaveBeenCalledTimes(1);
  });

  it("waits for close after an error and cancels the pending reconnect on unmount", async () => {
    const { result, unmount } = renderHook(() =>
      useWebSocket("/ws/training", vi.fn()),
    );
    act(() => FakeWebSocket.instances[0].fail());
    expect(result.current.status).toBe("connecting");
    expect(result.current.error).toBe("实时连接异常");
    await act(() => vi.advanceTimersByTimeAsync(30000));
    expect(FakeWebSocket.instances).toHaveLength(1);

    act(() => FakeWebSocket.instances[0].disconnect());
    expect(result.current.status).toBe("closed");
    unmount();
    await act(() => vi.advanceTimersByTimeAsync(30000));
    expect(FakeWebSocket.instances).toHaveLength(1);
  });
});
