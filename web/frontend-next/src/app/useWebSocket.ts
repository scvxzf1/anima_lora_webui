import { useEffect, useRef, useState } from 'react';

type WebSocketStatus = 'connecting' | 'open' | 'closed';

export function useWebSocket(
  path: string,
  onMessage: (message: Record<string, unknown>) => void,
  enabled = true,
) {
  const [status, setStatus] = useState<WebSocketStatus>('connecting');
  const [error, setError] = useState('');
  const callbackRef = useRef(onMessage);
  callbackRef.current = onMessage;

  useEffect(() => {
    if (!enabled) return;
    let disposed = false;
    let socket: WebSocket | null = null;
    let reconnectTimer: number | undefined;
    let attempts = 0;

    function connect() {
      if (disposed) return;
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      socket = new WebSocket(`${protocol}//${window.location.host}${path}`);
      setStatus('connecting');
      setError('');

      socket.onopen = () => {
        attempts = 0;
        if (!disposed) setStatus('open');
      };
      socket.onmessage = (event) => {
        if (disposed) return;
        try {
          callbackRef.current(JSON.parse(event.data as string) as Record<string, unknown>);
        } catch {
          // Ignore non-JSON frames.
        }
      };
      socket.onerror = () => {
        if (!disposed) setError('实时连接异常');
      };
      socket.onclose = () => {
        if (disposed) return;
        setStatus('closed');
        socket = null;
        scheduleReconnect();
      };
    }

    function scheduleReconnect() {
      if (disposed) return;
      const delay = Math.min(30000, 1000 * 2 ** attempts);
      attempts += 1;
      reconnectTimer = window.setTimeout(connect, delay);
    }

    connect();
    return () => {
      disposed = true;
      if (reconnectTimer) window.clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, [path, enabled]);

  return { status, error };
}
