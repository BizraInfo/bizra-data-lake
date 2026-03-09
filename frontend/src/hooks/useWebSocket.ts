/**
 * Sovereign WebSocket hook — connects to ghost_ws.py WS endpoint.
 * Auto-reconnect with exponential backoff (max 30s).
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import type { WSEvent } from '../types';

interface UseWebSocketOptions {
  url: string;
  onMessage?: (event: WSEvent) => void;
  enabled?: boolean;
  maxReconnectDelay?: number;
}

interface UseWebSocketReturn {
  connected: boolean;
  send: (data: unknown) => void;
  lastEvent: WSEvent | null;
}

export function useWebSocket({
  url,
  onMessage,
  enabled = true,
  maxReconnectDelay = 30_000,
}: UseWebSocketOptions): UseWebSocketReturn {
  const [connected, setConnected] = useState(false);
  const [lastEvent, setLastEvent] = useState<WSEvent | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const retriesRef = useRef(0);
  const timerRef = useRef<ReturnType<typeof setTimeout>>();
  const shouldReconnectRef = useRef(enabled);

  const connect = useCallback(() => {
    if (!enabled || !shouldReconnectRef.current) return;

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        retriesRef.current = 0;
      };

      ws.onmessage = (e) => {
        try {
          const event = JSON.parse(e.data) as WSEvent;
          setLastEvent(event);
          onMessage?.(event);
        } catch {
          // Non-JSON message — ignore
        }
      };

      ws.onclose = () => {
        setConnected(false);
        wsRef.current = null;
        if (!enabled || !shouldReconnectRef.current) {
          return;
        }
        // Exponential backoff reconnect
        const delay = Math.min(1000 * Math.pow(2, retriesRef.current), maxReconnectDelay);
        retriesRef.current++;
        timerRef.current = setTimeout(connect, delay);
      };

      ws.onerror = () => {
        ws.close();
      };
    } catch {
      if (!enabled || !shouldReconnectRef.current) {
        return;
      }
      // WebSocket construction failed
      const delay = Math.min(1000 * Math.pow(2, retriesRef.current), maxReconnectDelay);
      retriesRef.current++;
      timerRef.current = setTimeout(connect, delay);
    }
  }, [url, enabled, onMessage, maxReconnectDelay]);

  useEffect(() => {
    shouldReconnectRef.current = enabled;
    if (!enabled) {
      setConnected(false);
      return () => {
        shouldReconnectRef.current = false;
      };
    }

    connect();
    return () => {
      shouldReconnectRef.current = false;
      clearTimeout(timerRef.current);
      const ws = wsRef.current;
      wsRef.current = null;
      if (ws) {
        ws.onclose = null;
        ws.close();
      }
    };
  }, [connect, enabled]);

  const send = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  return { connected, send, lastEvent };
}
