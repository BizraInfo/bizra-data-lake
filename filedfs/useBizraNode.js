// ═══════════════════════════════════════════════════════════════
//  useBizraNode — React Hook for Node0 Protocol
//  Auto-detects: Tauri native IPC vs WebSocket bridge
//  Standing on: React hooks, Tauri invoke, EventEmitter pattern
// ═══════════════════════════════════════════════════════════════

import { useState, useCallback, useRef, useEffect } from "react";

// Detect if running inside Tauri
const IS_TAURI = typeof window !== "undefined" && window.__TAURI__;

// Parse tab-delimited protocol response
function parseResponse(raw) {
  const fields = raw.split("\t");
  const status = fields[0]; // OK or ERR
  const data = {};
  for (let i = 1; i < fields.length; i++) {
    const eq = fields[i].indexOf("=");
    if (eq > 0) {
      const key = fields[i].slice(0, eq);
      const val = fields[i].slice(eq + 1);
      if (val === "true") data[key] = true;
      else if (val === "false") data[key] = false;
      else if (/^-?\d+(\.\d+)?$/.test(val)) data[key] = parseFloat(val);
      else data[key] = val;
    }
  }
  return { status, ok: status === "OK", ...data };
}

// ─── Tauri Transport ───
class TauriTransport {
  constructor() {
    this.connected = false;
  }

  async connect(config = {}) {
    const { invoke } = await import("@tauri-apps/api/core");
    this.invoke = invoke;

    const result = await invoke("spawn_node", {
      binaryPath: config.binaryPath || null,
      userHash: config.userHash || "user_1",
      ihsanFloor: config.ihsanFloor || 9500,
    });

    this.connected = true;
    return parseResponse(result);
  }

  async send(command) {
    if (!this.connected || !this.invoke) {
      throw new Error("Not connected");
    }
    const raw = await this.invoke("send_command", { command });
    return { raw, parsed: parseResponse(raw) };
  }

  async disconnect() {
    if (this.invoke) {
      await this.invoke("shutdown_node");
    }
    this.connected = false;
  }
}

// ─── WebSocket Transport ───
class WebSocketTransport {
  constructor() {
    this.ws = null;
    this.connected = false;
    this.callbacks = [];
    this.responseQueue = [];
    this.resolveNext = null;
  }

  connect(config = {}) {
    const url = config.wsUrl || "ws://127.0.0.1:9470";

    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        this.connected = true;
        resolve({ ok: true, transport: "websocket" });
      };

      this.ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === "protocol" && this.resolveNext) {
            const resolver = this.resolveNext;
            this.resolveNext = null;
            resolver({ raw: msg.raw, parsed: msg.parsed });
          }
          for (const cb of this.callbacks) {
            cb(msg);
          }
        } catch {
          // ignore
        }
      };

      this.ws.onclose = () => {
        this.connected = false;
      };

      this.ws.onerror = (err) => {
        reject(err);
      };
    });
  }

  send(command) {
    if (!this.connected || !this.ws) {
      return Promise.reject(new Error("Not connected"));
    }

    return new Promise((resolve) => {
      this.resolveNext = resolve;
      this.ws.send(JSON.stringify({ type: "raw", line: command }));

      // Timeout fallback
      setTimeout(() => {
        if (this.resolveNext === resolve) {
          this.resolveNext = null;
          resolve({ raw: "ERR\ttimeout=true", parsed: { status: "ERR", ok: false, timeout: true } });
        }
      }, 5000);
    });
  }

  onMessage(cb) {
    this.callbacks.push(cb);
  }

  async disconnect() {
    if (this.ws) {
      this.ws.close();
    }
    this.connected = false;
  }
}

// ─── React Hook ───
export function useBizraNode(config = {}) {
  const [connected, setConnected] = useState(false);
  const [nodeState, setNodeState] = useState("Dormant");
  const [knowsMe, setKnowsMe] = useState(0);
  const [ihsan, setIhsan] = useState(config.ihsanFloor || 9500);
  const [protocolLog, setProtocolLog] = useState([]);
  const [traits, setTraits] = useState([]);
  const [error, setError] = useState(null);

  const transportRef = useRef(null);

  const addLog = useCallback((cmd, response) => {
    const ts = new Date().toLocaleTimeString("en-GB");
    setProtocolLog(prev => [...prev.slice(-100), { cmd, response, timestamp: ts }]);
  }, []);

  const connect = useCallback(async () => {
    try {
      const transport = IS_TAURI ? new TauriTransport() : new WebSocketTransport();
      await transport.connect(config);
      transportRef.current = transport;
      setConnected(true);
      setError(null);
      return true;
    } catch (err) {
      setError(err.message);
      return false;
    }
  }, [config]);

  const send = useCallback(async (command) => {
    if (!transportRef.current?.connected) {
      return { ok: false, error: "Not connected" };
    }
    try {
      const result = await transportRef.current.send(command);
      addLog(command, result.raw);

      // Update local state from responses
      const p = result.parsed;
      if (p.knows_me !== undefined) setKnowsMe(p.knows_me);
      if (p.ihsan !== undefined) setIhsan(p.ihsan);
      if (p.state !== undefined) setNodeState(p.state);

      return result.parsed;
    } catch (err) {
      addLog(command, `ERR\t${err.message}`);
      return { ok: false, error: err.message };
    }
  }, [addLog]);

  // Convenience methods
  const receive = useCallback((content) => send(`RECEIVE\t${content}`), [send]);
  const teach = useCallback((kind, content, confidence = 9500) =>
    send(`TEACH\t${kind}\t${content}\t${confidence}\t${Date.now()}`), [send]);
  const synthesize = useCallback(() => send("SYNTHESIZE"), [send]);
  const health = useCallback(() => send("HEALTH"), [send]);
  const queryKnowsMe = useCallback(() => send("KNOWS_ME"), [send]);
  const ping = useCallback(() => send("PING"), [send]);
  const version = useCallback(() => send("VERSION"), [send]);
  const shutdown = useCallback(async () => {
    await send("SHUTDOWN");
    setConnected(false);
    setNodeState("Dormant");
  }, [send]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (transportRef.current?.connected) {
        transportRef.current.disconnect();
      }
    };
  }, []);

  return {
    // State
    connected,
    nodeState,
    knowsMe,
    ihsan,
    protocolLog,
    traits,
    error,
    isTauri: IS_TAURI,

    // Actions
    connect,
    send,
    receive,
    teach,
    synthesize,
    health,
    queryKnowsMe,
    ping,
    version,
    shutdown,
  };
}

export { parseResponse, IS_TAURI };
