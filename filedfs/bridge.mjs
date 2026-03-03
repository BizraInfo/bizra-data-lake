// ═══════════════════════════════════════════════════════════════
//  BIZRA NODE BRIDGE — WebSocket ↔ stdio Protocol Bridge
//  Spawns bizra-node binary, proxies WS ↔ stdin/stdout
//  Standing on: VS Code LSP bridge, Jupyter kernel gateway
// ═══════════════════════════════════════════════════════════════

import { spawn } from "node:child_process";
import { WebSocketServer } from "ws";
import { resolve } from "node:path";
import { existsSync } from "node:fs";
import { EventEmitter } from "node:events";

const LOCAL_ORIGIN_PREFIXES = [
  "http://localhost",
  "https://localhost",
  "http://127.0.0.1",
  "https://127.0.0.1",
];

function isAllowedOrigin(origin) {
  if (!origin) return true;
  return LOCAL_ORIGIN_PREFIXES.some((prefix) => origin.startsWith(prefix));
}

function envFlag(name) {
  return ["1", "true", "yes", "on"].includes(
    String(process.env[name] || "").trim().toLowerCase()
  );
}

function extractAuthToken(req) {
  const auth = req?.headers?.authorization || "";
  if (auth.startsWith("Bearer ")) {
    return auth.slice("Bearer ".length).trim();
  }
  try {
    const url = new URL(req.url || "/", "http://127.0.0.1");
    return (url.searchParams.get("token") || "").trim();
  } catch {
    return "";
  }
}

function validateClientUpgrade(req) {
  const origin = req.headers.origin || null;
  if (!isAllowedOrigin(origin)) {
    return { allowed: false, reason: "origin_rejected" };
  }

  const expectedToken = String(process.env.BIZRA_BRIDGE_TOKEN || "").trim();
  const allowAnonymous = envFlag("BIZRA_BRIDGE_ALLOW_ANONYMOUS");
  if (!expectedToken && !allowAnonymous) {
    return { allowed: false, reason: "token_required" };
  }
  if (expectedToken) {
    const providedToken = extractAuthToken(req);
    if (providedToken !== expectedToken) {
      return { allowed: false, reason: "token_invalid" };
    }
  }

  return { allowed: true, reason: "ok" };
}

function sanitizeProtocolValue(value) {
  return String(value ?? "").replace(/[\t\n\r]/g, "");
}

// ─── Configuration ───
const DEFAULT_CONFIG = {
  port: 9470,                          // WebSocket port (B=9, I=4, Z=7, R=0 → BIZR)
  host: "127.0.0.1",                   // Localhost only — sovereign
  binaryPath: null,                    // Auto-detect
  userHash: "user_1",
  ihsanFloor: 9500,
  autoBanner: true,
  autoSession: true,
  healthInterval: 30_000,              // Health check every 30s
  reconnectDelay: 2_000,
  maxReconnects: 5,
  debug: process.env.BIZRA_DEBUG === "1",
};

// ─── Binary Locator ───
function findBinary() {
  const candidates = [
    resolve("./bizra-node"),
    resolve("./target/release/bizra-node"),
    resolve("./target/debug/bizra-node"),
    resolve("../bizra-workspace/target/release/bizra-node"),
    resolve("../bizra-node/target/release/bizra-node"),
  ];
  // Windows variants
  if (process.platform === "win32") {
    candidates.push(
      ...candidates.map(c => c + ".exe")
    );
  }
  for (const path of candidates) {
    if (existsSync(path)) return path;
  }
  return null;
}

// ─── Node Process Manager ───
class NodeProcess extends EventEmitter {
  constructor(config) {
    super();
    this.config = config;
    this.process = null;
    this.buffer = "";
    this.alive = false;
    this.reconnects = 0;
  }

  spawn() {
    const binaryPath = this.config.binaryPath || findBinary();
    if (!binaryPath) {
      throw new Error(
        "bizra-node binary not found. Build with: cd bizra-workspace && cargo build --release"
      );
    }

    const args = [
      "--user", this.config.userHash,
      "--ihsan", String(this.config.ihsanFloor),
    ];
    if (!this.config.autoBanner) args.push("--no-banner");
    if (!this.config.autoSession) args.push("--no-auto-session");

    this.log(`Spawning: ${binaryPath} ${args.join(" ")}`);

    this.process = spawn(binaryPath, args, {
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env },
    });

    this.alive = true;
    this.reconnects = 0;

    // stdout → parse protocol responses
    this.process.stdout.on("data", (chunk) => {
      this.buffer += chunk.toString();
      const lines = this.buffer.split("\n");
      this.buffer = lines.pop() || ""; // Keep incomplete last line
      for (const line of lines) {
        if (line.trim()) {
          this.log(`← ${line}`);
          this.emit("response", line.trim());
        }
      }
    });

    // stderr → debug log
    this.process.stderr.on("data", (chunk) => {
      this.log(`[stderr] ${chunk.toString().trim()}`);
    });

    // Process exit
    this.process.on("close", (code, signal) => {
      this.alive = false;
      this.log(`Node exited: code=${code} signal=${signal}`);
      this.emit("exit", { code, signal });

      // Auto-reconnect unless intentional shutdown
      if (code !== 0 && this.reconnects < this.config.maxReconnects) {
        this.reconnects++;
        this.log(`Reconnecting (${this.reconnects}/${this.config.maxReconnects})...`);
        setTimeout(() => this.spawn(), this.config.reconnectDelay);
      }
    });

    this.process.on("error", (err) => {
      this.log(`Process error: ${err.message}`);
      this.emit("error", err);
    });
  }

  send(command) {
    if (!this.alive || !this.process) {
      return false;
    }
    this.log(`→ ${command}`);
    this.process.stdin.write(command + "\n");
    return true;
  }

  shutdown() {
    if (this.alive && this.process) {
      this.reconnects = this.config.maxReconnects; // Prevent auto-reconnect
      this.send("SHUTDOWN");
      // Force kill after 3 seconds
      setTimeout(() => {
        if (this.process && !this.process.killed) {
          this.process.kill("SIGTERM");
        }
      }, 3000);
    }
  }

  log(msg) {
    if (this.config.debug) {
      const ts = new Date().toISOString().split("T")[1].slice(0, 12);
      console.log(`[${ts}] [node] ${msg}`);
    }
  }
}

// ─── WebSocket Server ───
class BridgeServer {
  constructor(config = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.node = new NodeProcess(this.config);
    this.wss = null;
    this.clients = new Set();
    this.pendingCallbacks = new Map();
    this.commandId = 0;
    this.healthTimer = null;
  }

  start() {
    // Spawn node
    this.node.spawn();

    // Listen for responses → broadcast to all clients
    this.node.on("response", (line) => {
      for (const ws of this.clients) {
        if (ws.readyState === 1) { // OPEN
          ws.send(JSON.stringify({
            type: "protocol",
            raw: line,
            parsed: this.parseResponse(line),
            timestamp: Date.now(),
          }));
        }
      }
    });

    this.node.on("exit", ({ code, signal }) => {
      for (const ws of this.clients) {
        if (ws.readyState === 1) {
          ws.send(JSON.stringify({
            type: "node_exit",
            code, signal,
            timestamp: Date.now(),
          }));
        }
      }
    });

    // Start WebSocket server
    this.wss = new WebSocketServer({
      port: this.config.port,
      host: this.config.host,
    });

    this.wss.on("connection", (ws, req) => {
      const validation = validateClientUpgrade(req);
      if (!validation.allowed) {
        ws.close(4001, validation.reason);
        return;
      }

      this.log(`Client connected from ${req.socket.remoteAddress}`);
      this.clients.add(ws);

      // Send welcome
      ws.send(JSON.stringify({
        type: "connected",
        node: "bizra-node",
        version: "0.1.0",
        protocol: "1.0",
        timestamp: Date.now(),
      }));

      ws.on("message", (data) => {
        try {
          const msg = JSON.parse(data.toString());
          this.handleClientMessage(ws, msg);
        } catch {
          ws.send(JSON.stringify({
            type: "error",
            code: "PARSE_ERROR",
            message: "JSON message required",
          }));
        }
      });

      ws.on("close", () => {
        this.clients.delete(ws);
        this.log("Client disconnected");
      });

      ws.on("error", (err) => {
        this.log(`Client error: ${err.message}`);
        this.clients.delete(ws);
      });
    });

    this.wss.on("listening", () => {
      console.log(`\n  ◎ BIZRA Node Bridge`);
      console.log(`  ├─ WebSocket: ws://${this.config.host}:${this.config.port}`);
      console.log(`  ├─ Node binary: ${this.config.binaryPath || findBinary() || "not found"}`);
      console.log(`  └─ Ready for connections\n`);
    });

    // Health checks
    this.healthTimer = setInterval(() => {
      if (this.node.alive) {
        this.node.send("HEALTH");
      }
    }, this.config.healthInterval);

    // Graceful shutdown
    const shutdown = () => {
      console.log("\n  Shutting down bridge...");
      clearInterval(this.healthTimer);
      this.node.shutdown();
      this.wss.close(() => {
        process.exit(0);
      });
    };
    process.on("SIGINT", shutdown);
    process.on("SIGTERM", shutdown);
  }

  handleClientMessage(ws, msg) {
    const shutdownTokenExpected = String(
      process.env.BIZRA_BRIDGE_SHUTDOWN_TOKEN || ""
    ).trim();
    const shutdownTokenProvided = String(
      msg?.shutdown_token || msg?.args?.shutdown_token || ""
    ).trim();

    const isShutdownCommand = () => {
      if (msg.type === "command") {
        return String(msg.cmd || "").toUpperCase() === "SHUTDOWN";
      }
      if (msg.type === "raw") {
        const raw = sanitizeProtocolValue(msg.line);
        return raw.toUpperCase().startsWith("SHUTDOWN");
      }
      return false;
    };

    if (isShutdownCommand()) {
      if (!shutdownTokenExpected || shutdownTokenProvided !== shutdownTokenExpected) {
        ws.send(JSON.stringify({
          type: "error",
          code: "UNAUTHORIZED_SHUTDOWN",
          message: "Valid shutdown token required",
        }));
        return;
      }
    }

    switch (msg.type) {
      case "command":
        // { type: "command", cmd: "RECEIVE", args: ["hello world"] }
        const parts = [
          sanitizeProtocolValue(msg.cmd),
          ...(msg.args || []).map((arg) => sanitizeProtocolValue(arg)),
        ];
        this.node.send(parts.join("\t"));
        break;

      case "raw":
        if (!envFlag("BIZRA_BRIDGE_ALLOW_RAW")) {
          ws.send(JSON.stringify({
            type: "error",
            code: "RAW_DISABLED",
            message: "Raw forwarding is disabled by policy",
          }));
          break;
        }
        // { type: "raw", line: "PING" } (opt-in only)
        this.node.send(sanitizeProtocolValue(msg.line));
        break;

      case "ping":
        ws.send(JSON.stringify({ type: "pong", timestamp: Date.now() }));
        break;

      default:
        ws.send(JSON.stringify({
          type: "error",
          message: `Unknown message type: ${msg.type}`,
        }));
    }
  }

  parseResponse(line) {
    const fields = line.split("\t");
    const status = fields[0]; // OK or ERR
    const data = {};
    for (let i = 1; i < fields.length; i++) {
      const eq = fields[i].indexOf("=");
      if (eq > 0) {
        const key = fields[i].slice(0, eq);
        const val = fields[i].slice(eq + 1);
        // Try to parse numbers and booleans
        if (val === "true") data[key] = true;
        else if (val === "false") data[key] = false;
        else if (/^-?\d+(\.\d+)?$/.test(val)) data[key] = parseFloat(val);
        else data[key] = val;
      }
    }
    return { status, ...data };
  }

  log(msg) {
    if (this.config.debug) {
      const ts = new Date().toISOString().split("T")[1].slice(0, 12);
      console.log(`[${ts}] [bridge] ${msg}`);
    }
  }
}

// ─── Client Library (for browser use) ───
export class BizraClient extends EventEmitter {
  constructor(url = "ws://127.0.0.1:9470") {
    super();
    this.url = url;
    this.ws = null;
    this.connected = false;
    this.reconnectTimer = null;
    this.commandQueue = [];
  }

  connect() {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        this.connected = true;
        this.emit("connected");
        // Flush queued commands
        while (this.commandQueue.length > 0) {
          const cmd = this.commandQueue.shift();
          this.ws.send(cmd);
        }
        resolve();
      };

      this.ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          this.emit("message", msg);
          if (msg.type === "protocol") {
            this.emit("protocol", msg.parsed, msg.raw);
          }
        } catch {
          this.emit("raw", event.data);
        }
      };

      this.ws.onclose = () => {
        this.connected = false;
        this.emit("disconnected");
      };

      this.ws.onerror = (err) => {
        this.emit("error", err);
        reject(err);
      };
    });
  }

  send(cmd, ...args) {
    const line = [cmd, ...args].join("\t");
    const msg = JSON.stringify({ type: "raw", line });
    if (this.connected && this.ws) {
      this.ws.send(msg);
    } else {
      this.commandQueue.push(msg);
    }
  }

  receive(content) { this.send("RECEIVE", content); }
  teach(kind, content, confidence = 9500) { this.send("TEACH", kind, content, String(confidence), String(Date.now())); }
  synthesize() { this.send("SYNTHESIZE"); }
  health() { this.send("HEALTH"); }
  knowsMe() { this.send("KNOWS_ME"); }
  ping() { this.send("PING"); }
  shutdown() { this.send("SHUTDOWN"); }

  disconnect() {
    if (this.ws) {
      this.ws.close();
    }
  }
}

// ─── CLI Entry ───
if (process.argv[1]?.endsWith("bridge.mjs") || process.argv[1]?.endsWith("bridge.js")) {
  const config = {};
  for (let i = 2; i < process.argv.length; i++) {
    switch (process.argv[i]) {
      case "--port": config.port = parseInt(process.argv[++i]); break;
      case "--binary": config.binaryPath = process.argv[++i]; break;
      case "--user": config.userHash = process.argv[++i]; break;
      case "--ihsan": config.ihsanFloor = parseInt(process.argv[++i]); break;
      case "--debug": config.debug = true; break;
    }
  }
  const bridge = new BridgeServer(config);
  bridge.start();
}

export { BridgeServer, NodeProcess };
