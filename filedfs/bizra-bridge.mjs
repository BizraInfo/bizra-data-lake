#!/usr/bin/env node
// ============================================================
// bizra-bridge.js — WebSocket ↔ stdio bridge
// ============================================================
// Standing on giants:
//   VS Code LSP    → spawn binary, pipe stdin/stdout
//   Jupyter Kernel  → WS gateway to compute process  
//   Chrome DevTools → WS protocol to running process
//
// Architecture:
//   Browser ←→ WebSocket ←→ Bridge ←→ stdio ←→ bizra-node
//
// The bridge is intentionally thin. It:
//   1. Spawns bizra-node as a child process
//   2. Opens a WebSocket server
//   3. Forwards WS messages → stdin (as protocol lines)
//   4. Forwards stdout lines → WS messages (as JSON)
//   5. That's it. All intelligence is in the binary.
//
// Usage:
//   node bizra-bridge.js                     # defaults
//   node bizra-bridge.js --port 9100         # custom port
//   node bizra-bridge.js --binary ./bizra-node --port 9100
//
// WS Protocol (JSON over WebSocket):
//   Client → Bridge:  { "verb": "RECEIVE", "args": { "content": "hello", "timestamp": 1234 } }
//   Bridge → Client:  { "ok": true, "fields": { "content": "...", "confidence": "0.85" } }
//                  or { "ok": false, "code": "BAD_COMMAND", "message": "..." }
//   Bridge → Client:  { "event": "started", ... }  (on connect)
// ============================================================

import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { WebSocketServer } from "ws"; // npm install ws
import { createInterface } from "node:readline";
import { resolve } from "node:path";

// ============================================================
// CONFIGURATION
// ============================================================

const DEFAULT_PORT = 9100;
const DEFAULT_BINARY = "./bizra-node";

function parseArgs() {
  const args = process.argv.slice(2);
  const config = {
    port: DEFAULT_PORT,
    binary: DEFAULT_BINARY,
    binaryArgs: ["--no-banner"],
  };

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case "--port":
        config.port = parseInt(args[++i], 10);
        break;
      case "--binary":
        config.binary = resolve(args[++i]);
        break;
      case "--ihsan":
        config.binaryArgs.push("--ihsan", args[++i]);
        break;
      case "--user":
        config.binaryArgs.push("--user", args[++i]);
        break;
      case "--help":
        console.log(`
bizra-bridge — WebSocket bridge to bizra-node

Usage: node bizra-bridge.js [options]

Options:
  --port <num>      WebSocket port (default: ${DEFAULT_PORT})
  --binary <path>   Path to bizra-node binary (default: ${DEFAULT_BINARY})
  --ihsan <score>   إحسان floor for the node (0-10000)
  --user <hash>     User identity hash
  --help            Show this help
        `);
        process.exit(0);
    }
  }

  return config;
}

// ============================================================
// PROTOCOL TRANSLATION
// ============================================================

/**
 * Convert a JSON WS message to a bizra-node protocol line.
 *
 * Input:  { verb: "RECEIVE", args: { content: "hello", timestamp: 1234 } }
 * Output: "RECEIVE\thello\t1234"
 */
function jsonToProtocol(msg) {
  const { verb, args = {} } = msg;

  switch (verb?.toUpperCase()) {
    case "RECEIVE":
      return `RECEIVE\t${escapeTab(args.content || "")}\t${args.timestamp || Date.now()}`;
    case "TEACH":
      return `TEACH\t${args.kind || "fact"}\t${escapeTab(args.content || "")}\t${args.confidence || 9000}\t${args.timestamp || Date.now()}`;
    case "SYNTHESIZE":
      return `SYNTHESIZE\t${args.timestamp || Date.now()}`;
    case "QUERY":
      return `QUERY\t${args.key || ""}`;
    case "START_SESSION":
      return `START_SESSION\t${args.timestamp || Date.now()}`;
    case "END_SESSION":
      return `END_SESSION\t${args.timestamp || Date.now()}`;
    case "IHSAN":
      return `IHSAN\t${args.score || 9900}`;
    case "PROFILE":
      return "PROFILE";
    case "KNOWS_ME":
      return "KNOWS_ME";
    case "HEALTH":
      return "HEALTH";
    case "PING":
      return "PING";
    case "VERSION":
      return "VERSION";
    case "SHUTDOWN":
      return "SHUTDOWN";

    // SAP v0 protocol verbs
    case "SAP_MEET_OPEN":
      return `SAP_MEET_OPEN\t${escapeTab(args.profile || "sap-ads-retail-v0")}\t${escapeTab(args.initiator_role || "visitor")}\t${args.timestamp || Date.now()}`;
    case "SAP_MESSAGE":
      return `SAP_MESSAGE\t${escapeTab(args.session_id || "")}\t${escapeTab(args.content || "")}\t${args.timestamp || Date.now()}`;
    case "SAP_DISCLOSURE":
      return `SAP_DISCLOSURE\t${escapeTab(args.session_id || "")}`;
    case "SAP_CONSENT_REQUEST":
      return `SAP_CONSENT_REQUEST\t${escapeTab(args.session_id || "")}\t${escapeTab(JSON.stringify(args.scopes || []))}`;
    case "SAP_CONSENT_REVOKE":
      return `SAP_CONSENT_REVOKE\t${escapeTab(args.session_id || "")}\t${escapeTab(args.receipt_id || "")}`;
    case "SAP_SESSION_CLOSE":
      return `SAP_SESSION_CLOSE\t${escapeTab(args.session_id || "")}\t${args.timestamp || Date.now()}`;

    default:
      return null; // Unknown verb
  }
}

/**
 * Parse a bizra-node protocol response line into JSON.
 *
 * Input:  "OK\tcontent=hello world\tconfidence=0.85"
 * Output: { ok: true, fields: { content: "hello world", confidence: "0.85" } }
 *
 * Input:  "ERR\tBAD_COMMAND\tUnknown command: BOGUS"
 * Output: { ok: false, code: "BAD_COMMAND", message: "Unknown command: BOGUS" }
 */
function protocolToJson(line) {
  const parts = line.split("\t");

  if (parts[0] === "OK") {
    const fields = {};
    for (let i = 1; i < parts.length; i++) {
      const eqIdx = parts[i].indexOf("=");
      if (eqIdx > 0) {
        const key = parts[i].slice(0, eqIdx);
        const value = unescapeTab(parts[i].slice(eqIdx + 1));
        fields[key] = value;
      }
    }
    return { ok: true, fields };
  }

  if (parts[0] === "ERR") {
    return {
      ok: false,
      code: parts[1] || "UNKNOWN",
      message: parts[2] || "Unknown error",
    };
  }

  // Unknown format — pass through
  return { ok: true, raw: line };
}

function escapeTab(s) {
  return s.replace(/\\/g, "\\\\").replace(/\t/g, "\\t").replace(/\n/g, "\\n").replace(/\r/g, "\\r");
}

function unescapeTab(s) {
  let result = "";
  let i = 0;
  while (i < s.length) {
    if (s[i] === "\\" && i + 1 < s.length) {
      switch (s[i + 1]) {
        case "t": result += "\t"; i += 2; continue;
        case "n": result += "\n"; i += 2; continue;
        case "r": result += "\r"; i += 2; continue;
        case "\\": result += "\\"; i += 2; continue;
      }
    }
    result += s[i];
    i++;
  }
  return result;
}

// ============================================================
// NODE PROCESS MANAGER
// ============================================================

class NodeProcess {
  constructor(binaryPath, args = []) {
    this.binaryPath = binaryPath;
    this.args = args;
    this.process = null;
    this.readline = null;
    this.responseQueue = []; // Pending response callbacks
    this.alive = false;
  }

  start() {
    return new Promise((resolve, reject) => {
      console.log(`[bridge] Spawning: ${this.binaryPath} ${this.args.join(" ")}`);

      this.process = spawn(this.binaryPath, this.args, {
        stdio: ["pipe", "pipe", "pipe"],
      });

      this.alive = true;

      // Read stdout line by line
      this.readline = createInterface({ input: this.process.stdout });
      this.readline.on("line", (line) => {
        this._onLine(line.trim());
      });

      // Stderr → log
      this.process.stderr.on("data", (data) => {
        console.error(`[node stderr] ${data.toString().trim()}`);
      });

      this.process.on("error", (err) => {
        console.error(`[bridge] Failed to spawn: ${err.message}`);
        this.alive = false;
        reject(err);
      });

      this.process.on("exit", (code, signal) => {
        console.log(`[bridge] Node exited: code=${code}, signal=${signal}`);
        this.alive = false;
        // Reject any pending responses
        while (this.responseQueue.length > 0) {
          const cb = this.responseQueue.shift();
          cb({ ok: false, code: "PROCESS_EXIT", message: "Node process exited" });
        }
      });

      // Give it a moment to start
      setTimeout(() => {
        if (this.alive) {
          resolve();
        }
      }, 200);
    });
  }

  /**
   * Send a protocol line and wait for the response.
   * Returns a promise that resolves with the JSON response.
   */
  send(protocolLine) {
    return new Promise((resolve) => {
      if (!this.alive) {
        resolve({ ok: false, code: "NOT_RUNNING", message: "Node is not running" });
        return;
      }

      this.responseQueue.push(resolve);
      this.process.stdin.write(protocolLine + "\n");
    });
  }

  _onLine(line) {
    if (!line) return;

    const json = protocolToJson(line);

    // If there's a pending request, resolve it
    if (this.responseQueue.length > 0) {
      const cb = this.responseQueue.shift();
      cb(json);
    } else {
      // Unsolicited message (e.g., startup banner)
      console.log(`[node] Unsolicited: ${JSON.stringify(json)}`);
    }
  }

  stop() {
    if (this.alive && this.process) {
      this.process.stdin.write("SHUTDOWN\n");
      setTimeout(() => {
        if (this.alive) {
          this.process.kill("SIGTERM");
        }
      }, 2000);
    }
  }
}

// ============================================================
// WEBSOCKET SERVER
// ============================================================

async function main() {
  const config = parseArgs();

  // Spawn the node
  const node = new NodeProcess(config.binary, config.binaryArgs);

  try {
    await node.start();
    console.log(`[bridge] Node started successfully`);
  } catch (err) {
    console.error(`[bridge] Failed to start node: ${err.message}`);
    console.error(`[bridge] Make sure bizra-node binary exists at: ${config.binary}`);
    process.exit(1);
  }

  // HTTP server (for potential REST endpoints later)
  const httpServer = createServer((req, res) => {
    if (req.url === "/health") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ status: "ok", node_alive: node.alive }));
      return;
    }
    res.writeHead(200, { "Content-Type": "text/plain" });
    res.end("bizra-bridge — WebSocket bridge to bizra-node\n");
  });

  // WebSocket server
  const wss = new WebSocketServer({ server: httpServer });

  wss.on("connection", (ws, req) => {
    const addr = req.socket.remoteAddress;
    console.log(`[bridge] Client connected: ${addr}`);

    // Send connected event
    ws.send(JSON.stringify({
      event: "connected",
      node: "bizra-node",
      version: "0.1.0",
      protocol: "1.0",
    }));

    ws.on("message", async (data) => {
      let msg;
      try {
        msg = JSON.parse(data.toString());
      } catch {
        ws.send(JSON.stringify({ ok: false, code: "PARSE_ERROR", message: "Invalid JSON" }));
        return;
      }

      // Translate to protocol
      const protocolLine = jsonToProtocol(msg);
      if (!protocolLine) {
        ws.send(JSON.stringify({ ok: false, code: "BAD_COMMAND", message: `Unknown verb: ${msg.verb}` }));
        return;
      }

      // Send to node and wait for response
      const response = await node.send(protocolLine);

      // Add the original verb for client-side routing
      response.verb = msg.verb;

      ws.send(JSON.stringify(response));

      // If shutdown, close gracefully
      if (msg.verb?.toUpperCase() === "SHUTDOWN") {
        setTimeout(() => {
          wss.clients.forEach((c) => c.close());
          httpServer.close();
          process.exit(0);
        }, 500);
      }
    });

    ws.on("close", () => {
      console.log(`[bridge] Client disconnected: ${addr}`);
    });

    ws.on("error", (err) => {
      console.error(`[bridge] WS error: ${err.message}`);
    });
  });

  httpServer.listen(config.port, () => {
    console.log(`[bridge] WebSocket bridge listening on ws://localhost:${config.port}`);
    console.log(`[bridge] Health check: http://localhost:${config.port}/health`);
    console.log(`[bridge] Ready for Alpha-100 connections`);
  });

  // Graceful shutdown
  process.on("SIGINT", () => {
    console.log("\n[bridge] Shutting down...");
    node.stop();
    wss.clients.forEach((c) => c.close());
    httpServer.close(() => process.exit(0));
  });

  process.on("SIGTERM", () => {
    node.stop();
    wss.clients.forEach((c) => c.close());
    httpServer.close(() => process.exit(0));
  });
}

main().catch(console.error);
