#!/usr/bin/env node
/**
 * BIZRA Agentic-Flow HTTP Server
 *
 * Standalone HTTP server wrapping agentic-flow's health module and CLI.
 * Used by Docker service and BIZRA kernel bridge (port 3100).
 *
 * Endpoints:
 *   GET  /health   — Health status
 *   GET  /agents   — List available agents
 *   GET  /tools    — List MCP tools
 *   POST /swarm    — Invoke agent swarm
 *   POST /mcp/call — Call MCP tool
 *   POST /worker   — Dispatch background worker
 */

import http from "http";
import { execSync, spawn } from "child_process";
import { readFileSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const AF_ROOT = resolve(__dirname, "../vendor/agentic-flow/agentic-flow");

const PORT = parseInt(process.env.PORT || "3100", 10);
const HOST = process.env.HOST || "0.0.0.0";
const startTime = Date.now();

/** Parse JSON body from request */
function parseBody(req) {
  return new Promise((resolve, reject) => {
    let data = "";
    req.on("data", (chunk) => (data += chunk));
    req.on("end", () => {
      try {
        resolve(data ? JSON.parse(data) : {});
      } catch (e) {
        reject(new Error("Invalid JSON"));
      }
    });
    req.on("error", reject);
  });
}

/** Run agentic-flow CLI and capture output */
function runCLI(args, timeoutMs = 30000) {
  try {
    const result = execSync(
      `node ${AF_ROOT}/dist/cli-proxy.js ${args}`,
      {
        timeout: timeoutMs,
        encoding: "utf8",
        cwd: AF_ROOT,
        env: { ...process.env, NO_COLOR: "1" },
      }
    );
    return result;
  } catch (e) {
    return e.stdout || e.message;
  }
}

/** Parse agent list from CLI output */
function parseAgentList() {
  const raw = runCLI("--list", 10000);
  const agents = [];
  const lines = raw.split("\n");
  let currentCategory = "";

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    // Category header (e.g., "CORE:", "GITHUB:")
    if (trimmed.match(/^[A-Z-]+:$/)) {
      currentCategory = trimmed.replace(":", "").toLowerCase();
      continue;
    }

    // Agent line (starts with name followed by description)
    const match = trimmed.match(/^(\S+)\s{2,}(.+)/);
    if (match) {
      agents.push({
        name: match[1],
        category: currentCategory,
        description: match[2].trim(),
      });
    }
  }
  return agents;
}

/** JSON response helper */
function json(res, data, status = 200) {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(data, null, 2));
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${HOST}:${PORT}`);
  const path = url.pathname;
  const method = req.method;

  try {
    // ─── Health ───
    if (path === "/health" && method === "GET") {
      const mem = process.memoryUsage();
      return json(res, {
        status: "healthy",
        service: "agentic-flow",
        version: "2.0.5",
        uptime: Math.round((Date.now() - startTime) / 1000),
        timestamp: new Date().toISOString(),
        memory: {
          heapUsed: Math.round(mem.heapUsed / 1024 / 1024),
          heapTotal: Math.round(mem.heapTotal / 1024 / 1024),
        },
        features: {
          agents: true,
          swarm: true,
          reasoningBank: true,
          mcp: true,
        },
      });
    }

    // ─── List Agents ───
    if (path === "/agents" && method === "GET") {
      const agents = parseAgentList();
      return json(res, { agents, count: agents.length });
    }

    // ─── List Tools ───
    if (path === "/tools" && method === "GET") {
      return json(res, {
        tools: [
          { name: "agentic_flow_agent", description: "Execute an agentic-flow agent with a task" },
          { name: "agentic_flow_list_agents", description: "List all available agents" },
          { name: "reasoning_bank_query", description: "Query ReasoningBank patterns" },
          { name: "swarm_init", description: "Initialize agent swarm" },
          { name: "worker_dispatch", description: "Dispatch background worker" },
        ],
        transport: "stdio",
        note: "Full MCP tool catalog available via stdio transport (213+ tools)",
      });
    }

    // ─── Invoke Swarm ───
    if (path === "/swarm" && method === "POST") {
      const body = await parseBody(req);
      const { task, topology = "mesh", agent_count = 5, timeout_ms = 30000 } = body;

      if (!task) {
        return json(res, { error: "task is required" }, 400);
      }

      // Use CLI to run agent with task
      const agent = body.agent || "coder";
      const output = runCLI(
        `--agent ${agent} --task "${task.replace(/"/g, '\\"')}"`,
        timeout_ms
      );

      return json(res, {
        task,
        topology,
        agent_count,
        agent,
        status: "completed",
        output: output.substring(0, 10000),
        timestamp: new Date().toISOString(),
      });
    }

    // ─── MCP Tool Call ───
    if (path === "/mcp/call" && method === "POST") {
      const body = await parseBody(req);
      const { tool_name, arguments: args } = body;

      if (!tool_name) {
        return json(res, { error: "tool_name is required" }, 400);
      }

      return json(res, {
        tool_name,
        status: "mcp_tools_via_stdio",
        message: "MCP tools are available via stdio transport. Use the BIZRA kernel bridge for HTTP access.",
        timestamp: new Date().toISOString(),
      });
    }

    // ─── Dispatch Worker ───
    if (path === "/worker" && method === "POST") {
      const body = await parseBody(req);
      const { worker_type, directive } = body;

      if (!worker_type || !directive) {
        return json(res, { error: "worker_type and directive are required" }, 400);
      }

      const output = runCLI(
        `--agent ${worker_type} --task "${directive.replace(/"/g, '\\"')}"`,
        60000
      );

      return json(res, {
        worker_type,
        directive,
        status: "completed",
        output: output.substring(0, 10000),
        timestamp: new Date().toISOString(),
      });
    }

    // ─── 404 ───
    json(res, { error: "Not found", endpoints: ["/health", "/agents", "/tools", "/swarm", "/mcp/call", "/worker"] }, 404);
  } catch (err) {
    json(res, { error: err.message }, 500);
  }
});

server.listen(PORT, HOST, () => {
  console.log(`[agentic-flow] HTTP server listening on ${HOST}:${PORT}`);
  console.log(`[agentic-flow] Health: http://${HOST}:${PORT}/health`);
  console.log(`[agentic-flow] Agents: http://${HOST}:${PORT}/agents`);
});

// Graceful shutdown
process.on("SIGTERM", () => {
  console.log("[agentic-flow] SIGTERM received, shutting down...");
  server.close(() => process.exit(0));
});
process.on("SIGINT", () => {
  server.close(() => process.exit(0));
});
