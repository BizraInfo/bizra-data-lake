import { createServer } from "http";
import { Server } from "socket.io";

// ═══════════════════════════════════════════════════════════════
// DEMA/BIZRA — Real-Time Orchestration WebSocket Service
// Port 3004 — Simulated multi-agent orchestration events
// ═══════════════════════════════════════════════════════════════

const PORT = 3004;

const httpServer = createServer();
const io = new Server(httpServer, {
  path: "/",
  cors: {
    origin: "*",
    methods: ["GET", "POST"],
  },
  pingTimeout: 60000,
  pingInterval: 25000,
});

// ── Simulated Agent Pool ──────────────────────────────────────

interface SimulatedAgent {
  id: string;
  name: string;
  role: string;
  status: "active" | "busy" | "idle" | "error" | "sleeping";
  cpu: number;
  memory: number;
  tasksInQueue: number;
}

const AGENT_POOL: SimulatedAgent[] = [
  { id: "ag-7a3f", name: "Nexus Prime", role: "coordinator", status: "active", cpu: 42, memory: 61, tasksInQueue: 3 },
  { id: "ag-9b2e", name: "Sentinel", role: "guardian", status: "active", cpu: 18, memory: 34, tasksInQueue: 0 },
  { id: "ag-4c1d", name: "Archer", role: "researcher", status: "busy", cpu: 73, memory: 82, tasksInQueue: 7 },
  { id: "ag-8e5a", name: "Forge", role: "executor", status: "busy", cpu: 89, memory: 76, tasksInQueue: 5 },
  { id: "ag-2f6b", name: "Oracle", role: "verifier", status: "idle", cpu: 12, memory: 28, tasksInQueue: 1 },
  { id: "ag-6d3c", name: "Spectra", role: "observer", status: "active", cpu: 31, memory: 45, tasksInQueue: 2 },
  { id: "ag-1a8e", name: "Tuner", role: "optimizer", status: "idle", cpu: 8, memory: 22, tasksInQueue: 0 },
  { id: "ag-5b7f", name: "Relay", role: "coordinator", status: "active", cpu: 55, memory: 58, tasksInQueue: 4 },
];

const TELEMETRY_MESSAGES: Record<string, string[]> = {
  gateway: [
    "Inbound request rate normalized after spike",
    "SSL certificate rotation completed successfully",
    "Rate limiter threshold adjusted: 1000 → 1200 req/s",
    "WebSocket connection pool expanded to 256 slots",
    "Health check endpoint responding within SLA",
    "CORS preflight cache refreshed for 12 origins",
  ],
  agent_runtime: [
    "Agent spawn latency: 42ms (p95)",
    "Capability registry synced — 10 capabilities active",
    "Agent heartbeat aggregation completed",
    "Task scheduler queue depth: 23 items",
    "Agent sandbox isolation verified",
    "Runtime memory pool reclaimed 128MB",
  ],
  trust_engine: [
    "Trust score recalculation completed for 3 agents",
    "Reputation graph edge weights updated",
    "Behavioral baseline drift detected: 0.3% deviation",
    "Cryptographic verification batch processed: 47 receipts",
    "Trust anchor chain validated: depth 5, integrity OK",
    "Confidence decay applied to 2 stale entries",
  ],
  receipt_chain: [
    "Receipt batch committed — block height 0x3f2a",
    "Merkle proof generated for verification request",
    "Receipt expiry sweep: 4 expired entries pruned",
    "Hash chain integrity verified: 1,247 consecutive blocks",
    "Cross-agent receipt correlation matched 3 dependencies",
    "Receipt anchoring to trust anchor completed",
  ],
  memory_store: [
    "Vector index rebuilt: 12,847 embeddings indexed",
    "Context window rotated for agent ag-4c1d",
    "Knowledge graph node count: 3,291 (+14 today)",
    "Memory consolidation completed — 23 merged entries",
    "Semantic similarity search latency: 12ms avg",
    "Memory eviction policy triggered: LRU, 8 entries freed",
  ],
  resource_registry: [
    "Resource credential rotation scheduled for 3 services",
    "Browser session pool: 2 idle, 1 active",
    "File I/O quota check passed: 67% used",
    "Terminal session timeout: cleaned 2 stale shells",
    "Knowledge base sync completed from upstream",
    "Resource dependency graph updated: 4 new edges",
  ],
  optimization_engine: [
    "Optimization cycle #847 completed — +2.3% throughput",
    "A/B test result: config-v3 outperforms baseline by 1.8%",
    "Hot path optimization applied to request handler",
    "Cache hit ratio improved to 94.2% after tuning",
    "Garbage collection pressure reduced by 18%",
    "Connection pooling optimized: 150 → 200 max connections",
  ],
  governance_layer: [
    "Boundary compliance check passed: all 12 rules OK",
    "Permission escalation request denied: insufficient trust",
    "Audit trail checkpoint written for cycle #847",
    "Governance rule G-003 triggered: rate limit warning",
    "Cryptographic boundary verification: no violations",
    "Privacy scan completed: 0 sensitive data leaks detected",
  ],
};

const ORCHESTRATION_EVENTS: {
  type: string;
  messages: string[];
  severity: string[];
  metadata: Record<string, string | number>;
}[] = [
  {
    type: "task_assigned",
    messages: [
      "Assigned data extraction task to Archer — priority high",
      "Delegated code review to Oracle for PR #237",
      "Routed optimization task to Tuner — medium priority",
      "Sent verification batch to Sentinel — 14 items",
      "Assigned research synthesis to Spectra",
    ],
    severity: ["info", "info", "info", "info", "success"],
    metadata: { priority: "high", queueDepth: 23 },
  },
  {
    type: "task_completed",
    messages: [
      "Code generation completed by Forge — 847 lines, 3 files",
      "Web research task finished by Archer — 12 sources cited",
      "Security audit completed by Sentinel — 0 findings",
      "Performance benchmark finished by Tuner — p95 improved 12%",
      "Data validation passed by Oracle — 99.7% accuracy",
    ],
    severity: ["success", "success", "success", "success", "success"],
    metadata: { duration: "4.2s", quality: "high" },
  },
  {
    type: "coordination",
    messages: [
      "Multi-agent handoff initiated: Nexus Prime → Relay for scaling",
      "Consensus reached on task ordering — 5 agents aligned",
      "Dependency resolution completed for pipeline stage 3",
      "Resource contention detected — reassigning 2 tasks to Forge",
      "Cross-agent state synchronization completed in 89ms",
    ],
    severity: ["info", "success", "info", "warning", "success"],
    metadata: { participants: 5, protocol: "raft" },
  },
  {
    type: "handoff",
    messages: [
      "Task ownership transferred: Archer → Forge (web_search results)",
      "Coordinator handoff: Nexus Prime → Relay (load balancing)",
      "Receipt chain custody transferred to Sentinel",
    ],
    severity: ["info", "info", "success"],
    metadata: { transferType: "hot", acknowledged: true },
  },
  {
    type: "task_failed",
    messages: [
      "Forge: file write failed — permission denied on /tmp/output",
      "Archer: web scrape timeout — target site rate limited",
      "Oracle: verification hash mismatch — retrying with fresh data",
    ],
    severity: ["error", "warning", "warning"],
    metadata: { retryCount: 2, maxRetries: 3 },
  },
  {
    type: "agent_spawn",
    messages: [
      "New executor agent 'Vortex' spawned for parallel processing",
      "Temporary researcher spawned — scope: limited web crawl",
    ],
    severity: ["info", "info"],
    metadata: { ttl: "1h", capabilities: 4 },
  },
  {
    type: "termination",
    messages: [
      "Agent 'Spark' terminated — TTL expired after 2h idle",
      "Temporary researcher decommissioned — task scope completed",
    ],
    severity: ["info", "info"],
    metadata: { reason: "ttl_expired", graceful: true },
  },
];

const GOVERNANCE_RULES = [
  { id: "G-001", name: "Agent Boundary Isolation", category: "boundary" },
  { id: "G-002", name: "Permission Escalation Control", category: "permission" },
  { id: "G-003", name: "Request Rate Threshold", category: "performance" },
  { id: "G-004", name: "Data Integrity Verification", category: "integrity" },
  { id: "G-005", name: "PII Redaction Enforcement", category: "privacy" },
  { id: "G-006", name: "Cryptographic Non-Repudiation", category: "integrity" },
  { id: "G-007", name: "Agent Trust Floor", category: "boundary" },
  { id: "G-008", name: "Resource Quota Compliance", category: "boundary" },
  { id: "G-009", name: "Audit Trail Completeness", category: "integrity" },
  { id: "G-010", name: "Concurrent Session Limit", category: "performance" },
  { id: "G-011", name: "Memory Consumption Cap", category: "performance" },
  { id: "G-012", name: "External Service Auth", category: "permission" },
];

// ── Utility Functions ─────────────────────────────────────────

function randomPick<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

function randomBetween(min: number, max: number): number {
  return Math.round((Math.random() * (max - min) + min) * 100) / 100;
}

function generateTraceId(): string {
  return `trace-${Date.now().toString(36)}-${Math.random().toString(36).substring(2, 10)}`;
}

function jitter(base: number, range: number): number {
  return Math.round(base + (Math.random() - 0.5) * 2 * range);
}

// ── Event Generators ──────────────────────────────────────────

function generateHeartbeat() {
  // Slightly mutate agent states for realism
  for (const agent of AGENT_POOL) {
    agent.cpu = Math.max(2, Math.min(99, jitter(agent.cpu, 8)));
    agent.memory = Math.max(10, Math.min(98, jitter(agent.memory, 5)));
    agent.tasksInQueue = Math.max(0, jitter(agent.tasksInQueue, 2));

    // Occasionally flip status
    const roll = Math.random();
    if (roll < 0.03) {
      agent.status = randomPick(["active", "busy", "idle", "active"]);
    } else if (roll < 0.01) {
      agent.status = "error";
    }
  }

  return {
    agents: AGENT_POOL.map((a) => ({
      id: a.id,
      name: a.name,
      role: a.role,
      status: a.status,
      cpu: a.cpu,
      memory: a.memory,
      tasksInQueue: a.tasksInQueue,
    })),
    totalAgents: AGENT_POOL.length,
    activeAgents: AGENT_POOL.filter((a) => a.status === "active").length,
    busyAgents: AGENT_POOL.filter((a) => a.status === "busy").length,
    idleAgents: AGENT_POOL.filter((a) => a.status === "idle").length,
    errorAgents: AGENT_POOL.filter((a) => a.status === "error").length,
    timestamp: new Date().toISOString(),
  };
}

function generateTelemetry() {
  const components = Object.keys(TELEMETRY_MESSAGES) as string[];
  const component = randomPick(components);
  const messages = TELEMETRY_MESSAGES[component];
  const level = randomPick(["debug", "info", "info", "info", "warn"] as const);
  const message = randomPick(messages);

  return {
    level,
    component,
    message,
    metadata: {
      agentCount: AGENT_POOL.length,
      activeTasks: AGENT_POOL.reduce((s, a) => s + a.tasksInQueue, 0),
      serviceUptime: process.uptime().toFixed(1) + "s",
      nodeId: `node-${Math.floor(Math.random() * 3)}`,
    },
    timestamp: new Date().toISOString(),
    traceId: level !== "debug" ? generateTraceId() : undefined,
    spanId: level !== "debug" ? `span-${Math.random().toString(36).substring(2, 10)}` : undefined,
  };
}

function generatePerformanceSnapshot() {
  return {
    cpu: randomBetween(25, 85),
    memory: randomBetween(40, 78),
    diskIo: randomBetween(0.5, 45),
    networkIo: randomBetween(1.2, 120),
    activeConnections: jitter(142, 30),
    requestRate: randomBetween(180, 950),
    errorRate: randomBetween(0.01, 2.5),
    p50: jitter(12, 6),
    p95: jitter(85, 25),
    p99: jitter(210, 60),
    timestamp: new Date().toISOString(),
  };
}

function generateOrchestrationEvent() {
  const template = randomPick(ORCHESTRATION_EVENTS);
  const agent = randomPick(AGENT_POOL);

  return {
    type: template.type,
    agentId: agent.id,
    agentName: agent.name,
    agentRole: agent.role,
    message: randomPick(template.messages),
    severity: randomPick(template.severity),
    metadata: {
      ...template.metadata,
      taskId: `task-${Math.random().toString(36).substring(2, 9)}`,
      coordinatorId: AGENT_POOL[0].id,
    },
    timestamp: new Date().toISOString(),
  };
}

function generateGovernanceCheck() {
  const rulesChecked = GOVERNANCE_RULES.length;
  // Occasionally create a violation for realism (10% chance)
  const hasViolation = Math.random() < 0.1;
  const violationCount = hasViolation ? jitter(1, 1) : 0;

  const violations = [];
  if (hasViolation) {
    const violatedRule = randomPick(GOVERNANCE_RULES);
    violations.push({
      ruleId: violatedRule.id,
      ruleName: violatedRule.name,
      category: violatedRule.category,
      severity: randomPick(["low", "medium"] as const),
      description:
        violatedRule.category === "performance"
          ? "Threshold exceeded — auto-mitigation applied"
          : "Anomalous activity detected — monitoring escalated",
      agentId: randomPick(AGENT_POOL).id,
      action: randomPick(["audit", "escalate"] as const),
    });
  }

  return {
    rulesChecked,
    violations,
    violationCount,
    status: hasViolation ? "warning" : "compliant",
    checkedAt: new Date().toISOString(),
    timestamp: new Date().toISOString(),
  };
}

// ── Socket.IO Server ─────────────────────────────────────────

const connectedClients = new Set<string>();

io.on("connection", (socket) => {
  connectedClients.add(socket.id);
  console.log(`[orchestration] Client connected: ${socket.id} (${connectedClients.size} total)`);

  // Send immediate welcome event with service info
  socket.emit("orchestration:connected", {
    service: "dema-orchestration",
    version: "1.0.0",
    port: PORT,
    agents: AGENT_POOL.length,
    timestamp: new Date().toISOString(),
    message: "Connected to DEMA/BIZRA orchestration real-time stream",
  });

  // Start all interval emitters for this socket
  const intervals: ReturnType<typeof setInterval>[] = [];

  // 1. Heartbeat — every 5 seconds
  intervals.push(
    setInterval(() => {
      try {
        socket.emit("agent:heartbeat", generateHeartbeat());
      } catch {
        // client may have disconnected
      }
    }, 5000)
  );

  // 2. Telemetry — every 3 seconds
  intervals.push(
    setInterval(() => {
      try {
        socket.emit("telemetry:event", generateTelemetry());
      } catch {
        // client may have disconnected
      }
    }, 3000)
  );

  // 3. Performance — every 10 seconds
  intervals.push(
    setInterval(() => {
      try {
        socket.emit("performance:snapshot", generatePerformanceSnapshot());
      } catch {
        // client may have disconnected
      }
    }, 10000)
  );

  // 4. Orchestration — every 8 seconds
  intervals.push(
    setInterval(() => {
      try {
        socket.emit("orchestration:event", generateOrchestrationEvent());
      } catch {
        // client may have disconnected
      }
    }, 8000)
  );

  // 5. Governance — every 15 seconds
  intervals.push(
    setInterval(() => {
      try {
        socket.emit("governance:check", generateGovernanceCheck());
      } catch {
        // client may have disconnected
      }
    }, 15000)
  );

  // Cleanup on disconnect
  socket.on("disconnect", (reason) => {
    connectedClients.delete(socket.id);
    console.log(`[orchestration] Client disconnected: ${socket.id} (${reason}) — ${connectedClients.size} remaining`);
    for (const interval of intervals) {
      clearInterval(interval);
    }
  });

  socket.on("error", (error) => {
    console.error(`[orchestration] Socket error (${socket.id}):`, error);
  });

  // Allow clients to request an immediate event burst
  socket.on("orchestration:burst", () => {
    console.log(`[orchestration] Burst requested by ${socket.id}`);
    socket.emit("agent:heartbeat", generateHeartbeat());
    socket.emit("telemetry:event", generateTelemetry());
    socket.emit("telemetry:event", generateTelemetry());
    socket.emit("performance:snapshot", generatePerformanceSnapshot());
    socket.emit("orchestration:event", generateOrchestrationEvent());
    socket.emit("orchestration:event", generateOrchestrationEvent());
    socket.emit("governance:check", generateGovernanceCheck());
  });
});

// ── Start Server ──────────────────────────────────────────────

httpServer.listen(PORT, () => {
  console.log("═══════════════════════════════════════════════════");
  console.log("  DEMA/BIZRA — Orchestration WebSocket Service");
  console.log(`  Port: ${PORT}`);
  console.log(`  Path: / (Caddy-forwarded)`);
  console.log("  Events:");
  console.log("    • agent:heartbeat      — every 5s");
  console.log("    • telemetry:event      — every 3s");
  console.log("    • performance:snapshot  — every 10s");
  console.log("    • orchestration:event   — every 8s");
  console.log("    • governance:check      — every 15s");
  console.log("═══════════════════════════════════════════════════");
});

// Graceful shutdown
function shutdown(signal: string) {
  console.log(`\n[orchestration] Received ${signal}, shutting down gracefully...`);
  for (const clientId of connectedClients) {
    console.log(`[orchestration] Closing connection: ${clientId}`);
  }
  io.close();
  httpServer.close(() => {
    console.log("[orchestration] Server closed.");
    process.exit(0);
  });
  // Force exit after 5s if graceful shutdown hangs
  setTimeout(() => process.exit(1), 5000);
}

process.on("SIGTERM", () => shutdown("SIGTERM"));
process.on("SIGINT", () => shutdown("SIGINT"));
