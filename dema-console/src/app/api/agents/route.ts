// ═══════════════════════════════════════════════════════════════
// BIZRA — Agents API
// Manage orchestration agents: list, filter, register.
// ═══════════════════════════════════════════════════════════════

import { NextRequest } from "next/server";
import { success, created, badRequest, rateLimited, internalError } from "@/lib/api/errors";
import { checkRateLimit, getClientIp } from "@/lib/api/rate-limit";

// ─── Demo Seed Data ───────────────────────────────────────────

interface Agent {
  id: string;
  name: string;
  role: string;
  status: "active" | "idle" | "offline" | "error";
  capabilities: string[];
  createdAt: string;
  lastHeartbeat: string;
  metrics: {
    tasksCompleted: number;
    successRate: number;
    avgLatencyMs: number;
  };
}

const demoAgents: Agent[] = [
  {
    id: "agent-001",
    name: "Atlas Coordinator",
    role: "coordinator",
    status: "active",
    capabilities: ["task_delegation", "resource_allocation", "conflict_resolution"],
    createdAt: "2025-06-01T00:00:00Z",
    lastHeartbeat: new Date().toISOString(),
    metrics: { tasksCompleted: 1247, successRate: 0.97, avgLatencyMs: 42 },
  },
  {
    id: "agent-002",
    name: "Sage Researcher",
    role: "researcher",
    status: "active",
    capabilities: ["web_search", "document_analysis", "data_synthesis", "citation_tracking"],
    createdAt: "2025-06-03T08:30:00Z",
    lastHeartbeat: new Date().toISOString(),
    metrics: { tasksCompleted: 892, successRate: 0.94, avgLatencyMs: 180 },
  },
  {
    id: "agent-003",
    name: "Forge Executor",
    role: "executor",
    status: "active",
    capabilities: ["code_generation", "file_operations", "terminal_commands", "api_calls"],
    createdAt: "2025-06-05T14:00:00Z",
    lastHeartbeat: new Date().toISOString(),
    metrics: { tasksCompleted: 2103, successRate: 0.91, avgLatencyMs: 320 },
  },
  {
    id: "agent-004",
    name: "Sentinel Monitor",
    role: "monitor",
    status: "idle",
    capabilities: ["health_check", "performance_tracking", "anomaly_detection", "alerting"],
    createdAt: "2025-06-07T10:15:00Z",
    lastHeartbeat: new Date(Date.now() - 120_000).toISOString(),
    metrics: { tasksCompleted: 560, successRate: 0.99, avgLatencyMs: 15 },
  },
  {
    id: "agent-005",
    name: "Cipher Auditor",
    role: "auditor",
    status: "active",
    capabilities: ["governance_check", "policy_validation", "compliance_review", "risk_assessment"],
    createdAt: "2025-06-10T06:00:00Z",
    lastHeartbeat: new Date().toISOString(),
    metrics: { tasksCompleted: 334, successRate: 0.98, avgLatencyMs: 85 },
  },
  {
    id: "agent-006",
    name: "Nexus Bridge",
    role: "coordinator",
    status: "offline",
    capabilities: ["cross_system_communication", "data_routing", "protocol_translation"],
    createdAt: "2025-06-12T20:45:00Z",
    lastHeartbeat: new Date(Date.now() - 3_600_000).toISOString(),
    metrics: { tasksCompleted: 156, successRate: 0.87, avgLatencyMs: 210 },
  },
  {
    id: "agent-007",
    name: "Prism Optimizer",
    role: "optimizer",
    status: "error",
    capabilities: ["performance_tuning", "resource_optimization", "bottleneck_analysis"],
    createdAt: "2025-06-14T11:30:00Z",
    lastHeartbeat: new Date(Date.now() - 7_200_000).toISOString(),
    metrics: { tasksCompleted: 89, successRate: 0.78, avgLatencyMs: 450 },
  },
  {
    id: "agent-008",
    name: "Echo Learner",
    role: "researcher",
    status: "idle",
    capabilities: ["pattern_recognition", "knowledge_extraction", "feedback_processing"],
    createdAt: "2025-06-16T03:00:00Z",
    lastHeartbeat: new Date(Date.now() - 300_000).toISOString(),
    metrics: { tasksCompleted: 412, successRate: 0.93, avgLatencyMs: 260 },
  },
];

// In-memory registry for POST-created agents
const agentRegistry = [...demoAgents];

// ─── GET /api/agents ───────────────────────────────────────────

export async function GET(request: NextRequest) {
  try {
    const { allowed, remaining } = checkRateLimit(getClientIp(request));
    if (!allowed) return rateLimited();

    const { searchParams } = new URL(request.url);
    const status = searchParams.get("status");
    const role = searchParams.get("role");
    const rawLimit = searchParams.get("limit");
    const limit = rawLimit ? Math.min(parseInt(rawLimit, 10) || 50, 200) : 50;

    let filtered = [...agentRegistry];

    if (status) {
      filtered = filtered.filter(
        (a) => a.status === status,
      );
    }
    if (role) {
      filtered = filtered.filter(
        (a) => a.role === role,
      );
    }

    const paginated = filtered.slice(0, limit);

    return success({
      agents: paginated,
      total: filtered.length,
      limit,
      remaining,
    });
  } catch {
    return internalError("Failed to fetch agents");
  }
}

// ─── POST /api/agents ──────────────────────────────────────────

export async function POST(request: NextRequest) {
  try {
    const { allowed } = checkRateLimit(getClientIp(request));
    if (!allowed) return rateLimited();

    const body = await request.json();
    const { name, role, capabilities } = body;

    // Validate required fields
    if (!name || typeof name !== "string" || name.trim().length === 0) {
      return badRequest("Agent name is required");
    }
    if (!role || typeof role !== "string" || role.trim().length === 0) {
      return badRequest("Agent role is required");
    }
    if (
      !Array.isArray(capabilities) ||
      capabilities.length === 0 ||
      !capabilities.every((c: unknown) => typeof c === "string")
    ) {
      return badRequest("Capabilities must be a non-empty array of strings");
    }

    const validRoles = [
      "coordinator",
      "researcher",
      "executor",
      "monitor",
      "auditor",
      "optimizer",
    ];
    if (!validRoles.includes(role)) {
      return badRequest(
        `Invalid role. Must be one of: ${validRoles.join(", ")}`,
      );
    }

    const newAgent: Agent = {
      id: `agent-${String(agentRegistry.length + 1).padStart(3, "0")}`,
      name: name.trim(),
      role: role.trim(),
      status: "idle",
      capabilities: capabilities.map((c: string) => c.trim()),
      createdAt: new Date().toISOString(),
      lastHeartbeat: new Date().toISOString(),
      metrics: {
        tasksCompleted: 0,
        successRate: 0,
        avgLatencyMs: 0,
      },
    };

    agentRegistry.push(newAgent);

    return created(newAgent);
  } catch {
    return internalError("Failed to create agent");
  }
}
