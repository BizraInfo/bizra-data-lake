// ═══════════════════════════════════════════════════════════════
// BIZRA — Graph API
// Dependency graph snapshots and impact propagation analysis.
// ═══════════════════════════════════════════════════════════════

import { NextRequest } from "next/server";
import { success, created, badRequest, rateLimited, internalError } from "@/lib/api/errors";
import { checkRateLimit, getClientIp } from "@/lib/api/rate-limit";

// ─── Types ────────────────────────────────────────────────────

interface GraphNode {
  id: string;
  type: "agent" | "service" | "resource" | "policy" | "workflow";
  label: string;
  status: "healthy" | "degraded" | "critical" | "unknown";
  metadata: {
    role?: string;
    version?: string;
    uptime?: number;
    load?: number;
  };
}

interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: "depends_on" | "provides_to" | "monitors" | "governs" | "optimizes";
  weight: number;
  label: string;
}

interface Propagation {
  id: string;
  sourceNodeId: string;
  targetType: string;
  maxDepth: number;
  affectedNodes: string[];
  insights: string[];
  calculatedAt: string;
  status: "completed" | "in_progress" | "failed";
}

// ─── Demo Seed Data ───────────────────────────────────────────

const nodes: GraphNode[] = [
  { id: "n-agent-001", type: "agent", label: "Atlas Coordinator", status: "healthy", metadata: { role: "coordinator", uptime: 172800, load: 0.65 } },
  { id: "n-agent-002", type: "agent", label: "Sage Researcher", status: "healthy", metadata: { role: "researcher", uptime: 148000, load: 0.42 } },
  { id: "n-agent-003", type: "agent", label: "Forge Executor", status: "degraded", metadata: { role: "executor", uptime: 120000, load: 0.88 } },
  { id: "n-svc-001", type: "service", label: "Task Scheduler", status: "healthy", metadata: { version: "2.4.1", uptime: 200000, load: 0.31 } },
  { id: "n-svc-002", type: "service", label: "Message Broker", status: "healthy", metadata: { version: "1.8.0", uptime: 200000, load: 0.55 } },
  { id: "n-svc-003", type: "service", label: "State Manager", status: "healthy", metadata: { version: "3.1.2", uptime: 200000, load: 0.28 } },
  { id: "n-res-001", type: "resource", label: "Knowledge Base", status: "healthy", metadata: { version: "1.0.0", load: 0.15 } },
  { id: "n-res-002", type: "resource", label: "Credential Vault", status: "healthy", metadata: { version: "2.0.3", load: 0.05 } },
  { id: "n-pol-001", type: "policy", label: "Boundary Policy", status: "healthy", metadata: {} },
  { id: "n-pol-002", type: "policy", label: "Escalation Rules", status: "healthy", metadata: {} },
  { id: "n-wf-001", type: "workflow", label: "Research Pipeline", status: "healthy", metadata: { version: "1.2.0" } },
  { id: "n-wf-002", type: "workflow", label: "Deployment Pipeline", status: "degraded", metadata: { version: "2.0.1" } },
  { id: "n-agent-004", type: "agent", label: "Sentinel Monitor", status: "healthy", metadata: { role: "monitor", uptime: 96000, load: 0.22 } },
  { id: "n-agent-005", type: "agent", label: "Cipher Auditor", status: "healthy", metadata: { role: "auditor", uptime: 72000, load: 0.38 } },
];

const edges: GraphEdge[] = [
  { id: "e-001", source: "n-agent-001", target: "n-agent-002", type: "depends_on", weight: 0.8, label: "delegates research" },
  { id: "e-002", source: "n-agent-001", target: "n-agent-003", type: "depends_on", weight: 0.9, label: "delegates execution" },
  { id: "e-003", source: "n-agent-002", target: "n-res-001", type: "depends_on", weight: 0.7, label: "queries knowledge" },
  { id: "e-004", source: "n-agent-003", target: "n-res-002", type: "depends_on", weight: 0.95, label: "requires credentials" },
  { id: "e-005", source: "n-agent-001", target: "n-svc-001", type: "depends_on", weight: 0.85, label: "schedules tasks" },
  { id: "e-006", source: "n-svc-001", target: "n-svc-002", type: "depends_on", weight: 0.9, label: "dispatches messages" },
  { id: "e-007", source: "n-svc-002", target: "n-svc-003", type: "provides_to", weight: 0.6, label: "state updates" },
  { id: "e-008", source: "n-agent-004", target: "n-agent-001", type: "monitors", weight: 0.5, label: "health monitoring" },
  { id: "e-009", source: "n-agent-004", target: "n-agent-003", type: "monitors", weight: 0.7, label: "load monitoring" },
  { id: "e-010", source: "n-agent-005", target: "n-pol-001", type: "governs", weight: 0.9, label: "enforces boundaries" },
  { id: "e-011", source: "n-agent-005", target: "n-pol-002", type: "governs", weight: 0.85, label: "escalation enforcement" },
  { id: "e-012", source: "n-agent-001", target: "n-wf-001", type: "depends_on", weight: 0.75, label: "uses research pipeline" },
  { id: "e-013", source: "n-agent-003", target: "n-wf-002", type: "depends_on", weight: 0.8, label: "uses deploy pipeline" },
  { id: "e-014", source: "n-agent-002", target: "n-wf-001", type: "provides_to", weight: 0.7, label: "feeds research data" },
  { id: "e-015", source: "n-pol-001", target: "n-agent-003", type: "governs", weight: 0.9, label: "execution boundaries" },
];

const propagations: Propagation[] = [
  {
    id: "prop-001",
    sourceNodeId: "n-agent-003",
    targetType: "service",
    maxDepth: 2,
    affectedNodes: ["n-svc-001", "n-svc-002", "n-wf-002"],
    insights: [
      "Forge Executor degradation propagates to 2 services within depth 1",
      "Deployment Pipeline is at risk due to executor dependency chain",
      "Recommend reducing Forge workload by 30% to stabilize",
    ],
    calculatedAt: new Date(Date.now() - 3_600_000).toISOString(),
    status: "completed",
  },
  {
    id: "prop-002",
    sourceNodeId: "n-svc-002",
    targetType: "agent",
    maxDepth: 3,
    affectedNodes: ["n-agent-001", "n-agent-002", "n-agent-003", "n-svc-003"],
    insights: [
      "Message Broker outage would impact all active agents",
      "Critical dependency: 85% of inter-agent communication flows through this service",
      "Failover to backup broker recommended within 30s SLA",
    ],
    calculatedAt: new Date(Date.now() - 7_200_000).toISOString(),
    status: "completed",
  },
];

// ─── GET /api/graph ────────────────────────────────────────────

export async function GET(request: NextRequest) {
  try {
    const { allowed, remaining } = checkRateLimit(getClientIp(request));
    if (!allowed) return rateLimited();

    const { searchParams } = new URL(request.url);
    const nodeType = searchParams.get("nodeType");
    const rawDepth = searchParams.get("depth");
    const depth = rawDepth ? parseInt(rawDepth, 10) || 2 : 2;

    // Filter nodes by type if specified
    const filteredNodes = nodeType
      ? nodes.filter((n) => n.type === nodeType)
      : nodes;

    // For depth-filtered views, only include edges where both endpoints exist in filtered nodes,
    // or where the edge distance from a filtered node is within depth
    const filteredNodeIds = new Set(filteredNodes.map((n) => n.id));
    const relevantEdges = edges.filter((e) => {
      if (filteredNodeIds.has(e.source) && filteredNodeIds.has(e.target)) return true;
      return false;
    });

    return success({
      snapshot: {
        nodes: filteredNodes,
        edges: relevantEdges,
      },
      propagations: propagations.slice(0, depth),
      meta: {
        totalNodes: nodes.length,
        totalEdges: edges.length,
        filteredNodes: filteredNodes.length,
        filteredEdges: relevantEdges.length,
        depth,
        remaining,
      },
    });
  } catch {
    return internalError("Failed to fetch graph snapshot");
  }
}

// ─── POST /api/graph ───────────────────────────────────────────

export async function POST(request: NextRequest) {
  try {
    const { allowed } = checkRateLimit(getClientIp(request));
    if (!allowed) return rateLimited();

    const body = await request.json();
    const { sourceNodeId, targetType, maxDepth } = body;

    if (!sourceNodeId || typeof sourceNodeId !== "string") {
      return badRequest("sourceNodeId is required and must be a string");
    }

    const validTargetTypes = ["agent", "service", "resource", "policy", "workflow", "all"];
    const resolvedTarget = targetType || "all";
    if (!validTargetTypes.includes(resolvedTarget)) {
      return badRequest(
        `Invalid targetType. Must be one of: ${validTargetTypes.join(", ")}`,
      );
    }

    const resolvedDepth = typeof maxDepth === "number" ? Math.min(maxDepth, 10) : 3;

    // Simulate BFS propagation from source node
    const sourceNode = nodes.find((n) => n.id === sourceNodeId);
    if (!sourceNode) {
      return badRequest(`Node "${sourceNodeId}" not found in graph`);
    }

    // BFS to find affected nodes
    const visited = new Set<string>([sourceNodeId]);
    const queue: Array<{ id: string; currentDepth: number }> = [
      { id: sourceNodeId, currentDepth: 0 },
    ];
    const affected: string[] = [];

    while (queue.length > 0) {
      const { id, currentDepth } = queue.shift()!;
      if (currentDepth >= resolvedDepth) continue;

      for (const edge of edges) {
        let neighbor: string | null = null;
        if (edge.source === id && !visited.has(edge.target)) {
          neighbor = edge.target;
        } else if (edge.target === id && !visited.has(edge.source)) {
          neighbor = edge.source;
        }

        if (neighbor) {
          const neighborNode = nodes.find((n) => n.id === neighbor);
          if (
            neighborNode &&
            (resolvedTarget === "all" || neighborNode.type === resolvedTarget)
          ) {
            visited.add(neighbor);
            affected.push(neighbor);
            queue.push({ id: neighbor, currentDepth: currentDepth + 1 });
          }
        }
      }
    }

    // Generate insights based on affected nodes
    const insights: string[] = [];
    const affectedAgentNodes = affected.filter((id) => id.startsWith("n-agent-"));
    const affectedServiceNodes = affected.filter((id) => id.startsWith("n-svc-"));
    const criticalNodes = affected.filter(
      (id) => nodes.find((n) => n.id === id)?.status === "critical",
    );

    if (affectedAgentNodes.length > 0) {
      insights.push(
        `${affectedAgentNodes.length} agent(s) impacted by changes from "${sourceNode.label}"`,
      );
    }
    if (affectedServiceNodes.length > 0) {
      insights.push(
        `${affectedServiceNodes.length} service(s) in the dependency chain at depth ≤ ${resolvedDepth}`,
      );
    }
    if (criticalNodes.length > 0) {
      insights.push(
        `WARNING: ${criticalNodes.length} critical node(s) detected in blast radius — escalation recommended`,
      );
    }
    if (affected.length === 0) {
      insights.push(
        "No downstream impact detected. Source node is isolated or depth is insufficient.",
      );
    } else {
      const highWeightEdges = edges.filter(
        (e) =>
          (e.source === sourceNodeId && affected.includes(e.target)) ||
          (e.target === sourceNodeId && affected.includes(e.source)),
      );
      const avgWeight =
        highWeightEdges.reduce((sum, e) => sum + e.weight, 0) /
        Math.max(highWeightEdges.length, 1);
      insights.push(
        `Average dependency weight: ${(avgWeight * 100).toFixed(1)}% — ${
          avgWeight > 0.8 ? "tight coupling detected" : "acceptable coupling"
        }`,
      );
    }

    const propagation: Propagation = {
      id: `prop-${String(propagations.length + 1).padStart(3, "0")}`,
      sourceNodeId,
      targetType: resolvedTarget,
      maxDepth: resolvedDepth,
      affectedNodes: affected,
      insights,
      calculatedAt: new Date().toISOString(),
      status: "completed",
    };

    propagations.push(propagation);

    return created(propagation);
  } catch {
    return internalError("Failed to calculate propagation");
  }
}
