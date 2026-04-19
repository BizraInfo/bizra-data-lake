// ═══════════════════════════════════════════════════════════════
// BIZRA — Optimization API
// Optimization cycles, system metrics, and improvement tracking.
// ═══════════════════════════════════════════════════════════════

import { NextRequest } from "next/server";
import { success, created, badRequest, rateLimited, internalError } from "@/lib/api/errors";
import { checkRateLimit, getClientIp } from "@/lib/api/rate-limit";

// ─── Types ────────────────────────────────────────────────────

interface OptimizationCycle {
  id: string;
  status: "scanning" | "analyzing" | "applying" | "completed" | "failed" | "rolled_back";
  targets: string[];
  initiatedBy: string;
  startedAt: string;
  completedAt: string | null;
  duration: number | null;
  improvements: {
    component: string;
    metricBefore: number;
    metricAfter: number;
    improvement: string;
    delta: string;
  }[];
  summary: string;
}

interface SystemMetrics {
  agentPool: {
    total: number;
    active: number;
    avgSuccessRate: number;
    avgLatencyMs: number;
  };
  graph: {
    totalNodes: number;
    totalEdges: number;
    avgNodeHealth: number;
  };
  governance: {
    totalRules: number;
    activeRules: number;
    recentDenials: number;
    complianceRate: number;
  };
  optimization: {
    totalCycles: number;
    successRate: number;
    avgImprovementPct: number;
  };
  timestamp: string;
}

// ─── Demo Seed Data ───────────────────────────────────────────

const cycles: OptimizationCycle[] = [
  {
    id: "opt-001",
    status: "completed",
    targets: ["agent_runtime", "task_scheduler"],
    initiatedBy: "system",
    startedAt: new Date(Date.now() - 172_800_000).toISOString(),
    completedAt: new Date(Date.now() - 171_600_000).toISOString(),
    duration: 7200,
    improvements: [
      {
        component: "agent_runtime",
        metricBefore: 320,
        metricAfter: 245,
        improvement: "Agent dispatch latency reduced",
        delta: "-23.4%",
      },
      {
        component: "task_scheduler",
        metricBefore: 85,
        metricAfter: 92,
        improvement: "Task scheduling throughput increased",
        delta: "+8.2%",
      },
    ],
    summary: "Optimized agent dispatch pipeline and task scheduler priority queue. Reduced average latency by 23%.",
  },
  {
    id: "opt-002",
    status: "completed",
    targets: ["message_broker", "state_manager"],
    initiatedBy: "admin",
    startedAt: new Date(Date.now() - 86_400_000).toISOString(),
    completedAt: new Date(Date.now() - 85_800_000).toISOString(),
    duration: 3600,
    improvements: [
      {
        component: "message_broker",
        metricBefore: 180,
        metricAfter: 95,
        improvement: "Message broker throughput doubled",
        delta: "+89.5%",
      },
      {
        component: "state_manager",
        metricBefore: 42,
        metricAfter: 38,
        improvement: "State sync latency improved",
        delta: "-9.5%",
      },
    ],
    summary: "Upgraded message broker batch processing and optimized state manager conflict resolution.",
  },
  {
    id: "opt-003",
    status: "completed",
    targets: ["governance_engine"],
    initiatedBy: "system",
    startedAt: new Date(Date.now() - 43_200_000).toISOString(),
    completedAt: new Date(Date.now() - 42_500_000).toISOString(),
    duration: 4200,
    improvements: [
      {
        component: "governance_engine",
        metricBefore: 120,
        metricAfter: 35,
        improvement: "Governance rule evaluation latency reduced",
        delta: "-70.8%",
      },
    ],
    summary: "Compiled governance rule engine for faster evaluation. Batch processing of policy checks now 3x faster.",
  },
  {
    id: "opt-004",
    status: "failed",
    targets: ["knowledge_base", "embedding_service"],
    initiatedBy: "admin",
    startedAt: new Date(Date.now() - 21_600_000).toISOString(),
    completedAt: new Date(Date.now() - 21_300_000).toISOString(),
    duration: 1800,
    improvements: [],
    summary: "Optimization failed — embedding service version conflict detected. Rolled back automatically.",
  },
  {
    id: "opt-005",
    status: "completed",
    targets: ["resource_allocator"],
    initiatedBy: "system",
    startedAt: new Date(Date.now() - 7_200_000).toISOString(),
    completedAt: new Date(Date.now() - 5_400_000).toISOString(),
    duration: 5400,
    improvements: [
      {
        component: "resource_allocator",
        metricBefore: 0.72,
        metricAfter: 0.89,
        improvement: "Resource utilization efficiency improved",
        delta: "+23.6%",
      },
    ],
    summary: "Implemented adaptive resource allocation algorithm. Better workload distribution across idle agents.",
  },
];

const currentMetrics: SystemMetrics = {
  agentPool: {
    total: 8,
    active: 4,
    avgSuccessRate: 0.92,
    avgLatencyMs: 195,
  },
  graph: {
    totalNodes: 14,
    totalEdges: 15,
    avgNodeHealth: 0.86,
  },
  governance: {
    totalRules: 7,
    activeRules: 6,
    recentDenials: 1,
    complianceRate: 0.97,
  },
  optimization: {
    totalCycles: 5,
    successRate: 0.8,
    avgImprovementPct: 35.7,
  },
  timestamp: new Date().toISOString(),
};

// ─── GET /api/optimization ────────────────────────────────────

export async function GET(request: NextRequest) {
  try {
    const { allowed, remaining } = checkRateLimit(getClientIp(request));
    if (!allowed) return rateLimited();

    const { searchParams } = new URL(request.url);
    const status = searchParams.get("status");
    const rawLimit = searchParams.get("limit");
    const limit = rawLimit ? Math.min(parseInt(rawLimit, 10) || 10, 100) : 10;

    // Filter cycles by status
    let filteredCycles = [...cycles];
    if (status) {
      filteredCycles = filteredCycles.filter((c) => c.status === status);
    }

    // Sort by most recent first
    filteredCycles.sort(
      (a, b) => new Date(b.startedAt).getTime() - new Date(a.startedAt).getTime(),
    );

    const paginatedCycles = filteredCycles.slice(0, limit);

    // Recalculate metrics based on current state
    const updatedMetrics = { ...currentMetrics };
    updatedMetrics.optimization.totalCycles = cycles.length;
    const completedCycles = cycles.filter((c) => c.status === "completed");
    updatedMetrics.optimization.successRate =
      completedCycles.length / Math.max(cycles.length, 1);

    if (completedCycles.length > 0) {
      const allDeltas = completedCycles.flatMap((c) =>
        c.improvements
          .filter((i) => i.delta.startsWith("-") && i.metricBefore > 50)
          .map((i) => {
            const pct = parseFloat(i.delta);
            return isNaN(pct) ? 0 : Math.abs(pct);
          }),
      );
      updatedMetrics.optimization.avgImprovementPct =
        allDeltas.length > 0
          ? allDeltas.reduce((s, d) => s + d, 0) / allDeltas.length
          : 0;
    }

    return success({
      cycles: paginatedCycles,
      metrics: updatedMetrics,
      meta: {
        totalCycles: filteredCycles.length,
        limit,
        remaining,
      },
    });
  } catch {
    return internalError("Failed to fetch optimization data");
  }
}

// ─── POST /api/optimization ───────────────────────────────────

export async function POST(request: NextRequest) {
  try {
    const { allowed } = checkRateLimit(getClientIp(request));
    if (!allowed) return rateLimited();

    const body = await request.json();
    const { targets } = body;

    if (!Array.isArray(targets) || targets.length === 0) {
      return badRequest("targets must be a non-empty array of component names");
    }

    if (!targets.every((t: unknown) => typeof t === "string" && t.trim().length > 0)) {
      return badRequest("Each target must be a non-empty string");
    }

    // Known optimizable components
    const knownComponents = [
      "agent_runtime",
      "task_scheduler",
      "message_broker",
      "state_manager",
      "governance_engine",
      "knowledge_base",
      "embedding_service",
      "resource_allocator",
      "cache_layer",
      "logging_pipeline",
    ];

    const validTargets = targets
      .map((t: string) => t.trim())
      .filter((t: string) => knownComponents.includes(t));

    if (validTargets.length === 0) {
      return badRequest(
        `No valid targets. Known components: ${knownComponents.join(", ")}`,
      );
    }

    // Create new optimization cycle
    const newCycle: OptimizationCycle = {
      id: `opt-${String(cycles.length + 1).padStart(3, "0")}`,
      status: "scanning",
      targets: validTargets,
      initiatedBy: "user",
      startedAt: new Date().toISOString(),
      completedAt: null,
      duration: null,
      improvements: [],
      summary: `Optimization cycle initiated for: ${validTargets.join(", ")}. Currently scanning target components for improvement opportunities.`,
    };

    cycles.push(newCycle);

    // Simulate async completion after a short delay
    // In production, this would be a background job
    setTimeout(() => {
      newCycle.status = "completed";
      newCycle.completedAt = new Date().toISOString();
      newCycle.duration = Math.floor(Math.random() * 3000) + 1500;

      for (const target of validTargets) {
        const improvement = Math.floor(Math.random() * 30) + 5;
        newCycle.improvements.push({
          component: target,
          metricBefore: Math.floor(Math.random() * 200) + 50,
          metricAfter: Math.floor(Math.random() * 50) + 20,
          improvement: `${target} optimized`,
          delta: `+${improvement}%`,
        });
      }

      newCycle.summary = `Optimization completed for ${validTargets.length} component(s). Applied ${newCycle.improvements.length} improvement(s). Average gain: ${Math.floor(validTargets.reduce((s) => s + Math.random() * 30 + 5, 0) / validTargets.length)}%.`;
    }, 2000);

    return created({
      cycle: newCycle,
      message:
        "Optimization cycle started. Poll GET /api/optimization?status=scanning to track progress.",
    });
  } catch {
    return internalError("Failed to start optimization cycle");
  }
}
