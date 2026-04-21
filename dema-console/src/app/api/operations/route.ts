// ═══════════════════════════════════════════════════════════════
// BIZRA — Operations API
// System health, telemetry, diagnostics, and performance data.
// ═══════════════════════════════════════════════════════════════

import { NextRequest } from "next/server";
import { success, created, badRequest, rateLimited, internalError } from "@/lib/api/errors";
import { checkRateLimit, getClientIp } from "@/lib/api/rate-limit";

// ─── Types ────────────────────────────────────────────────────

type HealthStatus = "healthy" | "degraded" | "critical" | "unknown";

interface ComponentHealth {
  name: string;
  status: HealthStatus;
  uptime: number;
  lastChecked: string;
  version?: string;
  details: string;
  metrics?: {
    cpu?: number;
    memory?: number;
    errorRate?: number;
    latencyP95?: number;
    throughput?: number;
  };
}

interface TelemetryEvent {
  id: string;
  component: string;
  level: "info" | "warn" | "error" | "critical";
  message: string;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

interface PerformanceData {
  responseTime: {
    p50: number;
    p95: number;
    p99: number;
  };
  throughput: {
    requestsPerSecond: number;
    eventsPerSecond: number;
  };
  resourceUsage: {
    cpu: number;
    memory: number;
    disk: number;
    network: number;
  };
  errorBudget: {
    budget: number;
    consumed: number;
    remaining: number;
    burnRate: number;
  };
  period: string;
}

interface DiagnosticResult {
  component: string;
  status: "pass" | "fail" | "warning";
  checks: {
    name: string;
    status: "pass" | "fail" | "warning";
    message: string;
    duration: number;
  }[];
  summary: string;
  timestamp: string;
}

// ─── Demo Seed Data ───────────────────────────────────────────

const componentHealth: ComponentHealth[] = [
  {
    name: "agent_runtime",
    status: "healthy",
    uptime: 259200,
    lastChecked: new Date().toISOString(),
    version: "2.6.0",
    details: "All agent processes operating normally. 4/8 agents active.",
    metrics: { cpu: 0.34, memory: 0.52, errorRate: 0.02, latencyP95: 195 },
  },
  {
    name: "task_scheduler",
    status: "healthy",
    uptime: 259200,
    lastChecked: new Date().toISOString(),
    version: "2.4.1",
    details: "Task queue depth: 12. Processing at normal rate.",
    metrics: { cpu: 0.12, memory: 0.28, errorRate: 0.00, latencyP95: 45 },
  },
  {
    name: "message_broker",
    status: "healthy",
    uptime: 259200,
    lastChecked: new Date().toISOString(),
    version: "1.8.0",
    details: "Message throughput stable. No backlog detected.",
    metrics: { cpu: 0.22, memory: 0.41, errorRate: 0.01, latencyP95: 12, throughput: 850 },
  },
  {
    name: "state_manager",
    status: "degraded",
    uptime: 259200,
    lastChecked: new Date(Date.now() - 30_000).toISOString(),
    version: "3.1.2",
    details: "Elevated write latency detected. Failover standby ready.",
    metrics: { cpu: 0.67, memory: 0.78, errorRate: 0.05, latencyP95: 320 },
  },
  {
    name: "governance_engine",
    status: "healthy",
    uptime: 259200,
    lastChecked: new Date().toISOString(),
    version: "1.3.0",
    details: "6/7 rules active. Recent compliance rate: 97%.",
    metrics: { cpu: 0.08, memory: 0.15, errorRate: 0.00, latencyP95: 35 },
  },
  {
    name: "optimization_service",
    status: "healthy",
    uptime: 172800,
    lastChecked: new Date().toISOString(),
    version: "1.1.0",
    details: "5 cycles completed (80% success rate). Last cycle 2h ago.",
    metrics: { cpu: 0.05, memory: 0.18, errorRate: 0.00, latencyP95: 120 },
  },
  {
    name: "knowledge_base",
    status: "healthy",
    uptime: 259200,
    lastChecked: new Date().toISOString(),
    version: "1.0.0",
    details: "Index healthy. 14,832 entries. Query latency nominal.",
    metrics: { cpu: 0.15, memory: 0.62, errorRate: 0.00, latencyP95: 85 },
  },
  {
    name: "credential_vault",
    status: "healthy",
    uptime: 259200,
    lastChecked: new Date().toISOString(),
    version: "2.0.3",
    details: "Vault sealed/unsealed normally. 3 active sessions.",
    metrics: { cpu: 0.03, memory: 0.08, errorRate: 0.00, latencyP95: 8 },
  },
];

const telemetryEvents: TelemetryEvent[] = [
  {
    id: "tel-001",
    component: "state_manager",
    level: "warn",
    message: "Write latency exceeded threshold (320ms > 200ms) for 3 consecutive samples",
    timestamp: new Date(Date.now() - 45_000).toISOString(),
    metadata: { threshold: 200, current: 320, samples: 3 },
  },
  {
    id: "tel-002",
    component: "agent_runtime",
    level: "info",
    message: "Agent Forge Executor transitioned to degraded state — high load detected (88%)",
    timestamp: new Date(Date.now() - 120_000).toISOString(),
    metadata: { agentId: "agent-003", load: 0.88 },
  },
  {
    id: "tel-003",
    component: "governance_engine",
    level: "info",
    message: "Credential access denied for executor agent (gov-001 triggered)",
    timestamp: new Date(Date.now() - 300_000).toISOString(),
    metadata: { ruleId: "gov-001", agentId: "agent-003" },
  },
  {
    id: "tel-004",
    component: "optimization_service",
    level: "info",
    message: "Optimization cycle opt-005 completed successfully",
    timestamp: new Date(Date.now() - 600_000).toISOString(),
    metadata: { cycleId: "opt-005", improvements: 1, duration: 5400 },
  },
  {
    id: "tel-005",
    component: "message_broker",
    level: "error",
    message: "Temporary connection pool exhaustion — scaled pool from 20 to 30 connections",
    timestamp: new Date(Date.now() - 1_800_000).toISOString(),
    metadata: { poolBefore: 20, poolAfter: 30, peakConcurrent: 27 },
  },
  {
    id: "tel-006",
    component: "task_scheduler",
    level: "info",
    message: "Scheduled maintenance window approaching in 4 hours",
    timestamp: new Date(Date.now() - 3_600_000).toISOString(),
    metadata: { maintenanceWindow: new Date(Date.now() + 14_400_000).toISOString() },
  },
  {
    id: "tel-007",
    component: "credential_vault",
    level: "critical",
    message: "Unusual access pattern detected — 5 failed auth attempts from single source",
    timestamp: new Date(Date.now() - 7_200_000).toISOString(),
    metadata: { attempts: 5, source: "internal-network", resolved: true },
  },
  {
    id: "tel-008",
    component: "knowledge_base",
    level: "info",
    message: "Index rebuild completed — 247 new entries indexed",
    timestamp: new Date(Date.now() - 14_400_000).toISOString(),
    metadata: { newEntries: 247, indexSize: "1.2GB", duration: 340 },
  },
];

const performanceData: PerformanceData = {
  responseTime: {
    p50: 42,
    p95: 195,
    p99: 480,
  },
  throughput: {
    requestsPerSecond: 156,
    eventsPerSecond: 42,
  },
  resourceUsage: {
    cpu: 0.28,
    memory: 0.45,
    disk: 0.33,
    network: 0.12,
  },
  errorBudget: {
    budget: 0.001,
    consumed: 0.0003,
    remaining: 0.0007,
    burnRate: 0.8,
  },
  period: "last_1h",
};

// ─── Helper: Run diagnostics ───────────────────────────────────

function runDiagnostic(component?: string): DiagnosticResult {
  const targetComponents = component
    ? componentHealth.filter((c) => c.name === component)
    : componentHealth;

  if (targetComponents.length === 0 && component) {
    return {
      component,
      status: "fail",
      checks: [
        {
          name: "component_exists",
          status: "fail",
          message: `Component "${component}" not found in system registry`,
          duration: 5,
        },
      ],
      summary: `Diagnostic failed: component "${component}" does not exist`,
      timestamp: new Date().toISOString(),
    };
  }

  const allChecks: DiagnosticResult["checks"] = [];
  let hasFailures = false;
  let hasWarnings = false;

  for (const comp of targetComponents) {
    const compChecks: DiagnosticResult["checks"] = [];

    // Check 1: Process alive
    compChecks.push({
      name: `${comp.name}:process_alive`,
      status: comp.status !== "critical" ? "pass" : "fail",
      message:
        comp.status !== "critical"
          ? "Process is running"
          : "Process is in critical state",
      duration: Math.floor(Math.random() * 20) + 2,
    });

    // Check 2: Memory usage
    if (comp.metrics?.memory) {
      const memOk = comp.metrics.memory < 0.8;
      compChecks.push({
        name: `${comp.name}:memory_usage`,
        status: memOk ? "pass" : comp.metrics.memory < 0.9 ? "warning" : "fail",
        message: `Memory usage at ${(comp.metrics.memory * 100).toFixed(1)}%`,
        duration: Math.floor(Math.random() * 15) + 1,
      });
      if (!memOk) hasWarnings = true;
    }

    // Check 3: Error rate
    if (comp.metrics?.errorRate !== undefined) {
      const errOk = comp.metrics.errorRate < 0.05;
      compChecks.push({
        name: `${comp.name}:error_rate`,
        status: errOk ? "pass" : "warning",
        message: `Error rate at ${(comp.metrics.errorRate * 100).toFixed(2)}%`,
        duration: Math.floor(Math.random() * 10) + 1,
      });
      if (!errOk) hasWarnings = true;
    }

    // Check 4: Latency
    if (comp.metrics?.latencyP95) {
      const latOk = comp.metrics.latencyP95 < 500;
      compChecks.push({
        name: `${comp.name}:latency_p95`,
        status: latOk ? "pass" : comp.metrics.latencyP95 < 1000 ? "warning" : "fail",
        message: `P95 latency at ${comp.metrics.latencyP95}ms`,
        duration: Math.floor(Math.random() * 25) + 2,
      });
      if (!latOk) {
        if (comp.metrics.latencyP95 >= 1000) hasFailures = true;
        else hasWarnings = true;
      }
    }

    // Check 5: Connectivity
    const connected = comp.status !== "critical";
    compChecks.push({
      name: `${comp.name}:connectivity`,
      status: connected ? "pass" : "fail",
      message: connected
        ? "All internal connections healthy"
        : "Connection failures detected",
      duration: Math.floor(Math.random() * 30) + 5,
    });

    if (!connected) hasFailures = true;
    allChecks.push(...compChecks);
  }

  const overallStatus: DiagnosticResult["status"] = hasFailures
    ? "fail"
    : hasWarnings
      ? "warning"
      : "pass";

  const passCount = allChecks.filter((c) => c.status === "pass").length;
  const totalCount = allChecks.length;

  return {
    component: component || "all",
    status: overallStatus,
    checks: allChecks,
    summary:
      overallStatus === "pass"
        ? `All systems nominal — ${passCount}/${totalCount} checks passed`
        : overallStatus === "warning"
          ? `Degraded performance detected — ${passCount}/${totalCount} checks passed, ${totalCount - passCount} warning(s)`
          : `Issues detected — ${passCount}/${totalCount} checks passed, ${totalCount - passCount} failure(s)`,
    timestamp: new Date().toISOString(),
  };
}

// ─── GET /api/operations ──────────────────────────────────────

export async function GET(request: NextRequest) {
  try {
    const { allowed, remaining } = checkRateLimit(getClientIp(request));
    if (!allowed) return rateLimited();

    const { searchParams } = new URL(request.url);
    const component = searchParams.get("component");
    const level = searchParams.get("level");

    // Filter health data by component
    let filteredHealth = [...componentHealth];
    if (component) {
      filteredHealth = filteredHealth.filter((h) => h.name === component);
      if (filteredHealth.length === 0) {
        return badRequest(`Component "${component}" not found`);
      }
    }

    // Filter telemetry by component and level
    let filteredTelemetry = [...telemetryEvents];
    if (component) {
      filteredTelemetry = filteredTelemetry.filter(
        (t) => t.component === component,
      );
    }
    if (level) {
      const validLevels = ["info", "warn", "error", "critical"];
      if (!validLevels.includes(level)) {
        return badRequest(
          `Invalid level. Must be one of: ${validLevels.join(", ")}`,
        );
      }
      filteredTelemetry = filteredTelemetry.filter((t) => t.level === level);
    }

    // Sort telemetry by most recent first
    filteredTelemetry.sort(
      (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime(),
    );

    // Calculate overall health status
    const statuses = filteredHealth.map((h) => h.status);
    const overallHealth: HealthStatus = statuses.includes("critical")
      ? "critical"
      : statuses.includes("degraded")
        ? "degraded"
        : statuses.includes("healthy")
          ? "healthy"
          : "unknown";

    return success({
      health: {
        overall: overallHealth,
        components: filteredHealth,
        checkedAt: new Date().toISOString(),
      },
      telemetry: {
        events: filteredTelemetry,
        total: filteredTelemetry.length,
      },
      performance: performanceData,
      meta: {
        remaining,
      },
    });
  } catch {
    return internalError("Failed to fetch operations data");
  }
}

// ─── POST /api/operations ─────────────────────────────────────

export async function POST(request: NextRequest) {
  try {
    const { allowed } = checkRateLimit(getClientIp(request));
    if (!allowed) return rateLimited();

    const body = await request.json();
    const { component, diagnostic } = body;

    // Determine operation type
    const isDiagnostic = diagnostic === true;
    const targetComponent =
      typeof component === "string" && component.trim().length > 0
        ? component.trim()
        : undefined;

    if (!isDiagnostic && !targetComponent) {
      return badRequest(
        "Either 'component' (string) or 'diagnostic' (true) must be provided",
      );
    }

    if (isDiagnostic) {
      // Run full diagnostic or component-specific diagnostic
      const result = runDiagnostic(targetComponent);
      return created(result);
    }

    // Manual health check for a specific component
    const target = componentHealth.find((c) => c.name === targetComponent);
    if (!target) {
      return badRequest(`Component "${targetComponent}" not found`);
    }

    // Simulate health check with minor metric fluctuations.
    const jitter = () => (Math.random() - 0.5) * 0.05;
    const clampUnitInterval = (value: number) => Math.max(0, Math.min(1, value));
    const currentMetrics = target.metrics;
    const checkedComponent: ComponentHealth = {
      ...target,
      lastChecked: new Date().toISOString(),
      metrics: currentMetrics
        ? {
            ...currentMetrics,
            ...(typeof currentMetrics.cpu === "number"
              ? { cpu: clampUnitInterval(currentMetrics.cpu + jitter()) }
              : {}),
            ...(typeof currentMetrics.memory === "number"
              ? { memory: clampUnitInterval(currentMetrics.memory + jitter()) }
              : {}),
          }
        : undefined,
    };

    // Update the in-memory health record
    const idx = componentHealth.findIndex((c) => c.name === targetComponent);
    if (idx !== -1) {
      componentHealth[idx] = checkedComponent;
    }

    // Add a telemetry event for the manual check
    telemetryEvents.push({
      id: `tel-${String(telemetryEvents.length + 1).padStart(3, "0")}`,
      component: target.name,
      level: "info",
      message: `Manual health check initiated for ${target.name} — status: ${checkedComponent.status}`,
      timestamp: new Date().toISOString(),
    });

    return created({
      healthCheck: checkedComponent,
      message: `Health check completed for "${targetComponent}"`,
      timestamp: new Date().toISOString(),
    });
  } catch {
    return internalError("Failed to process operations request");
  }
}
