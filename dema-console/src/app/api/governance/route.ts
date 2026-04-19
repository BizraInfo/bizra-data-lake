// ═══════════════════════════════════════════════════════════════
// BIZRA — Governance API
// Boundary enforcement, policy validation, and audit events.
// ═══════════════════════════════════════════════════════════════

import { NextRequest } from "next/server";
import { success, created, badRequest, rateLimited, internalError } from "@/lib/api/errors";
import { checkRateLimit, getClientIp } from "@/lib/api/rate-limit";

// ─── Types ────────────────────────────────────────────────────

interface GovernanceRule {
  id: string;
  name: string;
  category: "boundary" | "escalation" | "compliance" | "safety" | "performance";
  description: string;
  active: boolean;
  severity: "low" | "medium" | "high" | "critical";
  conditions: {
    agentRoles?: string[];
    resourceTypes?: string[];
    actions?: string[];
    thresholds?: Record<string, number>;
  };
  actions: string[];
  lastEvaluated: string;
  timesTriggered: number;
}

interface GovernanceEvent {
  id: string;
  ruleId: string;
  ruleName: string;
  decision: "allow" | "deny" | "escalate";
  agentId: string;
  action: string;
  resource: string;
  reason: string;
  timestamp: string;
  resolved: boolean;
}

// ─── Demo Seed Data ───────────────────────────────────────────

const rules: GovernanceRule[] = [
  {
    id: "gov-001",
    name: "Credential Access Boundary",
    category: "boundary",
    description: "Restricts credential vault access to auditors and coordinators only",
    active: true,
    severity: "critical",
    conditions: {
      agentRoles: ["auditor", "coordinator"],
      resourceTypes: ["credential"],
    },
    actions: ["deny", "alert", "log"],
    lastEvaluated: new Date(Date.now() - 600_000).toISOString(),
    timesTriggered: 12,
  },
  {
    id: "gov-002",
    name: "Task Delegation Limit",
    category: "performance",
    description: "Prevents any agent from delegating more than 10 concurrent tasks",
    active: true,
    severity: "medium",
    conditions: {
      agentRoles: ["coordinator"],
      actions: ["delegate"],
      thresholds: { maxConcurrent: 10 },
    },
    actions: ["throttle", "log"],
    lastEvaluated: new Date(Date.now() - 300_000).toISOString(),
    timesTriggered: 34,
  },
  {
    id: "gov-003",
    name: "External Service Safety Check",
    category: "safety",
    description: "Requires safety validation before any agent calls external APIs",
    active: true,
    severity: "high",
    conditions: {
      actions: ["external_api_call", "web_request"],
      resourceTypes: ["service"],
    },
    actions: ["validate", "approve", "log"],
    lastEvaluated: new Date(Date.now() - 1_800_000).toISOString(),
    timesTriggered: 567,
  },
  {
    id: "gov-004",
    name: "Escalation on Critical Failure",
    category: "escalation",
    description: "Auto-escalates to human operator when 3+ agents report critical status",
    active: true,
    severity: "critical",
    conditions: {
      thresholds: { criticalAgentCount: 3 },
    },
    actions: ["escalate", "alert", "pause_operations"],
    lastEvaluated: new Date(Date.now() - 3_600_000).toISOString(),
    timesTriggered: 2,
  },
  {
    id: "gov-005",
    name: "Data Retention Compliance",
    category: "compliance",
    description: "Enforces data lifecycle policies for all stored artifacts and logs",
    active: true,
    severity: "medium",
    conditions: {
      resourceTypes: ["knowledge", "file"],
    },
    actions: ["archive", "log", "notify"],
    lastEvaluated: new Date(Date.now() - 86_400_000).toISOString(),
    timesTriggered: 8,
  },
  {
    id: "gov-006",
    name: "Code Execution Sandbox",
    category: "safety",
    description: "All code execution must occur within sandboxed environments",
    active: true,
    severity: "critical",
    conditions: {
      agentRoles: ["executor"],
      actions: ["execute_code", "run_command"],
    },
    actions: ["sandbox", "validate", "log"],
    lastEvaluated: new Date(Date.now() - 900_000).toISOString(),
    timesTriggered: 1204,
  },
  {
    id: "gov-007",
    name: "Rate Limit Governance",
    category: "performance",
    description: "Throttles agents that exceed 100 actions per minute",
    active: false,
    severity: "low",
    conditions: {
      thresholds: { actionsPerMinute: 100 },
    },
    actions: ["throttle", "log"],
    lastEvaluated: new Date(Date.now() - 172_800_000).toISOString(),
    timesTriggered: 0,
  },
];

const events: GovernanceEvent[] = [
  {
    id: "evt-gov-001",
    ruleId: "gov-001",
    ruleName: "Credential Access Boundary",
    decision: "deny",
    agentId: "agent-003",
    action: "access_credential",
    resource: "Credential Vault",
    reason: "Agent role 'executor' is not authorized for credential access",
    timestamp: new Date(Date.now() - 1_200_000).toISOString(),
    resolved: true,
  },
  {
    id: "evt-gov-002",
    ruleId: "gov-003",
    ruleName: "External Service Safety Check",
    decision: "allow",
    agentId: "agent-002",
    action: "external_api_call",
    resource: "Research API Gateway",
    reason: "Safety validation passed — endpoint whitelisted, TLS verified",
    timestamp: new Date(Date.now() - 2_400_000).toISOString(),
    resolved: true,
  },
  {
    id: "evt-gov-003",
    ruleId: "gov-002",
    ruleName: "Task Delegation Limit",
    decision: "escalate",
    agentId: "agent-001",
    action: "delegate",
    resource: "Task Scheduler",
    reason: "Coordinator approaching delegation limit (9/10 concurrent tasks)",
    timestamp: new Date(Date.now() - 3_600_000).toISOString(),
    resolved: false,
  },
  {
    id: "evt-gov-004",
    ruleId: "gov-006",
    ruleName: "Code Execution Sandbox",
    decision: "allow",
    agentId: "agent-003",
    action: "execute_code",
    resource: "Sandbox Runtime",
    reason: "Code execution approved — sandbox environment verified, no elevated permissions requested",
    timestamp: new Date(Date.now() - 5_400_000).toISOString(),
    resolved: true,
  },
  {
    id: "evt-gov-005",
    ruleId: "gov-004",
    ruleName: "Escalation on Critical Failure",
    decision: "escalate",
    agentId: "system",
    action: "health_check",
    resource: "Agent Cluster",
    reason: "2 agents reporting critical status — approaching escalation threshold (2/3)",
    timestamp: new Date(Date.now() - 7_200_000).toISOString(),
    resolved: true,
  },
];

// ─── Helper: Evaluate governance rules ─────────────────────────

function evaluateAction(
  action: string,
  agentId: string,
  resource: string,
  metadata?: Record<string, unknown>,
): {
  decision: "allow" | "deny" | "escalate";
  matchedRules: GovernanceRule[];
  reasons: string[];
} {
  const agentRole = metadata?.agentRole as string | undefined;
  const matchedRules: GovernanceRule[] = [];
  const reasons: string[] = [];
  let decision: "allow" | "deny" | "escalate" = "allow";

  for (const rule of rules) {
    if (!rule.active) continue;

    let matches = true;

    // Check role restrictions
    if (
      rule.conditions.agentRoles?.length &&
      agentRole &&
      !rule.conditions.agentRoles.includes(agentRole)
    ) {
      if (rule.severity === "critical") {
        matches = false;
      }
    }

    // Check resource type restrictions
    if (rule.conditions.resourceTypes?.length) {
      const resourceType = metadata?.resourceType as string | undefined;
      if (resourceType && !rule.conditions.resourceTypes.includes(resourceType)) {
        matches = false;
      }
    }

    // Check action restrictions
    if (rule.conditions.actions?.length) {
      if (!rule.conditions.actions.some((a) => action.includes(a))) {
        matches = false;
      }
    }

    if (matches) {
      matchedRules.push(rule);
      rule.timesTriggered += 1;
      rule.lastEvaluated = new Date().toISOString();

      if (rule.severity === "critical" && rule.actions.includes("deny")) {
        decision = "deny";
        reasons.push(
          `[${rule.name}] Critical rule violation: ${rule.description}`,
        );
      } else if (
        rule.severity === "high" &&
        rule.actions.includes("escalate")
      ) {
        if (decision !== "deny") decision = "escalate";
        reasons.push(
          `[${rule.name}] High-severity rule matched — escalating for review`,
        );
      } else if (rule.severity === "medium") {
        reasons.push(
          `[${rule.name}] Advisory: ${rule.description}`,
        );
      }
    }
  }

  if (matchedRules.length === 0) {
    reasons.push("No matching governance rules — action permitted by default");
  }

  return { decision, matchedRules, reasons };
}

// ─── GET /api/governance ───────────────────────────────────────

export async function GET(request: NextRequest) {
  try {
    const { allowed, remaining } = checkRateLimit(getClientIp(request));
    if (!allowed) return rateLimited();

    const { searchParams } = new URL(request.url);
    const category = searchParams.get("category");
    const activeParam = searchParams.get("active");

    const validCategories = [
      "boundary",
      "escalation",
      "compliance",
      "safety",
      "performance",
    ];

    // Filter rules
    let filteredRules = [...rules];
    if (category && validCategories.includes(category)) {
      filteredRules = filteredRules.filter((r) => r.category === category);
    }
    if (activeParam !== null) {
      const activeFilter = activeParam === "true";
      filteredRules = filteredRules.filter((r) => r.active === activeFilter);
    }

    // Sort rules by severity
    const severityOrder = { critical: 0, high: 1, medium: 2, low: 3 };
    filteredRules.sort(
      (a, b) => severityOrder[a.severity] - severityOrder[b.severity],
    );

    // Return recent events (most recent first)
    const recentEvents = [...events].sort(
      (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime(),
    );

    return success({
      rules: filteredRules,
      events: recentEvents,
      meta: {
        totalRules: rules.length,
        filteredRules: filteredRules.length,
        recentEvents: recentEvents.length,
        remaining,
      },
    });
  } catch {
    return internalError("Failed to fetch governance data");
  }
}

// ─── POST /api/governance ──────────────────────────────────────

export async function POST(request: NextRequest) {
  try {
    const { allowed } = checkRateLimit(getClientIp(request));
    if (!allowed) return rateLimited();

    const body = await request.json();
    const { action, agentId, resource, metadata } = body;

    if (!action || typeof action !== "string") {
      return badRequest("action is required and must be a string");
    }
    if (!agentId || typeof agentId !== "string") {
      return badRequest("agentId is required and must be a string");
    }
    if (!resource || typeof resource !== "string") {
      return badRequest("resource is required and must be a string");
    }

    // Evaluate against governance rules
    const { decision, matchedRules, reasons } = evaluateAction(
      action,
      agentId,
      resource,
      metadata,
    );

    // Record the governance event
    const governanceEvent: GovernanceEvent = {
      id: `evt-gov-${String(events.length + 1).padStart(3, "0")}`,
      ruleId: matchedRules[0]?.id || "none",
      ruleName: matchedRules[0]?.name || "No rule matched",
      decision,
      agentId,
      action,
      resource,
      reason: reasons[0] || "Default permit",
      timestamp: new Date().toISOString(),
      resolved: decision === "allow",
    };
    events.push(governanceEvent);

    return created({
      decision,
      agentId,
      action,
      resource,
      matchedRules: matchedRules.map((r) => ({
        id: r.id,
        name: r.name,
        severity: r.severity,
      })),
      reasons,
      eventId: governanceEvent.id,
      timestamp: governanceEvent.timestamp,
    });
  } catch {
    return internalError("Failed to validate governance action");
  }
}
