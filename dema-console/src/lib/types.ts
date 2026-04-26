// ═══════════════════════════════════════════════════════════════
// DEMA — Core Type Definitions
// The one visible face of BIZRA
// ═══════════════════════════════════════════════════════════════

export type Screen =
  | "onboarding"
  | "dashboard"
  | "ask"
  | "receipts"
  | "resources"
  | "actions"
  | "orchestration"
  | "impact"
  | "governance"
  | "autopilot"
  | "operations"
  | "adk-factory"
  | "settings";

export type TrustLevel = "visitor" | "citizen" | "operator" | "admin";
export type ReceiptStatus = "pending" | "verified" | "rejected" | "expired";
export type ReceiptType = "action" | "verification" | "delegation" | "completion" | "error";
export type ManifestStatus = "draft" | "active" | "completed" | "archived";
export type ResourceStatus = "registered" | "active" | "revoked";
export type ResourceType = "file" | "url" | "credential" | "service" | "knowledge" | "browser" | "terminal";
export type ActionStatus = "pending" | "approved" | "executing" | "completed" | "failed" | "denied" | "stopped";
export type ActionMode = "browser" | "computer" | "code" | "research";
export type PermissionLevel = "auto" | "explicit" | "denied";
export type MemoryCategory = "preference" | "context" | "knowledge" | "poi";
export type ResearchDepth = "quick" | "deep" | "exhaustive";

export interface TrustState {
  principalId: string | null;
  principalName: string;
  level: TrustLevel | null;
  score: number | null;
  maxScore: number | null;
  lastVerified: string | null;
  sessionId: string | null;
  isActive: boolean;
  chainHead: string | null;
  missionId: string | null;
  missionReceiptId: string | null;
  activationReceiptId: string | null;
  finalStage: string | null;
  profileHash: string | null;
  cacheWarning: string | null;
}

export interface Receipt {
  id: string;
  missionId: string | null;
  type: ReceiptType;
  status: ReceiptStatus;
  title: string;
  description: string | null;
  evidence: string | null;
  issuedAt: string;
  verifiedAt: string | null;
  expiresAt: string | null;
}

export interface Manifest {
  id: string;
  missionId: string | null;
  title: string;
  description: string | null;
  status: ManifestStatus;
  artifactCount: number;
  createdAt: string;
  updatedAt: string;
}

export interface ManifestArtifact {
  id: string;
  manifestId: string;
  name: string;
  type: string;
  path: string | null;
  hash: string | null;
  createdAt: string;
}

export interface Resource {
  id: string;
  name: string;
  type: ResourceType;
  path: string | null;
  status: ResourceStatus;
  metadata: Record<string, unknown> | null;
  createdAt: string;
  updatedAt: string;
}

export interface ActionLog {
  id: string;
  mode: ActionMode;
  action: string;
  status: ActionStatus;
  description: string | null;
  permission: PermissionLevel;
  evidence: string | null;
  createdAt: string;
  completedAt: string | null;
}

export interface MemoryEntry {
  id: string;
  category: MemoryCategory;
  title: string;
  content: string;
  confidence: number;
  relevance: number;
  source: string | null;
  tags: string[];
  createdAt: string;
  updatedAt: string;
}

export interface StateGap {
  current: string | null;
  ideal: string | null;
  gapPercent: number | null;
  nextAction: string | null;
  urgency: "low" | "medium" | "high" | "critical";
}

export interface ResearchCitation {
  id: string;
  url: string;
  title: string;
  snippet: string;
  credibility: number;
  retrievedAt: string;
}

export interface AskMessage {
  id: string;
  role: "user" | "dema";
  content: string;
  citations?: ResearchCitation[];
  confidence?: number;
  trustState?: string;
  nextAction?: string;
  timestamp: string;
}

export interface BrowserSession {
  id: string;
  url: string;
  title: string;
  status: "idle" | "navigating" | "interacting" | "completed" | "error";
  actionLog: string[];
  startedAt: string;
  screenshotUrl?: string;
}

// ═══════════════════════════════════════════════════════════════
// BIZRA Orchestration System Types
// ═══════════════════════════════════════════════════════════════

export type AgentStatus = "idle" | "active" | "busy" | "error" | "sleeping" | "terminated";
export type AgentRole = "coordinator" | "researcher" | "executor" | "verifier" | "observer" | "optimizer" | "guardian";
export type AgentCapability = "reasoning" | "code_gen" | "web_search" | "file_io" | "browser_auto" | "system_exec" | "graph_analysis" | "crypto_verify" | "telemetry" | "memory_mgmt";

export interface Agent {
  id: string;
  name: string;
  role: AgentRole;
  status: AgentStatus;
  capabilities: AgentCapability[];
  trustScore: number;
  tasksCompleted: number;
  tasksFailed: number;
  lastActivity: string;
  uptime: number; // seconds
  metadata: Record<string, unknown>;
  parentId?: string;
  children?: string[];
}

export interface AgentTask {
  id: string;
  agentId: string;
  title: string;
  description: string;
  status: "queued" | "assigned" | "executing" | "completed" | "failed" | "cancelled";
  priority: "low" | "medium" | "high" | "critical";
  assignedAt: string | null;
  startedAt: string | null;
  completedAt: string | null;
  result?: string;
  error?: string;
}

export interface OrchestrationEvent {
  id: string;
  type: "agent_spawn" | "agent_status" | "task_assigned" | "task_completed" | "task_failed" | "coordination" | "handoff" | "termination" | "heartbeat";
  agentId?: string;
  message: string;
  metadata: Record<string, unknown>;
  timestamp: string;
  severity: "info" | "warning" | "error" | "success";
}

// ═══════════════════════════════════════════════════════════════
// Impact Calculation & Graph System Types
// ═══════════════════════════════════════════════════════════════

export type GraphNodeType = "action" | "agent" | "resource" | "receipt" | "mission" | "memory" | "boundary";
export type EdgeType = "depends_on" | "produces" | "verifies" | "delegates" | "blocks" | "informs";

export interface GraphNode {
  id: string;
  label: string;
  type: GraphNodeType;
  x: number;
  y: number;
  status?: string;
  weight: number;
  metadata: Record<string, unknown>;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: EdgeType;
  weight: number;
  label?: string;
}

export interface ImpactPropagation {
  id: string;
  sourceNodeId: string;
  targetType: GraphNodeType;
  affectedNodes: string[];
  depth: number;
  impactScore: number;
  propagationPath: string[];
  status: "calculating" | "completed" | "failed";
  timestamp: string;
  insights: string[];
}

export interface GraphSnapshot {
  id: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  propagations: ImpactPropagation[];
  totalNodes: number;
  totalEdges: number;
  density: number;
  timestamp: string;
}

// ═══════════════════════════════════════════════════════════════
// Governance & Cryptographic Validation Types
// ═══════════════════════════════════════════════════════════════

export type TrustAnchorType = "constitutional" | "cryptographic" | "reputation" | "behavioral";
export type ProofType = "receipt_chain" | "hash_verification" | "signature" | "merkle_proof" | "zero_knowledge";
export type GovernanceAction = "allow" | "deny" | "escalate" | "quarantine" | "audit" | "revoke";

export interface TrustAnchor {
  id: string;
  name: string;
  type: TrustAnchorType;
  publicKey?: string;
  algorithm?: string;
  active: boolean;
  lastUsed: string;
  verifications: number;
  failures: number;
}

export interface CryptoProof {
  id: string;
  type: ProofType;
  anchorId: string;
  subject: string;
  hash: string;
  signature?: string;
  verified: boolean;
  verifiedAt: string | null;
  expiresAt: string | null;
  metadata: Record<string, unknown>;
}

export interface GovernanceRule {
  id: string;
  name: string;
  description: string;
  category: "boundary" | "permission" | "integrity" | "privacy" | "performance";
  severity: "low" | "medium" | "high" | "critical";
  action: GovernanceAction;
  conditions: string[];
  active: boolean;
  violations: number;
  lastViolated: string | null;
}

export interface GovernanceEvent {
  id: string;
  ruleId: string;
  ruleName: string;
  action: GovernanceAction;
  subject: string;
  description: string;
  severity: "low" | "medium" | "high" | "critical";
  timestamp: string;
}

// ═══════════════════════════════════════════════════════════════
// Autopilot & Self-Optimization Types
// ═══════════════════════════════════════════════════════════════

export type OptimizationCycleStatus = "idle" | "scanning" | "analyzing" | "optimizing" | "validating" | "applying" | "completed" | "rollback";
export type MetricTrend = "improving" | "stable" | "degrading" | "volatile";

export interface OptimizationCycle {
  id: string;
  cycleNumber: number;
  status: OptimizationCycleStatus;
  startedAt: string;
  completedAt: string | null;
  duration: number;
  optimizations: OptimizationAction[];
  metricsBefore: Record<string, number>;
  metricsAfter: Record<string, number>;
  improvement: number;
  rollbackTriggered: boolean;
}

export interface OptimizationAction {
  id: string;
  target: string;
  action: string;
  description: string;
  impact: "low" | "medium" | "high";
  risk: "low" | "medium" | "high";
  status: "pending" | "applied" | "failed" | "reverted";
  result?: string;
}

export interface SystemMetric {
  id: string;
  name: string;
  category: "performance" | "reliability" | "security" | "efficiency" | "quality";
  value: number;
  target: number;
  unit: string;
  trend: MetricTrend;
  history: { timestamp: string; value: number }[];
}

export interface EvolutionProjection {
  id: string;
  horizon: "1h" | "6h" | "24h" | "7d" | "30d";
  confidence: number;
  predictions: {
    metric: string;
    currentValue: number;
    projectedValue: number;
    direction: "up" | "down" | "stable";
  }[];
  recommendations: string[];
  risks: string[];
  timestamp: string;
}

// ═══════════════════════════════════════════════════════════════
// Real-Time Operations & Telemetry Types
// ═══════════════════════════════════════════════════════════════

export type TelemetryLevel = "debug" | "info" | "warn" | "error" | "fatal";
export type SystemComponent = "gateway" | "agent_runtime" | "trust_engine" | "receipt_chain" | "memory_store" | "resource_registry" | "optimization_engine" | "governance_layer";

export interface TelemetryEvent {
  id: string;
  level: TelemetryLevel;
  component: SystemComponent;
  message: string;
  metadata: Record<string, unknown>;
  timestamp: string;
  traceId?: string;
  spanId?: string;
}

export interface SystemHealth {
  component: SystemComponent;
  status: "healthy" | "degraded" | "down" | "unknown";
  uptime: number;
  latency: number;
  errorRate: number;
  throughput: number;
  lastCheck: string;
}

export interface PerformanceSnapshot {
  id: string;
  timestamp: string;
  cpu: number;
  memory: number;
  diskIo: number;
  networkIo: number;
  activeConnections: number;
  requestRate: number;
  errorRate: number;
  p50Latency: number;
  p95Latency: number;
  p99Latency: number;
}

// ═══════════════════════════════════════════════════════════════
// BIZRA-ADK — Agent Factory Types
// ═══════════════════════════════════════════════════════════════

export type ADKAgentState = "draft" | "chartered" | "wired" | "exercised" | "sealed" | "frozen";
export type ADKCouncilType = "PAT-7" | "SAT-5" | "Custom";
export type ADKLifecycleStep = "NIYYAH" | "BAYYINAH" | "HADD" | "AMANAH" | "THAMARA" | "IISAL" | "RETROSPECTIVE";
export type ADKVerdictKind = "Pass" | "BlockedByIhsan" | "BlockedByEvidence" | "BlockedByDignity" | "BlockedByBudget" | "BlockedByAnchor" | "BlockedByCharter";
export type ADKMigrationPhase = "A" | "B" | "C" | "D" | "E";
export type ADKMigrationStatus = "pending" | "in_progress" | "completed" | "blocked" | "killed";
export type ADKToolSource = "local_corpus" | "local_file" | "wrapped_external" | "external_unverified";
export type ADKBudgetKind = "tokens" | "wall_seconds" | "tool_calls" | "evidence_fetches";

export interface ADKAgentIdentity {
  id: string;
  name: string;
  council: ADKCouncilType;
  charterHash: string;
  publicKey: string;
  governanceClass: "PAT" | "SAT" | "Frozen" | "Sovereign";
  state: ADKAgentState;
  model: string;
  frozen: boolean;
  charterText: string;
  tools: ADKTool[];
  locCount: number;
  testCount: number;
  testsPassing: number;
  lastLoopProofAt: string | null;
  createdAt: string;
}

export interface ADKTool {
  id: string;
  name: string;
  description: string;
  source: ADKToolSource;
  maxResults?: number;
  wrappedAt?: string | null;
}

export interface ADKMission {
  id: string;
  question: string;
  requesterAgentId: string;
  requesterAgentName: string;
  targetAgentId: string;
  targetAgentName: string;
  class: "Pat" | "Sat" | "Frozen" | "Sovereign";
  allowExternalUnwrapped: boolean;
  budget: ADKBudget;
  status: "created" | "executing" | "passed" | "blocked" | "sealed";
  createdAt: string;
  completedAt: string | null;
}

export interface ADKBudget {
  maxTokens: number;
  maxWallSeconds: number;
  maxToolCalls: number;
  maxEvidenceFetches: number;
}

export interface ADKReceiptNode {
  id: string;
  missionId: string;
  actorAgentName: string;
  actorAgentId: string;
  action: string;
  contentHash: string;
  parentReceiptHash: string | null;
  childReceiptHashes: string[];
  ihsanScore: number | null;
  verdict: ADKVerdictKind | null;
  sealed: boolean;
  evidenceCount: number;
  timestamp: string;
}

export interface ADKLifecycleTrace {
  missionId: string;
  agentName: string;
  steps: {
    step: ADKLifecycleStep;
    status: "pending" | "active" | "completed" | "failed" | "blocked";
    startedAt: string | null;
    completedAt: string | null;
    duration: number | null;
    output?: string;
  }[];
}

export interface ADKMigrationCheckpoint {
  phase: ADKMigrationPhase;
  title: string;
  description: string;
  status: ADKMigrationStatus;
  focusedDays: number;
  calendarWeeks: string;
  gate: string;
  killCondition?: string;
  deliverables: { label: string; done: boolean }[];
  startedAt: string | null;
  completedAt: string | null;
}

export interface ADKPrimitiveDef {
  name: string;
  arabic: string;
  english: string;
  description: string;
  color: string;
  icon: string;
}

export interface ADKTestSuite {
  category: "unit" | "property" | "integration" | "adversarial" | "regression" | "daughter_test";
  target: number;
  current: number;
  passing: number;
}

export interface ADKSchemaVersion {
  version: string;
  path: string;
  language: "CDDL" | "Python" | "Rust";
  driftDetected: boolean;
  lastBumpedAt: string | null;
}

// ═══════════════════════════════════════════════════════════════
// DEMA Constitutional Generative UI — Mission Lifecycle Types
// ═══════════════════════════════════════════════════════════════

export type MissionStage =
  | "idle"
  | "intent"
  | "admissibility"
  | "action"
  | "confirmation"
  | "receipt"
  | "blocked";

export type MissionType =
  | "organize"
  | "research"
  | "analyze"
  | "create"
  | "communicate"
  | "monitor";

export type MissionUrgency = "low" | "medium" | "high" | "critical";
export type MissionQuality = "draft" | "standard" | "precise";
export type MissionScope = "narrow" | "normal" | "wide";

export type GateId =
  | "ZANN_ZERO"
  | "CLAIM_MUST_BIND"
  | "RIBA_ZERO"
  | "NO_SHADOW_STATE"
  | "IHSAN_FLOOR";

export type GateStatus = "pending" | "evaluating" | "passed" | "blocked";

export type SurfaceType =
  | "mission-composer"
  | "gate-ladder"
  | "resource-truth"
  | "organize-preview"
  | "receipt-reveal"
  | "memory-constellation"
  | "reject-remediation"
  | "memory-delta";

export interface GateEvaluation {
  id: GateId;
  label: string;
  description: string;
  status: GateStatus;
  detail: string | null;
}

export interface MissionActionStep {
  id: string;
  label: string;
  description: string;
  type: "read" | "write" | "navigate" | "compute" | "verify";
  resource?: string;
  status: "pending" | "active" | "completed" | "skipped";
}

export interface MissionActionPlan {
  steps: MissionActionStep[];
  estimatedDuration: string;
  resourcesRequired: string[];
  dryRunAvailable: boolean;
  dryRunResult: {
    filesAffected: number;
    operationsPlanned: number;
    riskLevel: "low" | "medium" | "high";
    warnings: string[];
  } | null;
}

export interface Mission {
  id: string;
  intent: string;
  currentState: string;
  desiredState: string;
  missionType: MissionType;
  urgency: MissionUrgency;
  quality: MissionQuality;
  scope: MissionScope;
  stage: MissionStage;
  gates: GateEvaluation[];
  selectedResources: string[];
  actionPlan: MissionActionPlan | null;
  sealedReceipt: Receipt | null;
  createdAt: string;
  sealedAt: string | null;
}

export interface StageTransition {
  from: MissionStage;
  to: MissionStage;
  label: string;
  timestamp: string;
}

export const GATE_DEFINITIONS: GateEvaluation[] = [
  { id: "ZANN_ZERO", label: "ZANN_ZERO", description: "No fabricated data or hallucinated claims", status: "pending", detail: null },
  { id: "CLAIM_MUST_BIND", label: "CLAIM_MUST_BIND", description: "Every claim must bind to a verifiable receipt", status: "pending", detail: null },
  { id: "RIBA_ZERO", label: "RIBA_ZERO", description: "No extractive, exploitative, or unjust mechanisms", status: "pending", detail: null },
  { id: "NO_SHADOW_STATE", label: "NO_SHADOW_STATE", description: "No hidden or duplicated constitutional truth", status: "pending", detail: null },
  { id: "IHSAN_FLOOR", label: "IHSAN_FLOOR", description: "Ihsan (excellence) score meets minimum threshold (0.85)", status: "pending", detail: null },
];

export const SURFACE_LABELS: Record<SurfaceType, string> = {
  "mission-composer": "Mission Composer",
  "gate-ladder": "Gate Ladder",
  "resource-truth": "Resource Truth",
  "organize-preview": "Action Preview",
  "receipt-reveal": "Receipt Reveal",
  "memory-constellation": "Memory Constellation",
  "reject-remediation": "Remediation",
  "memory-delta": "Memory Update",
};

export const MISSION_TYPE_LABELS: Record<MissionType, string> = {
  organize: "Organize",
  research: "Research",
  analyze: "Analyze",
  create: "Create",
  communicate: "Communicate",
  monitor: "Monitor",
};
