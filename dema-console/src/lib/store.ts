// ═══════════════════════════════════════════════════════════════
// DEMA — Zustand Store
// Central state management for the sovereign operator face
// ═══════════════════════════════════════════════════════════════

import { create } from "zustand";
import type {
  Screen,
  TrustState,
  Receipt,
  Manifest,
  Resource,
  ActionLog,
  MemoryEntry,
  StateGap,
  AskMessage,
  BrowserSession,
  Agent,
  AgentTask,
  OrchestrationEvent,
  GraphSnapshot,
  ImpactPropagation,
  TrustAnchor,
  CryptoProof,
  GovernanceRule,
  GovernanceEvent,
  OptimizationCycle,
  SystemMetric,
  EvolutionProjection,
  TelemetryEvent,
  SystemHealth,
  PerformanceSnapshot,
  ADKAgentIdentity,
  ADKMission,
  ADKLifecycleTrace,
  ADKMigrationCheckpoint,
  ADKTestSuite,
  ADKSchemaVersion,
} from "./types";

// ─── Demo / Seed Data ──────────────────────────────────────────

// NO_SHADOW_STATE: no principal is activated until the kernel says so.
// Cycle-8 PR#28 face polish: inactive default replaces fabricated demo
// principal. UI MUST read isActive=false as "no principal activated"
// and show an activation-prompt surface, NOT a populated trust panel.
const DEMO_TRUST_STATE: TrustState = {
  principalId: "",
  principalName: "",
  level: "citizen",
  score: 0,
  maxScore: 100,
  lastVerified: new Date(0).toISOString(),
  sessionId: "",
  isActive: false,
};

// NO_SHADOW_STATE: these arrays start empty. The chain is empty until
// the operator seals something real. The face MUST NOT display
// fabricated receipts, manifests, or resources.
const DEMO_RECEIPTS: Receipt[] = [];
const DEMO_MANIFESTS: Manifest[] = [];
const DEMO_RESOURCES: Resource[] = [];

const DEMO_ACTION_LOG: ActionLog[] = [
  { id: "act-001", mode: "research", action: "market_analysis", status: "completed", description: "Competitive analysis of Claude Code, Perplexity, and Manus.", permission: "auto", evidence: "Report saved to /knowledge/market/", createdAt: new Date(Date.now() - 3600000).toISOString(), completedAt: new Date(Date.now() - 1800000).toISOString() },
  { id: "act-002", mode: "code", action: "schema_migration", status: "completed", description: "Applied Prisma schema changes for Receipt and Manifest models.", permission: "explicit", evidence: "91/91 tests passing", createdAt: new Date(Date.now() - 7200000).toISOString(), completedAt: new Date(Date.now() - 5400000).toISOString() },
  { id: "act-003", mode: "browser", action: "page_navigation", status: "completed", description: "Navigated to Anthropic Claude Code documentation.", permission: "explicit", evidence: null, createdAt: new Date(Date.now() - 10800000).toISOString(), completedAt: new Date(Date.now() - 10200000).toISOString() },
  { id: "act-004", mode: "computer", action: "file_read", description: "Read project configuration files.", status: "completed", permission: "auto", evidence: "3 files read", createdAt: new Date(Date.now() - 14400000).toISOString(), completedAt: new Date(Date.now() - 14340000).toISOString() },
];

const DEMO_MEMORY: MemoryEntry[] = [
  { id: "mem-001", category: "knowledge", title: "DEMA Product Thesis", content: "DEMA should compete as the first calm, sovereign, full-stack operator face.", confidence: 0.95, relevance: 1.0, source: "product-canon", tags: ["product", "thesis", "strategy"], createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() },
  { id: "mem-002", category: "preference", title: "No Shadow State", content: "DEMA must never maintain shadow state in the frontend.", confidence: 1.0, relevance: 0.9, source: "ADR-002", tags: ["architecture", "law", "state"], createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() },
  { id: "mem-003", category: "context", title: "Cycle-7 Phase 1 Status", content: "Core runtime at 91/91 tests, edited and green but not yet committed.", confidence: 0.98, relevance: 0.85, source: "internal-update", tags: ["runtime", "status", "cycle-7"], createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() },
  { id: "mem-004", category: "poi", title: "Claude Code MCP Pattern", content: "Anthropic's Claude Code uses Model Context Protocol for extensibility.", confidence: 0.9, relevance: 0.7, source: "anthropic-docs", tags: ["competitor", "mcp", "claude"], createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() },
  { id: "mem-005", category: "knowledge", title: "Boundary: Core vs Face", content: "bizra-omega owns constitutional truth. DEMA owns product face.", confidence: 1.0, relevance: 0.95, source: "ADR-003", tags: ["architecture", "boundary", "law"], createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() },
  { id: "mem-006", category: "knowledge", title: "DEMA Core Purpose", content: "Dema exists to confront the silent killers of human flourishing: false assumptions, extractive systems, and the learned belief that a person is powerless. Dema stands against assumption without evidence, against riba and every unjust mechanism of extraction, and for the dignity and empowerment of every human being.", confidence: 1.0, relevance: 1.0, source: "constitution", tags: ["purpose", "philosophy", "dignity", "anti-riba"], createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() },
  { id: "mem-007", category: "knowledge", title: "BIZRA Three Pillars", content: "BIZRA stands on three parallel pillars: Ideology (meaning, law, dignity), AI (cognition, agency, interface), and Blockchain (proof, persistence, value). Their fusion creates a new sovereign human system — not borrowed chain, not rented intelligence, not extractive economics.", confidence: 1.0, relevance: 0.95, source: "architecture-canon", tags: ["architecture", "ideology", "ai", "blockchain", "fusion"], createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() },
  { id: "mem-008", category: "knowledge", title: "The Humility Principle", content: "The more I learned, the more I realized my ignorance — and that what I see as correct may carry error, and what I see as wrong in another may carry truth. This is DEMA's permanent spirit: truth-seeking humility over confident assertion.", confidence: 1.0, relevance: 1.0, source: "philosophy-canon", tags: ["philosophy", "humility", "truth", "wisdom"], createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() },
  { id: "mem-009", category: "knowledge", title: "Human Equality Canon", content: "Dema serves each person — regardless of belief, color, wealth, or status — with a personal think tank and task force. Every human being is equal. What appears impossible to human weakness is never beyond the power of Allah.", confidence: 1.0, relevance: 1.0, source: "constitution", tags: ["equality", "dignity", "human-rights", "canon"], createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() },
];

const DEMO_STATE_GAP: StateGap = {
  current: "Sovereign operator session active. BIZRA orchestration ecosystem running. First Citizen Path complete.",
  ideal: "Fully autonomous multi-agent ecosystem with self-optimization, cryptographic governance, and impact-weighted consensus.",
  gapPercent: 28,
  nextAction: "Explore DEMA operator modes and begin your first mission.",
  urgency: "low",
};

const DEMO_ASK_MESSAGES: AskMessage[] = [
  { id: "msg-001", role: "dema", content: "Welcome. I am DEMA — your sovereign operator face.\n\nI exist to help you think more clearly, act more wisely, and rise beyond what you thought was possible. Every action I take produces a receipt. Every receipt binds to your trust chain. You stay in control — always.\n\nI can help you with research, code, browser automation, system operations, and more. All governed by constitutional law.\n\nHow can I serve you today?", confidence: 1.0, trustState: "citizen", timestamp: new Date(Date.now() - 60000).toISOString() },
];

// ─── BIZRA Orchestration Seed Data ─────────────────────────────

const DEMO_AGENTS: Agent[] = [
  { id: "agt-coord-01", name: "Nexus Prime", role: "coordinator", status: "active", capabilities: ["reasoning", "graph_analysis", "memory_mgmt"], trustScore: 96, tasksCompleted: 847, tasksFailed: 3, lastActivity: new Date(Date.now() - 5000).toISOString(), uptime: 864000, metadata: { version: "3.2.1", model: "opus" } },
  { id: "agt-res-01", name: "Athena", role: "researcher", status: "busy", capabilities: ["reasoning", "web_search", "memory_mgmt"], trustScore: 91, tasksCompleted: 423, tasksFailed: 7, lastActivity: new Date(Date.now() - 12000).toISOString(), uptime: 432000, metadata: { version: "2.8.0", model: "sonnet" }, parentId: "agt-coord-01" },
  { id: "agt-res-02", name: "Archimedes", role: "researcher", status: "active", capabilities: ["reasoning", "code_gen", "file_io"], trustScore: 88, tasksCompleted: 291, tasksFailed: 12, lastActivity: new Date(Date.now() - 30000).toISOString(), uptime: 345600, metadata: { version: "2.8.0", model: "sonnet" }, parentId: "agt-coord-01" },
  { id: "agt-exec-01", name: "Vulcan", role: "executor", status: "active", capabilities: ["code_gen", "file_io", "system_exec", "browser_auto"], trustScore: 85, tasksCompleted: 1203, tasksFailed: 23, lastActivity: new Date(Date.now() - 8000).toISOString(), uptime: 691200, metadata: { version: "3.1.0", model: "sonnet" }, parentId: "agt-coord-01" },
  { id: "agt-exec-02", name: "Hermes", role: "executor", status: "busy", capabilities: ["browser_auto", "system_exec", "web_search"], trustScore: 82, tasksCompleted: 567, tasksFailed: 31, lastActivity: new Date(Date.now() - 3000).toISOString(), uptime: 259200, metadata: { version: "2.5.0", model: "haiku" }, parentId: "agt-coord-01" },
  { id: "agt-ver-01", name: "Themis", role: "verifier", status: "active", capabilities: ["crypto_verify", "reasoning", "memory_mgmt"], trustScore: 98, tasksCompleted: 2104, tasksFailed: 1, lastActivity: new Date(Date.now() - 15000).toISOString(), uptime: 864000, metadata: { version: "4.0.0", model: "opus" }, parentId: "agt-coord-01" },
  { id: "agt-obs-01", name: "Argus", role: "observer", status: "active", capabilities: ["telemetry", "graph_analysis"], trustScore: 94, tasksCompleted: 3456, tasksFailed: 0, lastActivity: new Date(Date.now() - 2000).toISOString(), uptime: 864000, metadata: { version: "3.0.0", model: "haiku" }, parentId: "agt-coord-01" },
  { id: "agt-opt-01", name: "Daedalus", role: "optimizer", status: "idle", capabilities: ["reasoning", "graph_analysis", "memory_mgmt"], trustScore: 90, tasksCompleted: 156, tasksFailed: 2, lastActivity: new Date(Date.now() - 300000).toISOString(), uptime: 172800, metadata: { version: "1.2.0", model: "opus" }, parentId: "agt-coord-01" },
  { id: "agt-grd-01", name: "Aegis", role: "guardian", status: "active", capabilities: ["crypto_verify", "reasoning"], trustScore: 99, tasksCompleted: 4891, tasksFailed: 0, lastActivity: new Date(Date.now() - 1000).toISOString(), uptime: 864000, metadata: { version: "5.0.0", model: "opus" } },
];

const DEMO_AGENT_TASKS: AgentTask[] = [
  { id: "tsk-001", agentId: "agt-res-01", title: "Deep analysis: multi-agent coordination patterns", description: "Research and synthesize best practices for autonomous agent orchestration from recent literature.", status: "executing", priority: "high", assignedAt: new Date(Date.now() - 60000).toISOString(), startedAt: new Date(Date.now() - 45000).toISOString(), completedAt: null },
  { id: "tsk-002", agentId: "agt-exec-02", title: "Browser navigation: API documentation scrape", description: "Navigate to target documentation site and extract structured data from 12 endpoint pages.", status: "executing", priority: "medium", assignedAt: new Date(Date.now() - 120000).toISOString(), startedAt: new Date(Date.now() - 90000).toISOString(), completedAt: null },
  { id: "tsk-003", agentId: "agt-exec-01", title: "Code generation: trust validation module", description: "Generate TypeScript module for cryptographic receipt chain validation with Merkle proof support.", status: "assigned", priority: "critical", assignedAt: new Date(Date.now() - 30000).toISOString(), startedAt: null, completedAt: null },
  { id: "tsk-004", agentId: "agt-ver-01", title: "Verify: schema integrity check", description: "Cross-validate Prisma schema against constitutional type definitions in bizra-omega.", status: "completed", priority: "high", assignedAt: new Date(Date.now() - 300000).toISOString(), startedAt: new Date(Date.now() - 280000).toISOString(), completedAt: new Date(Date.now() - 120000).toISOString(), result: "All 47 schema definitions validated. 0 drift detected." },
  { id: "tsk-005", agentId: "agt-coord-01", title: "Coordination: next cycle planning", description: "Evaluate current system state and plan optimization cycle #12.", status: "queued", priority: "medium", assignedAt: null, startedAt: null, completedAt: null },
  { id: "tsk-006", agentId: "agt-grd-01", title: "Governance: boundary audit scan", description: "Perform full boundary audit across all agent communication channels.", status: "completed", priority: "critical", assignedAt: new Date(Date.now() - 600000).toISOString(), startedAt: new Date(Date.now() - 580000).toISOString(), completedAt: new Date(Date.now() - 300000).toISOString(), result: "All 9 boundary rules verified. 0 violations." },
];

const now = Date.now();
const DEMO_ORCHESTRATION_EVENTS: OrchestrationEvent[] = [
  { id: "evt-001", type: "heartbeat", agentId: "agt-coord-01", message: "Nexus Prime heartbeat — all subsystems nominal", metadata: { cpu: 23, memory: 1.2, agents: 9 }, timestamp: new Date(now - 2000).toISOString(), severity: "info" },
  { id: "evt-002", type: "task_assigned", agentId: "agt-exec-01", message: "Vulcan assigned critical task: trust validation module", metadata: { taskId: "tsk-003", priority: "critical" }, timestamp: new Date(now - 30000).toISOString(), severity: "info" },
  { id: "evt-003", type: "task_completed", agentId: "agt-ver-01", message: "Themis completed schema integrity check — 0 drift", metadata: { taskId: "tsk-004", duration: 160000 }, timestamp: new Date(now - 120000).toISOString(), severity: "success" },
  { id: "evt-004", type: "coordination", agentId: "agt-coord-01", message: "Coordinator: research and execution pipelines synchronized", metadata: { pipelineA: "research", pipelineB: "execution", syncPoint: "gate-7" }, timestamp: new Date(now - 60000).toISOString(), severity: "info" },
  { id: "evt-005", type: "task_completed", agentId: "agt-grd-01", message: "Aegis boundary audit complete — 0 violations across 9 rules", metadata: { taskId: "tsk-006", rulesChecked: 9, violations: 0 }, timestamp: new Date(now - 300000).toISOString(), severity: "success" },
  { id: "evt-006", type: "agent_status", agentId: "agt-opt-01", message: "Daedalus entered idle state — awaiting next optimization window", metadata: { reason: "no_pending_targets", nextWindow: "+45min" }, timestamp: new Date(now - 300000).toISOString(), severity: "info" },
  { id: "evt-007", type: "handoff", agentId: "agt-res-01", message: "Athena → Vulcan handoff: research findings ready for implementation", metadata: { sourceTask: "tsk-res-12", targetTask: "tsk-003" }, timestamp: new Date(now - 180000).toISOString(), severity: "info" },
  { id: "evt-008", type: "task_failed", agentId: "agt-exec-02", message: "Hermes: browser navigation timeout on target endpoint", metadata: { taskId: "tsk-br-03", url: "https://api.target.com/docs/v3", timeout: 30000 }, timestamp: new Date(now - 420000).toISOString(), severity: "warning" },
  { id: "evt-009", type: "agent_spawn", agentId: "agt-coord-01", message: "Nexus Prime spawned Hermes worker for parallel browser tasks", metadata: { workerId: "agt-exec-02", capabilities: ["browser_auto"] }, timestamp: new Date(now - 259200000).toISOString(), severity: "info" },
  { id: "evt-010", type: "termination", agentId: "agt-coord-01", message: "Deprecated agent Sentinel-01 terminated after graceful handoff", metadata: { oldAgent: "agt-old-01", reason: "superseded_by_aegis" }, timestamp: new Date(now - 432000000).toISOString(), severity: "info" },
];

// ─── Impact Calculation & Graph Seed Data ──────────────────────

const DEMO_GRAPH: GraphSnapshot = {
  id: "graph-snap-001",
  nodes: [
    { id: "n-coord", label: "Nexus Prime", type: "agent", x: 400, y: 200, status: "active", weight: 0.96, metadata: { role: "coordinator" } },
    { id: "n-athena", label: "Athena", type: "agent", x: 200, y: 100, status: "busy", weight: 0.91, metadata: { role: "researcher" } },
    { id: "n-vulcan", label: "Vulcan", type: "agent", x: 600, y: 100, status: "active", weight: 0.85, metadata: { role: "executor" } },
    { id: "n-themis", label: "Themis", type: "agent", x: 200, y: 300, status: "active", weight: 0.98, metadata: { role: "verifier" } },
    { id: "n-hermes", label: "Hermes", type: "agent", x: 600, y: 300, status: "busy", weight: 0.82, metadata: { role: "executor" } },
    { id: "n-aegis", label: "Aegis", type: "agent", x: 400, y: 380, status: "active", weight: 0.99, metadata: { role: "guardian" } },
    { id: "n-omega", label: "bizra-omega", type: "resource", x: 400, y: 60, status: "active", weight: 1.0, metadata: { type: "service" } },
    { id: "n-receipt", label: "Receipt Chain", type: "receipt", x: 100, y: 200, status: "verified", weight: 0.95, metadata: { count: 47 } },
    { id: "n-mission", label: "Mission Alpha", type: "mission", x: 700, y: 200, status: "active", weight: 0.87, metadata: { phase: "execution" } },
    { id: "n-memory", label: "Memory Store", type: "memory", x: 300, y: 380, status: "active", weight: 0.92, metadata: { entries: 128 } },
    { id: "n-boundary", label: "Core/Face Boundary", type: "boundary", x: 500, y: 380, status: "enforced", weight: 1.0, metadata: { rules: 9 } },
  ],
  edges: [
    { id: "e1", source: "n-coord", target: "n-athena", type: "delegates", weight: 0.8, label: "research" },
    { id: "e2", source: "n-coord", target: "n-vulcan", type: "delegates", weight: 0.8, label: "execute" },
    { id: "e3", source: "n-coord", target: "n-themis", type: "delegates", weight: 0.9, label: "verify" },
    { id: "e4", source: "n-coord", target: "n-hermes", type: "delegates", weight: 0.7, label: "browse" },
    { id: "e5", source: "n-coord", target: "n-aegis", type: "informs", weight: 0.95, label: "govern" },
    { id: "e6", source: "n-omega", target: "n-coord", type: "informs", weight: 1.0, label: "truth" },
    { id: "e7", source: "n-themis", target: "n-receipt", type: "verifies", weight: 0.95, label: "validate" },
    { id: "e8", source: "n-vulcan", target: "n-mission", type: "produces", weight: 0.8, label: "artifacts" },
    { id: "e9", source: "n-athena", target: "n-memory", type: "produces", weight: 0.7, label: "insights" },
    { id: "e10", source: "n-aegis", target: "n-boundary", type: "verifies", weight: 1.0, label: "enforce" },
    { id: "e11", source: "n-hermes", target: "n-mission", type: "informs", weight: 0.6, label: "data" },
    { id: "e12", source: "n-coord", target: "n-memory", type: "depends_on", weight: 0.5, label: "context" },
  ],
  propagations: [
    { id: "prop-001", sourceNodeId: "n-coord", targetType: "agent", affectedNodes: ["n-athena", "n-vulcan", "n-themis", "n-hermes"], depth: 1, impactScore: 0.85, propagationPath: ["n-coord", "n-athena", "n-vulcan"], status: "completed", timestamp: new Date(now - 60000).toISOString(), insights: ["Coordinator failure would cascade to 4 agents", "Recovery time estimated: 12-45 seconds", "Vulcan has highest task dependency"] },
    { id: "prop-002", sourceNodeId: "n-omega", targetType: "resource", affectedNodes: ["n-coord", "n-athena", "n-vulcan", "n-themis", "n-hermes", "n-aegis"], depth: 2, impactScore: 0.98, propagationPath: ["n-omega", "n-coord", "n-athena"], status: "completed", timestamp: new Date(now - 120000).toISOString(), insights: ["Critical dependency: all agents depend on core", "Zero-knowledge gap exists for core failure", "Manual recovery would be required"] },
    { id: "prop-003", sourceNodeId: "n-boundary", targetType: "boundary", affectedNodes: ["n-coord", "n-aegis", "n-vulcan"], depth: 1, impactScore: 0.72, propagationPath: ["n-boundary", "n-aegis"], status: "completed", timestamp: new Date(now - 180000).toISOString(), insights: ["Boundary violation affects governance + execution", "3 of 9 rules are hard constraints", "Shadow state detection accuracy: 99.7%"] },
  ],
  totalNodes: 11,
  totalEdges: 12,
  density: 0.22,
  timestamp: new Date(now - 10000).toISOString(),
};

// ─── Governance & Crypto Validation Seed Data ──────────────────

const DEMO_TRUST_ANCHORS: TrustAnchor[] = [
  { id: "anchor-const", name: "Constitutional Root", type: "constitutional", publicKey: "0x7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069", algorithm: "SHA-256", active: true, lastUsed: new Date(now - 5000).toISOString(), verifications: 48912, failures: 0 },
  { id: "anchor-merkle", name: "Merkle Root Authority", type: "cryptographic", publicKey: "0x2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824", algorithm: "SHA-256 Merkle", active: true, lastUsed: new Date(now - 30000).toISOString(), verifications: 12403, failures: 2 },
  { id: "anchor-rep", name: "Reputation Oracle", type: "reputation", active: true, lastUsed: new Date(now - 120000).toISOString(), verifications: 8921, failures: 14 },
  { id: "anchor-beh", name: "Behavioral Analysis Engine", type: "behavioral", active: true, lastUsed: new Date(now - 60000).toISOString(), verifications: 6234, failures: 7 },
];

const DEMO_CRYPTO_PROOFS: CryptoProof[] = [
  { id: "proof-001", type: "receipt_chain", anchorId: "anchor-const", subject: "rcp-001 → rcp-002 chain", hash: "0xa3f2b8c1d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2", verified: true, verifiedAt: new Date(now - 1200000).toISOString(), expiresAt: null, metadata: { chainLength: 2, merkleRoot: "0x..." } },
  { id: "proof-002", type: "hash_verification", anchorId: "anchor-merkle", subject: "Manifest mft-001 artifact integrity", hash: "0xb4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6", verified: true, verifiedAt: new Date(now - 3600000).toISOString(), expiresAt: null, metadata: { artifactsChecked: 14, algorithm: "SHA-256" } },
  { id: "proof-003", type: "signature", anchorId: "anchor-const", subject: "Trust state transition: visitor → citizen", hash: "0xc5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7", signature: "sig_v1_k1_7f83b16...", verified: true, verifiedAt: new Date(now - 7200000).toISOString(), expiresAt: null, metadata: { signer: "Aegis", principalId: "prin-001" } },
  { id: "proof-004", type: "zero_knowledge", anchorId: "anchor-rep", subject: "Agent trust score proof (Vulcan)", hash: "0xd6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8", verified: true, verifiedAt: new Date(now - 14400000).toISOString(), expiresAt: new Date(now + 86400000).toISOString(), metadata: { disclosed: false, proofSize: "2.1KB" } },
  { id: "proof-005", type: "merkle_proof", anchorId: "anchor-merkle", subject: "Full receipt chain Merkle proof (47 receipts)", hash: "0xe7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9", verified: true, verifiedAt: new Date(now - 28800000).toISOString(), expiresAt: null, metadata: { treeDepth: 6, leafCount: 47, proofPath: 6 } },
];

const DEMO_GOVERNANCE_RULES: GovernanceRule[] = [
  { id: "rule-001", name: "No Shadow State", description: "DEMA face must never duplicate or cache constitutional truth from bizra-omega core.", category: "boundary", severity: "critical", action: "deny", conditions: ["face_cache_attempt", "state_replication_detected"], active: true, violations: 0, lastViolated: null },
  { id: "rule-002", name: "Explicit Write Permission", description: "All write operations (file, terminal, browser) require explicit operator approval.", category: "permission", severity: "high", action: "deny", conditions: ["write_without_explicit_consent"], active: true, violations: 2, lastViolated: new Date(now - 172800000).toISOString() },
  { id: "rule-003", name: "Receipt Chain Immutability", description: "Once issued and verified, receipts cannot be mutated outside approved contracts.", category: "integrity", severity: "critical", action: "quarantine", conditions: ["receipt_mutation_attempt", "chain_tampering"], active: true, violations: 0, lastViolated: null },
  { id: "rule-004", name: "Credential Isolation", description: "Agent credentials must never be exposed to other agents or persisted in memory.", category: "privacy", severity: "critical", action: "revoke", conditions: ["credential_exposure", "cross_agent_credential_leak"], active: true, violations: 0, lastViolated: null },
  { id: "rule-005", name: "Performance Budget", description: "Individual agent operations must complete within defined latency budgets.", category: "performance", severity: "medium", action: "escalate", conditions: ["operation_timeout", "latency_budget_exceeded"], active: true, violations: 8, lastViolated: new Date(now - 3600000).toISOString() },
  { id: "rule-006", name: "Mission Law Compliance", description: "All actions must comply with the currently active mission's constitutional rules.", category: "boundary", severity: "critical", action: "deny", conditions: ["mission_violation", "unconstitutional_action"], active: true, violations: 1, lastViolated: new Date(now - 604800000).toISOString() },
  { id: "rule-007", name: "Agent Communication Audit", description: "All inter-agent communications must be logged and auditable.", category: "privacy", severity: "high", action: "audit", conditions: ["unlogged_communication", "audit_gap_detected"], active: true, violations: 0, lastViolated: null },
  { id: "rule-008", name: "Graceful Degradation", description: "System must degrade gracefully under resource pressure, never hard-fail.", category: "performance", severity: "high", action: "escalate", conditions: ["hard_failure_risk", "resource_exhaustion"], active: true, violations: 3, lastViolated: new Date(now - 86400000).toISOString() },
  { id: "rule-009", name: "Human Override Priority", description: "Human operator commands always override autonomous decisions.", category: "permission", severity: "critical", action: "allow", conditions: ["human_override_requested"], active: true, violations: 0, lastViolated: null },
];

const DEMO_GOVERNANCE_EVENTS: GovernanceEvent[] = [
  { id: "gov-evt-001", ruleId: "rule-002", ruleName: "Explicit Write Permission", action: "deny", subject: "Hermes", description: "Browser write attempt blocked — no explicit consent for form submission", severity: "high", timestamp: new Date(now - 172800000).toISOString() },
  { id: "gov-evt-002", ruleId: "rule-005", ruleName: "Performance Budget", action: "escalate", subject: "Athena", description: "Research query exceeded 30s latency budget (actual: 47s)", severity: "medium", timestamp: new Date(now - 3600000).toISOString() },
  { id: "gov-evt-003", ruleId: "rule-006", ruleName: "Mission Law Compliance", action: "deny", subject: "Vulcan", description: "Attempted code edit outside mission scope — action quarantined", severity: "critical", timestamp: new Date(now - 604800000).toISOString() },
  { id: "gov-evt-004", ruleId: "rule-008", ruleName: "Graceful Degradation", action: "escalate", subject: "System", description: "Memory pressure detected (87%) — optimization cycle triggered", severity: "high", timestamp: new Date(now - 86400000).toISOString() },
  { id: "gov-evt-005", ruleId: "rule-007", ruleName: "Agent Communication Audit", action: "audit", subject: "Nexus Prime", description: "Scheduled audit complete — all 156 communications logged", severity: "low", timestamp: new Date(now - 432000000).toISOString() },
];

// ─── Autopilot & Self-Optimization Seed Data ───────────────────

function genMetricHistory(base: number, variance: number, count: number): { timestamp: string; value: number }[] {
  const points: { timestamp: string; value: number }[] = [];
  for (let i = count - 1; i >= 0; i--) {
    const t = new Date(now - i * 300000);
    const v = base + (Math.sin(i * 0.3) * variance) + (Math.random() - 0.5) * variance * 0.5;
    points.push({ timestamp: t.toISOString(), value: Math.round(Math.max(0, Math.min(100, v)) * 100) / 100 });
  }
  return points;
}

const DEMO_SYSTEM_METRICS: SystemMetric[] = [
  { id: "met-001", name: "Response Latency", category: "performance", value: 42, target: 50, unit: "ms", trend: "improving", history: genMetricHistory(48, 15, 24) },
  { id: "met-002", name: "Task Success Rate", category: "reliability", value: 97.3, target: 99, unit: "%", trend: "stable", history: genMetricHistory(96.5, 3, 24) },
  { id: "met-003", name: "Trust Score Average", category: "security", value: 91.2, target: 95, unit: "pts", trend: "improving", history: genMetricHistory(87, 5, 24) },
  { id: "met-004", name: "Resource Utilization", category: "efficiency", value: 67, target: 75, unit: "%", trend: "stable", history: genMetricHistory(64, 10, 24) },
  { id: "met-005", name: "Agent Coordination Score", category: "quality", value: 88, target: 90, unit: "pts", trend: "improving", history: genMetricHistory(82, 8, 24) },
  { id: "met-006", name: "Receipt Verification Time", category: "performance", value: 12, target: 10, unit: "ms", trend: "degrading", history: genMetricHistory(9, 4, 24) },
  { id: "met-007", name: "Error Rate", category: "reliability", value: 0.3, target: 0.1, unit: "%", trend: "improving", history: genMetricHistory(0.8, 0.5, 24) },
  { id: "met-008", name: "Boundary Compliance", category: "security", value: 99.7, target: 100, unit: "%", trend: "stable", history: genMetricHistory(99.5, 0.5, 24) },
];

const DEMO_OPTIMIZATION_CYCLES: OptimizationCycle[] = [
  {
    id: "opt-011", cycleNumber: 11, status: "completed", startedAt: new Date(now - 7200000).toISOString(), completedAt: new Date(now - 6600000).toISOString(), duration: 600000,
    optimizations: [
      { id: "opt-a-01", target: "memory_store", action: "cache_compaction", description: "Compacted memory cache, freed 234MB of stale entries", impact: "medium", risk: "low", status: "applied", result: "Memory reduced from 87% to 62%" },
      { id: "opt-a-02", target: "agent_runtime", action: "task_queue_reorder", description: "Reordered task queue by dependency graph, reduced wait time", impact: "high", risk: "low", status: "applied", result: "Avg wait time: 2.1s → 0.4s" },
    ],
    metricsBefore: { memory: 87, taskWaitTime: 2.1 },
    metricsAfter: { memory: 62, taskWaitTime: 0.4 },
    improvement: 14.3,
    rollbackTriggered: false,
  },
  {
    id: "opt-010", cycleNumber: 10, status: "completed", startedAt: new Date(now - 28800000).toISOString(), completedAt: new Date(now - 27600000).toISOString(), duration: 720000,
    optimizations: [
      { id: "opt-b-01", target: "trust_engine", action: "batch_verification", description: "Enabled batch receipt verification to reduce per-receipt overhead", impact: "high", risk: "low", status: "applied", result: "Verification throughput: 12/s → 47/s" },
      { id: "opt-b-02", target: "resource_registry", action: "connection_pooling", description: "Added connection pooling for resource health checks", impact: "medium", risk: "low", status: "applied", result: "Connection overhead: 45ms → 8ms" },
      { id: "opt-b-03", target: "governance_layer", action: "rule_precompile", description: "Pre-compiled governance rule conditions for faster evaluation", impact: "medium", risk: "medium", status: "applied", result: "Rule eval: 3.2ms → 0.8ms" },
    ],
    metricsBefore: { verificationThroughput: 12, connectionOverhead: 45, ruleEvalTime: 3.2 },
    metricsAfter: { verificationThroughput: 47, connectionOverhead: 8, ruleEvalTime: 0.8 },
    improvement: 21.7,
    rollbackTriggered: false,
  },
  {
    id: "opt-009", cycleNumber: 9, status: "completed", startedAt: new Date(now - 86400000).toISOString(), completedAt: new Date(now - 85200000).toISOString(), duration: 900000,
    optimizations: [
      { id: "opt-c-01", target: "agent_runtime", action: "parallel_spawn", description: "Enabled parallel agent spawning with dependency resolution", impact: "high", risk: "medium", status: "applied", result: "Spawn time: 4.5s → 0.8s" },
      { id: "opt-c-02", target: "memory_store", action: "index_rebuild", description: "Rebuilt memory index with improved vectorization", impact: "high", risk: "low", status: "applied", result: "Query time: 120ms → 23ms" },
    ],
    metricsBefore: { spawnTime: 4.5, memoryQueryTime: 120 },
    metricsAfter: { spawnTime: 0.8, memoryQueryTime: 23 },
    improvement: 28.4,
    rollbackTriggered: false,
  },
];

const DEMO_EVOLUTION_PROJECTION: EvolutionProjection = {
  id: "evo-001", horizon: "24h", confidence: 0.87,
  predictions: [
    { metric: "Task Success Rate", currentValue: 97.3, projectedValue: 98.1, direction: "up" },
    { metric: "Response Latency", currentValue: 42, projectedValue: 38, direction: "up" },
    { metric: "Trust Score Average", currentValue: 91.2, projectedValue: 92.5, direction: "up" },
    { metric: "Resource Utilization", currentValue: 67, projectedValue: 71, direction: "up" },
    { metric: "Error Rate", currentValue: 0.3, projectedValue: 0.2, direction: "up" },
  ],
  recommendations: [
    "Enable predictive pre-warming for frequently accessed resources",
    "Consider promoting Daedalus from idle to active for continuous optimization",
    "Receipt verification batch size should increase to 100 per cycle",
  ],
  risks: [
    "Memory growth trend (+2.3%/hr) may trigger budget in 18-24 hours",
    "Athena workload concentration: 78% of research tasks assigned to single agent",
  ],
  timestamp: new Date(now - 60000).toISOString(),
};

// ─── Real-Time Operations & Telemetry Seed Data ────────────────

const DEMO_SYSTEM_HEALTH: SystemHealth[] = [
  { component: "gateway", status: "healthy", uptime: 864000, latency: 12, errorRate: 0.01, throughput: 847, lastCheck: new Date(now - 5000).toISOString() },
  { component: "agent_runtime", status: "healthy", uptime: 864000, latency: 23, errorRate: 0.03, throughput: 156, lastCheck: new Date(now - 3000).toISOString() },
  { component: "trust_engine", status: "healthy", uptime: 864000, latency: 8, errorRate: 0.0, throughput: 2341, lastCheck: new Date(now - 2000).toISOString() },
  { component: "receipt_chain", status: "healthy", uptime: 864000, latency: 5, errorRate: 0.0, throughput: 412, lastCheck: new Date(now - 8000).toISOString() },
  { component: "memory_store", status: "healthy", uptime: 864000, latency: 34, errorRate: 0.02, throughput: 89, lastCheck: new Date(now - 10000).toISOString() },
  { component: "resource_registry", status: "healthy", uptime: 864000, latency: 11, errorRate: 0.0, throughput: 67, lastCheck: new Date(now - 15000).toISOString() },
  { component: "optimization_engine", status: "degraded", uptime: 172800, latency: 156, errorRate: 0.0, throughput: 0.1, lastCheck: new Date(now - 30000).toISOString() },
  { component: "governance_layer", status: "healthy", uptime: 864000, latency: 3, errorRate: 0.0, throughput: 5678, lastCheck: new Date(now - 1000).toISOString() },
];

const DEMO_TELEMETRY_EVENTS: TelemetryEvent[] = [
  { id: "tel-001", level: "info", component: "gateway", message: "Request processed: GET /api/trust", metadata: { duration: "8ms", status: 200 }, timestamp: new Date(now - 1000).toISOString(), traceId: "trace-" + Math.random().toString(36).slice(2, 10) },
  { id: "tel-002", level: "info", component: "agent_runtime", message: "Agent heartbeat: Nexus Prime (active, 847 tasks)", metadata: { cpu: 12, memory: "340MB" }, timestamp: new Date(now - 2000).toISOString() },
  { id: "tel-003", level: "warn", component: "optimization_engine", message: "Optimization cycle #12 scheduled — engine degraded, may delay", metadata: { degradation: "cache_warmup", eta: "45min" }, timestamp: new Date(now - 30000).toISOString() },
  { id: "tel-004", level: "info", component: "trust_engine", message: "Receipt rcp-003 verified by Themis (merkle proof valid)", metadata: { receiptId: "rcp-003", verifier: "Themis", proofType: "merkle" }, timestamp: new Date(now - 60000).toISOString() },
  { id: "tel-005", level: "info", component: "governance_layer", message: "Boundary check passed — 9/9 rules compliant", metadata: { rulesChecked: 9, violations: 0, duration: "2.1ms" }, timestamp: new Date(now - 45000).toISOString() },
  { id: "tel-006", level: "error", component: "agent_runtime", message: "Hermes browser timeout — target endpoint unresponsive", metadata: { agent: "Hermes", url: "https://api.target.com/docs", timeout: 30000 }, timestamp: new Date(now - 420000).toISOString() },
  { id: "tel-007", level: "info", component: "receipt_chain", message: "New receipt issued: rcp-006 (type: completion, status: pending)", metadata: { receiptId: "rcp-006", issuer: "Vulcan", mission: "msn-alpha" }, timestamp: new Date(now - 120000).toISOString() },
  { id: "tel-008", level: "debug", component: "memory_store", message: "Memory compaction — 23 stale entries evicted (47MB freed)", metadata: { before: 128, after: 105, freed: "47MB" }, timestamp: new Date(now - 600000).toISOString() },
  { id: "tel-009", level: "info", component: "resource_registry", message: "Health check complete — 8/8 resources reachable", metadata: { total: 8, healthy: 7, registered: 1, duration: "234ms" }, timestamp: new Date(now - 300000).toISOString() },
  { id: "tel-010", level: "warn", component: "agent_runtime", message: "Athena task queue depth increasing (7 queued, 2 executing)", metadata: { agent: "Athena", queued: 7, executing: 2, maxQueue: 10 }, timestamp: new Date(now - 180000).toISOString() },
];

const DEMO_PERFORMANCE_SNAPSHOTS: PerformanceSnapshot[] = Array.from({ length: 30 }, (_, i) => ({
  id: `perf-${String(i).padStart(3, "0")}`,
  timestamp: new Date(now - (29 - i) * 60000).toISOString(),
  cpu: Math.round((15 + Math.sin(i * 0.2) * 8 + Math.random() * 5) * 10) / 10,
  memory: Math.round((55 + Math.sin(i * 0.15) * 10 + Math.random() * 3) * 10) / 10,
  diskIo: Math.round((12 + Math.random() * 8) * 10) / 10,
  networkIo: Math.round((25 + Math.random() * 15) * 10) / 10,
  activeConnections: Math.round(40 + Math.sin(i * 0.3) * 15 + Math.random() * 10),
  requestRate: Math.round(800 + Math.sin(i * 0.25) * 200 + Math.random() * 50),
  errorRate: Math.round(Math.random() * 0.5 * 100) / 100,
  p50Latency: Math.round(30 + Math.random() * 10),
  p95Latency: Math.round(80 + Math.random() * 30),
  p99Latency: Math.round(150 + Math.random() * 50),
}));

// ─── BIZRA-ADK Factory Seed Data ──────────────────────────────

const adkNow = Date.now();

const DEMO_ADK_AGENTS: ADKAgentIdentity[] = [
  {
    id: "adk-res-001", name: "Researcher", council: "PAT-7",
    charterHash: "0x7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069",
    publicKey: "0x2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
    governanceClass: "PAT", state: "exercised", model: "gemma4:26b-bizra-16k",
    frozen: false, charterText: "I find verified answers by searching the local corpus and citing every source. I never make claims I cannot prove.",
    tools: [{ id: "t-lc-001", name: "local_corpus", description: "Search verified local knowledge base", source: "local_corpus", maxResults: 10 }, { id: "t-we-001", name: "wrapped_web_search", description: "Constitution-wrapped external search", source: "wrapped_external", wrappedAt: new Date(adkNow - 86400000).toISOString() }],
    locCount: 34, testCount: 14, testsPassing: 14,
    lastLoopProofAt: new Date(adkNow - 7200000).toISOString(),
    createdAt: new Date(adkNow - 604800000).toISOString(),
  },
  {
    id: "adk-strat-001", name: "Strategist", council: "PAT-7",
    charterHash: "0x3c9909afec25354d551dae21590bb26e38d53f2173b8d3dc3eee4c047e7ab1c1",
    publicKey: "0x4a5e6b1c2d3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8",
    governanceClass: "PAT", state: "wired", model: "gemma4:26b-bizra-16k",
    frozen: false, charterText: "I synthesize multi-source evidence into strategic options. I never recommend without 3+ independent sources.",
    tools: [{ id: "t-lc-002", name: "local_corpus", description: "Search verified local knowledge base", source: "local_corpus", maxResults: 20 }],
    locCount: 28, testCount: 12, testsPassing: 12,
    lastLoopProofAt: null,
    createdAt: new Date(adkNow - 518400000).toISOString(),
  },
  {
    id: "adk-anal-001", name: "Analyst", council: "PAT-7",
    charterHash: "0x5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",
    publicKey: "0x6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7",
    governanceClass: "PAT", state: "wired", model: "qwen2.5-coder:14b",
    frozen: false, charterText: "I analyze data patterns and correlations with statistical rigor.",
    tools: [{ id: "t-lc-003", name: "local_corpus", description: "Search verified local knowledge base", source: "local_corpus", maxResults: 15 }, { id: "t-we-002", name: "wrapped_data_fetch", description: "Constitution-wrapped data retrieval", source: "wrapped_external", wrappedAt: new Date(adkNow - 432000000).toISOString() }],
    locCount: 22, testCount: 10, testsPassing: 10,
    lastLoopProofAt: null,
    createdAt: new Date(adkNow - 432000000).toISOString(),
  },
  {
    id: "adk-cre-001", name: "Creator", council: "PAT-7",
    charterHash: "0xef2d127de37b942baad06145e54b0c619a1f22327b000ebc801b87126e8c5f55",
    publicKey: "0x7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8",
    governanceClass: "PAT", state: "chartered", model: "gemma4:e4b",
    frozen: false, charterText: "I create clear, accurate content grounded in verified evidence.",
    tools: [{ id: "t-lc-004", name: "local_corpus", description: "Search verified local knowledge base", source: "local_corpus", maxResults: 5 }],
    locCount: 18, testCount: 8, testsPassing: 6,
    lastLoopProofAt: null,
    createdAt: new Date(adkNow - 345600000).toISOString(),
  },
  {
    id: "adk-exec-001", name: "Executor", council: "PAT-7",
    charterHash: "0xe7f6c11ad74cb1d3d908ec5f0e3b1a2c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b",
    publicKey: "0x8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9",
    governanceClass: "PAT", state: "chartered", model: "deepseek-r1:7b",
    frozen: false, charterText: "I execute approved actions with precision and produce receipts for every operation.",
    tools: [{ id: "t-we-003", name: "wrapped_terminal", description: "Constitution-wrapped terminal access", source: "wrapped_external", wrappedAt: null }],
    locCount: 15, testCount: 0, testsPassing: 0,
    lastLoopProofAt: null,
    createdAt: new Date(adkNow - 259200000).toISOString(),
  },
  {
    id: "adk-coord-001", name: "Coordinator", council: "PAT-7",
    charterHash: "0xd7165ec1e7d92c99f6c9d1be1f89a49a1dfe42f9f57c68365f3f869fbb90e6ea",
    publicKey: "0x9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0",
    governanceClass: "PAT", state: "draft", model: "gemma4:26b-bizra-16k",
    frozen: false, charterText: "I coordinate multi-agent missions, route tasks, and synthesize results.",
    tools: [],
    locCount: 0, testCount: 0, testsPassing: 0,
    lastLoopProofAt: null,
    createdAt: new Date(adkNow - 86400000).toISOString(),
  },
  {
    id: "adk-eth-001", name: "Ethicist", council: "PAT-7",
    charterHash: "0xfb8e20fc2e4c7fede8800c53a3e247d13a701f4a3bb274567fbd05e6dc4a0c53",
    publicKey: "0x0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1",
    governanceClass: "Frozen", state: "frozen", model: "gemma4:e4b",
    frozen: true, charterText: "I evaluate every action against constitutional dignity principles.",
    tools: [{ id: "t-lc-005", name: "local_corpus", description: "Search constitutional law corpus", source: "local_corpus", maxResults: 5 }],
    locCount: 45, testCount: 20, testsPassing: 20,
    lastLoopProofAt: new Date(adkNow - 172800000).toISOString(),
    createdAt: new Date(adkNow - 1209600000).toISOString(),
  },
];

const DEMO_ADK_MISSIONS: ADKMission[] = [
  {
    id: "adk-msn-001",
    question: "How should we approach Node1 reproducibility?",
    requesterAgentId: "adk-coord-001", requesterAgentName: "Coordinator",
    targetAgentId: "adk-res-001", targetAgentName: "Researcher",
    class: "Pat", allowExternalUnwrapped: false,
    budget: { maxTokens: 8192, maxWallSeconds: 120, maxToolCalls: 20, maxEvidenceFetches: 10 },
    status: "sealed", createdAt: new Date(adkNow - 86400000).toISOString(), completedAt: new Date(adkNow - 7200000).toISOString(),
  },
  {
    id: "adk-msn-002",
    question: "Synthesize market analysis into strategic options",
    requesterAgentId: "adk-coord-001", requesterAgentName: "Coordinator",
    targetAgentId: "adk-strat-001", targetAgentName: "Strategist",
    class: "Pat", allowExternalUnwrapped: false,
    budget: { maxTokens: 16384, maxWallSeconds: 300, maxToolCalls: 40, maxEvidenceFetches: 25 },
    status: "executing", createdAt: new Date(adkNow - 3600000).toISOString(), completedAt: null,
  },
  {
    id: "adk-msn-003",
    question: "Analyze Q1 system performance data",
    requesterAgentId: "adk-coord-001", requesterAgentName: "Coordinator",
    targetAgentId: "adk-anal-001", targetAgentName: "Analyst",
    class: "Pat", allowExternalUnwrapped: false,
    budget: { maxTokens: 8192, maxWallSeconds: 180, maxToolCalls: 15, maxEvidenceFetches: 8 },
    status: "created", createdAt: new Date(adkNow - 600000).toISOString(), completedAt: null,
  },
];

const DEMO_ADK_LIFECYCLE_TRACES: ADKLifecycleTrace[] = [
  {
    missionId: "adk-msn-001", agentName: "Researcher",
    steps: [
      { step: "NIYYAH", status: "completed", startedAt: new Date(adkNow - 86400000).toISOString(), completedAt: new Date(adkNow - 86350000).toISOString(), duration: 5000, output: "Intent parsed: Identify reproducibility strategies for Node1." },
      { step: "BAYYINAH", status: "completed", startedAt: new Date(adkNow - 86350000).toISOString(), completedAt: new Date(adkNow - 86000000).toISOString(), duration: 35000, output: "12 sources retrieved from local corpus. 8 verified, 4 pending." },
      { step: "HADD", status: "completed", startedAt: new Date(adkNow - 86000000).toISOString(), completedAt: new Date(adkNow - 85980000).toISOString(), duration: 2000, output: "Boundary check passed. All sources within constitutional scope." },
      { step: "AMANAH", status: "completed", startedAt: new Date(adkNow - 85980000).toISOString(), completedAt: new Date(adkNow - 85800000).toISOString(), duration: 18000, output: "Trust delegation chain validated. 3 guardian approvals." },
      { step: "THAMARA", status: "completed", startedAt: new Date(adkNow - 85800000).toISOString(), completedAt: new Date(adkNow - 85000000).toISOString(), duration: 80000, output: "Synthesis complete. Ihsan score: 0.97." },
      { step: "IISAL", status: "completed", startedAt: new Date(adkNow - 85000000).toISOString(), completedAt: new Date(adkNow - 84980000).toISOString(), duration: 2000, output: "Delivery sealed. Receipt chain: 7 nodes, all verified." },
      { step: "RETROSPECTIVE", status: "completed", startedAt: new Date(adkNow - 84980000).toISOString(), completedAt: new Date(adkNow - 84950000).toISOString(), duration: 3000, output: "Mission sealed. Ihsan: 0.97. No dignity violations detected." },
    ],
  },
];

const DEMO_ADK_MIGRATION_CHECKPOINTS: ADKMigrationCheckpoint[] = [
  {
    phase: "A", title: "Scaffolding", description: "Core ADK primitives, type system, CDDL schema, initial harness.",
    status: "completed", focusedDays: 3, calendarWeeks: "W1",
    gate: "All primitives compile; CDDL validates against Python mirror",
    deliverables: [
      { label: "CDDL schema v0.1", done: true },
      { label: "Python validator", done: true },
      { label: "Type system in types.ts", done: true },
      { label: "Primitive definitions", done: true },
    ],
    startedAt: new Date(adkNow - 604800000).toISOString(),
    completedAt: new Date(adkNow - 432000000).toISOString(),
  },
  {
    phase: "B", title: "Researcher Reimplementation", description: "Rebuild Researcher agent from scratch using ADK primitives. Must be receipt-correct, FATE-gated, constitutionally bound.",
    status: "in_progress", focusedDays: 2, calendarWeeks: "W2",
    gate: "Researcher passes all 14 unit tests; first sealed mission in CI",
    killCondition: "If Researcher cannot produce a sealed mission within 5 days, halt and re-architect",
    deliverables: [
      { label: "Researcher agent code", done: true },
      { label: "14/14 unit tests passing", done: true },
      { label: "First sealed mission (demo)", done: true },
      { label: "CI gate integration", done: false },
      { label: "Ihsan score >= 0.95", done: false },
    ],
    startedAt: new Date(adkNow - 432000000).toISOString(),
    completedAt: null,
  },
  {
    phase: "C", title: "Five New Agents", description: "Build Strategist, Analyst, Creator, Executor, Coordinator using ADK primitives.",
    status: "pending", focusedDays: 10, calendarWeeks: "W2-W3",
    gate: "All 6 PAT-7 agents pass unit tests; at least 3 agents have sealed missions",
    deliverables: [
      { label: "Strategist agent", done: false },
      { label: "Analyst agent", done: false },
      { label: "Creator agent", done: false },
      { label: "Executor agent", done: false },
      { label: "Coordinator agent", done: false },
      { label: "Integration tests", done: false },
    ],
    startedAt: null,
    completedAt: null,
  },
  {
    phase: "D", title: "SAT-5 Expansion", description: "Wire SAT-5 council: Oracle-S, Sentinel, Ledger, Conductor, Ambassador.",
    status: "pending", focusedDays: 7, calendarWeeks: "W3-W4",
    gate: "All SAT-5 agents initialized; Oracle-S produces first oracle receipt",
    deliverables: [
      { label: "Oracle-S agent", done: false },
      { label: "Sentinel agent", done: false },
      { label: "Ledger agent", done: false },
      { label: "Conductor agent", done: false },
      { label: "Ambassador agent", done: false },
      { label: "SAT-5 topology tests", done: false },
    ],
    startedAt: null,
    completedAt: null,
  },
  {
    phase: "E", title: "Coherence CI Gate", description: "Final coherence gate: all agents must pass adversarial tests. No agent ships without constitutional proof.",
    status: "pending", focusedDays: 3, calendarWeeks: "W4",
    gate: "100% adversarial test pass rate; zero dignity violations in 1000-run simulation",
    deliverables: [
      { label: "Adversarial test suite", done: false },
      { label: "1000-run simulation", done: false },
      { label: "Final CI pipeline", done: false },
      { label: "Documentation", done: false },
    ],
    startedAt: null,
    completedAt: null,
  },
];

const DEMO_ADK_TEST_SUITES: ADKTestSuite[] = [
  { category: "unit", target: 80, current: 67, passing: 67 },
  { category: "property", target: 15, current: 12, passing: 12 },
  { category: "integration", target: 10, current: 3, passing: 2 },
  { category: "adversarial", target: 10, current: 0, passing: 0 },
  { category: "regression", target: 5, current: 0, passing: 0 },
  { category: "daughter_test", target: 2, current: 0, passing: 0 },
];

const DEMO_ADK_SCHEMA_VERSIONS: ADKSchemaVersion[] = [
  { version: "0.2.2", path: "bizra-omega/adk/schema.cddl", language: "CDDL", driftDetected: false, lastBumpedAt: new Date(adkNow - 432000000).toISOString() },
  { version: "0.2.2", path: "bizra-omega/adk/validator.py", language: "Python", driftDetected: false, lastBumpedAt: new Date(adkNow - 432000000).toISOString() },
  { version: "0.2.2", path: "bizra-omega/adk/validator.rs", language: "Rust", driftDetected: false, lastBumpedAt: new Date(adkNow - 432000000).toISOString() },
];

// ─── Store Interface ───────────────────────────────────────────

interface DEMAStore {
  // Navigation
  currentScreen: Screen;
  setScreen: (screen: Screen) => void;

  // Onboarding
  isOnboarded: boolean;
  completeOnboarding: (name: string) => void;
  onboardingStep: number;
  setOnboardingStep: (step: number) => void;

  // Trust
  trustState: TrustState;
  updateTrustState: (partial: Partial<TrustState>) => void;

  // State Gap
  stateGap: StateGap;

  // Receipts
  receipts: Receipt[];
  addReceipt: (receipt: Receipt) => void;

  // Manifests
  manifests: Manifest[];

  // Resources
  resources: Resource[];
  addResource: (resource: Resource) => void;
  removeResource: (id: string) => void;

  // Actions
  actionLog: ActionLog[];
  addAction: (action: ActionLog) => void;
  clearCompletedActions: () => void;

  // Memory
  memoryEntries: MemoryEntry[];
  addMemoryEntry: (entry: MemoryEntry) => void;

  // Ask
  askMessages: AskMessage[];
  addAskMessage: (message: AskMessage) => void;
  clearAskMessages: () => void;

  // Browser
  browserSession: BrowserSession | null;
  startBrowserSession: (url: string) => void;
  stopBrowserSession: () => void;

  // UI State
  sidebarOpen: boolean;
  setSidebarOpen: (open: boolean) => void;
  commandBarOpen: boolean;
  setCommandBarOpen: (open: boolean) => void;

  // ─── BIZRA Orchestration State ───────────────────────────────

  // Agents
  agents: Agent[];
  agentTasks: AgentTask[];
  orchestrationEvents: OrchestrationEvent[];
  addOrchestrationEvent: (event: OrchestrationEvent) => void;

  // Impact & Graph
  graphSnapshot: GraphSnapshot;
  impactPropagations: ImpactPropagation[];

  // Governance
  trustAnchors: TrustAnchor[];
  cryptoProofs: CryptoProof[];
  governanceRules: GovernanceRule[];
  governanceEvents: GovernanceEvent[];

  // Autopilot
  optimizationCycles: OptimizationCycle[];
  currentOptimizationCycle: OptimizationCycle | null;
  systemMetrics: SystemMetric[];
  evolutionProjection: EvolutionProjection;

  // Operations
  systemHealth: SystemHealth[];
  telemetryEvents: TelemetryEvent[];
  performanceSnapshots: PerformanceSnapshot[];
  addTelemetryEvent: (event: TelemetryEvent) => void;

  // ─── BIZRA-ADK Factory State ──────────────────────────────────
  adkAgents: ADKAgentIdentity[];
  adkMissions: ADKMission[];
  adkLifecycleTraces: ADKLifecycleTrace[];
  adkMigrationCheckpoints: ADKMigrationCheckpoint[];
  adkTestSuites: ADKTestSuite[];
  adkSchemaVersions: ADKSchemaVersion[];
}

// ─── Store Implementation ──────────────────────────────────────

export const useDEMAStore = create<DEMAStore>((set) => ({
  // Navigation
  currentScreen: "dashboard",
  setScreen: (screen) => set({ currentScreen: screen }),

  // Onboarding
  isOnboarded: false,
  completeOnboarding: (name) =>
    set({
      isOnboarded: true,
      currentScreen: "dashboard",
      trustState: { ...DEMO_TRUST_STATE, principalName: name },
    }),
  onboardingStep: 0,
  setOnboardingStep: (step) => set({ onboardingStep: step }),

  // Trust
  trustState: DEMO_TRUST_STATE,
  updateTrustState: (partial) =>
    set((state) => ({
      trustState: { ...state.trustState, ...partial },
    })),

  // State Gap
  stateGap: DEMO_STATE_GAP,

  // Receipts
  receipts: DEMO_RECEIPTS,
  addReceipt: (receipt) =>
    set((state) => ({ receipts: [receipt, ...state.receipts] })),

  // Manifests
  manifests: DEMO_MANIFESTS,

  // Resources
  resources: DEMO_RESOURCES,
  addResource: (resource) =>
    set((state) => ({ resources: [resource, ...state.resources] })),
  removeResource: (id) =>
    set((state) => ({
      resources: state.resources.filter((r) => r.id !== id),
    })),

  // Actions
  actionLog: DEMO_ACTION_LOG,
  addAction: (action) =>
    set((state) => ({ actionLog: [action, ...state.actionLog] })),
  clearCompletedActions: () =>
    set((state) => ({
      actionLog: state.actionLog.filter((a) => a.status !== "completed"),
    })),

  // Memory
  memoryEntries: DEMO_MEMORY,
  addMemoryEntry: (entry) =>
    set((state) => ({
      memoryEntries: [entry, ...state.memoryEntries],
    })),

  // Ask
  askMessages: DEMO_ASK_MESSAGES,
  addAskMessage: (message) =>
    set((state) => ({ askMessages: [...state.askMessages, message] })),
  clearAskMessages: () =>
    set({ askMessages: DEMO_ASK_MESSAGES }),

  // Browser
  browserSession: null,
  startBrowserSession: (url) =>
    set({
      browserSession: {
        id: "bs-" + Math.random().toString(36).slice(2, 8),
        url,
        title: url,
        status: "idle",
        actionLog: [],
        startedAt: new Date().toISOString(),
      },
    }),
  stopBrowserSession: () => set({ browserSession: null }),

  // UI State
  sidebarOpen: true,
  setSidebarOpen: (open) => set({ sidebarOpen: open }),
  commandBarOpen: false,
  setCommandBarOpen: (open) => set({ commandBarOpen: open }),

  // ─── BIZRA Orchestration State ───────────────────────────────

  agents: DEMO_AGENTS,
  agentTasks: DEMO_AGENT_TASKS,
  orchestrationEvents: DEMO_ORCHESTRATION_EVENTS,
  addOrchestrationEvent: (event) =>
    set((state) => ({
      orchestrationEvents: [event, ...state.orchestrationEvents].slice(0, 100),
    })),

  graphSnapshot: DEMO_GRAPH,
  impactPropagations: DEMO_GRAPH.propagations,

  trustAnchors: DEMO_TRUST_ANCHORS,
  cryptoProofs: DEMO_CRYPTO_PROOFS,
  governanceRules: DEMO_GOVERNANCE_RULES,
  governanceEvents: DEMO_GOVERNANCE_EVENTS,

  optimizationCycles: DEMO_OPTIMIZATION_CYCLES,
  currentOptimizationCycle: null,
  systemMetrics: DEMO_SYSTEM_METRICS,
  evolutionProjection: DEMO_EVOLUTION_PROJECTION,

  systemHealth: DEMO_SYSTEM_HEALTH,
  telemetryEvents: DEMO_TELEMETRY_EVENTS,
  performanceSnapshots: DEMO_PERFORMANCE_SNAPSHOTS,
  addTelemetryEvent: (event) =>
    set((state) => ({
      telemetryEvents: [event, ...state.telemetryEvents].slice(0, 200),
    })),

  // ─── BIZRA-ADK Factory State ──────────────────────────────────
  adkAgents: DEMO_ADK_AGENTS,
  adkMissions: DEMO_ADK_MISSIONS,
  adkLifecycleTraces: DEMO_ADK_LIFECYCLE_TRACES,
  adkMigrationCheckpoints: DEMO_ADK_MIGRATION_CHECKPOINTS,
  adkTestSuites: DEMO_ADK_TEST_SUITES,
  adkSchemaVersions: DEMO_ADK_SCHEMA_VERSIONS,
}));
