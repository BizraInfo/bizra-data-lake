import { useState } from "react";

const PHASES_COMPLETED = [
  {
    id: "p46",
    name: "Phase 46",
    commit: "366774c",
    title: "Cognitive Resonance — Core Engines",
    tests: 170,
    date: "Today",
    modules: [
      { name: "VectorSearchEngine", domain: "Memory & Persistence", status: "shipped", desc: "FAISS-backed 102K vector index, semantic search with SNR scoring" },
      { name: "GoTBridge", domain: "Intelligence Layer", status: "shipped", desc: "Graph-of-Thoughts reasoning bridge with convergence verification" },
      { name: "HMMEngine", domain: "Intelligence Layer", status: "shipped", desc: "Hidden Markov Model for cognitive state prediction" },
      { name: "CognitiveResonance", domain: "Intelligence Layer", status: "shipped", desc: "Full pipeline orchestrator: search→reason→predict with combined_snr" },
      { name: "Apex Integration", domain: "Intelligence Layer", status: "shipped", desc: "GoT bridge wired into apex_engine.py" },
      { name: "Proactive Integration", domain: "Intelligence Layer", status: "shipped", desc: "HMM hooks wired into proactive.py" },
    ]
  },
  {
    id: "p46.1",
    name: "Phase 46.1",
    commit: "477e9d9",
    title: "MCP Integration — Cognitive Resonance Exposed",
    tests: 15,
    date: "Today",
    modules: [
      { name: "sovereign_search tool", domain: "Integration Protocols", status: "shipped", desc: "FAISS search accessible via MCP, cacheable" },
      { name: "sovereign_resonance tool", domain: "Integration Protocols", status: "shipped", desc: "Full pipeline via MCP, returns combined_snr" },
      { name: "sovereign_predict tool", domain: "Integration Protocols", status: "shipped", desc: "HMM state tracking via MCP" },
      { name: "Phase46Interface", domain: "Integration Protocols", status: "shipped", desc: "Lazy singleton, graceful degradation, 3/3 components" },
      { name: "mcp_health extended", domain: "Operations & DevOps", status: "shipped", desc: "Reports Phase 46 component availability" },
      { name: "Server v1.3.0", domain: "Integration Protocols", status: "shipped", desc: "10 total MCP tools on sovereign surface" },
    ]
  },
  {
    id: "p47.1",
    name: "Phase 47.1",
    commit: "—",
    title: "Safe Activation — Canary Rollout",
    tests: 0,
    date: "Planned",
    modules: [
      { name: "Release isolation", domain: "Operations & DevOps", status: "planned", desc: "Cherry-pick branch with semantic integrity checks" },
      { name: "Canary routing", domain: "Operations & DevOps", status: "planned", desc: "Deterministic hash-based percentage gates per component" },
      { name: "Rollback automation", domain: "Operations & DevOps", status: "planned", desc: "Strict 2-window breach → auto rollback with receipt" },
      { name: "Observability surface", domain: "Operations & DevOps", status: "planned", desc: "9 metrics including combined_snr, observation_entropy" },
      { name: "HMM caller isolation", domain: "Operations & DevOps", status: "planned", desc: "Single-caller mode for staging purity" },
      { name: "Failure baseline", domain: "Operations & DevOps", status: "planned", desc: "Frozen 28-failure snapshot for regression comparison" },
    ]
  }
];

const DOMAIN_HEALTH = [
  {
    domain: "Philosophy & Vision",
    before: 95, after: 95, delta: 0,
    color: "#10B981",
    components: ["Constitutional Framework ✓", "Third Fact Whitepaper ✓", "إحسان Standard ✓", "Interest-Debt Theorem ✓"],
    note: "Unchanged. Strongest domain. Foundation is rock solid."
  },
  {
    domain: "Architecture Design",
    before: 80, after: 80, delta: 0,
    color: "#3B82F6",
    components: ["7-Layer DDAGI OS", "Dual Agentic (PAT/SAT)", "Dual Token (SEED/BLOOM)", "HyperBlockTree/BlockGraph"],
    note: "Unchanged. Design is complete. Implementation catching up."
  },
  {
    domain: "Intelligence Layer",
    before: 40, after: 62, delta: 22,
    color: "#8B5CF6",
    components: ["VectorSearchEngine ✓ NEW", "GoTBridge ✓ NEW", "HMMEngine ✓ NEW", "CognitiveResonance ✓ NEW", "MOE+HRM (designed)", "إحسان Scorer (partial)"],
    note: "Biggest jump today. 4 production engines shipped. First real inference capability."
  },
  {
    domain: "Memory & Persistence",
    before: 25, after: 38, delta: 13,
    color: "#F59E0B",
    components: ["persistence.log_conversation() (partial)", "VectorSearchEngine ✓ NEW (Layer 2 retrieval)", "Episodic logging (partial)", "Semantic memory (GAP)", "Memory synthesis pipeline (GAP)", "Context budget manager (GAP)"],
    note: "Search engine is Layer 2 retrieval memory. But Layer 3 (semantic) and synthesis pipeline still missing."
  },
  {
    domain: "Integration Protocols",
    before: 10, after: 35, delta: 25,
    color: "#EC4899",
    components: ["Sovereign MCP Server v1.3.0 ✓ (10 tools)", "sovereign_search ✓ NEW", "sovereign_resonance ✓ NEW", "sovereign_predict ✓ NEW", "AHK Bridge (partial)", "A2A Protocol (GAP)", "REST API (GAP)", "MCP Client (GAP)"],
    note: "MCP SERVER exists and works (10 tools). MCP CLIENT still missing — Node0 can be called but can't call out."
  },
  {
    domain: "Operations & DevOps",
    before: 15, after: 22, delta: 7,
    color: "#F97316",
    components: ["mcp_health ✓ (extended)", "CI test suites (210 Phase 46 tests)", "Canary framework (Phase 47.1 planned)", "Monitoring (GAP)", "Error recovery (GAP)", "CD pipeline (GAP)"],
    note: "Phase 47.1 will be the first real ops infrastructure. Currently just test suites and health endpoints."
  },
  {
    domain: "Security & Trust",
    before: 30, after: 30, delta: 0,
    color: "#EF4444",
    components: ["Ed25519 (designed)", "FATE Engine (designed)", "TMP (designed)", "Sovereign Data Vault (GAP)"],
    note: "No change today. Critical for Node0 MVP Phase 4 (sovereignty)."
  },
  {
    domain: "UX & Application",
    before: 20, after: 20, delta: 0,
    color: "#6366F1",
    components: ["Sacred Geometry UI (partial)", "Onboarding Flow (GAP)", "Memory Dashboard (GAP)", "Continuity UI (GAP)"],
    note: "Untouched. Phase 3 of MVP roadmap. User-facing experience undefined."
  },
  {
    domain: "Business / GTM",
    before: 35, after: 35, delta: 0,
    color: "#14B8A6",
    components: ["Third Fact Whitepaper ✓", "Investor Materials (partial)", "Pitch Deck (partial)", "Team Plan (GAP)"],
    note: "Unchanged. Solid foundation docs but no active fundraising infrastructure."
  },
  {
    domain: "Network / Federation",
    before: 5, after: 5, delta: 0,
    color: "#64748B",
    components: ["URP (designed)", "Proof-of-Impact (designed)", "Node Discovery (GAP)", "Federation Protocol (GAP)"],
    note: "Phase 2+. By design, not needed for Node0."
  },
];

const MVP_ROADMAP = [
  {
    phase: 0, name: "Foundation", weeks: "1-3", status: "not_started",
    items: [
      { name: "Hook System + Event Bus", critical: true, status: "gap", progress: 0, note: "THE foundation. Nothing connects without this." },
      { name: "Local LLM Runtime", critical: true, status: "gap", progress: 0, note: "LM Studio / Ollama integration for sovereign inference." },
    ]
  },
  {
    phase: 1, name: "Memory", weeks: "3-7", status: "partial",
    items: [
      { name: "Episodic Memory + Vector Store", critical: true, status: "partial", progress: 55, note: "VectorSearchEngine (Phase 46) IS this component. Needs conversation chunking + indexing pipeline." },
      { name: "Memory Synthesis Pipeline", critical: true, status: "gap", progress: 0, note: "THE critical gap. Transforms storage into memory. Extract→detect→resolve→update." },
      { name: "Context Budget Manager", critical: true, status: "gap", progress: 0, note: "Allocates 200K tokens optimally. Controls quality of every response." },
    ]
  },
  {
    phase: 2, name: "Agent", weeks: "7-10", status: "partial",
    items: [
      { name: "PAT Core (3 of 7 agents)", critical: true, status: "partial", progress: 25, note: "CognitiveResonance is proto-orchestration. Need Memory Agent, Task Agent, Desktop Agent." },
      { name: "AHK Desktop Bridge (Hardened)", critical: true, status: "partial", progress: 40, note: "Exists but needs hardening, verification loop, rollback on failure." },
      { name: "MCP Client", critical: false, status: "gap", progress: 0, note: "Node0 can be called (server works). Can't call out yet (no client)." },
    ]
  },
  {
    phase: 3, name: "Experience", weeks: "10-12", status: "not_started",
    items: [
      { name: "Onboarding Flow", critical: true, status: "gap", progress: 0, note: "First 5 minutes. The 'wow this knows me' moment." },
      { name: "Conversation Continuity UI", critical: true, status: "gap", progress: 0, note: "'I remember we discussed X last Tuesday' — this sells." },
      { name: "إحسان Quality Dashboard", critical: false, status: "gap", progress: 0, note: "Transparency builds trust." },
    ]
  },
  {
    phase: 4, name: "Sovereignty", weeks: "12-14", status: "not_started",
    items: [
      { name: "Sovereign Data Vault", critical: true, status: "gap", progress: 0, note: "Ed25519 encryption at rest. Your data never leaves without consent." },
      { name: "Memory Controls UI", critical: true, status: "gap", progress: 0, note: "See, edit, delete, export everything the agent knows about you." },
    ]
  },
];

const STRATEGIC_INSIGHTS = [
  {
    title: "What Just Changed",
    color: "#10B981",
    icon: "⬆",
    points: [
      "BIZRA now has its first production inference capability — real engines, not designs",
      "The MCP surface grew from 7 tools to 10 — external agents can access cognitive resonance",
      "210 tests provide the first real safety net for production deployment",
      "Phase 47.1 introduces BIZRA's first deployment discipline (canary + rollback)",
      "combined_snr gives you a single number measuring intelligence quality — your إحسان for AI"
    ]
  },
  {
    title: "What Hasn't Changed",
    color: "#F59E0B",
    icon: "→",
    points: [
      "Hook System still doesn't exist — the nervous system that connects everything",
      "Memory Synthesis Pipeline is still the #1 gap — storage ≠ memory",
      "No onboarding flow — Alpha-100 has no front door",
      "No MCP client — Node0 can be called but can't call external tools",
      "No local LLM runtime — inference depends on cloud APIs",
      "UX layer is untouched — no human-facing experience beyond CLI"
    ]
  },
  {
    title: "The Honest Assessment",
    color: "#8B5CF6",
    icon: "◇",
    points: [
      "Phase 46 built powerful engines. Phase 46.1 gave them a doorway. Phase 47.1 will prove they're safe.",
      "But engines without the Hook System are like F1 engines without a chassis — power with no vehicle.",
      "The gap between 'intelligence components exist' and 'agent works for a human' is still the entire MVP roadmap.",
      "Today's work moved Intelligence Layer from 40%→62% and Integration from 10%→35%. Meaningful.",
      "But Memory & Persistence (the soul) is still 38%. That's the Make-or-Break domain for Node0.",
    ]
  }
];

function ProgressBar({ value, max = 100, color = "#3B82F6", height = 6 }) {
  return (
    <div style={{ background: "rgba(255,255,255,0.08)", borderRadius: height/2, height, width: "100%", overflow: "hidden" }}>
      <div style={{ background: color, height: "100%", width: `${(value/max)*100}%`, borderRadius: height/2, transition: "width 0.4s ease" }} />
    </div>
  );
}

export default function BIZRAStatus() {
  const [activeTab, setActiveTab] = useState("overview");
  const [expandedDomain, setExpandedDomain] = useState(null);
  const [expandedPhase, setExpandedPhase] = useState(null);

  const totalTests = PHASES_COMPLETED.reduce((s, p) => s + p.tests, 0);
  const totalModules = PHASES_COMPLETED.reduce((s, p) => s + p.modules.length, 0);
  const shippedModules = PHASES_COMPLETED.reduce((s, p) => s + p.modules.filter(m => m.status === "shipped").length, 0);

  const overallBefore = Math.round(DOMAIN_HEALTH.reduce((s, d) => s + d.before, 0) / DOMAIN_HEALTH.length);
  const overallAfter = Math.round(DOMAIN_HEALTH.reduce((s, d) => s + d.after, 0) / DOMAIN_HEALTH.length);

  const mvpTotal = MVP_ROADMAP.reduce((s, p) => s + p.items.length, 0);
  const mvpGaps = MVP_ROADMAP.reduce((s, p) => s + p.items.filter(i => i.status === "gap").length, 0);
  const mvpPartial = MVP_ROADMAP.reduce((s, p) => s + p.items.filter(i => i.status === "partial").length, 0);

  return (
    <div style={{
      fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
      background: "#07070C",
      color: "#E2E8F0",
      minHeight: "100vh",
      padding: "16px",
      boxSizing: "border-box"
    }}>
      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: 12 }}>
        <div style={{ fontSize: 9, letterSpacing: 6, color: "#475569", textTransform: "uppercase" }}>After Phase 46 + 46.1 + 47.1 Plan</div>
        <h1 style={{ fontSize: 22, fontWeight: 800, margin: "4px 0", background: "linear-gradient(135deg, #10B981, #8B5CF6, #D4A574)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
          Where BIZRA Stands
        </h1>
      </div>

      {/* Key Metrics */}
      <div style={{ display: "flex", gap: 8, justifyContent: "center", marginBottom: 14, flexWrap: "wrap" }}>
        {[
          { label: "Tests Passing", value: totalTests, color: "#10B981" },
          { label: "Modules Shipped", value: `${shippedModules}/${totalModules}`, color: "#3B82F6" },
          { label: "Commits", value: "2 today", color: "#8B5CF6" },
          { label: "Overall Health", value: `${overallBefore}%→${overallAfter}%`, color: "#F59E0B" },
          { label: "MVP Gaps", value: `${mvpGaps}/${mvpTotal}`, color: "#EF4444" },
        ].map((m, i) => (
          <div key={i} style={{
            background: `${m.color}0A`, border: `1px solid ${m.color}25`,
            borderRadius: 8, padding: "8px 16px", textAlign: "center", minWidth: 100
          }}>
            <div style={{ fontSize: 18, fontWeight: 800, color: m.color }}>{m.value}</div>
            <div style={{ fontSize: 9, color: "#64748B", marginTop: 2 }}>{m.label}</div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: 2, justifyContent: "center", marginBottom: 14, flexWrap: "wrap" }}>
        {[
          { id: "overview", label: "◉ What Shipped" },
          { id: "health", label: "⬡ Domain Health" },
          { id: "mvp", label: "▸ MVP Roadmap" },
          { id: "strategic", label: "◇ Strategic Read" },
        ].map(t => (
          <button key={t.id} onClick={() => setActiveTab(t.id)} style={{
            background: activeTab === t.id ? "rgba(139,92,246,0.15)" : "rgba(255,255,255,0.03)",
            border: `1px solid ${activeTab === t.id ? "rgba(139,92,246,0.4)" : "rgba(255,255,255,0.06)"}`,
            color: activeTab === t.id ? "#C4B5FD" : "#64748B",
            padding: "6px 16px", borderRadius: 6, cursor: "pointer", fontSize: 11, fontFamily: "inherit"
          }}>{t.label}</button>
        ))}
      </div>

      {/* What Shipped */}
      {activeTab === "overview" && (
        <div style={{ maxWidth: 800, margin: "0 auto" }}>
          {PHASES_COMPLETED.map(phase => (
            <div key={phase.id} style={{
              marginBottom: 12, borderRadius: 10, overflow: "hidden",
              border: `1px solid ${phase.id === "p47.1" ? "rgba(249,115,22,0.25)" : "rgba(16,185,129,0.25)"}`,
              background: phase.id === "p47.1" ? "rgba(249,115,22,0.04)" : "rgba(16,185,129,0.04)"
            }}>
              <div style={{ padding: "12px 14px", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
                <div>
                  <span style={{ fontSize: 14, fontWeight: 700, color: phase.id === "p47.1" ? "#F97316" : "#10B981" }}>{phase.name}</span>
                  <span style={{ fontSize: 12, color: "#94A3B8", marginLeft: 8 }}>{phase.title}</span>
                </div>
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  {phase.commit !== "—" && (
                    <span style={{ fontSize: 10, fontFamily: "inherit", color: "#64748B", background: "rgba(255,255,255,0.05)", padding: "2px 8px", borderRadius: 4 }}>{phase.commit}</span>
                  )}
                  <span style={{
                    fontSize: 9, padding: "2px 8px", borderRadius: 4, fontWeight: 700, letterSpacing: 1,
                    background: phase.id === "p47.1" ? "rgba(249,115,22,0.15)" : "rgba(16,185,129,0.15)",
                    color: phase.id === "p47.1" ? "#FB923C" : "#34D399",
                    border: `1px solid ${phase.id === "p47.1" ? "rgba(249,115,22,0.3)" : "rgba(16,185,129,0.3)"}`
                  }}>{phase.id === "p47.1" ? "PLANNED" : `${phase.tests} TESTS`}</span>
                </div>
              </div>
              <div style={{ padding: "0 14px 12px" }}>
                {phase.modules.map((m, i) => (
                  <div key={i} style={{
                    display: "flex", alignItems: "center", gap: 8, padding: "4px 0",
                    borderTop: i > 0 ? "1px solid rgba(255,255,255,0.04)" : "none"
                  }}>
                    <span style={{
                      fontSize: 8, width: 6, height: 6, borderRadius: "50%", flexShrink: 0,
                      background: m.status === "shipped" ? "#10B981" : m.status === "planned" ? "#F97316" : "#64748B"
                    }} />
                    <span style={{ fontSize: 11, color: "#E2E8F0", fontWeight: 600, minWidth: 180 }}>{m.name}</span>
                    <span style={{ fontSize: 10, color: "#64748B", flex: 1 }}>{m.desc}</span>
                    <span style={{
                      fontSize: 8, padding: "1px 6px", borderRadius: 3, flexShrink: 0,
                      background: "rgba(255,255,255,0.05)", color: "#94A3B8"
                    }}>{m.domain}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Domain Health */}
      {activeTab === "health" && (
        <div style={{ maxWidth: 800, margin: "0 auto" }}>
          {DOMAIN_HEALTH.map((d, i) => (
            <div key={i}
              onClick={() => setExpandedDomain(expandedDomain === i ? null : i)}
              style={{
                marginBottom: 6, borderRadius: 8, cursor: "pointer",
                background: expandedDomain === i ? `${d.color}0A` : "rgba(255,255,255,0.02)",
                border: `1px solid ${expandedDomain === i ? `${d.color}30` : "rgba(255,255,255,0.05)"}`,
                padding: "10px 14px", transition: "all 0.15s"
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
                <span style={{ fontSize: 12, fontWeight: 700, color: d.color, flex: 1 }}>{d.domain}</span>
                <span style={{ fontSize: 11, color: "#64748B" }}>{d.before}%</span>
                <span style={{ fontSize: 10, color: "#475569" }}>→</span>
                <span style={{ fontSize: 11, color: d.delta > 0 ? "#10B981" : "#94A3B8", fontWeight: 700 }}>{d.after}%</span>
                {d.delta > 0 && <span style={{ fontSize: 9, color: "#10B981", background: "rgba(16,185,129,0.12)", padding: "1px 5px", borderRadius: 3 }}>+{d.delta}</span>}
              </div>
              <ProgressBar value={d.after} color={d.color} />
              {expandedDomain === i && (
                <div style={{ marginTop: 10 }}>
                  <div style={{ fontSize: 10, color: "#94A3B8", lineHeight: 1.6, marginBottom: 6, fontStyle: "italic" }}>{d.note}</div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                    {d.components.map((c, j) => (
                      <span key={j} style={{
                        fontSize: 9, padding: "2px 8px", borderRadius: 4,
                        background: c.includes("✓") ? "rgba(16,185,129,0.1)" : c.includes("GAP") ? "rgba(239,68,68,0.1)" : "rgba(255,255,255,0.04)",
                        color: c.includes("✓") ? "#34D399" : c.includes("GAP") ? "#FCA5A5" : "#94A3B8",
                        border: `1px solid ${c.includes("✓") ? "rgba(16,185,129,0.2)" : c.includes("GAP") ? "rgba(239,68,68,0.2)" : "rgba(255,255,255,0.06)"}`
                      }}>{c}</span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* MVP Roadmap */}
      {activeTab === "mvp" && (
        <div style={{ maxWidth: 800, margin: "0 auto" }}>
          {MVP_ROADMAP.map((phase, pi) => (
            <div key={pi} style={{ marginBottom: 12 }}>
              <div
                onClick={() => setExpandedPhase(expandedPhase === pi ? null : pi)}
                style={{
                  display: "flex", alignItems: "center", gap: 10, padding: "10px 14px",
                  borderRadius: 8, cursor: "pointer",
                  background: phase.status === "partial" ? "rgba(245,158,11,0.06)" : "rgba(255,255,255,0.02)",
                  border: `1px solid ${phase.status === "partial" ? "rgba(245,158,11,0.2)" : "rgba(255,255,255,0.06)"}`
                }}
              >
                <div style={{
                  width: 28, height: 28, borderRadius: 6, display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 12, fontWeight: 800, flexShrink: 0,
                  background: phase.status === "partial" ? "rgba(245,158,11,0.15)" : "rgba(255,255,255,0.05)",
                  color: phase.status === "partial" ? "#F59E0B" : "#64748B"
                }}>P{phase.phase}</div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 13, fontWeight: 700, color: "#E2E8F0" }}>{phase.name}</div>
                  <div style={{ fontSize: 10, color: "#64748B" }}>Weeks {phase.weeks} · {phase.items.length} components</div>
                </div>
                <span style={{
                  fontSize: 8, padding: "2px 8px", borderRadius: 4, letterSpacing: 1, fontWeight: 700,
                  background: phase.status === "partial" ? "rgba(245,158,11,0.12)" : phase.status === "not_started" ? "rgba(239,68,68,0.1)" : "rgba(16,185,129,0.1)",
                  color: phase.status === "partial" ? "#FCD34D" : phase.status === "not_started" ? "#FCA5A5" : "#34D399"
                }}>{phase.status === "partial" ? "IN PROGRESS" : phase.status === "not_started" ? "NOT STARTED" : "DONE"}</span>
              </div>
              {(expandedPhase === pi || true) && (
                <div style={{ paddingLeft: 24, marginTop: 4 }}>
                  {phase.items.map((item, ii) => (
                    <div key={ii} style={{
                      display: "flex", alignItems: "center", gap: 8, padding: "6px 10px",
                      borderLeft: `2px solid ${item.status === "gap" ? "rgba(239,68,68,0.3)" : item.status === "partial" ? "rgba(245,158,11,0.4)" : "rgba(16,185,129,0.4)"}`,
                      marginBottom: 2
                    }}>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                          <span style={{ fontSize: 11, fontWeight: 600, color: "#E2E8F0" }}>{item.name}</span>
                          {item.critical && <span style={{ fontSize: 7, color: "#EF4444", letterSpacing: 1, fontWeight: 700 }}>CRITICAL</span>}
                        </div>
                        <div style={{ fontSize: 9, color: "#64748B", marginTop: 1 }}>{item.note}</div>
                      </div>
                      <div style={{ width: 60, flexShrink: 0 }}>
                        <ProgressBar value={item.progress} color={item.status === "gap" ? "#EF4444" : item.status === "partial" ? "#F59E0B" : "#10B981"} height={4} />
                        <div style={{ fontSize: 8, color: "#475569", textAlign: "right", marginTop: 2 }}>{item.progress}%</div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Strategic Read */}
      {activeTab === "strategic" && (
        <div style={{ maxWidth: 800, margin: "0 auto" }}>
          {STRATEGIC_INSIGHTS.map((s, i) => (
            <div key={i} style={{
              marginBottom: 12, padding: 14, borderRadius: 10,
              background: `${s.color}06`, border: `1px solid ${s.color}20`
            }}>
              <div style={{ fontSize: 14, fontWeight: 700, color: s.color, marginBottom: 8, display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ fontSize: 16 }}>{s.icon}</span> {s.title}
              </div>
              {s.points.map((p, j) => (
                <div key={j} style={{
                  fontSize: 11, color: "#CBD5E1", lineHeight: 1.6, paddingLeft: 10,
                  borderLeft: `2px solid ${s.color}30`, marginBottom: 4
                }}>{p}</div>
              ))}
            </div>
          ))}

          {/* The Bottom Line */}
          <div style={{
            marginTop: 8, padding: 16, borderRadius: 10, textAlign: "center",
            background: "linear-gradient(135deg, rgba(16,185,129,0.06), rgba(139,92,246,0.06), rgba(212,165,116,0.06))",
            border: "1px solid rgba(212,165,116,0.2)"
          }}>
            <div style={{ fontSize: 9, color: "#D4A574", letterSpacing: 3, marginBottom: 8, fontWeight: 700 }}>THE BOTTOM LINE</div>
            <div style={{ fontSize: 13, color: "#E2E8F0", lineHeight: 1.8, maxWidth: 550, margin: "0 auto" }}>
              Today gave BIZRA its first cognitive engines and its first external interface.
              But the bridge from "engines exist" to "agent works for a human" is the
              <span style={{ color: "#F59E0B", fontWeight: 700 }}> Hook System</span> →
              <span style={{ color: "#EC4899", fontWeight: 700 }}> Memory Synthesis</span> →
              <span style={{ color: "#8B5CF6", fontWeight: 700 }}> Context Budget</span> path.
              That's still Week 1 of the 14-week roadmap.
            </div>
            <div style={{ fontSize: 11, color: "#94A3B8", marginTop: 8 }}>
              Phase 46 built the muscle. The skeleton (hooks) still needs to be built for the muscle to attach to.
            </div>
          </div>
        </div>
      )}

      <style>{`
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #1E293B; border-radius: 4px; }
      `}</style>
    </div>
  );
}
