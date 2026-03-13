import { useState } from "react";

const SYSTEM = [
  {
    id: "core-arch",
    name: "Core Architecture",
    icon: "◆",
    color: "#8B5CF6",
    items: [
      { name: "7-Layer DDAGI OS", status: "designed", desc: "Hardware → Network → Storage → Compute → Intelligence → Agent → Application", notes: "Full spec exists. Implementation partial." },
      { name: "Node0 Sovereignty Model", status: "designed", desc: "Single-user-first architecture. Prove one node works before scaling.", notes: "Philosophy clear. Runtime needs completion." },
      { name: "Dual Agentic System", status: "designed", desc: "PAT-7 (Personal) + SAT-49 (System). Personal agents local, system agents in URP.", notes: "Agent specs exist. Orchestration runtime needed." },
      { name: "RDVE Orchestrator", status: "partial", desc: "Reason → Decide → Validate → Execute loop", notes: "Design complete. Needs hook system integration." },
      { name: "Hook System / Event Bus", status: "gap", desc: "Pub/sub nervous system connecting all layers. 15+ lifecycle hooks.", notes: "🔴 CRITICAL GAP — nothing connects without this." },
      { name: "Autonomous Empowerment Loop", status: "designed", desc: "Diffusion reasoning → AHK execution → Receipt verification → Memory → Learning", notes: "The closed loop that makes BIZRA unique. Needs runtime." },
    ]
  },
  {
    id: "memory",
    name: "Memory & Persistence",
    icon: "◈",
    color: "#6366F1",
    items: [
      { name: "Sovereign Memory Architecture (SMA)", status: "designed", desc: "Privacy-preserving distributed memory. Local-first, user-owned.", notes: "Full spec written. Needs implementation." },
      { name: "Episodic Memory Store", status: "partial", desc: "Conversation logs with إحسان scores. persistence.log_conversation() exists.", notes: "Basic logging works. Needs vector indexing for retrieval." },
      { name: "Semantic Memory (User Model)", status: "gap", desc: "Distilled understanding of user — identity, beliefs, preferences, evolving over time.", notes: "🔴 CRITICAL GAP — this is what makes the agent KNOW you." },
      { name: "Procedural Memory", status: "partial", desc: "Behavioral patterns, command protocols, interaction preferences.", notes: "Manual via userPreferences JSON. Needs auto-learning." },
      { name: "Memory Synthesis Pipeline", status: "gap", desc: "Post-conversation: extract facts → resolve conflicts → compress → update user model.", notes: "🔴 CRITICAL GAP — without this, storage ≠ memory." },
      { name: "Context Budget Manager", status: "gap", desc: "Allocates 200K token window optimally across memory types + current conversation.", notes: "🔴 CRITICAL GAP — controls quality of every response." },
      { name: "Memory Retrieval (RAG)", status: "partial", desc: "Vector search + knowledge graph query to find relevant past context.", notes: "HyperGraphRAG designed. Needs implementation + SNR scoring." },
    ]
  },
  {
    id: "intelligence",
    name: "Intelligence Layer",
    icon: "◇",
    color: "#EC4899",
    items: [
      { name: "MOE (Mixture of Experts)", status: "designed", desc: "Routes to specialized models based on task type — coding, reasoning, creative, etc.", notes: "Architecture designed. Needs model loading pipeline." },
      { name: "HRM (Hierarchical Reasoning)", status: "designed", desc: "Multi-step reasoning chains with intermediate verification.", notes: "Spec complete. Integrates with FATE Engine." },
      { name: "HyperGraphRAG", status: "designed", desc: "Knowledge graph + vector store hybrid. Relationship-aware retrieval.", notes: "Novel architecture. Needs Rust implementation." },
      { name: "Diffusion Reasoning", status: "designed", desc: "Iterative refinement reasoning — explore → narrow → crystallize.", notes: "Theoretical framework solid. Needs runtime." },
      { name: "Graph-of-Thoughts", status: "designed", desc: "Non-linear reasoning across interconnected thought nodes.", notes: "Visualization built (React artifact from prior session)." },
      { name: "إحسان Scoring Engine", status: "partial", desc: "Quality evaluation: accuracy × completeness × relevance × SNR. Target: 0.99+", notes: "Scoring function exists. Needs integration into every hook." },
    ]
  },
  {
    id: "blockchain",
    name: "Blockchain & Economics",
    icon: "⬡",
    color: "#F59E0B",
    items: [
      { name: "Dual Token Economy", status: "designed", desc: "SEED (utility/resource access) + BLOOM (governance/impact rewards).", notes: "Tokenomics designed. Smart contracts needed." },
      { name: "HyperBlockTree / BlockGraph", status: "designed", desc: "Native blockchain — not EVM fork. Tree-structured for parallel processing.", notes: "Novel consensus structure. Rust implementation needed." },
      { name: "Proof-of-Impact Consensus", status: "designed", desc: "Nodes earn rewards by demonstrating real human empowerment, not compute waste.", notes: "Impact measurement framework designed. Validation logic needed." },
      { name: "Reverse Scaling Economics", status: "designed", desc: "Quality + performance IMPROVE as network grows (opposite of traditional systems).", notes: "Mathematical proof exists. Needs empirical validation." },
      { name: "Interest-Debt Impossibility Theorem", status: "complete", desc: "Mathematical proof that interest-based debt systems are structurally unsustainable.", notes: "Published in Third Fact whitepaper." },
      { name: "SEED Distribution Engine", status: "gap", desc: "How SEED tokens are minted, distributed, burned based on resource contribution.", notes: "Rules designed. Distribution smart contracts needed." },
    ]
  },
  {
    id: "security",
    name: "Security & Trust",
    icon: "◉",
    color: "#EF4444",
    items: [
      { name: "Ed25519 Cryptographic Identity", status: "designed", desc: "Every node has a unique keypair. All actions are signed.", notes: "Standard chosen. Key management system needed." },
      { name: "FATE Engine (Formal Verification)", status: "designed", desc: "Bounded utility, Crown Proofs, Causal Drag — formal safety guarantees.", notes: "Mathematical framework complete. Runtime gate needed." },
      { name: "TMP (Temporal Measurement Protocol)", status: "designed", desc: "Bounded recursive self-improvement with cryptographic safety guarantees.", notes: "Prevents runaway optimization. Needs integration." },
      { name: "Sovereign Data Vault", status: "gap", desc: "Encrypted local storage. User's data never leaves node without explicit consent.", notes: "Architecture clear. Encryption layer needs building." },
      { name: "Zero-Knowledge Proofs", status: "gap", desc: "Prove computation correctness without revealing data. For federated learning privacy.", notes: "Phase 2 requirement. Research phase." },
      { name: "Threat Model / Attack Surface Map", status: "gap", desc: "Systematic analysis of attack vectors across all 7 layers.", notes: "Needed for investor confidence + security audit." },
    ]
  },
  {
    id: "integration",
    name: "Integration & Protocols",
    icon: "⇌",
    color: "#10B981",
    items: [
      { name: "MCP Server (BIZRA)", status: "gap", desc: "Expose Node0 capabilities as tools via Model Context Protocol.", notes: "Makes BIZRA interoperable with any MCP-compatible agent." },
      { name: "MCP Client", status: "gap", desc: "Connect to external MCP servers — Notion, GitHub, Slack, etc.", notes: "Enables Node0 to use the entire MCP ecosystem." },
      { name: "A2A Protocol (PAT↔SAT)", status: "gap", desc: "Agent-to-Agent communication between personal + system agents.", notes: "Phase 2 critical. Enables federation." },
      { name: "AHK Bridge (Desktop Automation)", status: "partial", desc: "AutoHotkey execution layer — real desktop actions, verified outcomes.", notes: "BIZRA's unique moat. Exists but needs hardening." },
      { name: "REST/gRPC API", status: "gap", desc: "External API for Node0 — programmatic access to agent capabilities.", notes: "Needed for developer ecosystem." },
      { name: "Plugin SDK / Skill SDK", status: "gap", desc: "Let third parties build skills for Node0.", notes: "Phase 2+. Marketplace enabler." },
    ]
  },
  {
    id: "network",
    name: "Network & Federation",
    icon: "⊛",
    color: "#0EA5E9",
    items: [
      { name: "Universal Resource Pool (URP)", status: "designed", desc: "Every node contributes compute/storage/bandwidth → receives SEED tokens.", notes: "Core value prop. Phase 2 implementation." },
      { name: "Node Discovery Protocol", status: "gap", desc: "How nodes find each other on the network. DHT or gossip-based.", notes: "Phase 2. Standard P2P approaches applicable." },
      { name: "Federated Learning Pipeline", status: "gap", desc: "Train shared models without sharing raw data. Privacy-preserving.", notes: "Phase 2. Depends on ZK proofs + SMA." },
      { name: "Skill Sharing / Marketplace", status: "gap", desc: "Nodes share learned skills with network. Earn BLOOM tokens.", notes: "Phase 2+. Killer feature for network effects." },
      { name: "Reputation System", status: "gap", desc: "Node quality scores based on uptime, contribution quality, impact generated.", notes: "Feeds into Proof-of-Impact. Needs design." },
      { name: "Cross-Node State Sync", status: "gap", desc: "Selective, encrypted sync of approved data between federated nodes.", notes: "Phase 2. Builds on SMA." },
    ]
  },
  {
    id: "ux",
    name: "UX & Application",
    icon: "◐",
    color: "#A855F7",
    items: [
      { name: "Sacred Geometry UI", status: "partial", desc: "Flower of Life patterns, golden ratio layouts, glassmorphism.", notes: "Three.js/WebGL designs exist. Integration with runtime needed." },
      { name: "Onboarding Flow", status: "gap", desc: "First-time user experience: create node → set identity → first conversation → memory seeds.", notes: "🔴 CRITICAL for Alpha-100. First impression defines retention." },
      { name: "Memory Dashboard", status: "gap", desc: "User sees what agent remembers, can edit/delete/correct memories.", notes: "Transparency + sovereignty. Users must control their data." },
      { name: "Agent Status Panel", status: "gap", desc: "Shows PAT-7 status, active skills, memory health, إحسان scores.", notes: "Operational visibility. Builds trust." },
      { name: "Conversation Continuity UI", status: "gap", desc: "Visual indicator of memory state — 'I remember X from last session'.", notes: "The UX that SOLVES the cold-start problem visually." },
      { name: "Settings / Sovereignty Controls", status: "gap", desc: "User controls: what to remember, what to share, federation permissions.", notes: "Non-negotiable for sovereign architecture." },
    ]
  },
  {
    id: "ops",
    name: "Operations & DevOps",
    icon: "⚙",
    color: "#78716C",
    items: [
      { name: "Local Runtime (Node0)", status: "partial", desc: "Run everything on user's machine. No cloud dependency for core functions.", notes: "LM Studio integration exists. Orchestrator needs completion." },
      { name: "Monitoring / Observability", status: "gap", desc: "Logs, metrics, traces across all 7 layers. Performance dashboards.", notes: "Engineering necessity. Can use OpenTelemetry." },
      { name: "CI/CD Pipeline", status: "gap", desc: "Automated testing, building, deployment for 144 repos.", notes: "Engineering hygiene. GitHub Actions." },
      { name: "Error Recovery System", status: "gap", desc: "Graceful degradation when components fail. Fallback chains.", notes: "Production readiness requirement." },
      { name: "Update / Migration System", status: "gap", desc: "How nodes update without losing state. Versioned migrations.", notes: "Critical for long-running sovereign nodes." },
      { name: "Telemetry (Privacy-Preserving)", status: "gap", desc: "Anonymous performance metrics for system improvement. User opt-in.", notes: "Network health monitoring without compromising sovereignty." },
    ]
  },
  {
    id: "governance",
    name: "Governance & Philosophy",
    icon: "☽",
    color: "#D4A574",
    items: [
      { name: "Constitutional Framework", status: "complete", desc: "الرسالة (The Message) + البذرة (The Seed) — foundational philosophy.", notes: "Ramadan 2023 origin. Spiritual + technical fusion." },
      { name: "The Third Fact Whitepaper", status: "complete", desc: "Academic paper: mathematical framework for verifiable truth + regenerative economics.", notes: "Fortified v2 with corrected proofs. Publication-ready." },
      { name: "إحسان Standard", status: "complete", desc: "99%+ quality threshold on all deliverables. Excellence as spiritual practice.", notes: "Operating principle. Embedded across system." },
      { name: "DAO Governance Model", status: "gap", desc: "How network decisions are made. BLOOM token voting mechanics.", notes: "Phase 3. Needs design for decentralized governance." },
      { name: "Ethics Framework", status: "partial", desc: "Beneficence, non-maleficence, autonomy, justice — embedded in FATE Engine.", notes: "Principles set. Enforcement mechanisms needed." },
      { name: "Regulatory Compliance Map", status: "gap", desc: "UAE, EU (AI Act), US — regulatory positioning for each jurisdiction.", notes: "Investor requirement. Legal review needed." },
    ]
  },
  {
    id: "business",
    name: "Business & Go-to-Market",
    icon: "▲",
    color: "#F97316",
    items: [
      { name: "Investor Materials ($5M Seed)", status: "partial", desc: "Deck, whitepaper, technical docs, financial projections.", notes: "Whitepaper strong. Deck + projections need completion." },
      { name: "Alpha-100 Rollout Plan", status: "partial", desc: "First 100 users. Onboarding, feedback loops, iteration.", notes: "Strategy exists. Execution plan needs detail." },
      { name: "Competitive Analysis", status: "partial", desc: "BIZRA vs Fetch.ai, SingularityNET, Ocean Protocol, centralized AI.", notes: "Positioning as category creator, not competitor." },
      { name: "Revenue Model", status: "designed", desc: "SEED token utility fees + enterprise licensing + marketplace commission.", notes: "Multiple revenue streams designed. Needs financial modeling." },
      { name: "Team Building Plan", status: "gap", desc: "First 5-10 hires. Roles, compensation, equity structure.", notes: "Solo architect → team. Critical for investor confidence." },
      { name: "IP Protection Strategy", status: "gap", desc: "Patents, trade secrets, open-source vs proprietary decisions.", notes: "144 repos need IP classification." },
    ]
  },
];

const STATUS_CONFIG = {
  complete: { label: "COMPLETE", color: "#10B981", bg: "rgba(16,185,129,0.12)", border: "rgba(16,185,129,0.3)" },
  designed: { label: "DESIGNED", color: "#3B82F6", bg: "rgba(59,130,246,0.12)", border: "rgba(59,130,246,0.3)" },
  partial: { label: "PARTIAL", color: "#F59E0B", bg: "rgba(245,158,11,0.12)", border: "rgba(245,158,11,0.3)" },
  gap: { label: "GAP", color: "#EF4444", bg: "rgba(239,68,68,0.12)", border: "rgba(239,68,68,0.3)" },
};

export default function BIZRASystemMap() {
  const [selectedDomain, setSelectedDomain] = useState(null);
  const [filterStatus, setFilterStatus] = useState(null);
  const [view, setView] = useState("domains");

  const allItems = SYSTEM.flatMap(d => d.items.map(i => ({ ...i, domain: d.name, domainColor: d.color })));
  const stats = {
    complete: allItems.filter(i => i.status === "complete").length,
    designed: allItems.filter(i => i.status === "designed").length,
    partial: allItems.filter(i => i.status === "partial").length,
    gap: allItems.filter(i => i.status === "gap").length,
    total: allItems.length,
  };

  const domain = selectedDomain ? SYSTEM.find(d => d.id === selectedDomain) : null;
  const filteredItems = filterStatus ? allItems.filter(i => i.status === filterStatus) : null;

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
      <div style={{ textAlign: "center", marginBottom: 16 }}>
        <div style={{ fontSize: 9, letterSpacing: 6, color: "#475569", textTransform: "uppercase" }}>
          Complete System Inventory
        </div>
        <h1 style={{
          fontSize: 20, fontWeight: 700, margin: "4px 0",
          background: "linear-gradient(135deg, #8B5CF6, #EC4899, #F59E0B)",
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent"
        }}>
          BIZRA — What Exists, What's Missing
        </h1>
        <div style={{ fontSize: 10, color: "#475569" }}>
          {stats.total} components · {stats.complete} complete · {stats.designed} designed · {stats.partial} partial · {stats.gap} gaps
        </div>
      </div>

      {/* Stats Bar */}
      <div style={{ display: "flex", gap: 6, justifyContent: "center", marginBottom: 16, flexWrap: "wrap" }}>
        {Object.entries(STATUS_CONFIG).map(([key, cfg]) => (
          <button
            key={key}
            onClick={() => { setFilterStatus(filterStatus === key ? null : key); setSelectedDomain(null); setView("filtered"); }}
            style={{
              background: filterStatus === key ? cfg.bg : "rgba(255,255,255,0.02)",
              border: `1px solid ${filterStatus === key ? cfg.border : "rgba(255,255,255,0.06)"}`,
              color: filterStatus === key ? cfg.color : "#64748B",
              padding: "5px 12px", borderRadius: 6, cursor: "pointer",
              fontSize: 11, fontFamily: "inherit", display: "flex", alignItems: "center", gap: 6
            }}
          >
            <span style={{
              width: 7, height: 7, borderRadius: 2, background: cfg.color,
              opacity: filterStatus === key ? 1 : 0.4
            }} />
            {cfg.label}
            <span style={{ fontWeight: 700 }}>{stats[key]}</span>
          </button>
        ))}
        <button
          onClick={() => { setFilterStatus(null); setSelectedDomain(null); setView("domains"); }}
          style={{
            background: !filterStatus && view === "domains" ? "rgba(139,92,246,0.15)" : "rgba(255,255,255,0.02)",
            border: `1px solid ${!filterStatus && view === "domains" ? "rgba(139,92,246,0.3)" : "rgba(255,255,255,0.06)"}`,
            color: !filterStatus ? "#C4B5FD" : "#64748B",
            padding: "5px 12px", borderRadius: 6, cursor: "pointer",
            fontSize: 11, fontFamily: "inherit"
          }}
        >
          ALL
        </button>
      </div>

      {/* Health Bar */}
      <div style={{
        maxWidth: 600, margin: "0 auto 20px", height: 6, borderRadius: 3,
        background: "rgba(255,255,255,0.05)", display: "flex", overflow: "hidden"
      }}>
        <div style={{ width: `${(stats.complete/stats.total)*100}%`, background: "#10B981", transition: "width 0.3s" }} />
        <div style={{ width: `${(stats.designed/stats.total)*100}%`, background: "#3B82F6", transition: "width 0.3s" }} />
        <div style={{ width: `${(stats.partial/stats.total)*100}%`, background: "#F59E0B", transition: "width 0.3s" }} />
        <div style={{ width: `${(stats.gap/stats.total)*100}%`, background: "#EF4444", transition: "width 0.3s" }} />
      </div>

      {/* Filtered View */}
      {filterStatus && filteredItems && (
        <div style={{ maxWidth: 800, margin: "0 auto" }}>
          <div style={{ fontSize: 12, color: STATUS_CONFIG[filterStatus].color, fontWeight: 700, marginBottom: 12 }}>
            {filteredItems.length} items with status: {STATUS_CONFIG[filterStatus].label}
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {filteredItems.map((item, i) => (
              <div key={i} style={{
                background: "rgba(255,255,255,0.02)",
                border: `1px solid ${STATUS_CONFIG[item.status].border}`,
                borderRadius: 6, padding: "8px 12px"
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 3 }}>
                  <span style={{ fontSize: 12, fontWeight: 600, color: "#E2E8F0" }}>{item.name}</span>
                  <span style={{
                    fontSize: 8, padding: "1px 5px", borderRadius: 3,
                    background: `${item.domainColor}22`, color: item.domainColor,
                    border: `1px solid ${item.domainColor}44`, letterSpacing: 1
                  }}>{item.domain}</span>
                </div>
                <div style={{ fontSize: 10, color: "#94A3B8", marginBottom: 2 }}>{item.desc}</div>
                <div style={{ fontSize: 10, color: filterStatus === "gap" ? "#FCA5A5" : "#64748B", fontStyle: "italic" }}>{item.notes}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Domain Grid View */}
      {!filterStatus && view === "domains" && (
        <div style={{ maxWidth: 1000, margin: "0 auto" }}>
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
            gap: 8
          }}>
            {SYSTEM.map(d => {
              const dStats = {
                complete: d.items.filter(i => i.status === "complete").length,
                designed: d.items.filter(i => i.status === "designed").length,
                partial: d.items.filter(i => i.status === "partial").length,
                gap: d.items.filter(i => i.status === "gap").length,
              };
              const isSelected = selectedDomain === d.id;
              return (
                <div key={d.id}>
                  <div
                    onClick={() => setSelectedDomain(isSelected ? null : d.id)}
                    style={{
                      background: isSelected ? `${d.color}0D` : "rgba(255,255,255,0.02)",
                      border: `1px solid ${isSelected ? `${d.color}40` : "rgba(255,255,255,0.06)"}`,
                      borderRadius: 8, padding: "10px 12px", cursor: "pointer",
                      transition: "all 0.2s"
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        <span style={{ fontSize: 14, color: d.color }}>{d.icon}</span>
                        <span style={{ fontSize: 12, fontWeight: 700, color: d.color }}>{d.name}</span>
                      </div>
                      <span style={{ fontSize: 10, color: "#475569" }}>{d.items.length}</span>
                    </div>
                    {/* Mini health bar */}
                    <div style={{ height: 3, borderRadius: 2, background: "rgba(255,255,255,0.05)", display: "flex", overflow: "hidden", marginBottom: 6 }}>
                      <div style={{ width: `${(dStats.complete/d.items.length)*100}%`, background: "#10B981" }} />
                      <div style={{ width: `${(dStats.designed/d.items.length)*100}%`, background: "#3B82F6" }} />
                      <div style={{ width: `${(dStats.partial/d.items.length)*100}%`, background: "#F59E0B" }} />
                      <div style={{ width: `${(dStats.gap/d.items.length)*100}%`, background: "#EF4444" }} />
                    </div>
                    <div style={{ display: "flex", gap: 6, fontSize: 9, color: "#64748B" }}>
                      {dStats.complete > 0 && <span style={{ color: "#10B981" }}>✓{dStats.complete}</span>}
                      {dStats.designed > 0 && <span style={{ color: "#3B82F6" }}>◆{dStats.designed}</span>}
                      {dStats.partial > 0 && <span style={{ color: "#F59E0B" }}>◐{dStats.partial}</span>}
                      {dStats.gap > 0 && <span style={{ color: "#EF4444" }}>○{dStats.gap}</span>}
                    </div>
                  </div>

                  {/* Expanded items */}
                  {isSelected && (
                    <div style={{
                      marginTop: 4, display: "flex", flexDirection: "column", gap: 3,
                      animation: "fadeIn 0.2s ease"
                    }}>
                      {d.items.map((item, i) => {
                        const s = STATUS_CONFIG[item.status];
                        return (
                          <div key={i} style={{
                            background: s.bg,
                            border: `1px solid ${s.border}`,
                            borderRadius: 6, padding: "8px 10px"
                          }}>
                            <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 2 }}>
                              <span style={{
                                width: 6, height: 6, borderRadius: 2, background: s.color
                              }} />
                              <span style={{ fontSize: 11, fontWeight: 600, color: "#E2E8F0" }}>{item.name}</span>
                              <span style={{
                                fontSize: 7, padding: "1px 4px", borderRadius: 2,
                                background: s.bg, color: s.color,
                                border: `1px solid ${s.border}`,
                                fontWeight: 700, letterSpacing: 1
                              }}>{s.label}</span>
                            </div>
                            <div style={{ fontSize: 10, color: "#94A3B8", marginBottom: 2 }}>{item.desc}</div>
                            <div style={{ fontSize: 9, color: item.status === "gap" ? "#FCA5A5" : "#64748B", fontStyle: "italic" }}>
                              {item.notes}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Summary */}
      <div style={{
        maxWidth: 800, margin: "24px auto 0",
        background: "rgba(139,92,246,0.06)",
        border: "1px solid rgba(139,92,246,0.2)",
        borderRadius: 10, padding: 14
      }}>
        <div style={{ fontSize: 9, color: "#8B5CF6", letterSpacing: 3, marginBottom: 8, textTransform: "uppercase", fontWeight: 700 }}>
          System Health Summary
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 8 }}>
          {[
            { label: "Philosophy & Vision", pct: 95, color: "#10B981" },
            { label: "Architecture Design", pct: 80, color: "#3B82F6" },
            { label: "Memory & Persistence", pct: 25, color: "#EF4444" },
            { label: "Integration Protocols", pct: 10, color: "#EF4444" },
            { label: "Intelligence Layer", pct: 40, color: "#F59E0B" },
            { label: "Production Readiness", pct: 15, color: "#EF4444" },
            { label: "Business / GTM", pct: 35, color: "#F59E0B" },
            { label: "Network / Federation", pct: 5, color: "#EF4444" },
          ].map((m, i) => (
            <div key={i}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "#94A3B8", marginBottom: 3 }}>
                <span>{m.label}</span>
                <span style={{ color: m.color, fontWeight: 700 }}>{m.pct}%</span>
              </div>
              <div style={{ height: 3, borderRadius: 2, background: "rgba(255,255,255,0.05)" }}>
                <div style={{ height: "100%", borderRadius: 2, background: m.color, width: `${m.pct}%`, transition: "width 0.5s" }} />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div style={{
        textAlign: "center", marginTop: 24, fontSize: 9, color: "#1E293B"
      }}>
        BIZRA System Inventory · {stats.total} components across {SYSTEM.length} domains · Feb 2026
      </div>

      <style>{`
        @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
        * { box-sizing: border-box; }
      `}</style>
    </div>
  );
}
