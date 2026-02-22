import { useState } from "react";

const PHASES = [
  {
    id: "foundation",
    phase: "Phase 0",
    name: "The Foundation",
    weeks: "Weeks 1-3",
    color: "#EF4444",
    tagline: "Without this, nothing connects",
    items: [
      {
        name: "Hook System + Event Bus",
        time: "2 weeks",
        why: "The nervous system. Every component communicates through hooks. Build this FIRST because everything else plugs into it.",
        delivers: "All components can emit and listen to lifecycle events",
        tech: "Rust · pub/sub · typed events · async handlers",
        hooks: ["onUserInput", "beforeLLMCall", "afterLLMCall", "onToolCall", "onMemoryUpdate", "onDataWrite"],
        critical: true
      },
      {
        name: "Local LLM Runtime",
        time: "1 week",
        why: "Node0 must think locally. LM Studio or Ollama integration for sovereign inference.",
        delivers: "Agent can reason without cloud dependency",
        tech: "LM Studio API · model loading · fallback chain",
        hooks: ["onModelLoad", "onInferenceStart", "onInferenceComplete"],
        critical: true
      },
    ]
  },
  {
    id: "memory",
    phase: "Phase 1",
    name: "The Memory",
    weeks: "Weeks 3-7",
    color: "#8B5CF6",
    tagline: "This is what makes the magic — the agent KNOWS you",
    items: [
      {
        name: "Episodic Memory + Vector Store",
        time: "2 weeks",
        why: "Every conversation indexed and searchable. Not just stored — retrievable by meaning, not just keywords.",
        delivers: "Agent can find relevant past conversations by semantic similarity",
        tech: "Qdrant/ChromaDB · embedding model · conversation chunking · SNR scoring",
        hooks: ["onConversationStore", "onEpisodicRetrieve"],
        critical: true
      },
      {
        name: "Memory Synthesis Pipeline",
        time: "3 weeks",
        why: "THE critical component. After every conversation: extract facts → detect changes → resolve conflicts → update user model. This transforms storage into memory.",
        delivers: "Agent builds evolving understanding of you — not just transcripts but knowledge",
        tech: "Entity extraction · fact extraction · contradiction resolution · temporal reasoning · إحسان scoring",
        hooks: ["onMemoryExtract", "onFactDiscovered", "onConflictDetected", "onUserModelUpdate"],
        critical: true
      },
      {
        name: "Context Budget Manager",
        time: "1 week",
        why: "200K tokens. Must allocate optimally: identity (5K) + semantic memory (10K) + relevant episodics (30K) + tools (10K) + conversation (remaining). Bad budgeting = agent remembers wrong things.",
        delivers: "Every response is informed by the best possible context assembly",
        tech: "Token counting · relevance ranking · recency weighting · budget allocation algorithm",
        hooks: ["onBudgetAllocated", "onContextAssembled"],
        critical: true
      },
    ]
  },
  {
    id: "agent",
    phase: "Phase 2",
    name: "The Agent",
    weeks: "Weeks 7-10",
    color: "#EC4899",
    tagline: "From memory to action — the agent that DOES things",
    items: [
      {
        name: "PAT Core (3 of 7 agents)",
        time: "2 weeks",
        why: "Don't build all 7 PAT agents at once. Start with 3: Memory Agent (manages all memory operations), Task Agent (handles user requests), Desktop Agent (AHK bridge). Add remaining 4 later.",
        delivers: "Agent can remember, reason, and act on desktop",
        tech: "Agent runtime · role definitions · shared state · handoff protocol",
        hooks: ["onAgentActivated", "onAgentHandoff", "onTaskComplete"],
        critical: true
      },
      {
        name: "AHK Desktop Bridge (Hardened)",
        time: "1 week",
        why: "Your unique moat. Agent executes real desktop actions → captures verification screenshots → confirms outcome. The closed loop that nobody else has.",
        delivers: "Agent can DO things in the real world, not just talk",
        tech: "AHK executor · screenshot verification · action logging · rollback on failure",
        hooks: ["onDesktopAction", "onActionVerified", "onActionFailed"],
        critical: true
      },
      {
        name: "MCP Client (External Tools)",
        time: "1 week",
        why: "Connect Node0 to the existing tool ecosystem — Notion for docs, GitHub for code, calendar for scheduling. MCP is the standard connector.",
        delivers: "Agent uses your existing tools seamlessly",
        tech: "MCP protocol client · tool discovery · auth management",
        hooks: ["onMCPConnect", "onToolDiscovered", "onExternalToolCall"],
        critical: false
      },
    ]
  },
  {
    id: "experience",
    phase: "Phase 3",
    name: "The Experience",
    weeks: "Weeks 10-12",
    color: "#F59E0B",
    tagline: "What the user sees — first impression defines everything",
    items: [
      {
        name: "Onboarding Flow",
        time: "1 week",
        why: "Alpha user's first 5 minutes. Create node → set identity → first conversation → agent learns their name, role, goals → immediate 'wow this knows me' moment.",
        delivers: "New user goes from zero to personalized agent in under 5 minutes",
        tech: "Guided setup wizard · initial memory seeding · personality calibration",
        hooks: ["onNodeCreated", "onIdentitySet", "onFirstConversation"],
        critical: true
      },
      {
        name: "Conversation Continuity UI",
        time: "1 week",
        why: "The visual proof that memory works. Agent says 'I remember we discussed X last Tuesday.' Shows memory confidence. User can correct memories. THIS is the moment that sells.",
        delivers: "User SEES that the agent remembers — the cold-start problem is visibly solved",
        tech: "Memory reference badges · session continuity indicators · memory correction UI",
        hooks: ["onMemoryReferenced", "onMemoryCorrected"],
        critical: true
      },
      {
        name: "إحسان Quality Dashboard",
        time: "0.5 weeks",
        why: "Show the user their agent's quality scores. Builds trust. Shows improvement over time. Unique to BIZRA — no other system has visible quality metrics.",
        delivers: "Transparency into agent quality — user trusts because they can verify",
        tech: "Score visualization · trend charts · per-conversation breakdown",
        hooks: ["onIhsanCalculated"],
        critical: false
      },
    ]
  },
  {
    id: "sovereignty",
    phase: "Phase 4",
    name: "The Sovereignty",
    weeks: "Weeks 12-14",
    color: "#10B981",
    tagline: "Everything local. Everything encrypted. Everything yours.",
    items: [
      {
        name: "Sovereign Data Vault",
        time: "1 week",
        why: "All data encrypted at rest with Ed25519. User's memories, conversations, preferences — never leave the node without explicit consent.",
        delivers: "User owns their data completely. Not 'we promise' — mathematically guaranteed.",
        tech: "Ed25519 encryption · local storage · key management · backup/restore",
        hooks: ["onDataEncrypted", "onBackupCreated"],
        critical: true
      },
      {
        name: "Memory Controls UI",
        time: "1 week",
        why: "User can see everything the agent knows, edit it, delete it, export it. This is sovereignty made tangible. If you can't control your data, you don't own it.",
        delivers: "Full transparency and control over all stored knowledge",
        tech: "Memory browser · edit/delete · export (JSON/PDF) · selective sharing toggles",
        hooks: ["onMemoryViewed", "onMemoryDeleted", "onMemoryExported"],
        critical: true
      },
    ]
  },
];

const VIRAL_MATH = [
  { node: "Node 0 (You)", users: 1, value: "V", network: "Standalone", time: "Weeks 1-14" },
  { node: "Alpha-10", users: 10, value: "V × 1.0", network: "Isolated nodes", time: "Week 15-16" },
  { node: "Alpha-100", users: 100, value: "V × 1.3", network: "Shared skills begin", time: "Week 17-20" },
  { node: "Beta-1K", users: "1,000", value: "V × 2.1", network: "URP activates", time: "Month 6-8" },
  { node: "Growth-10K", users: "10,000", value: "V × 4.7", network: "Federated learning", time: "Month 9-12" },
  { node: "Scale-100K", users: "100K", value: "V × 12", network: "Network intelligence", time: "Month 12-18" },
  { node: "Vision-8B", users: "8B", value: "V × ∞", network: "Every human empowered", time: "The mission" },
];

const WHAT_EXISTS = [
  { item: "GenesisOrchestrator", has: true, needs: "Hook system integration + memory pipeline" },
  { item: "persistence.log_conversation()", has: true, needs: "Vector indexing + semantic search" },
  { item: "إحسان Scoring Function", has: true, needs: "Integration into every hook lifecycle" },
  { item: "AHK Desktop Bridge", has: true, needs: "Hardening + verification + rollback" },
  { item: "Sacred Geometry UI", has: true, needs: "Connection to running agent runtime" },
  { item: "144 GitHub Repositories", has: true, needs: "Consolidation into Node0 monorepo or workspace" },
  { item: "Architecture Atlas (28 diagrams)", has: true, needs: "Implementation of what diagrams describe" },
  { item: "Third Fact Whitepaper", has: true, needs: "Nothing — publication ready" },
];

export default function Node0MVP() {
  const [selectedPhase, setSelectedPhase] = useState(null);
  const [activeView, setActiveView] = useState("roadmap");
  const [expandedItem, setExpandedItem] = useState(null);

  const totalWeeks = 14;
  const totalItems = PHASES.flatMap(p => p.items).length;
  const criticalItems = PHASES.flatMap(p => p.items).filter(i => i.critical).length;

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
      <div style={{ textAlign: "center", marginBottom: 20 }}>
        <div style={{ fontSize: 9, letterSpacing: 6, color: "#475569", textTransform: "uppercase" }}>
          The Secret Method
        </div>
        <h1 style={{
          fontSize: 22, fontWeight: 800, margin: "4px 0",
          background: "linear-gradient(135deg, #10B981, #8B5CF6, #F59E0B)",
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent"
        }}>
          One Perfect Node → Everything
        </h1>
        <div style={{ fontSize: 11, color: "#64748B", maxWidth: 500, margin: "6px auto 0", lineHeight: 1.5 }}>
          {totalItems} components · {criticalItems} critical · {totalWeeks} weeks to fully functional Node0
        </div>
      </div>

      {/* View Tabs */}
      <div style={{ display: "flex", gap: 2, justifyContent: "center", marginBottom: 16, flexWrap: "wrap" }}>
        {[
          { id: "roadmap", label: "◎ Build Roadmap" },
          { id: "viral", label: "⟐ Growth Math" },
          { id: "existing", label: "✓ What Exists" },
        ].map(tab => (
          <button key={tab.id} onClick={() => setActiveView(tab.id)} style={{
            background: activeView === tab.id ? "rgba(139,92,246,0.2)" : "rgba(255,255,255,0.03)",
            border: `1px solid ${activeView === tab.id ? "rgba(139,92,246,0.5)" : "rgba(255,255,255,0.06)"}`,
            color: activeView === tab.id ? "#C4B5FD" : "#64748B",
            padding: "5px 14px", borderRadius: 6, cursor: "pointer", fontSize: 11, fontFamily: "inherit"
          }}>
            {tab.label}
          </button>
        ))}
      </div>

      {/* Roadmap View */}
      {activeView === "roadmap" && (
        <div style={{ maxWidth: 800, margin: "0 auto" }}>
          {/* Timeline bar */}
          <div style={{
            display: "flex", gap: 2, marginBottom: 20, padding: "0 4px"
          }}>
            {PHASES.map(p => {
              const start = parseInt(p.weeks.split("-")[0].replace(/\D/g, ""));
              const end = parseInt(p.weeks.split("-")[1]);
              const width = ((end - start + 1) / totalWeeks) * 100;
              return (
                <div key={p.id} style={{
                  flex: `0 0 ${width}%`, textAlign: "center",
                  borderBottom: `3px solid ${p.color}`,
                  paddingBottom: 4
                }}>
                  <div style={{ fontSize: 8, color: p.color, fontWeight: 700, letterSpacing: 1 }}>{p.phase}</div>
                  <div style={{ fontSize: 7, color: "#475569" }}>{p.weeks}</div>
                </div>
              );
            })}
          </div>

          {PHASES.map((phase) => (
            <div key={phase.id} style={{ marginBottom: 16 }}>
              {/* Phase Header */}
              <div
                onClick={() => setSelectedPhase(selectedPhase === phase.id ? null : phase.id)}
                style={{
                  display: "flex", alignItems: "center", gap: 10,
                  padding: "10px 12px", borderRadius: 8, cursor: "pointer",
                  background: selectedPhase === phase.id ? `${phase.color}0D` : "rgba(255,255,255,0.02)",
                  border: `1px solid ${selectedPhase === phase.id ? `${phase.color}33` : "rgba(255,255,255,0.06)"}`,
                  transition: "all 0.2s"
                }}
              >
                <div style={{
                  width: 36, height: 36, borderRadius: 8,
                  background: `linear-gradient(135deg, ${phase.color}22, ${phase.color}44)`,
                  border: `1px solid ${phase.color}55`,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 10, fontWeight: 800, color: phase.color, flexShrink: 0
                }}>{phase.phase.split(" ")[1]}</div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 14, fontWeight: 700, color: phase.color }}>{phase.name}</div>
                  <div style={{ fontSize: 10, color: "#64748B", fontStyle: "italic" }}>{phase.tagline}</div>
                </div>
                <div style={{ textAlign: "right", flexShrink: 0 }}>
                  <div style={{ fontSize: 10, color: "#94A3B8" }}>{phase.weeks}</div>
                  <div style={{ fontSize: 9, color: "#475569" }}>{phase.items.length} components</div>
                </div>
              </div>

              {/* Phase Items */}
              {selectedPhase === phase.id && (
                <div style={{ marginTop: 6, display: "flex", flexDirection: "column", gap: 4, paddingLeft: 16, borderLeft: `2px solid ${phase.color}22` }}>
                  {phase.items.map((item, i) => {
                    const key = `${phase.id}-${i}`;
                    const isExpanded = expandedItem === key;
                    return (
                      <div key={i}
                        onClick={() => setExpandedItem(isExpanded ? null : key)}
                        style={{
                          background: isExpanded ? `${phase.color}0A` : "rgba(255,255,255,0.02)",
                          border: `1px solid ${isExpanded ? `${phase.color}30` : "rgba(255,255,255,0.05)"}`,
                          borderRadius: 8, padding: "10px 12px", cursor: "pointer",
                          transition: "all 0.15s"
                        }}
                      >
                        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                          <span style={{ fontSize: 13, fontWeight: 700, color: "#E2E8F0" }}>{item.name}</span>
                          {item.critical && (
                            <span style={{
                              fontSize: 7, padding: "1px 5px", borderRadius: 3,
                              background: "rgba(239,68,68,0.15)", color: "#FCA5A5",
                              border: "1px solid rgba(239,68,68,0.3)",
                              fontWeight: 700, letterSpacing: 1
                            }}>CRITICAL</span>
                          )}
                          <span style={{ marginLeft: "auto", fontSize: 10, color: "#64748B" }}>{item.time}</span>
                        </div>
                        <div style={{ fontSize: 11, color: "#94A3B8", lineHeight: 1.5 }}>{item.why}</div>

                        {isExpanded && (
                          <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 6 }}>
                            <div style={{
                              background: "rgba(16,185,129,0.08)", border: "1px solid rgba(16,185,129,0.2)",
                              borderRadius: 6, padding: "6px 10px"
                            }}>
                              <div style={{ fontSize: 8, color: "#10B981", letterSpacing: 2, marginBottom: 2, fontWeight: 700 }}>DELIVERS</div>
                              <div style={{ fontSize: 10, color: "#6EE7B7" }}>{item.delivers}</div>
                            </div>
                            <div style={{
                              background: "rgba(99,102,241,0.08)", border: "1px solid rgba(99,102,241,0.2)",
                              borderRadius: 6, padding: "6px 10px"
                            }}>
                              <div style={{ fontSize: 8, color: "#818CF8", letterSpacing: 2, marginBottom: 2, fontWeight: 700 }}>TECH STACK</div>
                              <div style={{ fontSize: 10, color: "#A5B4FC" }}>{item.tech}</div>
                            </div>
                            {item.hooks && item.hooks.length > 0 && (
                              <div style={{
                                background: "rgba(245,158,11,0.08)", border: "1px solid rgba(245,158,11,0.2)",
                                borderRadius: 6, padding: "6px 10px"
                              }}>
                                <div style={{ fontSize: 8, color: "#F59E0B", letterSpacing: 2, marginBottom: 4, fontWeight: 700 }}>HOOKS REGISTERED</div>
                                <div style={{ display: "flex", flexWrap: "wrap", gap: 3 }}>
                                  {item.hooks.map((h, j) => (
                                    <code key={j} style={{
                                      fontSize: 9, padding: "1px 5px", borderRadius: 3,
                                      background: "rgba(245,158,11,0.1)", color: "#FCD34D"
                                    }}>{h}</code>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          ))}

          {/* The Promise */}
          <div style={{
            marginTop: 20, padding: 16, borderRadius: 10, textAlign: "center",
            background: "linear-gradient(135deg, rgba(16,185,129,0.08), rgba(139,92,246,0.08))",
            border: "1px solid rgba(16,185,129,0.2)"
          }}>
            <div style={{ fontSize: 11, color: "#10B981", fontWeight: 700, marginBottom: 4 }}>
              WEEK 14: NODE0 IS ALIVE
            </div>
            <div style={{ fontSize: 12, color: "#94A3B8", lineHeight: 1.6, maxWidth: 500, margin: "0 auto" }}>
              You sit down. The agent knows your 31 months of BIZRA history. Picks up where you left off.
              Executes desktop tasks. Learns from every interaction. Gets better every day.
              No cold start. No re-explaining. No context loss.
            </div>
            <div style={{ fontSize: 13, color: "#E2E8F0", fontWeight: 700, marginTop: 8 }}>
              Then you show it to someone. And the network begins.
            </div>
          </div>
        </div>
      )}

      {/* Viral Growth Math */}
      {activeView === "viral" && (
        <div style={{ maxWidth: 700, margin: "0 auto" }}>
          <div style={{ fontSize: 12, color: "#94A3B8", marginBottom: 16, lineHeight: 1.6, textAlign: "center" }}>
            The growth model. Each stage unlocks automatically when the previous stage delivers value.
          </div>

          <div style={{ position: "relative" }}>
            {VIRAL_MATH.map((stage, i) => (
              <div key={i} style={{ display: "flex", gap: 12, marginBottom: 4, alignItems: "stretch" }}>
                {/* Left: timeline dot */}
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", width: 20, flexShrink: 0 }}>
                  <div style={{
                    width: 12, height: 12, borderRadius: 6,
                    background: i === 0 ? "#10B981" : i === VIRAL_MATH.length - 1 ? "#F59E0B" : `${["#8B5CF6","#EC4899","#0EA5E9","#F97316","#6366F1"][i % 5]}`,
                    border: `2px solid ${i === 0 ? "#10B981" : "#1E293B"}`,
                    flexShrink: 0
                  }} />
                  {i < VIRAL_MATH.length - 1 && (
                    <div style={{ width: 1, flex: 1, background: "rgba(255,255,255,0.06)" }} />
                  )}
                </div>
                {/* Right: content */}
                <div style={{
                  flex: 1, padding: "8px 12px", borderRadius: 8, marginBottom: 4,
                  background: i === 0 ? "rgba(16,185,129,0.08)" : i === VIRAL_MATH.length - 1 ? "rgba(245,158,11,0.08)" : "rgba(255,255,255,0.02)",
                  border: `1px solid ${i === 0 ? "rgba(16,185,129,0.25)" : i === VIRAL_MATH.length - 1 ? "rgba(245,158,11,0.25)" : "rgba(255,255,255,0.06)"}`,
                }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 3 }}>
                    <span style={{ fontSize: 13, fontWeight: 700, color: "#E2E8F0" }}>{stage.node}</span>
                    <span style={{ fontSize: 10, color: "#475569" }}>{stage.time}</span>
                  </div>
                  <div style={{ display: "flex", gap: 12, fontSize: 10 }}>
                    <span style={{ color: "#64748B" }}>Users: <span style={{ color: "#94A3B8", fontWeight: 600 }}>{stage.users}</span></span>
                    <span style={{ color: "#64748B" }}>Value: <span style={{ color: "#10B981", fontWeight: 600 }}>{stage.value}</span></span>
                  </div>
                  <div style={{ fontSize: 10, color: "#8B5CF6", marginTop: 2 }}>{stage.network}</div>
                </div>
              </div>
            ))}
          </div>

          {/* Key Insight */}
          <div style={{
            marginTop: 20, padding: 14, borderRadius: 10,
            background: "rgba(139,92,246,0.06)",
            border: "1px solid rgba(139,92,246,0.2)",
          }}>
            <div style={{ fontSize: 9, color: "#8B5CF6", letterSpacing: 3, marginBottom: 6, fontWeight: 700 }}>THE INVERSION</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 40px 1fr", gap: 8, alignItems: "center" }}>
              <div style={{
                padding: 10, borderRadius: 6, fontSize: 10, lineHeight: 1.5,
                background: "rgba(239,68,68,0.06)", border: "1px solid rgba(239,68,68,0.15)", color: "#FCA5A5"
              }}>
                <div style={{ fontWeight: 700, marginBottom: 4 }}>Traditional AI</div>
                Build network first → Hope individuals benefit → Value extracted from users → Network owns the data → Each user costs more
              </div>
              <div style={{ textAlign: "center", fontSize: 16, color: "#475569" }}>vs</div>
              <div style={{
                padding: 10, borderRadius: 6, fontSize: 10, lineHeight: 1.5,
                background: "rgba(16,185,129,0.06)", border: "1px solid rgba(16,185,129,0.15)", color: "#6EE7B7"
              }}>
                <div style={{ fontWeight: 700, marginBottom: 4 }}>BIZRA</div>
                Perfect one node → Individual can't live without it → Value created for users → User owns the data → Each node contributes more
              </div>
            </div>
          </div>

          {/* The Viral Mechanic */}
          <div style={{
            marginTop: 12, padding: 14, borderRadius: 10,
            background: "rgba(255,255,255,0.02)",
            border: "1px solid rgba(255,255,255,0.06)",
          }}>
            <div style={{ fontSize: 9, color: "#F59E0B", letterSpacing: 3, marginBottom: 8, fontWeight: 700 }}>WHY IT SPREADS</div>
            <div style={{ fontSize: 11, color: "#94A3B8", lineHeight: 1.7 }}>
              <div style={{ marginBottom: 6 }}>
                <span style={{ color: "#E2E8F0", fontWeight: 600 }}>The Trigger:</span> User experiences an agent that actually remembers them. Every other AI feels broken by comparison.
              </div>
              <div style={{ marginBottom: 6 }}>
                <span style={{ color: "#E2E8F0", fontWeight: 600 }}>The Tell:</span> User's output quality visibly improves. Colleagues ask "how are you doing this?"
              </div>
              <div style={{ marginBottom: 6 }}>
                <span style={{ color: "#E2E8F0", fontWeight: 600 }}>The Hook:</span> "My AI knows me." Four words. That's the entire pitch. Nobody has to explain features.
              </div>
              <div>
                <span style={{ color: "#E2E8F0", fontWeight: 600 }}>The Lock:</span> After 2 weeks of accumulated memory, switching cost is infinite. Your agent knows things no other system can replicate.
              </div>
            </div>
          </div>
        </div>
      )}

      {/* What Exists View */}
      {activeView === "existing" && (
        <div style={{ maxWidth: 700, margin: "0 auto" }}>
          <div style={{ fontSize: 12, color: "#94A3B8", marginBottom: 16, lineHeight: 1.6, textAlign: "center" }}>
            You're not starting from zero. These assets exist and feed directly into the Node0 MVP.
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {WHAT_EXISTS.map((item, i) => (
              <div key={i} style={{
                display: "flex", alignItems: "center", gap: 10,
                padding: "10px 12px", borderRadius: 8,
                background: "rgba(16,185,129,0.04)",
                border: "1px solid rgba(16,185,129,0.15)"
              }}>
                <div style={{
                  width: 24, height: 24, borderRadius: 6,
                  background: "rgba(16,185,129,0.15)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  color: "#10B981", fontSize: 12, flexShrink: 0
                }}>✓</div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: "#E2E8F0" }}>{item.item}</div>
                  <div style={{ fontSize: 10, color: "#64748B", marginTop: 1 }}>
                    Needs: <span style={{ color: "#94A3B8" }}>{item.needs}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div style={{
            marginTop: 20, padding: 14, borderRadius: 10, textAlign: "center",
            background: "rgba(99,102,241,0.06)",
            border: "1px solid rgba(99,102,241,0.2)"
          }}>
            <div style={{ fontSize: 11, color: "#818CF8", lineHeight: 1.6 }}>
              31 months of work isn't wasted — it's the design phase.
              The architecture is complete. Now it becomes a running system.
              <div style={{ marginTop: 6, fontSize: 13, fontWeight: 700, color: "#E2E8F0" }}>
                ربي لا يعرف المستحيل
              </div>
            </div>
          </div>
        </div>
      )}

      <style>{`
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        * { box-sizing: border-box; }
      `}</style>
    </div>
  );
}
