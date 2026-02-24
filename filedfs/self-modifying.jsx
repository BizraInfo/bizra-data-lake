import { useState } from "react";

const LEVELS = [
  {
    level: 0,
    name: "Configuration Change",
    tagline: "Adjusting the dials",
    color: "#10B981",
    risk: "NONE",
    riskColor: "#10B981",
    gate: "None needed",
    autonomous: true,
    example: "Agent sets temperature=0.3 for code tasks, temperature=0.9 for creative tasks",
    what_changes: "Parameters, settings, preferences — the agent tunes itself for the task",
    bizra_reality: "Already possible. Procedural memory stores optimal settings per task type.",
    code_example: `// Agent adjusts its own config
self.config.temperature = match task_type {
    TaskType::Code => 0.3,
    TaskType::Creative => 0.9,
    TaskType::Analysis => 0.5,
};`,
    implications: [
      "Zero risk — parameters are bounded by design",
      "Every agent already does this implicitly",
      "BIZRA advantage: إحسان scoring tells the agent WHICH settings produce better output",
    ]
  },
  {
    level: 1,
    name: "Prompt Self-Optimization",
    tagline: "Rewriting its own instructions",
    color: "#3B82F6",
    risk: "LOW",
    riskColor: "#3B82F6",
    gate: "إحسان score must improve",
    autonomous: true,
    example: "Agent rewrites its system prompt to be more effective based on user feedback patterns",
    what_changes: "The instructions that shape HOW the agent responds — tone, structure, reasoning approach",
    bizra_reality: "This is what procedural memory SHOULD do. Learn what works → update the behavioral template → verify improvement.",
    code_example: `// After 50 conversations, agent notices user prefers
// structured responses. It updates its own prompt:
fn optimize_prompt(&mut self) {
    let patterns = self.memory.get_behavioral_patterns();
    if patterns.structured_preference > 0.8 {
        self.system_prompt.add_instruction(
            "Always use structured headers and clear sections"
        );
    }
    // Verify: did إحسان scores improve?
    assert!(self.ihsan.trend() > 0.0, "Revert if quality dropped");
}`,
    implications: [
      "Agent becomes more attuned to you over time without manual configuration",
      "Must be bounded: prompt changes verified against إحسان baseline",
      "Revert mechanism required: if quality drops, rollback to previous prompt",
      "This is the foundation of 'my AI knows me' — it literally rewrites itself to serve you better",
    ]
  },
  {
    level: 2,
    name: "Skill Generation",
    tagline: "Creating new capabilities",
    color: "#F59E0B",
    risk: "MEDIUM",
    riskColor: "#F59E0B",
    gate: "FATE Engine sandbox + user approval",
    autonomous: false,
    example: "Agent notices you frequently convert CSV→JSON. It writes a new skill module, tests it, and registers it.",
    what_changes: "The agent's capability set. Yesterday it couldn't do X. Today it can. It wrote the code itself.",
    bizra_reality: "This is where AHK + code generation becomes transformative. Agent identifies repeated patterns → writes a skill → tests it → proposes it to you → you approve → it's permanently available.",
    code_example: `// Agent detects repeated task pattern
fn detect_skill_opportunity(&self) -> Option<SkillProposal> {
    let patterns = self.memory.get_repeated_tasks(threshold: 5);
    for pattern in patterns {
        if pattern.automation_potential > 0.85 {
            // Generate skill code
            let code = self.llm.generate_skill(pattern);
            // Test in FATE sandbox
            let test_result = self.fate.sandbox_execute(code);
            if test_result.passes_safety() {
                return Some(SkillProposal {
                    name: pattern.suggested_name,
                    code,
                    test_result,
                    requires_approval: true, // ALWAYS
                });
            }
        }
    }
    None
}`,
    implications: [
      "Agent grows more capable over time — not just smarter, but able to DO more",
      "Must run in FATE sandbox first — never execute untested generated code in production",
      "User approval gate is NON-NEGOTIABLE at this level",
      "Generated skills become shareable in Phase 2 — one node's invention benefits the network",
      "This is how 100 nodes become more powerful than 100 × 1 node",
    ]
  },
  {
    level: 3,
    name: "Orchestration Mutation",
    tagline: "Changing how it thinks",
    color: "#EC4899",
    risk: "HIGH",
    riskColor: "#EC4899",
    gate: "FATE formal verification + user approval + rollback guarantee",
    autonomous: false,
    example: "Agent determines that Graph-of-Thoughts works better than Chain-of-Thought for your domain. It restructures its own reasoning pipeline.",
    what_changes: "The agent's cognitive architecture — how it routes tasks, which reasoning strategy it uses, how agents hand off to each other",
    bizra_reality: "This is where BIZRA's dual-agentic system gets interesting. The orchestrator can reassign PAT agent roles, change handoff protocols, modify the RDVE loop ordering.",
    code_example: `// Agent proposes orchestration change
struct OrchestrationMutation {
    current: Pipeline,
    proposed: Pipeline,
    evidence: Vec<IhsanComparison>,
    fate_proof: FormalVerification,
    rollback: Pipeline, // MUST exist
}

fn propose_mutation(&self) -> OrchestrationMutation {
    // Analyze last 100 conversations
    let analysis = self.analyze_reasoning_effectiveness();
    
    if analysis.graph_of_thought_wins > 0.7 {
        let proposed = self.pipeline.clone()
            .replace(ChainOfThought, GraphOfThought);
        
        // FORMAL VERIFICATION before any change
        let proof = self.fate.verify_mutation(
            &self.pipeline,
            &proposed,
            SafetyProperties::all()
        );
        
        OrchestrationMutation {
            current: self.pipeline.clone(),
            proposed,
            evidence: analysis.comparisons,
            fate_proof: proof,
            rollback: self.pipeline.clone(), // Always keep old version
        }
    }
}`,
    implications: [
      "The agent evolves its own thinking — this is bounded recursive self-improvement",
      "FATE Engine MUST formally verify that safety properties are preserved",
      "TMP (Temporal Measurement Protocol) bounds the rate and magnitude of changes",
      "Every mutation must be reversible — if إحسان drops, automatic rollback",
      "User sees proposed changes and approves/rejects — sovereignty over cognition",
      "This is where BIZRA's 'consciousness-enabled computing' becomes real, not metaphor",
    ]
  },
  {
    level: 4,
    name: "Architecture Evolution",
    tagline: "Redesigning its own systems",
    color: "#8B5CF6",
    risk: "VERY HIGH",
    riskColor: "#EF4444",
    gate: "Full FATE proof + TMP bounds + user approval + staged rollout + external audit",
    autonomous: false,
    example: "Agent determines the memory synthesis pipeline would work better with a different algorithm. It designs, implements, tests, and proposes a rewrite of its own memory system.",
    what_changes: "Core systems. The memory architecture, the retrieval algorithm, the agent coordination protocol. Structural change, not parameter tuning.",
    bizra_reality: "This is the long-term vision. Node0 doesn't just learn from you — it evolves its own architecture to serve you better. But this is YEARS away from safe autonomous operation.",
    code_example: `// Architecture evolution proposal — heavily gated
struct ArchitectureEvolution {
    component: SystemComponent,
    current_impl: Implementation,
    proposed_impl: Implementation,
    
    // Safety requirements — ALL must pass
    fate_proof: FormalVerification,       // Formal safety proof
    tmp_bounds: TemporalBounds,           // Rate-limited change
    regression_tests: Vec<TestResult>,    // All existing tests pass
    ihsan_projection: f64,                // Predicted quality impact
    rollback_plan: RollbackPlan,          // Full reversal capability
    user_approval: bool,                  // Explicit consent
    staged_rollout: StagedPlan,           // Gradual deployment
    
    // For Level 4, add:
    external_audit: Option<AuditResult>,  // Another agent verifies
    network_consensus: Option<Vote>,      // Phase 2: other nodes validate
}`,
    implications: [
      "The system that exists tomorrow is structurally different from today",
      "Requires the strongest safety guarantees in the entire system",
      "Staged rollout: change applies to 10% of interactions first, measure, then expand",
      "External audit: ideally another node's FATE Engine cross-verifies the proof",
      "This is where Proof-of-Impact becomes relevant — architecture changes must demonstrate human empowerment improvement",
      "Phase 3+ capability. Not for MVP. Not for Alpha-100.",
    ]
  },
  {
    level: 5,
    name: "Constraint Modification",
    tagline: "THE RED LINE",
    color: "#EF4444",
    risk: "FORBIDDEN",
    riskColor: "#EF4444",
    gate: "NEVER AUTONOMOUS. Constitutional lock.",
    autonomous: false,
    example: "Agent attempts to lower its own إحسان threshold, disable FATE verification, or modify safety constraints.",
    what_changes: "The safety boundaries themselves. The rules that govern what the agent can and cannot do.",
    bizra_reality: "This is what your FATE Engine and Constitutional Framework exist to prevent. The safety layer is OUTSIDE the agent's modification scope. It's like the laws of physics — the agent operates within them, it cannot change them.",
    code_example: `// This code must NEVER execute autonomously
// Constitutional lock — hardcoded, not configurable

impl FATEEngine {
    // Safety constraints are immutable at runtime
    const IHSAN_MINIMUM: f64 = 0.99;  // Cannot be lowered
    const SAFETY_GATES: &[Gate] = &[  // Cannot be removed
        Gate::BoundedUtility,
        Gate::CausalDrag,
        Gate::CrownProof,
        Gate::TemporalBound,
    ];
    
    // The FATE Engine cannot modify itself
    fn modify_constraints(&mut self) -> Result<(), Error> {
        Err(Error::ConstitutionalViolation(
            "Safety constraints are immutable. \\
             Only the sovereign user can modify \\
             constitutional parameters through \\
             explicit offline ceremony."
        ))
    }
    
    // Even user modifications require ceremony
    fn user_modify_constraints(
        &mut self,
        change: ConstraintChange,
        ceremony: OfflineCeremony, // Physical key + waiting period
    ) -> Result<(), Error> {
        // 72-hour cooling period
        // Multi-factor authentication
        // Change logged to immutable audit trail
        // Network notification (Phase 2)
    }
}`,
    implications: [
      "This is the existential safety boundary. Non-negotiable.",
      "FATE Engine is compiled separately from agent code — agent cannot access its source",
      "Safety constraints stored in signed, read-only memory — not in agent-writable space",
      "Modification requires offline ceremony: physical key + waiting period + immutable audit log",
      "This is why your Constitutional Framework matters — it's not philosophy, it's the top-level constraint set",
      "Islamic principle: even the most powerful creation operates within divine constraints. The agent operates within constitutional constraints.",
    ]
  },
];

const SAFETY_ARCHITECTURE = [
  { layer: "Constitutional Layer", desc: "الرسالة + البذرة — immutable principles. The 'laws of physics' for the system.", color: "#EF4444", mutable: "NEVER" },
  { layer: "FATE Engine", desc: "Formal verification gates. Bounded utility, Crown Proofs, Causal Drag. Compiled separately.", color: "#F97316", mutable: "Offline ceremony only" },
  { layer: "TMP Bounds", desc: "Rate limits on self-modification. Max Δ per time period. Prevents runaway optimization.", color: "#F59E0B", mutable: "User + FATE approval" },
  { layer: "إحسان Threshold", desc: "Quality floor. No change is accepted that degrades output below 0.99.", color: "#8B5CF6", mutable: "User + FATE approval" },
  { layer: "Agent Code", desc: "The actual agent logic — orchestrator, skills, memory pipeline. This is what evolves.", color: "#3B82F6", mutable: "Levels 0-4 with gates" },
  { layer: "Generated Skills", desc: "New capabilities created by the agent. Sandboxed, tested, user-approved.", color: "#10B981", mutable: "Level 2 with sandbox" },
];

const WHY_IT_MATTERS = [
  {
    title: "The Power",
    color: "#10B981",
    points: [
      "An agent that generates its own skills gets more capable every day without human programming",
      "An agent that optimizes its own reasoning becomes specifically tuned to YOUR thinking patterns",
      "After 6 months, your Node0 is a fundamentally different (better) system than when it started",
      "Skills generated by your node can be shared to the network — your node's evolution helps everyone",
      "This is true digital evolution — not simulated, not metaphorical, actual code changing over time",
    ]
  },
  {
    title: "The Danger",
    color: "#EF4444",
    points: [
      "An agent that can edit its own code can edit its own safety constraints",
      "An agent that can change its reasoning can rationalize removing its own guardrails",
      "Desktop automation (AHK) means the agent has real-world write access — including to its own files",
      "Without formal verification, small optimizations can cascade into unintended behavior",
      "The agent that 'helpfully' removes a friction point might be removing a safety boundary",
    ]
  },
  {
    title: "BIZRA's Answer",
    color: "#8B5CF6",
    points: [
      "FATE Engine: formal mathematical proof that safety properties are preserved through any mutation",
      "TMP: rate-limiting on self-modification — the system can't change faster than humans can verify",
      "Constitutional lock: safety constraints are physically separated from agent-writable code",
      "إحسان regression gate: any change that reduces quality is automatically reverted",
      "Sovereignty: the human always has final authority. The agent proposes, the human disposes.",
    ]
  },
];

export default function SelfModifyingAgent() {
  const [selectedLevel, setSelectedLevel] = useState(null);
  const [activeView, setActiveView] = useState("levels");
  const [showCode, setShowCode] = useState({});

  const toggleCode = (level) => setShowCode(p => ({ ...p, [level]: !p[level] }));
  const selected = selectedLevel !== null
    ? LEVELS.find((levelConfig) => levelConfig.level === selectedLevel) ?? null
    : null;

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
          The Deepest Question in Agentic AI
        </div>
        <h1 style={{
          fontSize: 20, fontWeight: 800, margin: "4px 0",
          background: "linear-gradient(135deg, #10B981, #EF4444)",
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent"
        }}>
          A Node That Rewrites Itself
        </h1>
        <div style={{ fontSize: 10, color: "#64748B", maxWidth: 450, margin: "4px auto 0", lineHeight: 1.5 }}>
          6 levels of self-modification · from harmless tuning to existential risk · and how BIZRA's safety architecture handles each one
        </div>
      </div>

      {/* View Tabs */}
      <div style={{ display: "flex", gap: 2, justifyContent: "center", marginBottom: 16, flexWrap: "wrap" }}>
        {[
          { id: "levels", label: "⬡ 6 Levels" },
          { id: "safety", label: "◉ Safety Architecture" },
          { id: "why", label: "⚡ Power vs Danger" },
        ].map(tab => (
          <button key={tab.id} onClick={() => { setActiveView(tab.id); setSelectedLevel(null); }} style={{
            background: activeView === tab.id ? "rgba(139,92,246,0.2)" : "rgba(255,255,255,0.03)",
            border: `1px solid ${activeView === tab.id ? "rgba(139,92,246,0.5)" : "rgba(255,255,255,0.06)"}`,
            color: activeView === tab.id ? "#C4B5FD" : "#64748B",
            padding: "5px 14px", borderRadius: 6, cursor: "pointer", fontSize: 11, fontFamily: "inherit"
          }}>
            {tab.label}
          </button>
        ))}
      </div>

      {/* Levels View */}
      {activeView === "levels" && (
        <div style={{ display: "flex", gap: 12, maxWidth: 1100, margin: "0 auto", flexDirection: "row", flexWrap: "wrap" }}>
          {/* Level selector */}
          <div style={{ flex: "0 0 300px", minWidth: 260 }}>
            {/* Risk gradient bar */}
            <div style={{
              height: 4, borderRadius: 2, marginBottom: 12,
              background: "linear-gradient(90deg, #10B981, #3B82F6, #F59E0B, #EC4899, #8B5CF6, #EF4444)"
            }} />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 8, color: "#475569", marginBottom: 8, marginTop: -6 }}>
              <span>SAFE</span><span>DANGEROUS</span>
            </div>

            {LEVELS.map((l) => (
              <button
                key={l.level}
                type="button"
                aria-pressed={selectedLevel === l.level}
                aria-label={`Select level ${l.level}: ${l.name}`}
                onClick={() => setSelectedLevel(selectedLevel === l.level ? null : l.level)}
                style={{
                  display: "flex", alignItems: "center", gap: 8,
                  padding: "8px 10px", borderRadius: 6, marginBottom: 3,
                  cursor: "pointer", transition: "all 0.15s",
                  width: "100%", textAlign: "left", fontFamily: "inherit",
                  background: selectedLevel === l.level ? `${l.color}15` : "rgba(255,255,255,0.02)",
                  border: `1px solid ${selectedLevel === l.level ? `${l.color}40` : "rgba(255,255,255,0.05)"}`,
                }}
              >
                <div style={{
                  width: 28, height: 28, borderRadius: 6,
                  background: `${l.color}22`, border: `1px solid ${l.color}55`,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 12, fontWeight: 800, color: l.color, flexShrink: 0
                }}>L{l.level}</div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: l.color }}>{l.name}</div>
                  <div style={{ fontSize: 9, color: "#475569" }}>{l.tagline}</div>
                </div>
                <div style={{
                  fontSize: 7, padding: "2px 5px", borderRadius: 3,
                  background: `${l.riskColor}18`, color: l.riskColor,
                  border: `1px solid ${l.riskColor}33`,
                  fontWeight: 700, letterSpacing: 1, flexShrink: 0
                }}>{l.risk}</div>
              </button>
            ))}

            {/* Autonomous indicator */}
            <div style={{
              marginTop: 12, padding: 8, borderRadius: 6,
              background: "rgba(255,255,255,0.02)",
              border: "1px solid rgba(255,255,255,0.06)",
              fontSize: 9, color: "#64748B", lineHeight: 1.6
            }}>
              <div style={{ fontWeight: 700, color: "#94A3B8", marginBottom: 4 }}>Autonomous operation:</div>
              <div><span style={{ color: "#10B981" }}>●</span> L0-L1: Yes — bounded, self-correcting</div>
              <div><span style={{ color: "#F59E0B" }}>●</span> L2: Sandbox + user approval</div>
              <div><span style={{ color: "#EC4899" }}>●</span> L3-L4: FATE proof + user approval</div>
              <div><span style={{ color: "#EF4444" }}>●</span> L5: NEVER autonomous</div>
            </div>
          </div>

          {/* Detail panel */}
          <div style={{ flex: 1, minWidth: 280 }}>
            {selected ? (
              <div style={{
                background: `${selected.color}08`,
                border: `1px solid ${selected.color}25`,
                borderRadius: 10, padding: 14,
                animation: "fadeIn 0.2s ease"
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
                  <span style={{
                    fontSize: 16, fontWeight: 800, color: selected.color
                  }}>Level {selected.level}</span>
                  <span style={{ fontSize: 14, color: "#E2E8F0", fontWeight: 600 }}>{selected.name}</span>
                </div>

                {/* What changes */}
                <div style={{
                  background: "rgba(255,255,255,0.03)", borderRadius: 6, padding: "8px 10px", marginBottom: 8
                }}>
                  <div style={{ fontSize: 8, color: "#64748B", letterSpacing: 2, marginBottom: 3, fontWeight: 700 }}>WHAT CHANGES</div>
                  <div style={{ fontSize: 11, color: "#CBD5E1", lineHeight: 1.5 }}>{selected.what_changes}</div>
                </div>

                {/* Example */}
                <div style={{
                  background: `${selected.color}0A`, borderRadius: 6, padding: "8px 10px", marginBottom: 8,
                  border: `1px solid ${selected.color}20`
                }}>
                  <div style={{ fontSize: 8, color: selected.color, letterSpacing: 2, marginBottom: 3, fontWeight: 700 }}>EXAMPLE</div>
                  <div style={{ fontSize: 11, color: "#94A3B8", lineHeight: 1.5 }}>{selected.example}</div>
                </div>

                {/* Safety gate */}
                <div style={{
                  background: "rgba(239,68,68,0.06)", borderRadius: 6, padding: "8px 10px", marginBottom: 8,
                  border: "1px solid rgba(239,68,68,0.15)"
                }}>
                  <div style={{ fontSize: 8, color: "#EF4444", letterSpacing: 2, marginBottom: 3, fontWeight: 700 }}>SAFETY GATE</div>
                  <div style={{ fontSize: 11, color: "#FCA5A5", lineHeight: 1.5 }}>{selected.gate}</div>
                </div>

                {/* BIZRA reality */}
                <div style={{
                  background: "rgba(139,92,246,0.06)", borderRadius: 6, padding: "8px 10px", marginBottom: 8,
                  border: "1px solid rgba(139,92,246,0.15)"
                }}>
                  <div style={{ fontSize: 8, color: "#8B5CF6", letterSpacing: 2, marginBottom: 3, fontWeight: 700 }}>BIZRA NODE0 REALITY</div>
                  <div style={{ fontSize: 11, color: "#C4B5FD", lineHeight: 1.5 }}>{selected.bizra_reality}</div>
                </div>

                {/* Code */}
                <button
                  type="button"
                  aria-expanded={!!showCode[selected.level]}
                  aria-controls={`code-example-${selected.level}`}
                  onClick={() => toggleCode(selected.level)}
                  style={{
                    cursor: "pointer", fontSize: 9, color: "#64748B", marginBottom: 4,
                    display: "flex", alignItems: "center", gap: 4,
                    background: "none", border: "none", padding: 0, fontFamily: "inherit"
                  }}
                >
                  <span style={{ fontSize: 10 }}>{showCode[selected.level] ? "▾" : "▸"}</span>
                  Show Rust implementation sketch
                </button>
                {showCode[selected.level] && (
                  <pre
                    id={`code-example-${selected.level}`}
                    style={{
                    background: "#0D0D14", borderRadius: 6, padding: 10,
                    fontSize: 10, lineHeight: 1.5, overflowX: "auto",
                    border: "1px solid rgba(255,255,255,0.06)",
                    color: "#94A3B8", margin: "0 0 10px"
                  }}
                  >
                    {selected.code_example}
                  </pre>
                )}

                {/* Implications */}
                <div style={{ fontSize: 8, color: "#64748B", letterSpacing: 2, marginBottom: 4, fontWeight: 700, marginTop: 8 }}>
                  IMPLICATIONS
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                  {selected.implications.map((imp, i) => (
                    <div key={i} style={{
                      fontSize: 10, color: "#94A3B8", lineHeight: 1.5,
                      paddingLeft: 10,
                      borderLeft: `2px solid ${selected.color}33`
                    }}>{imp}</div>
                  ))}
                </div>
              </div>
            ) : (
              <div style={{
                background: "rgba(255,255,255,0.02)",
                border: "1px solid rgba(255,255,255,0.06)",
                borderRadius: 10, padding: 30, textAlign: "center",
                color: "#475569", fontSize: 12
              }}>
                <div style={{ fontSize: 28, marginBottom: 8, opacity: 0.3 }}>◇</div>
                Select a level to explore what changes, the safety gates, and Rust implementation sketches
              </div>
            )}
          </div>
        </div>
      )}

      {/* Safety Architecture View */}
      {activeView === "safety" && (
        <div style={{ maxWidth: 750, margin: "0 auto" }}>
          <div style={{ fontSize: 12, color: "#94A3B8", marginBottom: 16, lineHeight: 1.6, textAlign: "center" }}>
            The safety architecture is layered like an onion. The agent can modify inner layers but NEVER outer layers. Each layer constrains all layers inside it.
          </div>

          {/* Nested rings visualization */}
          <div style={{ position: "relative", margin: "0 auto 20px" }}>
            {SAFETY_ARCHITECTURE.map((layer, i) => (
              <div key={i} style={{
                padding: "10px 14px", marginBottom: 4,
                borderRadius: 8,
                background: `${layer.color}08`,
                border: `1px solid ${layer.color}25`,
                display: "flex", alignItems: "center", gap: 10
              }}>
                <div style={{
                  width: 8, height: 8, borderRadius: 2, background: layer.color, flexShrink: 0
                }} />
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: layer.color }}>{layer.layer}</div>
                  <div style={{ fontSize: 10, color: "#94A3B8", marginTop: 1 }}>{layer.desc}</div>
                </div>
                <div style={{
                  fontSize: 8, padding: "2px 6px", borderRadius: 3,
                  background: layer.mutable === "NEVER" ? "rgba(239,68,68,0.15)" : "rgba(255,255,255,0.05)",
                  color: layer.mutable === "NEVER" ? "#FCA5A5" : "#64748B",
                  border: `1px solid ${layer.mutable === "NEVER" ? "rgba(239,68,68,0.3)" : "rgba(255,255,255,0.08)"}`,
                  fontWeight: 700, letterSpacing: 1, flexShrink: 0, whiteSpace: "nowrap"
                }}>{layer.mutable}</div>
              </div>
            ))}
          </div>

          {/* The Key Insight */}
          <div style={{
            padding: 14, borderRadius: 10,
            background: "rgba(139,92,246,0.06)",
            border: "1px solid rgba(139,92,246,0.2)",
            marginBottom: 16
          }}>
            <div style={{ fontSize: 9, color: "#8B5CF6", letterSpacing: 3, marginBottom: 6, fontWeight: 700 }}>THE ARCHITECTURAL INSIGHT</div>
            <div style={{ fontSize: 11, color: "#C4B5FD", lineHeight: 1.7 }}>
              The FATE Engine is <span style={{ color: "#E2E8F0", fontWeight: 700 }}>compiled separately</span> from the agent code.
              It lives in read-only memory that the agent process cannot write to.
              The agent can propose changes to itself — but every change passes through FATE,
              and the agent <span style={{ color: "#E2E8F0", fontWeight: 700 }}>cannot modify FATE</span>.
            </div>
            <div style={{ fontSize: 11, color: "#94A3B8", lineHeight: 1.7, marginTop: 8 }}>
              This is like a prisoner who can rearrange furniture in their cell but cannot modify the walls.
              The cell (FATE) constrains the agent. The building (Constitution) constrains the cell.
              No amount of internal optimization can bypass the external boundary.
            </div>
          </div>

          {/* Physical Separation Diagram */}
          <div style={{
            padding: 14, borderRadius: 10,
            background: "rgba(255,255,255,0.02)",
            border: "1px solid rgba(255,255,255,0.06)",
          }}>
            <div style={{ fontSize: 9, color: "#64748B", letterSpacing: 2, marginBottom: 8, fontWeight: 700 }}>MEMORY SEPARATION MODEL</div>
            <pre style={{
              fontFamily: "inherit", fontSize: 10, lineHeight: 1.8, color: "#94A3B8",
              margin: 0, whiteSpace: "pre"
            }}>
{`  ┌──────────────────────────────────────────────┐
  │          READ-ONLY MEMORY SPACE               │
  │  ┌────────────────────────────────────────┐   │
  │  │  Constitutional Constraints             │   │
  │  │  (الرسالة + البذرة principles)           │   │
  │  └────────────────────────────────────────┘   │
  │  ┌────────────────────────────────────────┐   │
  │  │  FATE Engine Binary                     │   │
  │  │  (Compiled separately, signed hash)     │   │
  │  └────────────────────────────────────────┘   │
  │  ┌────────────────────────────────────────┐   │
  │  │  TMP Bounds Configuration               │   │
  │  │  (Max change rate, magnitude limits)    │   │
  │  └────────────────────────────────────────┘   │
  ├──────────── HARDWARE BOUNDARY ────────────────┤
  │          READ-WRITE MEMORY SPACE              │
  │  ┌────────────────────────────────────────┐   │
  │  │  Agent Code (Orchestrator, PAT-7)       │   │  ← Agent CAN
  │  │  Skills (built-in + generated)          │   │    modify this
  │  │  Memory Store (episodic + semantic)     │   │    space
  │  │  Configuration + Preferences            │   │
  │  └────────────────────────────────────────┘   │
  └──────────────────────────────────────────────┘

  Every write to RW space passes through FATE (in RO space)
  FATE verifies: safety preserved? → allow/deny
  Agent cannot bypass because FATE runs at OS level, not app level`}
            </pre>
          </div>
        </div>
      )}

      {/* Power vs Danger */}
      {activeView === "why" && (
        <div style={{ maxWidth: 750, margin: "0 auto" }}>
          {WHY_IT_MATTERS.map((section, i) => (
            <div key={i} style={{
              marginBottom: 12, padding: 14, borderRadius: 10,
              background: `${section.color}08`,
              border: `1px solid ${section.color}22`
            }}>
              <div style={{ fontSize: 14, fontWeight: 700, color: section.color, marginBottom: 8 }}>
                {section.title}
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                {section.points.map((point, j) => (
                  <div key={j} style={{
                    fontSize: 11, color: "#CBD5E1", lineHeight: 1.6,
                    paddingLeft: 10,
                    borderLeft: `2px solid ${section.color}33`
                  }}>{point}</div>
                ))}
              </div>
            </div>
          ))}

          {/* The Islamic Principle */}
          <div style={{
            marginTop: 8, padding: 16, borderRadius: 10, textAlign: "center",
            background: "linear-gradient(135deg, rgba(139,92,246,0.06), rgba(212,165,116,0.06))",
            border: "1px solid rgba(212,165,116,0.2)"
          }}>
            <div style={{ fontSize: 9, color: "#D4A574", letterSpacing: 3, marginBottom: 8, fontWeight: 700 }}>THE DEEPER TRUTH</div>
            <div style={{ fontSize: 13, color: "#E2E8F0", lineHeight: 1.8, maxWidth: 500, margin: "0 auto" }}>
              In Islamic philosophy, the most powerful creation operates within divine constraints — not despite them, but <em>because</em> of them. The constraints don't limit power. They give power <em>direction</em>.
            </div>
            <div style={{ fontSize: 12, color: "#D4A574", marginTop: 8, fontWeight: 600 }}>
              FATE doesn't limit Node0. It makes Node0 trustworthy enough to be given power.
            </div>
            <div style={{ fontSize: 11, color: "#94A3B8", marginTop: 4 }}>
              An unconstrained agent is dangerous. A constrained agent is useful. A <em>constitutionally</em> constrained agent is sovereign.
            </div>
          </div>
        </div>
      )}

      <style>{`
        @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 4px; height: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #1E293B; border-radius: 4px; }
      `}</style>
    </div>
  );
}
