import { useState, useEffect, useRef } from "react";

const LAYERS = [
  {
    id: "ui",
    name: "UI / App Shell",
    subtitle: "What You See & Touch",
    color: "#0EA5E9",
    bgColor: "rgba(14,165,233,0.06)",
    borderColor: "rgba(14,165,233,0.25)",
    bizraFocus: "low",
    components: [
      { name: "Chat Interface", desc: "Message input, response display, streaming output", type: "core" },
      { name: "Artifact Renderer", desc: "Renders code, React, HTML, SVG, Mermaid inline", type: "core" },
      { name: "Tool Result UI", desc: "Maps, charts, calendars, file viewers — tool outputs rendered as widgets", type: "core" },
      { name: "Memory Indicators", desc: "Shows what the agent remembers, memory confidence signals", type: "bizra" },
    ],
    hooks: [
      { name: "onUserInput", desc: "Fires before message sent — can intercept, transform, attach context" },
      { name: "onStreamChunk", desc: "Fires on each token — enables real-time UI updates, typing indicators" },
      { name: "onToolResult", desc: "Fires when tool returns — triggers UI widget rendering" },
      { name: "onResponseComplete", desc: "Fires after full response — triggers memory extraction pipeline" },
    ],
    explanation: "This is where you interact. The UI is a thin shell — it sends messages and renders responses. The real intelligence is below. For BIZRA Node0, this becomes your sovereign interface — no cloud dependency."
  },
  {
    id: "hooks",
    name: "Hooks & Lifecycle System",
    subtitle: "The Nervous System",
    color: "#F59E0B",
    bgColor: "rgba(245,158,11,0.06)",
    borderColor: "rgba(245,158,11,0.25)",
    bizraFocus: "critical",
    components: [
      { name: "Pre-Processing Hooks", desc: "Run BEFORE the LLM sees your message — memory retrieval, context injection, intent classification", type: "bizra" },
      { name: "Mid-Processing Hooks", desc: "Run DURING generation — tool calls, skill activation, MCP routing", type: "core" },
      { name: "Post-Processing Hooks", desc: "Run AFTER response — memory extraction, إحسان scoring, conversation logging", type: "bizra" },
      { name: "Event Bus", desc: "Pub/sub system connecting all layers — any component can emit/listen to events", type: "bizra" },
    ],
    hooks: [
      { name: "beforeLLMCall", desc: "Assembles final prompt from memory + context + user input — THE critical hook" },
      { name: "onToolCall", desc: "Intercepts tool invocations — can route to MCP, local skills, or A2A" },
      { name: "afterLLMCall", desc: "Extracts new memories, scores quality, updates user model" },
      { name: "onMemoryUpdate", desc: "Fires when semantic memory changes — cascading updates" },
    ],
    explanation: "THIS IS WHERE BIZRA SHOULD FOCUS MOST. Hooks are the nervous system — they control what happens before, during, and after every interaction. The hook system is what transforms a stateless LLM into a persistent agent. Without hooks, you have a chatbot. With hooks, you have a partner."
  },
  {
    id: "orchestrator",
    name: "Orchestrator / Agent Runtime",
    subtitle: "The Brain",
    color: "#8B5CF6",
    bgColor: "rgba(139,92,246,0.06)",
    borderColor: "rgba(139,92,246,0.25)",
    bizraFocus: "critical",
    components: [
      { name: "Context Budget Manager", desc: "Allocates 200K token window: identity (5K) + memories (40K) + tools (10K) + conversation (rest)", type: "bizra" },
      { name: "Intent Router", desc: "Classifies user intent → routes to appropriate skill/tool/agent", type: "core" },
      { name: "PAT-7 Coordinator", desc: "Manages your 7 Personal Agent Team — each agent has specialty + shared memory", type: "bizra" },
      { name: "Tool Orchestration", desc: "Decides which tools to call, in what order, handles multi-step chains", type: "core" },
    ],
    hooks: [
      { name: "onIntentClassified", desc: "After intent detection — triggers appropriate agent from PAT-7" },
      { name: "onBudgetExceeded", desc: "When context window fills — triggers compression/pruning" },
      { name: "onAgentHandoff", desc: "When one PAT agent delegates to another — maintains shared state" },
      { name: "onChainComplete", desc: "When multi-tool chain finishes — aggregates results" },
    ],
    explanation: "Your GenesisOrchestrator lives here. This is the brain that coordinates everything. It doesn't reason — it manages the reasoning process. It decides what the LLM sees, which tools to use, which agent handles the task, and how to budget the context window."
  },
  {
    id: "skills",
    name: "Skills / Capabilities Layer",
    subtitle: "What The Agent Can DO",
    color: "#10B981",
    bgColor: "rgba(16,185,129,0.06)",
    borderColor: "rgba(16,185,129,0.25)",
    bizraFocus: "medium",
    components: [
      { name: "Built-in Skills", desc: "File creation (docx/pptx/xlsx), code execution, web search, image gen — native capabilities", type: "core" },
      { name: "Custom Skills", desc: "User-defined capability modules with SKILL.md specs — your extensions", type: "bizra" },
      { name: "Skill Registry", desc: "Catalog of available skills + metadata + activation conditions", type: "core" },
      { name: "AHK Bridge (Node0)", desc: "BIZRA's secret weapon — desktop automation skills via AutoHotkey execution", type: "bizra" },
    ],
    hooks: [
      { name: "onSkillActivated", desc: "Logs which skill was used — feeds procedural memory" },
      { name: "onSkillResult", desc: "Captures output — can trigger further skills or memory updates" },
      { name: "onSkillError", desc: "Graceful degradation — tries alternative skill or reports limitation" },
    ],
    explanation: "Skills are self-contained capability modules. Each skill has a SKILL.md definition file that tells the agent HOW to use it. The key insight: skills are composable. The agent can chain multiple skills together. Your AHK bridge is unique — no other system can execute real desktop actions."
  },
  {
    id: "mcp",
    name: "MCP (Model Context Protocol)",
    subtitle: "Universal Tool Connector",
    color: "#EC4899",
    bgColor: "rgba(236,72,153,0.06)",
    borderColor: "rgba(236,72,153,0.25)",
    bizraFocus: "high",
    components: [
      { name: "MCP Client", desc: "Lives in the agent — sends requests to MCP servers using standardized protocol", type: "core" },
      { name: "MCP Servers", desc: "External services exposed as tools — Notion, Asana, GitHub, Slack, your own services", type: "core" },
      { name: "Tool Discovery", desc: "Agent auto-discovers available tools from connected MCP servers at runtime", type: "core" },
      { name: "BIZRA MCP Server", desc: "Your own MCP server exposing Node0 capabilities to any MCP-compatible agent", type: "bizra" },
    ],
    hooks: [
      { name: "onMCPConnect", desc: "When new MCP server connects — registers tools into skill registry" },
      { name: "onMCPToolCall", desc: "Routes tool call through MCP protocol to external server" },
      { name: "onMCPResponse", desc: "Processes response — can transform before returning to agent" },
    ],
    explanation: "MCP is Anthropic's open standard for connecting AI to external tools. Think of it as USB-C for AI — one protocol, any service. BIZRA should implement its own MCP server so Node0 can be connected to ANY MCP-compatible agent. This is how you federate without dependency."
  },
  {
    id: "a2a",
    name: "A2A (Agent-to-Agent Protocol)",
    subtitle: "Agent Communication Highway",
    color: "#F97316",
    bgColor: "rgba(249,115,22,0.06)",
    borderColor: "rgba(249,115,22,0.25)",
    bizraFocus: "high",
    components: [
      { name: "Agent Card", desc: "JSON-LD identity document — declares agent capabilities, auth, endpoints", type: "core" },
      { name: "Task Protocol", desc: "Structured request/response for delegating tasks between agents", type: "core" },
      { name: "PAT ↔ SAT Bridge", desc: "BIZRA-specific: how your personal agents communicate with system agents in the URP", type: "bizra" },
      { name: "Capability Negotiation", desc: "Agents discover what each other can do before delegating tasks", type: "core" },
    ],
    hooks: [
      { name: "onAgentDiscovered", desc: "When new agent found on network — evaluates capabilities" },
      { name: "onTaskDelegated", desc: "When task sent to external agent — tracks state + timeout" },
      { name: "onTaskCompleted", desc: "When delegated task returns — integrates result into context" },
    ],
    explanation: "A2A is Google's open protocol for agents talking to agents. MCP connects agents to TOOLS. A2A connects agents to OTHER AGENTS. For BIZRA Phase 2: this is how Node0's PAT-7 talks to SAT agents in the Universal Resource Pool. Each node's PAT agents can delegate to specialized SAT agents across the network."
  },
  {
    id: "memory",
    name: "Memory Stack",
    subtitle: "The Soul of Persistence",
    color: "#6366F1",
    bgColor: "rgba(99,102,241,0.06)",
    borderColor: "rgba(99,102,241,0.25)",
    bizraFocus: "critical",
    components: [
      { name: "Procedural Memory", desc: "Learned behaviors — how to work with you, your commands, your preferences", type: "bizra" },
      { name: "Semantic Memory", desc: "Who you are — distilled identity model, evolving beliefs, compressed understanding", type: "bizra" },
      { name: "Episodic Memory", desc: "What happened — conversation logs, decisions, milestones, indexed + vectorized", type: "bizra" },
      { name: "Working Memory", desc: "Right now — the assembled context window for current interaction", type: "core" },
    ],
    hooks: [
      { name: "onMemoryExtract", desc: "Post-conversation: extracts facts, updates, patterns from raw transcript" },
      { name: "onMemorySynthesize", desc: "Compresses episodic → semantic: raw events become structured knowledge" },
      { name: "onMemoryRetrieve", desc: "Pre-conversation: pulls relevant memories based on current context" },
      { name: "onMemoryConflict", desc: "When new info contradicts stored memory — triggers resolution" },
    ],
    explanation: "This is where BIZRA's Sovereign Memory Architecture lives. The four memory types work together: procedural shapes HOW the agent responds, semantic shapes WHAT it knows about you, episodic provides EVIDENCE for its knowledge, working memory is the active assembly. The synthesis pipeline between episodic → semantic is the hardest engineering problem and the highest value."
  },
  {
    id: "llm",
    name: "LLM / Foundation Model",
    subtitle: "The Reasoning Engine",
    color: "#64748B",
    bgColor: "rgba(100,116,139,0.06)",
    borderColor: "rgba(100,116,139,0.25)",
    bizraFocus: "low",
    components: [
      { name: "Inference Engine", desc: "The model itself — processes assembled context, generates response tokens", type: "core" },
      { name: "Tool Use Parser", desc: "Detects when model wants to call a tool — extracts tool name + parameters", type: "core" },
      { name: "MOE Router (Node0)", desc: "BIZRA's Mixture-of-Experts — routes to specialized models based on task type", type: "bizra" },
      { name: "HRM (Node0)", desc: "Hierarchical Reasoning Module — multi-step reasoning chains with verification", type: "bizra" },
    ],
    hooks: [],
    explanation: "The LLM is stateless. It knows NOTHING between calls. It is a pure function: tokens in → tokens out. Every bit of 'memory' and 'personality' comes from the layers above. BIZRA's MOE+HRM architecture enhances reasoning quality, but persistence is NOT the LLM's job."
  },
  {
    id: "infra",
    name: "Infrastructure / Storage",
    subtitle: "The Foundation",
    color: "#78716C",
    bgColor: "rgba(120,113,108,0.06)",
    borderColor: "rgba(120,113,108,0.25)",
    bizraFocus: "medium",
    components: [
      { name: "Vector Store", desc: "Embeddings database for semantic search over memories — Qdrant, ChromaDB, or custom", type: "bizra" },
      { name: "Knowledge Graph", desc: "HyperGraphRAG — structured relationships between entities, concepts, decisions", type: "bizra" },
      { name: "Encrypted Storage", desc: "Sovereign data vault — Ed25519 signed, locally encrypted, user-owned", type: "bizra" },
      { name: "Blockchain Layer", desc: "HyperBlockTree/BlockGraph — immutable audit trail, Proof-of-Impact consensus", type: "bizra" },
    ],
    hooks: [
      { name: "onDataWrite", desc: "Every write is encrypted + signed — sovereignty guarantee" },
      { name: "onDataSync", desc: "Phase 2: federated sync with URP — encrypted, selective, user-controlled" },
    ],
    explanation: "Everything above runs on this. For Node0 Phase 1: all storage is local. Your sovereign memory never leaves your device unless you explicitly choose to federate in Phase 2. The vector store enables fast semantic search. The knowledge graph enables structured reasoning over relationships."
  }
];

const PROTOCOL_COMPARISON = [
  { protocol: "Skills", connects: "Agent → Capabilities", direction: "Internal", standard: "Proprietary", example: "File creation, code exec, AHK", bizra: "Custom skill modules" },
  { protocol: "MCP", connects: "Agent → Tools/Services", direction: "Outbound", standard: "Open (Anthropic)", example: "Notion, GitHub, Slack", bizra: "BIZRA MCP Server" },
  { protocol: "A2A", connects: "Agent → Agent", direction: "Bidirectional", standard: "Open (Google)", example: "PAT → SAT delegation", bizra: "PAT-7 ↔ SAT-49 bridge" },
  { protocol: "Plugins", connects: "Agent → Extensions", direction: "Inbound", standard: "Varies", example: "ChatGPT plugins (deprecated)", bizra: "Replaced by MCP+Skills" },
  { protocol: "Hooks", connects: "Layer → Layer", direction: "Internal Bus", standard: "Custom", example: "beforeLLM, afterLLM", bizra: "Core nervous system" },
];

const FOCUS_MAP = [
  { area: "Hook System + Event Bus", priority: "🔴 NOW", reason: "Controls ALL data flow. Without hooks, nothing connects. Build this first.", effort: "2-3 weeks", impact: "Unlocks everything" },
  { area: "Memory Synthesis Pipeline", priority: "🔴 NOW", reason: "Turns raw conversations into structured knowledge. The difference between storage and memory.", effort: "3-4 weeks", impact: "Agent remembers you" },
  { area: "Context Budget Manager", priority: "🔴 NOW", reason: "Decides what the LLM sees. Bad budgeting = agent forgets important context.", effort: "1-2 weeks", impact: "Quality of every response" },
  { area: "BIZRA MCP Server", priority: "🟡 NEXT", reason: "Exposes Node0 to the MCP ecosystem. Any compatible agent can use BIZRA's capabilities.", effort: "2-3 weeks", impact: "Interoperability" },
  { area: "A2A PAT↔SAT Bridge", priority: "🟡 NEXT", reason: "Phase 2 enabler. How personal agents talk to system agents across the network.", effort: "3-4 weeks", impact: "Federation" },
  { area: "AHK Skill Bridge", priority: "🟢 BUILT", reason: "Your unique differentiator. Desktop automation gives BIZRA a closed perception-action loop.", effort: "Refine", impact: "Competitive moat" },
  { area: "HyperGraphRAG", priority: "🟡 NEXT", reason: "Structured knowledge graph for relationship-aware retrieval. Upgrades flat vector search.", effort: "4-6 weeks", impact: "Retrieval quality" },
  { area: "Sovereign Encryption", priority: "🟢 DESIGNED", reason: "Ed25519 + local-first. Architecture exists, needs implementation.", effort: "2 weeks", impact: "Trust & sovereignty" },
];

const focusColors = { "critical": "#EF4444", "high": "#F59E0B", "medium": "#3B82F6", "low": "#6B7280" };
const focusLabels = { "critical": "CRITICAL", "high": "HIGH", "medium": "MEDIUM", "low": "LOW" };

export default function AgenticArchitecture() {
  const [selectedLayer, setSelectedLayer] = useState(null);
  const [activeTab, setActiveTab] = useState("architecture");
  const [hoveredComponent, setHoveredComponent] = useState(null);
  const [expandedHooks, setExpandedHooks] = useState({});

  const toggleHooks = (layerId) => {
    setExpandedHooks(prev => ({ ...prev, [layerId]: !prev[layerId] }));
  };

  const layer = selectedLayer ? LAYERS.find(l => l.id === selectedLayer) : null;

  return (
    <div style={{
      fontFamily: "'JetBrains Mono', 'SF Mono', 'Fira Code', monospace",
      background: "#0A0A0F",
      color: "#E2E8F0",
      minHeight: "100vh",
      padding: "20px",
      boxSizing: "border-box"
    }}>
      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: 24, position: "relative" }}>
        <div style={{ fontSize: 9, letterSpacing: 6, color: "#64748B", textTransform: "uppercase", marginBottom: 4 }}>
          BIZRA Node0 · Agentic System Architecture
        </div>
        <h1 style={{
          fontSize: 22,
          fontWeight: 700,
          margin: 0,
          background: "linear-gradient(135deg, #8B5CF6, #EC4899, #F59E0B)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          letterSpacing: 1
        }}>
          From UI to Backend — Every Layer Mapped
        </h1>
        <div style={{ fontSize: 10, color: "#475569", marginTop: 4 }}>
          Hooks · Skills · MCP · A2A · Memory · Persistence
        </div>
      </div>

      {/* Tab Navigation */}
      <div style={{ display: "flex", gap: 2, marginBottom: 20, justifyContent: "center", flexWrap: "wrap" }}>
        {[
          { id: "architecture", label: "⬡ Architecture Stack" },
          { id: "protocols", label: "⇌ Protocol Comparison" },
          { id: "focus", label: "◎ Action Map" },
          { id: "dataflow", label: "→ Data Flow" },
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => { setActiveTab(tab.id); setSelectedLayer(null); }}
            style={{
              background: activeTab === tab.id ? "rgba(139,92,246,0.2)" : "rgba(255,255,255,0.03)",
              border: `1px solid ${activeTab === tab.id ? "rgba(139,92,246,0.5)" : "rgba(255,255,255,0.06)"}`,
              color: activeTab === tab.id ? "#C4B5FD" : "#64748B",
              padding: "6px 14px",
              borderRadius: 6,
              cursor: "pointer",
              fontSize: 11,
              fontFamily: "inherit",
              transition: "all 0.2s"
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Architecture Stack View */}
      {activeTab === "architecture" && (
        <div style={{ display: "flex", gap: 16, maxWidth: 1100, margin: "0 auto", flexDirection: window.innerWidth < 700 ? "column" : "row" }}>
          {/* Layer Stack */}
          <div style={{ flex: "0 0 380px", minWidth: 280 }}>
            <div style={{ fontSize: 9, color: "#475569", letterSpacing: 3, marginBottom: 8, textTransform: "uppercase" }}>
              ↓ User Input Flows Down · Response Flows Up ↑
            </div>
            {LAYERS.map((l, i) => (
              <div key={l.id} style={{ position: "relative" }}>
                <div
                  onClick={() => setSelectedLayer(selectedLayer === l.id ? null : l.id)}
                  style={{
                    background: selectedLayer === l.id ? l.bgColor : "rgba(255,255,255,0.02)",
                    border: `1px solid ${selectedLayer === l.id ? l.borderColor : "rgba(255,255,255,0.06)"}`,
                    borderRadius: 8,
                    padding: "10px 12px",
                    marginBottom: 4,
                    cursor: "pointer",
                    transition: "all 0.2s",
                    display: "flex",
                    alignItems: "center",
                    gap: 10
                  }}
                >
                  <div style={{
                    width: 32, height: 32, borderRadius: 6,
                    background: `linear-gradient(135deg, ${l.color}22, ${l.color}44)`,
                    border: `1px solid ${l.color}66`,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 11, fontWeight: 700, color: l.color, flexShrink: 0
                  }}>
                    L{i}
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 12, fontWeight: 600, color: l.color }}>{l.name}</div>
                    <div style={{ fontSize: 9, color: "#64748B" }}>{l.subtitle}</div>
                  </div>
                  <div style={{
                    fontSize: 8, padding: "2px 6px", borderRadius: 4,
                    background: `${focusColors[l.bizraFocus]}22`,
                    color: focusColors[l.bizraFocus],
                    border: `1px solid ${focusColors[l.bizraFocus]}44`,
                    fontWeight: 700, letterSpacing: 1, flexShrink: 0
                  }}>
                    {focusLabels[l.bizraFocus]}
                  </div>
                </div>
                {i < LAYERS.length - 1 && (
                  <div style={{ textAlign: "center", color: "#1E293B", fontSize: 10, lineHeight: 1, margin: "-1px 0" }}>↕</div>
                )}
              </div>
            ))}
          </div>

          {/* Detail Panel */}
          <div style={{ flex: 1, minWidth: 0 }}>
            {layer ? (
              <div style={{
                background: layer.bgColor,
                border: `1px solid ${layer.borderColor}`,
                borderRadius: 12,
                padding: 16,
                animation: "fadeIn 0.2s ease"
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
                  <div style={{
                    width: 10, height: 10, borderRadius: 3,
                    background: layer.color
                  }} />
                  <h2 style={{ fontSize: 16, fontWeight: 700, color: layer.color, margin: 0 }}>
                    {layer.name}
                  </h2>
                </div>

                <p style={{ fontSize: 11, color: "#94A3B8", lineHeight: 1.6, margin: "0 0 16px" }}>
                  {layer.explanation}
                </p>

                {/* Components */}
                <div style={{ fontSize: 9, color: "#64748B", letterSpacing: 2, marginBottom: 6, textTransform: "uppercase" }}>
                  Components
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 4, marginBottom: 16 }}>
                  {layer.components.map((c, i) => (
                    <div
                      key={i}
                      onMouseEnter={() => setHoveredComponent(`${layer.id}-${i}`)}
                      onMouseLeave={() => setHoveredComponent(null)}
                      style={{
                        background: hoveredComponent === `${layer.id}-${i}` ? "rgba(255,255,255,0.06)" : "rgba(255,255,255,0.02)",
                        border: `1px solid ${c.type === "bizra" ? `${layer.color}33` : "rgba(255,255,255,0.05)"}`,
                        borderRadius: 6,
                        padding: "8px 10px",
                        transition: "all 0.15s"
                      }}
                    >
                      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        <span style={{ fontSize: 11, fontWeight: 600, color: "#E2E8F0" }}>{c.name}</span>
                        {c.type === "bizra" && (
                          <span style={{
                            fontSize: 7, padding: "1px 5px", borderRadius: 3,
                            background: "rgba(139,92,246,0.2)", color: "#C4B5FD",
                            border: "1px solid rgba(139,92,246,0.3)",
                            fontWeight: 700, letterSpacing: 1
                          }}>BIZRA</span>
                        )}
                      </div>
                      <div style={{ fontSize: 10, color: "#64748B", marginTop: 2 }}>{c.desc}</div>
                    </div>
                  ))}
                </div>

                {/* Hooks */}
                {layer.hooks.length > 0 && (
                  <>
                    <div
                      onClick={() => toggleHooks(layer.id)}
                      style={{
                        fontSize: 9, color: "#F59E0B", letterSpacing: 2, marginBottom: 6,
                        textTransform: "uppercase", cursor: "pointer", display: "flex", alignItems: "center", gap: 4
                      }}
                    >
                      ⚡ Hooks ({layer.hooks.length})
                      <span style={{ fontSize: 10 }}>{expandedHooks[layer.id] ? "▾" : "▸"}</span>
                    </div>
                    {expandedHooks[layer.id] && (
                      <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                        {layer.hooks.map((h, i) => (
                          <div key={i} style={{
                            background: "rgba(245,158,11,0.05)",
                            border: "1px solid rgba(245,158,11,0.15)",
                            borderRadius: 6, padding: "6px 10px"
                          }}>
                            <div style={{ fontSize: 11, fontWeight: 600, color: "#F59E0B", fontFamily: "inherit" }}>
                              {h.name}()
                            </div>
                            <div style={{ fontSize: 10, color: "#64748B", marginTop: 1 }}>{h.desc}</div>
                          </div>
                        ))}
                      </div>
                    )}
                  </>
                )}
              </div>
            ) : (
              <div style={{
                background: "rgba(255,255,255,0.02)",
                border: "1px solid rgba(255,255,255,0.06)",
                borderRadius: 12,
                padding: 32,
                textAlign: "center",
                color: "#475569",
                fontSize: 12
              }}>
                <div style={{ fontSize: 28, marginBottom: 8, opacity: 0.3 }}>⬡</div>
                Click any layer to explore its components, hooks, and BIZRA focus areas
              </div>
            )}
          </div>
        </div>
      )}

      {/* Protocol Comparison */}
      {activeTab === "protocols" && (
        <div style={{ maxWidth: 900, margin: "0 auto" }}>
          <div style={{ fontSize: 13, color: "#94A3B8", marginBottom: 16, lineHeight: 1.6 }}>
            Five integration patterns, each solving a different problem.
            The key insight: <span style={{ color: "#C4B5FD" }}>Skills</span> = internal capabilities,{" "}
            <span style={{ color: "#EC4899" }}>MCP</span> = connecting to external tools,{" "}
            <span style={{ color: "#F97316" }}>A2A</span> = agents talking to agents,{" "}
            <span style={{ color: "#64748B" }}>Plugins</span> = legacy (being replaced by MCP),{" "}
            <span style={{ color: "#F59E0B" }}>Hooks</span> = the internal nervous system connecting everything.
          </div>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <thead>
                <tr>
                  {["Protocol", "Connects", "Direction", "Standard", "Example", "BIZRA Use"].map(h => (
                    <th key={h} style={{
                      textAlign: "left", padding: "8px 10px", fontSize: 9,
                      color: "#64748B", letterSpacing: 2, textTransform: "uppercase",
                      borderBottom: "1px solid rgba(255,255,255,0.08)"
                    }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {PROTOCOL_COMPARISON.map((p, i) => (
                  <tr key={i} style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                    <td style={{ padding: "10px", fontWeight: 700, color: 
                      p.protocol === "Skills" ? "#10B981" :
                      p.protocol === "MCP" ? "#EC4899" :
                      p.protocol === "A2A" ? "#F97316" :
                      p.protocol === "Plugins" ? "#64748B" : "#F59E0B"
                    }}>{p.protocol}</td>
                    <td style={{ padding: "10px", color: "#CBD5E1" }}>{p.connects}</td>
                    <td style={{ padding: "10px" }}>
                      <span style={{
                        fontSize: 9, padding: "2px 6px", borderRadius: 3,
                        background: "rgba(255,255,255,0.05)",
                        border: "1px solid rgba(255,255,255,0.08)"
                      }}>{p.direction}</span>
                    </td>
                    <td style={{ padding: "10px", color: "#94A3B8" }}>{p.standard}</td>
                    <td style={{ padding: "10px", color: "#94A3B8" }}>{p.example}</td>
                    <td style={{ padding: "10px", color: "#C4B5FD", fontWeight: 600 }}>{p.bizra}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Visual Protocol Map */}
          <div style={{
            marginTop: 24, padding: 20,
            background: "rgba(255,255,255,0.02)",
            border: "1px solid rgba(255,255,255,0.06)",
            borderRadius: 12
          }}>
            <div style={{ fontSize: 9, color: "#64748B", letterSpacing: 2, marginBottom: 12, textTransform: "uppercase" }}>
              How They Connect
            </div>
            <div style={{ fontFamily: "inherit", fontSize: 11, lineHeight: 2, color: "#94A3B8", whiteSpace: "pre" }}>
{`  ┌─────────────────────────────────────────────┐
  │              YOUR NODE0                      │
  │                                              │
  │  ┌─────────┐    ┌──────────┐    ┌────────┐  │
  │  │  PAT-7  │◄──►│  HOOKS   │◄──►│ SKILLS │  │
  │  │ Agents  │    │  (Bus)   │    │ (AHK+) │  │
  │  └────┬────┘    └────┬─────┘    └────────┘  │
  │       │              │                       │
  │       │         ┌────┴─────┐                 │
  │       │         │ MEMORY   │                 │
  │       │         │ STACK    │                 │
  │       │         └──────────┘                 │
  └───────┼──────────────────────────────────────┘
          │              │
     A2A  │         MCP  │
          │              │
  ┌───────┼──────────────┼───────────────────────┐
  │       ▼              ▼        URP / Network   │
  │  ┌─────────┐    ┌──────────┐                  │
  │  │  SAT-49 │    │ External │  (Notion, Slack, │
  │  │ Agents  │    │ Services │   GitHub, etc.)  │
  │  └─────────┘    └──────────┘                  │
  └───────────────────────────────────────────────┘`}
            </div>
          </div>
        </div>
      )}

      {/* Focus / Action Map */}
      {activeTab === "focus" && (
        <div style={{ maxWidth: 900, margin: "0 auto" }}>
          <div style={{ fontSize: 13, color: "#94A3B8", marginBottom: 16, lineHeight: 1.6 }}>
            Prioritized action items for Node0. Red = build now, it blocks everything.
            Yellow = build next, it enables Phase 2. Green = already done or designed.
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {FOCUS_MAP.map((item, i) => (
              <div key={i} style={{
                background: "rgba(255,255,255,0.02)",
                border: `1px solid ${
                  item.priority.includes("🔴") ? "rgba(239,68,68,0.2)" :
                  item.priority.includes("🟡") ? "rgba(245,158,11,0.2)" :
                  "rgba(16,185,129,0.2)"
                }`,
                borderRadius: 8,
                padding: "12px 14px",
                display: "grid",
                gridTemplateColumns: "1fr auto",
                gap: 8,
                alignItems: "start"
              }}>
                <div>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                    <span style={{ fontSize: 13, fontWeight: 700, color: "#E2E8F0" }}>{item.area}</span>
                    <span style={{ fontSize: 10 }}>{item.priority.split(" ")[0]}</span>
                    <span style={{
                      fontSize: 8, padding: "1px 6px", borderRadius: 3,
                      background: item.priority.includes("🔴") ? "rgba(239,68,68,0.15)" :
                                  item.priority.includes("🟡") ? "rgba(245,158,11,0.15)" :
                                  "rgba(16,185,129,0.15)",
                      color: item.priority.includes("🔴") ? "#FCA5A5" :
                             item.priority.includes("🟡") ? "#FCD34D" : "#6EE7B7",
                      fontWeight: 700, letterSpacing: 1
                    }}>
                      {item.priority.split(" ").slice(1).join(" ")}
                    </span>
                  </div>
                  <div style={{ fontSize: 11, color: "#94A3B8", lineHeight: 1.5 }}>{item.reason}</div>
                </div>
                <div style={{ textAlign: "right", flexShrink: 0 }}>
                  <div style={{ fontSize: 10, color: "#64748B" }}>{item.effort}</div>
                  <div style={{ fontSize: 10, color: "#8B5CF6", fontWeight: 600, marginTop: 2 }}>{item.impact}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Data Flow View */}
      {activeTab === "dataflow" && (
        <div style={{ maxWidth: 900, margin: "0 auto" }}>
          <div style={{ fontSize: 13, color: "#94A3B8", marginBottom: 20, lineHeight: 1.6 }}>
            What happens from the moment you type a message to the moment the agent remembers it forever.
            Every numbered step is a hook point where BIZRA can intercept and control.
          </div>
          {[
            { phase: "INPUT PHASE", color: "#0EA5E9", steps: [
              { n: "01", hook: "onUserInput", desc: "You type a message. The UI captures it and fires the input hook.", who: "UI Layer" },
              { n: "02", hook: "beforeMemoryRetrieve", desc: "Orchestrator extracts keywords/intent from your message to query memory.", who: "Orchestrator" },
              { n: "03", hook: "onMemoryRetrieve", desc: "Memory system searches vector store + knowledge graph for relevant memories. Returns ranked results.", who: "Memory Stack" },
              { n: "04", hook: "beforeLLMCall", desc: "Context Budget Manager assembles the final prompt: system prompt + semantic memory + retrieved episodics + procedural config + current conversation + your message. Must fit in 200K tokens.", who: "Orchestrator" },
            ]},
            { phase: "PROCESSING PHASE", color: "#8B5CF6", steps: [
              { n: "05", hook: "onLLMCall", desc: "Assembled prompt sent to LLM. Model begins generating tokens.", who: "LLM" },
              { n: "06", hook: "onToolCall", desc: "If model decides to use a tool: hook intercepts → routes to Skill (internal), MCP (external service), or A2A (another agent).", who: "Hooks Layer" },
              { n: "07", hook: "onToolResult", desc: "Tool result returns. Hook can transform it, log it, or trigger additional tools.", who: "Hooks Layer" },
              { n: "08", hook: "onStreamChunk", desc: "Each generated token flows to UI for real-time display.", who: "UI Layer" },
            ]},
            { phase: "OUTPUT PHASE", color: "#10B981", steps: [
              { n: "09", hook: "onResponseComplete", desc: "Full response assembled. Post-processing begins.", who: "Hooks Layer" },
              { n: "10", hook: "onIhsanScore", desc: "إحسان scoring evaluates response quality. Logged to conversation record.", who: "BIZRA Quality" },
              { n: "11", hook: "onMemoryExtract", desc: "Memory extraction agent parses the conversation for: new facts, updated beliefs, decisions made, action items.", who: "Memory Synthesis" },
              { n: "12", hook: "onMemorySynthesize", desc: "Extracted info compressed into semantic memory. Conflicts resolved. User model updated. Procedural patterns learned.", who: "Memory Stack" },
            ]},
            { phase: "PERSISTENCE PHASE", color: "#F59E0B", steps: [
              { n: "13", hook: "onDataWrite", desc: "Full conversation + extracted memories encrypted with Ed25519 and written to sovereign local storage.", who: "Infrastructure" },
              { n: "14", hook: "onStateUpdate", desc: "Agent state updated: conversation count, token usage, إحسان running average, behavioral pattern updates.", who: "Orchestrator" },
              { n: "15", hook: "onDataSync (Phase 2)", desc: "If federated: selected memories sync to URP via encrypted channels. User controls what leaves Node0.", who: "Federation Layer" },
            ]},
          ].map((phase) => (
            <div key={phase.phase} style={{ marginBottom: 20 }}>
              <div style={{
                fontSize: 9, fontWeight: 700, letterSpacing: 3,
                color: phase.color, marginBottom: 8,
                textTransform: "uppercase",
                display: "flex", alignItems: "center", gap: 8
              }}>
                <div style={{ width: 8, height: 8, borderRadius: 2, background: phase.color }} />
                {phase.phase}
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 3, paddingLeft: 4, borderLeft: `2px solid ${phase.color}22` }}>
                {phase.steps.map((step) => (
                  <div key={step.n} style={{
                    display: "flex", gap: 10, padding: "8px 12px",
                    background: "rgba(255,255,255,0.02)",
                    borderRadius: 6,
                    alignItems: "start"
                  }}>
                    <div style={{
                      fontSize: 11, fontWeight: 700, color: phase.color,
                      fontFamily: "inherit", flexShrink: 0, width: 20
                    }}>{step.n}</div>
                    <div style={{ flex: 1 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 2 }}>
                        <code style={{
                          fontSize: 10, color: "#F59E0B",
                          background: "rgba(245,158,11,0.1)",
                          padding: "1px 5px", borderRadius: 3
                        }}>{step.hook}</code>
                        <span style={{ fontSize: 9, color: "#475569" }}>→ {step.who}</span>
                      </div>
                      <div style={{ fontSize: 11, color: "#94A3B8", lineHeight: 1.5 }}>{step.desc}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Footer */}
      <div style={{
        textAlign: "center", marginTop: 32, paddingTop: 16,
        borderTop: "1px solid rgba(255,255,255,0.04)",
        fontSize: 9, color: "#334155"
      }}>
        BIZRA Node0 · Agentic Architecture Reference · Feb 2026
      </div>

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
