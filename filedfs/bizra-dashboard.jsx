import { useState, useEffect, useRef, useCallback } from "react";

// ============================================================
// BIZRA Node0 Dashboard — Alpha-100
// "My AI Knows Me" Interface
// ============================================================

// Sacred geometry seed-of-life SVG pattern
const SeedOfLife = ({ size = 120, opacity = 0.08, color = "#D4A547" }) => (
  <svg width={size} height={size} viewBox="0 0 120 120" style={{ opacity }}>
    {[0, 60, 120, 180, 240, 300].map((angle, i) => {
      const cx = 60 + 30 * Math.cos((angle * Math.PI) / 180);
      const cy = 60 + 30 * Math.sin((angle * Math.PI) / 180);
      return <circle key={i} cx={cx} cy={cy} r="30" fill="none" stroke={color} strokeWidth="0.5" />;
    })}
    <circle cx="60" cy="60" r="30" fill="none" stroke={color} strokeWidth="0.5" />
  </svg>
);

// Animated radial score gauge
const KnowsMeGauge = ({ score, size = 180 }) => {
  const radius = (size - 20) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - score * circumference;
  const gradientId = "gauge-grad";

  return (
    <div style={{ position: "relative", width: size, height: size }}>
      <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
        <defs>
          <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#D4A547" />
            <stop offset="50%" stopColor="#F0D68A" />
            <stop offset="100%" stopColor="#D4A547" />
          </linearGradient>
        </defs>
        <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="rgba(212,165,71,0.08)" strokeWidth="6" />
        <circle
          cx={size / 2} cy={size / 2} r={radius}
          fill="none" stroke={`url(#${gradientId})`} strokeWidth="6"
          strokeDasharray={circumference} strokeDashoffset={offset}
          strokeLinecap="round"
          style={{ transition: "stroke-dashoffset 1.2s cubic-bezier(0.4,0,0.2,1)" }}
        />
      </svg>
      <div style={{
        position: "absolute", inset: 0, display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center",
      }}>
        <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 36, fontWeight: 700, color: "#F0D68A", letterSpacing: -1 }}>
          {(score * 100).toFixed(1)}
        </span>
        <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "rgba(212,165,71,0.6)", letterSpacing: 2, textTransform: "uppercase", marginTop: 2 }}>
          knows me
        </span>
      </div>
    </div>
  );
};

// Ihsan bar
const IhsanBar = ({ score }) => {
  const pct = (score / 10000) * 100;
  const color = score >= 9500 ? "#5BBA6F" : score >= 8000 ? "#D4A547" : "#E85D4A";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "rgba(255,255,255,0.4)", width: 50, letterSpacing: 1 }}>إحسان</span>
      <div style={{ flex: 1, height: 4, background: "rgba(255,255,255,0.06)", borderRadius: 2, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 2, transition: "width 0.6s ease" }} />
      </div>
      <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color, width: 50, textAlign: "right" }}>
        {(score / 100).toFixed(1)}%
      </span>
    </div>
  );
};

// Agent role badge
const AgentBadge = ({ name, active, vetoes }) => {
  const colors = {
    Navigator: "#6B9BF7", Scholar: "#A78BFA", Artisan: "#F59E42",
    Guardian: "#E85D4A", Mentor: "#5BBA6F", Diplomat: "#38BDF8", Oracle: "#F0D68A"
  };
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 6, padding: "4px 10px",
      background: active ? `${colors[name]}18` : "rgba(255,255,255,0.02)",
      border: `1px solid ${active ? colors[name] + "40" : "rgba(255,255,255,0.05)"}`,
      borderRadius: 6, transition: "all 0.3s ease",
    }}>
      <div style={{
        width: 6, height: 6, borderRadius: "50%",
        background: active ? colors[name] : "rgba(255,255,255,0.15)",
        boxShadow: active ? `0 0 8px ${colors[name]}60` : "none",
      }} />
      <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: active ? colors[name] : "rgba(255,255,255,0.3)", letterSpacing: 0.5 }}>
        {name}
      </span>
      {name === "Guardian" && vetoes > 0 && (
        <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "#E85D4A", marginLeft: "auto" }}>
          {vetoes}🛡
        </span>
      )}
    </div>
  );
};

// Message bubble
const MessageBubble = ({ role, content, meta }) => {
  const isUser = role === "user";
  return (
    <div style={{
      display: "flex", flexDirection: "column",
      alignItems: isUser ? "flex-end" : "flex-start",
      marginBottom: 12, maxWidth: "85%",
      alignSelf: isUser ? "flex-end" : "flex-start",
    }}>
      <div style={{
        padding: "10px 14px",
        background: isUser ? "rgba(212,165,71,0.12)" : "rgba(255,255,255,0.04)",
        border: `1px solid ${isUser ? "rgba(212,165,71,0.2)" : "rgba(255,255,255,0.06)"}`,
        borderRadius: isUser ? "14px 14px 4px 14px" : "14px 14px 14px 4px",
        color: "rgba(255,255,255,0.88)",
        fontFamily: "'DM Sans', sans-serif", fontSize: 13.5, lineHeight: 1.55,
      }}>
        {content}
      </div>
      {meta && (
        <div style={{
          display: "flex", gap: 10, marginTop: 4, padding: "0 4px",
        }}>
          {meta.agents && (
            <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "rgba(212,165,71,0.4)" }}>
              {meta.agents} agents
            </span>
          )}
          {meta.fragments > 0 && (
            <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "rgba(91,186,111,0.5)" }}>
              +{meta.fragments} learned
            </span>
          )}
          {meta.confidence && (
            <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "rgba(255,255,255,0.25)" }}>
              {(meta.confidence * 100).toFixed(0)}% conf
            </span>
          )}
        </div>
      )}
    </div>
  );
};

// Trait pill
const TraitPill = ({ label, value, confidence }) => (
  <div style={{
    display: "flex", alignItems: "center", gap: 8,
    padding: "6px 10px", background: "rgba(212,165,71,0.06)",
    border: "1px solid rgba(212,165,71,0.12)", borderRadius: 8,
  }}>
    <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "rgba(212,165,71,0.5)", textTransform: "uppercase", letterSpacing: 0.8 }}>
      {label}
    </span>
    <span style={{ fontFamily: "'DM Sans', sans-serif", fontSize: 12, color: "rgba(255,255,255,0.75)", flex: 1 }}>
      {value}
    </span>
    <div style={{ width: 20, height: 3, background: "rgba(255,255,255,0.06)", borderRadius: 2, overflow: "hidden" }}>
      <div style={{ width: `${(confidence / 10000) * 100}%`, height: "100%", background: "#D4A547", borderRadius: 2 }} />
    </div>
  </div>
);

// Metric card
const Metric = ({ label, value, sub }) => (
  <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
    <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "rgba(255,255,255,0.3)", letterSpacing: 1.2, textTransform: "uppercase" }}>
      {label}
    </span>
    <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 20, fontWeight: 600, color: "rgba(255,255,255,0.85)" }}>
      {value}
    </span>
    {sub && (
      <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "rgba(255,255,255,0.2)" }}>
        {sub}
      </span>
    )}
  </div>
);

// ============================================================
// SIMULATED NODE BRIDGE
// In production: WebSocket → spawned bizra-node binary
// Here: simulates protocol responses for the artifact demo
// ============================================================
const createNodeBridge = () => {
  let state = {
    knowsMe: 0.0, ihsan: 9900, messages: 0, fragments: 0, insights: 0,
    traits: [], vetoes: 0, sessions: 0, tasks: 0,
    agentsActive: ["Navigator", "Scholar", "Artisan", "Guardian", "Mentor", "Diplomat", "Oracle"],
  };

  const extractFragments = (text) => {
    let count = 0;
    const lower = text.toLowerCase();
    if (/i (prefer|like|love|enjoy|want|hate|need)/.test(lower)) count++;
    if (/i('m| am) (working on|building|trying|a )/.test(lower)) count++;
    if (/my (goal|project|expertise|name|favorite)/.test(lower)) count++;
    if (/i (specialize|work with|work at|live in)/.test(lower)) count++;
    return count;
  };

  const classifyIntent = (text) => {
    const lower = text.toLowerCase();
    if (/\b(code|function|program|debug|implement|script|api)\b/.test(lower)) return "Code";
    if (/\b(what|why|how|when|where|who|explain|tell me)\b/.test(lower)) return "Question";
    if (/\b(create|write|generate|make|design|build|draft)\b/.test(lower)) return "Create";
    if (/\b(analyze|compare|evaluate|review|assess)\b/.test(lower)) return "Analyze";
    if (/\b(plan|strategy|roadmap|schedule|next steps)\b/.test(lower)) return "Plan";
    return "Chat";
  };

  const agentsForIntent = {
    Chat: ["Navigator", "Diplomat", "Oracle"],
    Question: ["Scholar", "Oracle", "Navigator"],
    Code: ["Artisan", "Scholar", "Guardian"],
    Create: ["Artisan", "Diplomat", "Navigator"],
    Analyze: ["Scholar", "Oracle", "Artisan"],
    Plan: ["Oracle", "Navigator", "Scholar"],
  };

  return {
    send: (verb, args = {}) => {
      switch (verb) {
        case "RECEIVE": {
          const frags = extractFragments(args.content);
          state.messages++;
          state.fragments += frags;
          state.tasks += 2;
          state.knowsMe = Math.min(1.0, state.knowsMe + frags * 0.008 + 0.002);
          const intent = classifyIntent(args.content);
          const agents = agentsForIntent[intent] || ["Navigator"];

          // Simulate knowledge extraction
          if (frags > 0) {
            const lower = args.content.toLowerCase();
            if (/i (prefer|like|love)/.test(lower)) {
              const match = args.content.match(/(?:prefer|like|love)\s+(.+?)(?:\.|$)/i);
              if (match) state.traits = [...state.traits.filter(t => t.label !== "preference"), { label: "preference", value: match[1].trim().slice(0, 40), confidence: 8500 }];
            }
            if (/i('m| am) (a |an |the )/.test(lower)) {
              const match = args.content.match(/(?:i'm|i am)\s+(?:a|an|the)\s+(.+?)(?:\.|,|$)/i);
              if (match) state.traits = [...state.traits.filter(t => t.label !== "identity"), { label: "identity", value: match[1].trim().slice(0, 40), confidence: 9000 }];
            }
            if (/my goal/.test(lower)) {
              const match = args.content.match(/my goal\s+(?:is\s+)?(?:to\s+)?(.+?)(?:\.|$)/i);
              if (match) state.traits = [...state.traits.filter(t => t.label !== "goal"), { label: "goal", value: match[1].trim().slice(0, 40), confidence: 8800 }];
            }
            if (/i (specialize|work with|work at)/.test(lower)) {
              const match = args.content.match(/(?:specialize in|work with|work at)\s+(.+?)(?:\.|$)/i);
              if (match) state.traits = [...state.traits.filter(t => t.label !== "expertise"), { label: "expertise", value: match[1].trim().slice(0, 40), confidence: 9200 }];
            }
            if (/i live in/.test(lower)) {
              const match = args.content.match(/live in\s+(.+?)(?:\.|$)/i);
              if (match) state.traits = [...state.traits.filter(t => t.label !== "location"), { label: "location", value: match[1].trim().slice(0, 40), confidence: 9500 }];
            }
          }

          // Generate contextual response
          let responseText;
          const hasTraits = state.traits.length > 0;
          if (hasTraits && state.knowsMe > 0.05) {
            const personalization = state.traits.map(t => t.value).join(", ");
            responseText = `I understand — with your background in ${personalization}, this connects well. ${intent === "Question" ? "Let me research this thoroughly." : intent === "Code" ? "I'll draft an implementation aligned with your style." : intent === "Create" ? "I'll create something tailored to your vision." : "Let's explore this together."}`;
          } else {
            responseText = intent === "Question" ? "Let me look into that for you." : intent === "Code" ? "I'll work on an implementation." : intent === "Create" ? "I'll start crafting that." : "I hear you. Tell me more so I can understand you better.";
          }

          return {
            ok: true,
            content: responseText,
            confidence: hasTraits ? 0.85 + state.knowsMe * 0.1 : 0.75,
            agents: agents.length,
            agentNames: agents,
            fragments: frags,
            knowsMe: state.knowsMe,
            vetoed: false,
            intent,
          };
        }
        case "TEACH": {
          state.fragments++;
          state.knowsMe = Math.min(1.0, state.knowsMe + 0.012);
          state.traits = [...state.traits.filter(t => t.label !== args.kind),
            { label: args.kind, value: args.content.slice(0, 40), confidence: args.confidence || 9000 }];
          return { ok: true, knowsMe: state.knowsMe };
        }
        case "SYNTHESIZE": {
          const newInsights = Math.min(3, Math.floor(state.fragments / 3));
          state.insights += newInsights;
          state.knowsMe = Math.min(1.0, state.knowsMe + newInsights * 0.015);
          return { ok: true, insights: newInsights, knowsMe: state.knowsMe };
        }
        case "HEALTH": {
          return { ok: true, ...state };
        }
        case "KNOWS_ME": {
          return { ok: true, score: state.knowsMe };
        }
        case "PING": {
          return { ok: true, pong: true };
        }
        default:
          return { ok: true };
      }
    },
    getState: () => ({ ...state }),
  };
};

// ============================================================
// MAIN DASHBOARD COMPONENT
// ============================================================
export default function Node0Dashboard() {
  const [bridge] = useState(() => createNodeBridge());
  const [nodeState, setNodeState] = useState(bridge.getState());
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [connected, setConnected] = useState(false);
  const [activeAgents, setActiveAgents] = useState([]);
  const [pulseKey, setPulseKey] = useState(0);
  const chatEndRef = useRef(null);
  const inputRef = useRef(null);

  // Boot sequence
  useEffect(() => {
    const timer = setTimeout(() => {
      setConnected(true);
      bridge.send("PING");
    }, 800);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = useCallback(() => {
    if (!input.trim() || !connected) return;
    const text = input.trim();
    setInput("");

    // Add user message
    setMessages(prev => [...prev, { role: "user", content: text }]);

    // Process through bridge
    setTimeout(() => {
      const result = bridge.send("RECEIVE", { content: text, timestamp: Date.now() });
      setActiveAgents(result.agentNames || []);
      setPulseKey(k => k + 1);

      setTimeout(() => {
        setMessages(prev => [...prev, {
          role: "node",
          content: result.content,
          meta: {
            agents: result.agents,
            fragments: result.fragments,
            confidence: result.confidence,
            intent: result.intent,
          },
        }]);
        setNodeState(bridge.getState());
        setTimeout(() => setActiveAgents([]), 1500);
      }, 400 + Math.random() * 300);
    }, 100);
  }, [input, connected, bridge]);

  const handleTeach = useCallback((kind, content) => {
    bridge.send("TEACH", { kind, content, confidence: 9000 });
    setNodeState(bridge.getState());
    setPulseKey(k => k + 1);
  }, [bridge]);

  const handleSynthesize = useCallback(() => {
    const result = bridge.send("SYNTHESIZE", {});
    setNodeState(bridge.getState());
    setPulseKey(k => k + 1);
    if (result.insights > 0) {
      setMessages(prev => [...prev, {
        role: "system",
        content: `🧬 Synthesis complete — ${result.insights} new insight${result.insights > 1 ? "s" : ""} generated. Knows-me score: ${(result.knowsMe * 100).toFixed(1)}%`,
      }]);
    }
  }, [bridge]);

  // Quick teach suggestions
  const suggestions = [
    { kind: "preference", text: "I prefer dark mode and minimal UI" },
    { kind: "expertise", text: "I specialize in distributed systems" },
    { kind: "goal", text: "My goal is to democratize AI for everyone" },
    { kind: "fact", text: "I live in Dubai and work in GMT+4" },
  ];

  const allAgents = ["Navigator", "Scholar", "Artisan", "Guardian", "Mentor", "Diplomat", "Oracle"];

  return (
    <div style={{
      minHeight: "100vh", background: "#0A0B0F",
      fontFamily: "'DM Sans', sans-serif", color: "rgba(255,255,255,0.88)",
      display: "flex", flexDirection: "column", position: "relative", overflow: "hidden",
    }}>
      {/* Load fonts */}
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet" />

      {/* Background sacred geometry */}
      <div style={{ position: "fixed", inset: 0, pointerEvents: "none", zIndex: 0 }}>
        <div style={{ position: "absolute", top: -40, right: -20, opacity: 0.03 }}>
          <SeedOfLife size={400} opacity={1} />
        </div>
        <div style={{ position: "absolute", bottom: -60, left: -30, opacity: 0.02 }}>
          <SeedOfLife size={500} opacity={1} />
        </div>
        {/* Gradient overlay */}
        <div style={{
          position: "absolute", inset: 0,
          background: "radial-gradient(ellipse at 30% 20%, rgba(212,165,71,0.03) 0%, transparent 60%), radial-gradient(ellipse at 70% 80%, rgba(107,155,247,0.02) 0%, transparent 50%)",
        }} />
      </div>

      {/* Header */}
      <header style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "14px 24px", borderBottom: "1px solid rgba(255,255,255,0.04)",
        background: "rgba(10,11,15,0.8)", backdropFilter: "blur(20px)",
        position: "relative", zIndex: 10,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 28, height: 28, borderRadius: 8,
            background: "linear-gradient(135deg, #D4A547, #8B6914)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 14, fontWeight: 700, color: "#0A0B0F",
            fontFamily: "'JetBrains Mono', monospace",
          }}>
            B
          </div>
          <div>
            <div style={{ fontWeight: 600, fontSize: 14, letterSpacing: -0.3, color: "rgba(255,255,255,0.9)" }}>
              Node0
              <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "rgba(212,165,71,0.5)", marginLeft: 8, fontWeight: 400 }}>
                v0.1.0
              </span>
            </div>
            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "rgba(255,255,255,0.2)", letterSpacing: 1.5, marginTop: 1 }}>
              SOVEREIGN AI NODE
            </div>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <IhsanBar score={nodeState.ihsan} />
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{
              width: 7, height: 7, borderRadius: "50%",
              background: connected ? "#5BBA6F" : "#E85D4A",
              boxShadow: connected ? "0 0 10px rgba(91,186,111,0.5)" : "none",
              animation: connected ? "pulse 2s infinite" : "none",
            }} />
            <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: connected ? "rgba(91,186,111,0.7)" : "rgba(232,93,74,0.7)" }}>
              {connected ? "CONNECTED" : "BOOTING"}
            </span>
          </div>
        </div>
      </header>

      {/* Main grid */}
      <div style={{
        display: "grid", gridTemplateColumns: "260px 1fr 240px",
        flex: 1, gap: 0, position: "relative", zIndex: 5, minHeight: 0,
      }}>
        {/* LEFT PANEL — Agents + Metrics */}
        <div style={{
          borderRight: "1px solid rgba(255,255,255,0.04)",
          padding: 16, display: "flex", flexDirection: "column", gap: 20,
          overflowY: "auto",
        }}>
          {/* PAT Team */}
          <div>
            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "rgba(255,255,255,0.25)", letterSpacing: 1.5, marginBottom: 10, textTransform: "uppercase" }}>
              Personal Agent Team
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {allAgents.map(name => (
                <AgentBadge
                  key={name}
                  name={name}
                  active={activeAgents.includes(name)}
                  vetoes={name === "Guardian" ? nodeState.vetoes : 0}
                />
              ))}
            </div>
          </div>

          {/* Metrics */}
          <div>
            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "rgba(255,255,255,0.25)", letterSpacing: 1.5, marginBottom: 12, textTransform: "uppercase" }}>
              Runtime Metrics
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
              <Metric label="Messages" value={nodeState.messages} />
              <Metric label="Fragments" value={nodeState.fragments} />
              <Metric label="Insights" value={nodeState.insights} />
              <Metric label="Tasks" value={nodeState.tasks} />
            </div>
          </div>

          {/* Quick Teach */}
          <div>
            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "rgba(255,255,255,0.25)", letterSpacing: 1.5, marginBottom: 10, textTransform: "uppercase" }}>
              Quick Teach
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {suggestions.map((s, i) => (
                <button
                  key={i}
                  onClick={() => {
                    handleTeach(s.kind, s.text);
                    setMessages(prev => [...prev, { role: "system", content: `📚 Taught: "${s.text}"` }]);
                  }}
                  style={{
                    background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)",
                    borderRadius: 6, padding: "6px 10px", textAlign: "left", cursor: "pointer",
                    fontFamily: "'DM Sans', sans-serif", fontSize: 11, color: "rgba(255,255,255,0.5)",
                    transition: "all 0.2s ease",
                  }}
                  onMouseEnter={e => { e.target.style.background = "rgba(212,165,71,0.08)"; e.target.style.borderColor = "rgba(212,165,71,0.2)"; }}
                  onMouseLeave={e => { e.target.style.background = "rgba(255,255,255,0.02)"; e.target.style.borderColor = "rgba(255,255,255,0.06)"; }}
                >
                  <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "rgba(212,165,71,0.4)", marginRight: 6 }}>
                    {s.kind}
                  </span>
                  {s.text.slice(0, 35)}...
                </button>
              ))}
            </div>
          </div>

          {/* Synthesize button */}
          <button
            onClick={handleSynthesize}
            style={{
              background: "linear-gradient(135deg, rgba(212,165,71,0.15), rgba(212,165,71,0.05))",
              border: "1px solid rgba(212,165,71,0.25)", borderRadius: 8,
              padding: "10px 16px", cursor: "pointer",
              fontFamily: "'JetBrains Mono', monospace", fontSize: 11, fontWeight: 600,
              color: "#D4A547", letterSpacing: 0.5,
              transition: "all 0.2s ease",
            }}
            onMouseEnter={e => { e.target.style.background = "linear-gradient(135deg, rgba(212,165,71,0.25), rgba(212,165,71,0.1))"; }}
            onMouseLeave={e => { e.target.style.background = "linear-gradient(135deg, rgba(212,165,71,0.15), rgba(212,165,71,0.05))"; }}
          >
            🧬 Synthesize Memory
          </button>
        </div>

        {/* CENTER — Chat */}
        <div style={{
          display: "flex", flexDirection: "column", minHeight: 0,
        }}>
          {/* Chat messages */}
          <div style={{
            flex: 1, overflowY: "auto", padding: "20px 24px",
            display: "flex", flexDirection: "column",
          }}>
            {messages.length === 0 && (
              <div style={{
                flex: 1, display: "flex", flexDirection: "column",
                alignItems: "center", justifyContent: "center", gap: 16, opacity: 0.4,
              }}>
                <SeedOfLife size={80} opacity={0.3} color="#D4A547" />
                <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "rgba(212,165,71,0.6)", letterSpacing: 2, textTransform: "uppercase" }}>
                  Node0 Ready
                </div>
                <div style={{ fontSize: 13, color: "rgba(255,255,255,0.3)", textAlign: "center", maxWidth: 300, lineHeight: 1.6 }}>
                  Start a conversation. Every message teaches me who you are. The more we talk, the better I know you.
                </div>
              </div>
            )}
            {messages.map((msg, i) => (
              msg.role === "system" ? (
                <div key={i} style={{
                  textAlign: "center", padding: "8px 16px", marginBottom: 12,
                  fontFamily: "'JetBrains Mono', monospace", fontSize: 10,
                  color: "rgba(212,165,71,0.5)", background: "rgba(212,165,71,0.04)",
                  borderRadius: 20, alignSelf: "center",
                }}>
                  {msg.content}
                </div>
              ) : (
                <MessageBubble key={i} role={msg.role} content={msg.content} meta={msg.meta} />
              )
            ))}
            <div ref={chatEndRef} />
          </div>

          {/* Input area */}
          <div style={{
            padding: "12px 20px 16px", borderTop: "1px solid rgba(255,255,255,0.04)",
            background: "rgba(10,11,15,0.6)", backdropFilter: "blur(10px)",
          }}>
            <div style={{
              display: "flex", alignItems: "center", gap: 10,
              background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)",
              borderRadius: 12, padding: "6px 6px 6px 16px",
              transition: "border-color 0.2s ease",
            }}>
              <input
                ref={inputRef}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => e.key === "Enter" && sendMessage()}
                placeholder={connected ? "Talk to your node..." : "Connecting..."}
                disabled={!connected}
                style={{
                  flex: 1, background: "none", border: "none", outline: "none",
                  fontFamily: "'DM Sans', sans-serif", fontSize: 14, color: "rgba(255,255,255,0.88)",
                  padding: "6px 0",
                }}
              />
              <button
                onClick={sendMessage}
                disabled={!input.trim() || !connected}
                style={{
                  width: 36, height: 36, borderRadius: 8, border: "none",
                  background: input.trim() ? "linear-gradient(135deg, #D4A547, #8B6914)" : "rgba(255,255,255,0.04)",
                  color: input.trim() ? "#0A0B0F" : "rgba(255,255,255,0.15)",
                  cursor: input.trim() ? "pointer" : "default",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 16, fontWeight: 700, transition: "all 0.2s ease",
                }}
              >
                ↑
              </button>
            </div>
            <div style={{
              display: "flex", justifyContent: "center", gap: 16, marginTop: 8,
              fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "rgba(255,255,255,0.15)",
            }}>
              <span>Try: "I'm a software architect who loves Rust"</span>
              <span>•</span>
              <span>"My goal is to build sovereign AI"</span>
            </div>
          </div>
        </div>

        {/* RIGHT PANEL — Knowledge */}
        <div style={{
          borderLeft: "1px solid rgba(255,255,255,0.04)",
          padding: 16, display: "flex", flexDirection: "column", gap: 20, alignItems: "center",
          overflowY: "auto",
        }}>
          {/* Knows Me Gauge */}
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
            <KnowsMeGauge score={nodeState.knowsMe} />
            <div style={{
              fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "rgba(255,255,255,0.2)",
              letterSpacing: 1.5, textTransform: "uppercase", marginTop: 4,
            }}>
              Understanding Depth
            </div>
          </div>

          {/* Profile Traits */}
          <div style={{ width: "100%" }}>
            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "rgba(255,255,255,0.25)", letterSpacing: 1.5, marginBottom: 10, textTransform: "uppercase" }}>
              Learned Traits
              <span style={{ color: "rgba(212,165,71,0.4)", marginLeft: 6 }}>
                {nodeState.traits.length}
              </span>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {nodeState.traits.length === 0 ? (
                <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "rgba(255,255,255,0.12)", textAlign: "center", padding: "20px 0" }}>
                  No traits learned yet.
                  <br />Talk to teach me.
                </div>
              ) : (
                nodeState.traits.map((t, i) => (
                  <TraitPill key={`${t.label}-${i}`} label={t.label} value={t.value} confidence={t.confidence} />
                ))
              )}
            </div>
          </div>

          {/* Session info */}
          <div style={{
            width: "100%", padding: "10px 12px",
            background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.04)",
            borderRadius: 8,
          }}>
            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "rgba(255,255,255,0.25)", letterSpacing: 1.5, marginBottom: 8, textTransform: "uppercase" }}>
              Architecture
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {["bizra-hooks → nerves", "bizra-memory → brain", "bizra-agent → being", "bizra-node → process"].map((line, i) => (
                <div key={i} style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "rgba(255,255,255,0.18)" }}>
                  {line}
                </div>
              ))}
            </div>
            <div style={{
              marginTop: 8, paddingTop: 8, borderTop: "1px solid rgba(255,255,255,0.04)",
              fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "rgba(212,165,71,0.3)",
              textAlign: "center", lineHeight: 1.6,
            }}>
              10,000 lines • 205 tests
              <br />Zero external dependencies
              <br />ربي لا يعرف المستحيل
            </div>
          </div>
        </div>
      </div>

      {/* CSS animations */}
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.15); }
        ::placeholder { color: rgba(255,255,255,0.25); }
        * { box-sizing: border-box; margin: 0; padding: 0; }
      `}</style>
    </div>
  );
}
