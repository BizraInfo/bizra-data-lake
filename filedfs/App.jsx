// ============================================================
// BIZRA Node0 — Alpha-100 Dashboard (Production)
// ============================================================
// Wired to bizra-node via useNode hook.
// In Tauri: invoke → Rust → Node.execute()
// In browser: simulated bridge (demo mode)
// ============================================================

import { useState, useEffect, useRef, useCallback } from "react";
import { useNode } from "./hooks/useNode";
import OnboardingFlow from "./onboarding/OnboardingFlow";

// ── Sacred Geometry ─────────────────────────────────────────

const SeedOfLife = ({ size = 120, opacity = 0.08, color = "#D4A547" }) => (
  <svg width={size} height={size} viewBox="0 0 120 120" style={{ opacity }}>
    {[0, 60, 120, 180, 240, 300].map((a, i) => (
      <circle key={i} cx={60 + 30 * Math.cos((a * Math.PI) / 180)} cy={60 + 30 * Math.sin((a * Math.PI) / 180)} r="30" fill="none" stroke={color} strokeWidth="0.5" />
    ))}
    <circle cx="60" cy="60" r="30" fill="none" stroke={color} strokeWidth="0.5" />
  </svg>
);

// ── Score Gauge ─────────────────────────────────────────────

const KnowsMeGauge = ({ score, size = 180 }) => {
  const r = (size - 20) / 2;
  const c = 2 * Math.PI * r;
  return (
    <div style={{ position: "relative", width: size, height: size }}>
      <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
        <defs>
          <linearGradient id="ggrad" x1="0%" y1="0%" x2="100%"><stop offset="0%" stopColor="#D4A547" /><stop offset="50%" stopColor="#F0D68A" /><stop offset="100%" stopColor="#D4A547" /></linearGradient>
        </defs>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgba(212,165,71,0.08)" strokeWidth="6" />
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="url(#ggrad)" strokeWidth="6" strokeDasharray={c} strokeDashoffset={c - score * c} strokeLinecap="round" style={{ transition: "stroke-dashoffset 1.2s cubic-bezier(0.4,0,0.2,1)" }} />
      </svg>
      <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
        <span style={{ fontFamily: "var(--mono)", fontSize: 36, fontWeight: 700, color: "#F0D68A", letterSpacing: -1 }}>{(score * 100).toFixed(1)}</span>
        <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "rgba(212,165,71,0.6)", letterSpacing: 2, textTransform: "uppercase", marginTop: 2 }}>knows me</span>
      </div>
    </div>
  );
};

// ── Ihsan Bar ───────────────────────────────────────────────

const IhsanBar = ({ score }) => {
  const pct = (score / 10000) * 100;
  const color = score >= 9500 ? "#5BBA6F" : score >= 8000 ? "#D4A547" : "#E85D4A";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "rgba(255,255,255,0.4)", width: 50, letterSpacing: 1 }}>إحسان</span>
      <div style={{ flex: 1, height: 4, background: "rgba(255,255,255,0.06)", borderRadius: 2, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 2, transition: "width 0.6s ease" }} />
      </div>
      <span style={{ fontFamily: "var(--mono)", fontSize: 11, color, width: 50, textAlign: "right" }}>{(score / 100).toFixed(1)}%</span>
    </div>
  );
};

// ── Agent Badge ─────────────────────────────────────────────

const AGENT_COLORS = {
  Navigator: "#6B9BF7", Scholar: "#A78BFA", Artisan: "#F59E42",
  Guardian: "#E85D4A", Mentor: "#5BBA6F", Diplomat: "#38BDF8", Oracle: "#F0D68A",
};
const ALL_AGENTS = Object.keys(AGENT_COLORS);

const AgentBadge = ({ name, active }) => {
  const c = AGENT_COLORS[name] || "#888";
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 6, padding: "4px 10px",
      background: active ? `${c}18` : "rgba(255,255,255,0.02)",
      border: `1px solid ${active ? c + "40" : "rgba(255,255,255,0.05)"}`,
      borderRadius: 6, transition: "all 0.3s ease",
    }}>
      <div style={{ width: 6, height: 6, borderRadius: "50%", background: active ? c : "rgba(255,255,255,0.15)", boxShadow: active ? `0 0 8px ${c}60` : "none" }} />
      <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: active ? c : "rgba(255,255,255,0.3)", letterSpacing: 0.5 }}>{name}</span>
    </div>
  );
};

// ── Message Bubble ──────────────────────────────────────────

const Bubble = ({ role, content, meta }) => {
  const isUser = role === "user";
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: isUser ? "flex-end" : "flex-start", marginBottom: 12, maxWidth: "85%", alignSelf: isUser ? "flex-end" : "flex-start" }}>
      <div style={{
        padding: "10px 14px",
        background: isUser ? "rgba(212,165,71,0.12)" : "rgba(255,255,255,0.04)",
        border: `1px solid ${isUser ? "rgba(212,165,71,0.2)" : "rgba(255,255,255,0.06)"}`,
        borderRadius: isUser ? "14px 14px 4px 14px" : "14px 14px 14px 4px",
        color: "rgba(255,255,255,0.88)", fontFamily: "var(--sans)", fontSize: 13.5, lineHeight: 1.55,
      }}>{content}</div>
      {meta && (
        <div style={{ display: "flex", gap: 10, marginTop: 4, padding: "0 4px" }}>
          {meta.agents && <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: "rgba(212,165,71,0.4)" }}>{meta.agents} agents</span>}
          {meta.fragments > 0 && <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: "rgba(91,186,111,0.5)" }}>+{meta.fragments} learned</span>}
          {meta.confidence && <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: "rgba(255,255,255,0.25)" }}>{(parseFloat(meta.confidence) * 100).toFixed(0)}% conf</span>}
        </div>
      )}
    </div>
  );
};

// ── Trait Pill ───────────────────────────────────────────────

const TraitPill = ({ label, value, confidence }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 10px", background: "rgba(212,165,71,0.06)", border: "1px solid rgba(212,165,71,0.12)", borderRadius: 8 }}>
    <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: "rgba(212,165,71,0.5)", textTransform: "uppercase", letterSpacing: 0.8 }}>{label}</span>
    <span style={{ fontFamily: "var(--sans)", fontSize: 12, color: "rgba(255,255,255,0.75)", flex: 1 }}>{value}</span>
    <div style={{ width: 20, height: 3, background: "rgba(255,255,255,0.06)", borderRadius: 2, overflow: "hidden" }}>
      <div style={{ width: `${(confidence / 10000) * 100}%`, height: "100%", background: "#D4A547", borderRadius: 2 }} />
    </div>
  </div>
);

// ── Metric ──────────────────────────────────────────────────

const Metric = ({ label, value }) => (
  <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
    <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: "rgba(255,255,255,0.3)", letterSpacing: 1.2, textTransform: "uppercase" }}>{label}</span>
    <span style={{ fontFamily: "var(--mono)", fontSize: 20, fontWeight: 600, color: "rgba(255,255,255,0.85)" }}>{value}</span>
  </div>
);

// ── Section Header ──────────────────────────────────────────

const SectionLabel = ({ children, extra }) => (
  <div style={{ fontFamily: "var(--mono)", fontSize: 9, color: "rgba(255,255,255,0.25)", letterSpacing: 1.5, marginBottom: 10, textTransform: "uppercase" }}>
    {children}
    {extra && <span style={{ color: "rgba(212,165,71,0.4)", marginLeft: 6 }}>{extra}</span>}
  </div>
);

// ============================================================
// MAIN DASHBOARD
// ============================================================

export default function App() {
  const node = useNode();
  const { connected, mode, send, receive, teach, synthesize, refreshHealth } = node;
  const [onboarded, setOnboarded] = useState(() => {
    try { return localStorage.getItem("bizra_onboarded") === "1"; } catch { return false; }
  });

  if (!onboarded) {
    return (
      <OnboardingFlow
        node={node}
        onComplete={() => {
          try { localStorage.setItem("bizra_onboarded", "1"); } catch {}
          setOnboarded(true);
        }}
      />
    );
  }

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [activeAgents, setActiveAgents] = useState([]);
  const [nodeData, setNodeData] = useState({
    knowsMe: 0, ihsan: 9900, messages: 0, fragments: 0, insights: 0, traits: [],
  });
  const chatEndRef = useRef(null);

  // Refresh state from node
  const syncState = useCallback(async () => {
    const h = await send("HEALTH");
    if (h?.ok && h.fields) {
      const f = h.fields;
      setNodeData((prev) => ({
        ...prev,
        knowsMe: parseFloat(f.knows_me || "0"),
        ihsan: parseInt(f.ihsan || "9900", 10),
        messages: parseInt(f.messages || "0", 10),
        fragments: parseInt(f.fragments || "0", 10),
        insights: parseInt(f.insights || "0", 10),
      }));
    }
    // Also refresh profile
    const p = await send("PROFILE");
    if (p?.ok && p.fields?.traits) {
      const traits = p.fields.traits
        .split("|")
        .filter(Boolean)
        .map((entry) => {
          const [label, value, conf] = entry.split(":");
          return { label, value, confidence: parseInt(conf || "8000", 10) };
        });
      setNodeData((prev) => ({ ...prev, traits }));
    }
  }, [send]);

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Send message
  const sendMessage = useCallback(async () => {
    if (!input.trim() || !connected) return;
    const text = input.trim();
    setInput("");

    setMessages((prev) => [...prev, { role: "user", content: text }]);

    const result = await receive(text);

    if (result?.ok && result.fields) {
      const f = result.fields;
      // Flash active agents
      const agentCount = parseInt(f.agents_consulted || "0", 10);
      const activeNames = ALL_AGENTS.slice(0, agentCount);
      setActiveAgents(activeNames);
      setTimeout(() => setActiveAgents([]), 1500);

      setMessages((prev) => [
        ...prev,
        {
          role: "node",
          content: f.content || "...",
          meta: {
            agents: f.agents_consulted,
            fragments: parseInt(f.fragments_extracted || "0", 10),
            confidence: f.confidence,
            intent: f.intent,
          },
        },
      ]);
    }

    await syncState();
  }, [input, connected, receive, syncState]);

  // Teach shortcut
  const handleTeach = useCallback(
    async (kind, content) => {
      await teach(kind, content);
      setMessages((prev) => [...prev, { role: "system", content: `📚 Taught: "${content}"` }]);
      await syncState();
    },
    [teach, syncState]
  );

  // Synthesize
  const handleSynthesize = useCallback(async () => {
    const result = await synthesize();
    if (result?.ok && result.fields) {
      const n = parseInt(result.fields.insights_generated || "0", 10);
      if (n > 0) {
        setMessages((prev) => [
          ...prev,
          { role: "system", content: `🧬 Synthesis complete — ${n} new insight${n > 1 ? "s" : ""} generated. Knows-me: ${(parseFloat(result.fields.knows_me || "0") * 100).toFixed(1)}%` },
        ]);
      }
    }
    await syncState();
  }, [synthesize, syncState]);

  const quickTeach = [
    { kind: "preference", text: "I prefer dark mode and minimal UI" },
    { kind: "expertise", text: "I specialize in distributed systems" },
    { kind: "goal", text: "My goal is to democratize AI for everyone" },
    { kind: "fact", text: "I live in Dubai and work in GMT+4" },
  ];

  return (
    <div style={{
      "--sans": "'DM Sans', sans-serif",
      "--mono": "'JetBrains Mono', monospace",
      minHeight: "100vh", background: "#0A0B0F",
      fontFamily: "var(--sans)", color: "rgba(255,255,255,0.88)",
      display: "flex", flexDirection: "column", position: "relative", overflow: "hidden",
    }}>
      {/* Background */}
      <div style={{ position: "fixed", inset: 0, pointerEvents: "none", zIndex: 0 }}>
        <div style={{ position: "absolute", top: -40, right: -20, opacity: 0.03 }}><SeedOfLife size={400} opacity={1} /></div>
        <div style={{ position: "absolute", bottom: -60, left: -30, opacity: 0.02 }}><SeedOfLife size={500} opacity={1} /></div>
        <div style={{ position: "absolute", inset: 0, background: "radial-gradient(ellipse at 30% 20%, rgba(212,165,71,0.03) 0%, transparent 60%), radial-gradient(ellipse at 70% 80%, rgba(107,155,247,0.02) 0%, transparent 50%)" }} />
      </div>

      {/* Header */}
      <header style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "14px 24px", borderBottom: "1px solid rgba(255,255,255,0.04)",
        background: "rgba(10,11,15,0.8)", backdropFilter: "blur(20px)", position: "relative", zIndex: 10,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ width: 28, height: 28, borderRadius: 8, background: "linear-gradient(135deg, #D4A547, #8B6914)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14, fontWeight: 700, color: "#0A0B0F", fontFamily: "var(--mono)" }}>B</div>
          <div>
            <div style={{ fontWeight: 600, fontSize: 14, letterSpacing: -0.3 }}>
              Node0
              <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "rgba(212,165,71,0.5)", marginLeft: 8, fontWeight: 400 }}>v0.1.0</span>
              <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: "rgba(107,155,247,0.4)", marginLeft: 8 }}>
                {mode === "tauri" ? "NATIVE" : "DEMO"}
              </span>
            </div>
            <div style={{ fontFamily: "var(--mono)", fontSize: 9, color: "rgba(255,255,255,0.2)", letterSpacing: 1.5, marginTop: 1 }}>SOVEREIGN AI NODE</div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <IhsanBar score={nodeData.ihsan} />
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{ width: 7, height: 7, borderRadius: "50%", background: connected ? "#5BBA6F" : "#E85D4A", boxShadow: connected ? "0 0 10px rgba(91,186,111,0.5)" : "none", animation: connected ? "pulse 2s infinite" : "none" }} />
            <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: connected ? "rgba(91,186,111,0.7)" : "rgba(232,93,74,0.7)" }}>
              {connected ? "CONNECTED" : "BOOTING"}
            </span>
          </div>
        </div>
      </header>

      {/* Main Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "260px 1fr 240px", flex: 1, gap: 0, position: "relative", zIndex: 5, minHeight: 0 }}>

        {/* LEFT — Agents + Metrics */}
        <div style={{ borderRight: "1px solid rgba(255,255,255,0.04)", padding: 16, display: "flex", flexDirection: "column", gap: 20, overflowY: "auto" }}>
          <div>
            <SectionLabel>Personal Agent Team</SectionLabel>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {ALL_AGENTS.map((name) => <AgentBadge key={name} name={name} active={activeAgents.includes(name)} />)}
            </div>
          </div>

          <div>
            <SectionLabel>Runtime Metrics</SectionLabel>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
              <Metric label="Messages" value={nodeData.messages} />
              <Metric label="Fragments" value={nodeData.fragments} />
              <Metric label="Insights" value={nodeData.insights} />
              <Metric label="Traits" value={nodeData.traits.length} />
            </div>
          </div>

          <div>
            <SectionLabel>Quick Teach</SectionLabel>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {quickTeach.map((s, i) => (
                <button key={i} onClick={() => handleTeach(s.kind, s.text)} style={{
                  background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)",
                  borderRadius: 6, padding: "6px 10px", textAlign: "left", cursor: "pointer",
                  fontFamily: "var(--sans)", fontSize: 11, color: "rgba(255,255,255,0.5)", transition: "all 0.2s",
                }}
                  onMouseEnter={(e) => { e.currentTarget.style.background = "rgba(212,165,71,0.08)"; e.currentTarget.style.borderColor = "rgba(212,165,71,0.2)"; }}
                  onMouseLeave={(e) => { e.currentTarget.style.background = "rgba(255,255,255,0.02)"; e.currentTarget.style.borderColor = "rgba(255,255,255,0.06)"; }}
                >
                  <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: "rgba(212,165,71,0.4)", marginRight: 6 }}>{s.kind}</span>
                  {s.text.slice(0, 35)}...
                </button>
              ))}
            </div>
          </div>

          <button onClick={handleSynthesize} style={{
            background: "linear-gradient(135deg, rgba(212,165,71,0.15), rgba(212,165,71,0.05))",
            border: "1px solid rgba(212,165,71,0.25)", borderRadius: 8,
            padding: "10px 16px", cursor: "pointer",
            fontFamily: "var(--mono)", fontSize: 11, fontWeight: 600, color: "#D4A547", letterSpacing: 0.5,
          }}>
            🧬 Synthesize Memory
          </button>
        </div>

        {/* CENTER — Chat */}
        <div style={{ display: "flex", flexDirection: "column", minHeight: 0 }}>
          <div style={{ flex: 1, overflowY: "auto", padding: "20px 24px", display: "flex", flexDirection: "column" }}>
            {messages.length === 0 && (
              <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 16, opacity: 0.4 }}>
                <SeedOfLife size={80} opacity={0.3} color="#D4A547" />
                <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "rgba(212,165,71,0.6)", letterSpacing: 2, textTransform: "uppercase" }}>Node0 Ready</div>
                <div style={{ fontSize: 13, color: "rgba(255,255,255,0.3)", textAlign: "center", maxWidth: 300, lineHeight: 1.6 }}>
                  Start a conversation. Every message teaches me who you are.
                </div>
              </div>
            )}
            {messages.map((msg, i) =>
              msg.role === "system" ? (
                <div key={i} style={{ textAlign: "center", padding: "8px 16px", marginBottom: 12, fontFamily: "var(--mono)", fontSize: 10, color: "rgba(212,165,71,0.5)", background: "rgba(212,165,71,0.04)", borderRadius: 20, alignSelf: "center" }}>{msg.content}</div>
              ) : (
                <Bubble key={i} role={msg.role} content={msg.content} meta={msg.meta} />
              )
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Input */}
          <div style={{ padding: "12px 20px 16px", borderTop: "1px solid rgba(255,255,255,0.04)", background: "rgba(10,11,15,0.6)", backdropFilter: "blur(10px)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 12, padding: "6px 6px 6px 16px" }}>
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && sendMessage()}
                placeholder={connected ? "Talk to your node..." : "Connecting..."}
                disabled={!connected}
                style={{ flex: 1, background: "none", border: "none", outline: "none", fontFamily: "var(--sans)", fontSize: 14, color: "rgba(255,255,255,0.88)", padding: "6px 0" }}
              />
              <button
                onClick={sendMessage}
                disabled={!input.trim() || !connected}
                style={{
                  width: 36, height: 36, borderRadius: 8, border: "none",
                  background: input.trim() ? "linear-gradient(135deg, #D4A547, #8B6914)" : "rgba(255,255,255,0.04)",
                  color: input.trim() ? "#0A0B0F" : "rgba(255,255,255,0.15)",
                  cursor: input.trim() ? "pointer" : "default",
                  display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, fontWeight: 700,
                }}>↑</button>
            </div>
            <div style={{ display: "flex", justifyContent: "center", gap: 16, marginTop: 8, fontFamily: "var(--mono)", fontSize: 9, color: "rgba(255,255,255,0.15)" }}>
              <span>Try: "I'm a software architect who loves Rust"</span>
              <span>•</span>
              <span>"My goal is to build sovereign AI"</span>
            </div>
          </div>
        </div>

        {/* RIGHT — Knowledge */}
        <div style={{ borderLeft: "1px solid rgba(255,255,255,0.04)", padding: 16, display: "flex", flexDirection: "column", gap: 20, alignItems: "center", overflowY: "auto" }}>
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
            <KnowsMeGauge score={nodeData.knowsMe} />
            <div style={{ fontFamily: "var(--mono)", fontSize: 9, color: "rgba(255,255,255,0.2)", letterSpacing: 1.5, textTransform: "uppercase", marginTop: 4 }}>Understanding Depth</div>
          </div>

          <div style={{ width: "100%" }}>
            <SectionLabel extra={nodeData.traits.length}>Learned Traits</SectionLabel>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {nodeData.traits.length === 0 ? (
                <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "rgba(255,255,255,0.12)", textAlign: "center", padding: "20px 0" }}>No traits yet.<br />Talk to teach me.</div>
              ) : (
                nodeData.traits.map((t, i) => <TraitPill key={`${t.label}-${i}`} label={t.label} value={t.value} confidence={t.confidence} />)
              )}
            </div>
          </div>

          <div style={{ width: "100%", padding: "10px 12px", background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.04)", borderRadius: 8 }}>
            <SectionLabel>Architecture</SectionLabel>
            {["bizra-hooks → nerves", "bizra-memory → brain", "bizra-agent → being", "bizra-node → process"].map((l, i) => (
              <div key={i} style={{ fontFamily: "var(--mono)", fontSize: 9, color: "rgba(255,255,255,0.18)", marginBottom: 2 }}>{l}</div>
            ))}
            <div style={{ marginTop: 8, paddingTop: 8, borderTop: "1px solid rgba(255,255,255,0.04)", fontFamily: "var(--mono)", fontSize: 9, color: "rgba(212,165,71,0.3)", textAlign: "center", lineHeight: 1.6 }}>
              10,000 lines • 205 tests<br />Zero dependencies<br />ربي لا يعرف المستحيل
            </div>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
      `}</style>
    </div>
  );
}
