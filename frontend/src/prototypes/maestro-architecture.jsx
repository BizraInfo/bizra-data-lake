import { useState, useEffect, useRef } from "react";

const GOLD = "#C4A35A";
const NAVY = "#1A1F36";
const DARK = "#0D1117";
const SURFACE = "#161B22";
const BORDER = "#30363D";
const TEXT = "#E6EDF3";
const DIM = "#8B949E";
const GREEN = "#3FB950";
const AMBER = "#D29922";
const RED = "#F85149";
const BLUE = "#58A6FF";
const PURPLE = "#BC8CFF";
const CYAN = "#39D2C0";

const agents = [
  { id: "planner", label: "Planner", icon: "🗺️", color: BLUE, desc: "Strategy & sequencing" },
  { id: "critic", label: "Critic", icon: "🔍", color: AMBER, desc: "Risk & edge cases" },
  { id: "ethicist", label: "Ethicist", icon: "⚖️", color: PURPLE, desc: "Moral evaluation" },
  { id: "executor", label: "Executor", icon: "⚡", color: GREEN, desc: "Implementation" },
  { id: "verifier", label: "Verifier", icon: "✓", color: CYAN, desc: "Success criteria" },
  { id: "security", label: "Security", icon: "🛡️", color: RED, desc: "Sovereignty & TeleScript" },
  { id: "optimizer", label: "Optimizer", icon: "📊", color: GOLD, desc: "Efficiency & cost" },
];

const tiers = {
  s1: { label: "S1 · Reflex", agents: [], color: GREEN, desc: "Maestro responds directly. No agents." },
  s15: { label: "S1.5 · Moderate", agents: ["executor"], color: CYAN, desc: "One specialist." },
  s2: { label: "S2 · Deliberative", agents: ["planner", "executor", "verifier"], color: BLUE, desc: "Core triangle." },
  s2p: { label: "S2+ · Complex", agents: agents.map(a => a.id), color: PURPLE, desc: "Full ensemble." },
};

const emotions = [
  { id: "neutral", label: "Neutral", emoji: "😐", tone: "warm_professional" },
  { id: "frustrated", label: "Frustrated", emoji: "😤", tone: "patient_empathetic", boost: ["critic"] },
  { id: "urgent", label: "Urgent", emoji: "⏰", tone: "direct_action_oriented", boost: ["optimizer"] },
  { id: "curious", label: "Curious", emoji: "🤔", tone: "enthusiastic_detailed", boost: ["planner"] },
  { id: "overwhelmed", label: "Overwhelmed", emoji: "😵", tone: "calm_simplified", reduce: true },
  { id: "confident", label: "Confident", emoji: "💪", tone: "collaborative_peer" },
  { id: "playful", label: "Playful", emoji: "😄", tone: "light_creative" },
];

const trustLevels = [
  { id: "stranger", label: "Stranger", days: "0-1", autonomy: 0.1, icon: "👤" },
  { id: "acquaintance", label: "Acquaintance", days: "1-7", autonomy: 0.3, icon: "🤝" },
  { id: "colleague", label: "Colleague", days: "7-30", autonomy: 0.5, icon: "👥" },
  { id: "partner", label: "Partner", days: "30-90", autonomy: 0.7, icon: "🤲" },
  { id: "extension", label: "Extension", days: "90+", autonomy: 0.9, icon: "🧬" },
];

export default function MaestroViz() {
  const [selectedTier, setSelectedTier] = useState("s2");
  const [selectedEmotion, setSelectedEmotion] = useState("neutral");
  const [trustIdx, setTrustIdx] = useState(0);
  const [animPhase, setAnimPhase] = useState(0);
  const [showFlow, setShowFlow] = useState(false);
  const canvasRef = useRef(null);

  const tier = tiers[selectedTier];
  const emotion = emotions.find(e => e.id === selectedEmotion);
  const trust = trustLevels[trustIdx];

  // Compute active agents based on tier + emotion
  let activeAgents = new Set(tier.agents);
  if (emotion.boost) emotion.boost.forEach(a => activeAgents.add(a));
  if (emotion.reduce) activeAgents = new Set([...activeAgents].filter(a => a === "executor"));
  if (selectedTier === "s1") activeAgents = new Set();

  useEffect(() => {
    if (!showFlow) return;
    const interval = setInterval(() => {
      setAnimPhase(p => (p + 1) % 7);
    }, 1200);
    return () => clearInterval(interval);
  }, [showFlow]);

  const phaseLabels = [
    "User speaks",
    "Emotion detected",
    "Agents selected",
    "Agents deliberate",
    "Maestro synthesizes",
    "Tone adapted",
    "Response delivered",
  ];

  return (
    <div style={{
      background: DARK, color: TEXT, fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
      minHeight: "100vh", padding: "24px", boxSizing: "border-box",
    }}>
      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: 32 }}>
        <div style={{ fontSize: 11, color: GOLD, letterSpacing: 6, marginBottom: 8 }}>
          B I Z R A
        </div>
        <h1 style={{
          fontSize: 28, fontWeight: 300, margin: 0, color: TEXT,
          fontFamily: "'Georgia', serif",
        }}>
          The Maestro Layer
        </h1>
        <p style={{ color: DIM, fontSize: 13, marginTop: 8, fontStyle: "italic" }}>
          One voice. One personality. One relationship. Seven agents behind the curtain.
        </p>
      </div>

      {/* Main Architecture */}
      <div style={{ maxWidth: 900, margin: "0 auto" }}>

        {/* User → Maestro → Agents flow */}
        <div style={{
          background: SURFACE, border: `1px solid ${BORDER}`, borderRadius: 12,
          padding: 24, marginBottom: 24,
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 20 }}>
            <div style={{ fontSize: 11, color: GOLD, letterSpacing: 3 }}>ARCHITECTURE</div>
            <div style={{ flex: 1, height: 1, background: BORDER }} />
            <button
              onClick={() => setShowFlow(!showFlow)}
              style={{
                background: showFlow ? GOLD : "transparent",
                color: showFlow ? DARK : GOLD,
                border: `1px solid ${GOLD}`,
                borderRadius: 6, padding: "4px 12px", fontSize: 11,
                cursor: "pointer", letterSpacing: 1,
              }}
            >
              {showFlow ? "⏸ PAUSE" : "▶ ANIMATE"}
            </button>
          </div>

          {/* Flow diagram */}
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8, flexWrap: "wrap" }}>
            {/* User */}
            <FlowNode
              label="User"
              icon="👤"
              active={showFlow && animPhase === 0}
              color={TEXT}
              sub="speaks"
            />
            <Arrow active={showFlow && animPhase >= 1} />

            {/* Maestro */}
            <div style={{
              background: animPhase >= 1 && animPhase <= 5 && showFlow
                ? `${GOLD}15` : `${GOLD}08`,
              border: `2px solid ${GOLD}`,
              borderRadius: 12, padding: "16px 20px", textAlign: "center",
              transition: "all 0.4s",
              boxShadow: showFlow && animPhase >= 1 && animPhase <= 5
                ? `0 0 20px ${GOLD}30` : "none",
              minWidth: 160,
            }}>
              <div style={{ fontSize: 24, marginBottom: 4 }}>🎭</div>
              <div style={{ fontSize: 15, fontWeight: 600, color: GOLD }}>MAESTRO</div>
              <div style={{ fontSize: 10, color: DIM, marginTop: 4 }}>
                {showFlow ? phaseLabels[animPhase] : "Communication Persona"}
              </div>
              <div style={{
                fontSize: 9, color: emotion.tone === "warm_professional" ? DIM : AMBER,
                marginTop: 6, fontStyle: "italic",
              }}>
                tone: {emotion.tone}
              </div>
            </div>
            <Arrow active={showFlow && animPhase >= 3} />

            {/* Agent Ensemble */}
            <div style={{
              border: `1px solid ${BORDER}`, borderRadius: 12,
              padding: 12, minWidth: 260,
              opacity: selectedTier === "s1" ? 0.3 : 1,
              transition: "opacity 0.3s",
            }}>
              <div style={{ fontSize: 10, color: DIM, marginBottom: 8, textAlign: "center" }}>
                PAT ENSEMBLE ({activeAgents.size} active)
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6, justifyContent: "center" }}>
                {agents.map(a => {
                  const isActive = activeAgents.has(a.id);
                  const isAnimating = showFlow && animPhase === 3 && isActive;
                  return (
                    <div key={a.id} style={{
                      background: isActive ? `${a.color}20` : `${BORDER}40`,
                      border: `1px solid ${isActive ? a.color : BORDER}`,
                      borderRadius: 8, padding: "6px 10px",
                      fontSize: 10, color: isActive ? a.color : `${DIM}60`,
                      transition: "all 0.3s",
                      transform: isAnimating ? "scale(1.1)" : "scale(1)",
                      boxShadow: isAnimating ? `0 0 12px ${a.color}40` : "none",
                    }}>
                      <span style={{ marginRight: 4 }}>{a.icon}</span>
                      {a.label}
                    </div>
                  );
                })}
              </div>
              {selectedTier === "s1" && (
                <div style={{
                  textAlign: "center", fontSize: 10, color: GREEN,
                  marginTop: 8, fontStyle: "italic",
                }}>
                  Maestro responds directly — agents sleeping
                </div>
              )}
            </div>
            <Arrow active={showFlow && animPhase >= 4} />

            {/* FATE Gate */}
            <FlowNode
              label="FATE"
              icon="⛩️"
              active={showFlow && animPhase >= 5}
              color={GREEN}
              sub="gate"
            />
            <Arrow active={showFlow && animPhase >= 6} />

            {/* Response */}
            <FlowNode
              label="Response"
              icon="💬"
              active={showFlow && animPhase === 6}
              color={GOLD}
              sub="single voice"
            />
          </div>
        </div>

        {/* Controls */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 24 }}>

          {/* Complexity Tier */}
          <div style={{
            background: SURFACE, border: `1px solid ${BORDER}`,
            borderRadius: 12, padding: 20,
          }}>
            <div style={{ fontSize: 11, color: GOLD, letterSpacing: 3, marginBottom: 16 }}>
              COMPLEXITY TIER
            </div>
            {Object.entries(tiers).map(([key, t]) => (
              <button
                key={key}
                onClick={() => setSelectedTier(key)}
                style={{
                  display: "block", width: "100%", textAlign: "left",
                  background: selectedTier === key ? `${t.color}15` : "transparent",
                  border: `1px solid ${selectedTier === key ? t.color : "transparent"}`,
                  borderRadius: 8, padding: "10px 14px", marginBottom: 6,
                  color: selectedTier === key ? t.color : DIM,
                  cursor: "pointer", fontSize: 12, transition: "all 0.2s",
                }}
              >
                <div style={{ fontWeight: 600 }}>{t.label}</div>
                <div style={{ fontSize: 10, opacity: 0.7, marginTop: 2 }}>
                  {t.desc} → {t.agents.length === 0 ? "0" : t.agents.length} agent{t.agents.length !== 1 ? "s" : ""}
                </div>
              </button>
            ))}
          </div>

          {/* Emotion State */}
          <div style={{
            background: SURFACE, border: `1px solid ${BORDER}`,
            borderRadius: 12, padding: 20,
          }}>
            <div style={{ fontSize: 11, color: GOLD, letterSpacing: 3, marginBottom: 16 }}>
              USER EMOTION → TONE
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {emotions.map(e => (
                <button
                  key={e.id}
                  onClick={() => setSelectedEmotion(e.id)}
                  style={{
                    background: selectedEmotion === e.id ? `${GOLD}20` : `${BORDER}40`,
                    border: `1px solid ${selectedEmotion === e.id ? GOLD : BORDER}`,
                    borderRadius: 8, padding: "8px 12px",
                    color: selectedEmotion === e.id ? GOLD : DIM,
                    cursor: "pointer", fontSize: 11, transition: "all 0.2s",
                  }}
                >
                  <span style={{ marginRight: 4 }}>{e.emoji}</span>
                  {e.label}
                </button>
              ))}
            </div>
            <div style={{
              marginTop: 16, padding: 12, background: `${GOLD}08`,
              borderRadius: 8, border: `1px solid ${GOLD}30`,
            }}>
              <div style={{ fontSize: 10, color: DIM }}>Selected tone:</div>
              <div style={{ fontSize: 13, color: GOLD, marginTop: 4 }}>
                {emotion.tone.replace(/_/g, " ")}
              </div>
              {emotion.boost && (
                <div style={{ fontSize: 10, color: AMBER, marginTop: 6 }}>
                  +agents: {emotion.boost.join(", ")}
                </div>
              )}
              {emotion.reduce && (
                <div style={{ fontSize: 10, color: RED, marginTop: 6 }}>
                  Reduces to executor only (أمك simplification)
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Trust Evolution */}
        <div style={{
          background: SURFACE, border: `1px solid ${BORDER}`,
          borderRadius: 12, padding: 20, marginBottom: 24,
        }}>
          <div style={{ fontSize: 11, color: GOLD, letterSpacing: 3, marginBottom: 16 }}>
            TRUST EVOLUTION
          </div>
          <div style={{ display: "flex", gap: 4, alignItems: "stretch" }}>
            {trustLevels.map((t, i) => (
              <button
                key={t.id}
                onClick={() => setTrustIdx(i)}
                style={{
                  flex: 1, textAlign: "center",
                  background: i === trustIdx ? `${GOLD}15` : i <= trustIdx ? `${GREEN}08` : "transparent",
                  border: `1px solid ${i === trustIdx ? GOLD : i <= trustIdx ? `${GREEN}40` : BORDER}`,
                  borderRadius: 8, padding: "12px 8px",
                  color: i === trustIdx ? GOLD : i <= trustIdx ? GREEN : DIM,
                  cursor: "pointer", fontSize: 11, transition: "all 0.2s",
                }}
              >
                <div style={{ fontSize: 20 }}>{t.icon}</div>
                <div style={{ fontWeight: 600, marginTop: 4 }}>{t.label}</div>
                <div style={{ fontSize: 9, opacity: 0.6 }}>Day {t.days}</div>
              </button>
            ))}
          </div>
          <div style={{
            display: "grid", gridTemplateColumns: "1fr 1fr",
            gap: 12, marginTop: 16,
          }}>
            <div style={{ padding: 12, background: `${GOLD}08`, borderRadius: 8 }}>
              <div style={{ fontSize: 10, color: DIM }}>Autonomy Budget</div>
              <div style={{ fontSize: 22, color: GOLD, fontWeight: 300 }}>
                {(trust.autonomy * 100).toFixed(0)}%
              </div>
              <div style={{ fontSize: 10, color: DIM, marginTop: 4 }}>
                {trust.autonomy < 0.3 ? "Ask before everything" :
                 trust.autonomy < 0.6 ? "Handle routine silently" :
                 "Only ask for novel/risky"}
              </div>
            </div>
            <div style={{ padding: 12, background: `${GOLD}08`, borderRadius: 8 }}>
              <div style={{ fontSize: 10, color: DIM }}>Proactive Threshold</div>
              <div style={{ fontSize: 22, color: GOLD, fontWeight: 300 }}>
                {((1 - trust.autonomy) * 100 + 5).toFixed(0)}%
              </div>
              <div style={{ fontSize: 10, color: DIM, marginTop: 4 }}>
                {trust.autonomy < 0.3 ? "Only surface high-confidence" :
                 trust.autonomy < 0.6 ? "Share moderate hunches" :
                 "Surface even early intuitions"}
              </div>
            </div>
          </div>
        </div>

        {/* The JARVIS Principle */}
        <div style={{
          background: `${GOLD}08`, border: `1px solid ${GOLD}30`,
          borderRadius: 12, padding: 24, textAlign: "center",
        }}>
          <div style={{ fontSize: 11, color: GOLD, letterSpacing: 3, marginBottom: 12 }}>
            THE JARVIS PRINCIPLE
          </div>
          <div style={{
            fontSize: 15, color: TEXT, lineHeight: 1.8,
            fontFamily: "'Georgia', serif", maxWidth: 600, margin: "0 auto",
          }}>
            The user talks to <span style={{ color: GOLD }}>one person</span>.
            Behind that person, <span style={{ color: BLUE }}>seven specialists</span> deliberate.
            The Maestro reads <span style={{ color: AMBER }}>emotion</span>,
            selects <span style={{ color: CYAN }}>agents</span>,
            synthesizes into <span style={{ color: GOLD }}>one voice</span>,
            and evolves <span style={{ color: GREEN }}>trust</span> over time.
          </div>
          <div style={{
            fontSize: 12, color: DIM, marginTop: 16, fontStyle: "italic",
          }}>
            أمك wouldn't talk to a committee. She'd talk to someone she trusts.
          </div>
        </div>
      </div>
    </div>
  );
}

function FlowNode({ label, icon, active, color, sub }) {
  return (
    <div style={{
      textAlign: "center", padding: "10px 14px",
      border: `1px solid ${active ? color : BORDER}`,
      borderRadius: 10,
      background: active ? `${color}15` : "transparent",
      transition: "all 0.4s",
      boxShadow: active ? `0 0 15px ${color}25` : "none",
      minWidth: 70,
    }}>
      <div style={{ fontSize: 18 }}>{icon}</div>
      <div style={{ fontSize: 11, color: active ? color : DIM, fontWeight: 600 }}>{label}</div>
      <div style={{ fontSize: 9, color: DIM }}>{sub}</div>
    </div>
  );
}

function Arrow({ active }) {
  return (
    <div style={{
      color: active ? GOLD : `${BORDER}`,
      fontSize: 16, transition: "color 0.4s",
      padding: "0 2px",
    }}>
      →
    </div>
  );
}
