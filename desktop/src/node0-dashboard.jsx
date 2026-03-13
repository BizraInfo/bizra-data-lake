import { useState, useEffect, useRef, useCallback, useMemo } from "react";

// ═══════════════════════════════════════════════════════════════
//  NODE0 ALPHA-100 DASHBOARD — "My AI Knows Me"
//  Sacred Geometry × Sovereign AI × إحسان
//  Standing on: bizra-node v0.1.0 protocol
// ═══════════════════════════════════════════════════════════════

const PALETTE = {
  void: "#08080e",
  surface: "#0f0f18",
  panel: "#141422",
  border: "#1e1e30",
  borderLight: "#2a2a44",
  gold: "#d4a853",
  goldDim: "#a07830",
  goldGlow: "#d4a85340",
  amber: "#c17f24",
  lightGold: "#e8d5a3",
  warmWhite: "#e8e0d0",
  dimText: "#8888a0",
  green: "#4ade80",
  greenDim: "#22c55e40",
  red: "#f87171",
  redDim: "#ef444440",
  blue: "#60a5fa",
  purple: "#a78bfa",
};

const PAT_AGENTS = [
  { id: "navigator", name: "Navigator", icon: "⊛", role: "Conversation pilot", color: "#60a5fa" },
  { id: "scholar", name: "Scholar", icon: "◈", role: "Knowledge synthesis", color: "#a78bfa" },
  { id: "artisan", name: "Artisan", icon: "◇", role: "Creative expression", color: "#f472b6" },
  { id: "guardian", name: "Guardian", icon: "◆", role: "Safety & ethics", color: "#4ade80" },
  { id: "mentor", name: "Mentor", icon: "◎", role: "Growth guidance", color: "#fbbf24" },
  { id: "diplomat", name: "Diplomat", icon: "◉", role: "Tone calibration", color: "#fb923c" },
  { id: "oracle", name: "Oracle", icon: "◊", role: "Pattern prediction", color: "#c084fc" },
];

// Sacred geometry SVG patterns
function SeedOfLife({ size = 200, opacity = 0.08 }) {
  const r = size * 0.2;
  const cx = size / 2;
  const cy = size / 2;
  const circles = [{ x: cx, y: cy }];
  for (let i = 0; i < 6; i++) {
    const angle = (Math.PI / 3) * i - Math.PI / 2;
    circles.push({ x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) });
  }
  return (
    <svg width={size} height={size} style={{ position: "absolute", top: 0, left: 0, opacity, pointerEvents: "none" }}>
      {circles.map((c, i) => (
        <circle key={i} cx={c.x} cy={c.y} r={r} fill="none" stroke={PALETTE.gold} strokeWidth="0.5">
          <animate attributeName="r" values={`${r};${r * 1.02};${r}`} dur={`${4 + i * 0.5}s`} repeatCount="indefinite" />
        </circle>
      ))}
    </svg>
  );
}

function HexGrid({ width = 300, height = 300, cellSize = 30, opacity = 0.04 }) {
  const hexes = [];
  const h = cellSize * Math.sqrt(3);
  for (let row = 0; row < height / h + 1; row++) {
    for (let col = 0; col < width / (cellSize * 1.5) + 1; col++) {
      const x = col * cellSize * 1.5;
      const y = row * h + (col % 2 === 1 ? h / 2 : 0);
      const pts = [];
      for (let i = 0; i < 6; i++) {
        const a = (Math.PI / 3) * i;
        pts.push(`${x + cellSize * Math.cos(a)},${y + cellSize * Math.sin(a)}`);
      }
      hexes.push(<polygon key={`${row}-${col}`} points={pts.join(" ")} fill="none" stroke={PALETTE.gold} strokeWidth="0.3" />);
    }
  }
  return (
    <svg width={width} height={height} style={{ position: "absolute", top: 0, right: 0, opacity, pointerEvents: "none" }}>
      {hexes}
    </svg>
  );
}

// Radial "Knows Me" Score — the living heart
function KnowsMeOrb({ score, size = 180 }) {
  const pct = Math.min(score, 1);
  const circumference = 2 * Math.PI * 70;
  const dashOffset = circumference * (1 - pct);

  return (
    <div style={{ position: "relative", width: size, height: size, margin: "0 auto" }}>
      <svg width={size} height={size} viewBox="0 0 180 180">
        {/* Outer glow */}
        <defs>
          <radialGradient id="orbGlow">
            <stop offset="0%" stopColor={PALETTE.gold} stopOpacity="0.15" />
            <stop offset="100%" stopColor={PALETTE.gold} stopOpacity="0" />
          </radialGradient>
          <filter id="goldBlur">
            <feGaussianBlur stdDeviation="3" />
          </filter>
        </defs>
        <circle cx="90" cy="90" r="85" fill="url(#orbGlow)">
          <animate attributeName="r" values="82;88;82" dur="4s" repeatCount="indefinite" />
        </circle>
        {/* Track */}
        <circle cx="90" cy="90" r="70" fill="none" stroke={PALETTE.border} strokeWidth="4" />
        {/* Progress arc */}
        <circle
          cx="90" cy="90" r="70" fill="none"
          stroke={PALETTE.gold}
          strokeWidth="4"
          strokeDasharray={circumference}
          strokeDashoffset={dashOffset}
          strokeLinecap="round"
          transform="rotate(-90 90 90)"
          filter="url(#goldBlur)"
          style={{ transition: "stroke-dashoffset 1.5s cubic-bezier(0.4,0,0.2,1)" }}
        />
        <circle
          cx="90" cy="90" r="70" fill="none"
          stroke={PALETTE.gold}
          strokeWidth="2"
          strokeDasharray={circumference}
          strokeDashoffset={dashOffset}
          strokeLinecap="round"
          transform="rotate(-90 90 90)"
          style={{ transition: "stroke-dashoffset 1.5s cubic-bezier(0.4,0,0.2,1)" }}
        />
        {/* Sacred inner ring */}
        <circle cx="90" cy="90" r="55" fill="none" stroke={PALETTE.goldDim} strokeWidth="0.5" opacity="0.4" />
        {/* Tick marks */}
        {Array.from({ length: 24 }).map((_, i) => {
          const a = (Math.PI * 2 / 24) * i - Math.PI / 2;
          const r1 = 62, r2 = 66;
          return (
            <line key={i}
              x1={90 + r1 * Math.cos(a)} y1={90 + r1 * Math.sin(a)}
              x2={90 + r2 * Math.cos(a)} y2={90 + r2 * Math.sin(a)}
              stroke={i < pct * 24 ? PALETTE.gold : PALETTE.border}
              strokeWidth={i % 6 === 0 ? "1.5" : "0.5"}
              opacity={i < pct * 24 ? 0.8 : 0.3}
            />
          );
        })}
      </svg>
      {/* Score text */}
      <div style={{
        position: "absolute", top: "50%", left: "50%", transform: "translate(-50%, -50%)",
        textAlign: "center",
      }}>
        <div style={{
          fontFamily: "'Cormorant Garamond', Georgia, serif",
          fontSize: "32px", fontWeight: 600, color: PALETTE.gold,
          lineHeight: 1,
        }}>
          {(pct * 100).toFixed(1)}
        </div>
        <div style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: "9px", color: PALETTE.dimText,
          letterSpacing: "2px", textTransform: "uppercase", marginTop: "4px",
        }}>
          knows me
        </div>
      </div>
    </div>
  );
}

// Protocol log entry
function ProtocolLine({ cmd, response, timestamp }) {
  const isOk = response?.startsWith("OK");
  return (
    <div style={{
      fontFamily: "'JetBrains Mono', monospace", fontSize: "11px",
      padding: "6px 10px", borderBottom: `1px solid ${PALETTE.border}`,
      display: "flex", gap: "8px", alignItems: "flex-start",
    }}>
      <span style={{ color: PALETTE.dimText, flexShrink: 0, fontSize: "9px", marginTop: "2px" }}>
        {timestamp}
      </span>
      <span style={{ color: PALETTE.amber, flexShrink: 0 }}>→</span>
      <span style={{ color: PALETTE.lightGold, flexShrink: 0 }}>{cmd}</span>
      {response && (
        <>
          <span style={{ color: PALETTE.dimText }}>│</span>
          <span style={{ color: isOk ? PALETTE.green : PALETTE.red, wordBreak: "break-all" }}>
            {response}
          </span>
        </>
      )}
    </div>
  );
}

// Agent card
function AgentCard({ agent, active, consulted }) {
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: "8px",
      padding: "8px 10px", borderRadius: "6px",
      background: consulted ? `${agent.color}10` : "transparent",
      border: `1px solid ${consulted ? agent.color + "40" : PALETTE.border}`,
      transition: "all 0.4s ease",
    }}>
      <span style={{
        fontSize: "16px", color: agent.color,
        opacity: active ? 1 : 0.3,
        filter: consulted ? `drop-shadow(0 0 4px ${agent.color})` : "none",
        transition: "all 0.4s ease",
      }}>
        {agent.icon}
      </span>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: "10px", color: active ? PALETTE.warmWhite : PALETTE.dimText,
          letterSpacing: "1px", textTransform: "uppercase",
        }}>
          {agent.name}
        </div>
        <div style={{
          fontSize: "9px", color: PALETTE.dimText, marginTop: "1px",
        }}>
          {agent.role}
        </div>
      </div>
      <div style={{
        width: "6px", height: "6px", borderRadius: "50%",
        background: active ? PALETTE.green : PALETTE.border,
        boxShadow: active ? `0 0 6px ${PALETTE.green}` : "none",
      }} />
    </div>
  );
}

// Trait badge
function TraitBadge({ label, value, confidence }) {
  return (
    <div style={{
      display: "inline-flex", alignItems: "center", gap: "6px",
      padding: "4px 10px", borderRadius: "12px",
      background: `${PALETTE.gold}0a`, border: `1px solid ${PALETTE.gold}20`,
      fontSize: "11px", fontFamily: "'JetBrains Mono', monospace",
    }}>
      <span style={{ color: PALETTE.dimText }}>{label}</span>
      <span style={{ color: PALETTE.lightGold }}>{value}</span>
      <span style={{
        color: PALETTE.gold, fontSize: "9px", opacity: 0.6,
      }}>
        {(confidence * 100).toFixed(0)}%
      </span>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
//  MAIN DASHBOARD
// ═══════════════════════════════════════════════════════════════

export default function Node0Dashboard() {
  const [connected, setConnected] = useState(false);
  const [nodeState, setNodeState] = useState("Dormant");
  const [ihsan, setIhsan] = useState(9900);
  const [knowsMe, setKnowsMe] = useState(0);
  const [messages, setMessages] = useState([]);
  const [protocolLog, setProtocolLog] = useState([]);
  const [traits, setTraits] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [teachInput, setTeachInput] = useState("");
  const [teachKind, setTeachKind] = useState("preference");
  const [activeAgents, setActiveAgents] = useState(new Set());
  const [consultedAgents, setConsultedAgents] = useState(new Set());
  const [commandsProcessed, setCommandsProcessed] = useState(0);
  const [sessionActive, setSessionActive] = useState(false);
  const [view, setView] = useState("chat"); // chat | protocol | agents | teach
  const chatEndRef = useRef(null);
  const logEndRef = useRef(null);
  const [bootPhase, setBootPhase] = useState(0);

  const now = () => {
    const d = new Date();
    return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}`;
  };

  const addProtocol = useCallback((cmd, response) => {
    setProtocolLog((p) => [...p.slice(-50), { cmd, response, timestamp: now() }]);
    setCommandsProcessed((c) => c + 1);
  }, []);

  // Simulated protocol bridge (in production: WebSocket → stdio)
  const sendCommand = useCallback((cmd) => {
    const parts = cmd.split("\t");
    const verb = parts[0];

    // Simulate protocol responses
    const delay = 100 + Math.random() * 300;
    return new Promise((resolve) => {
      setTimeout(() => {
        let response = "";
        switch (verb) {
          case "VERSION":
            response = "OK\tnode=bizra-node\tversion=0.1.0\tprotocol=1.0";
            break;
          case "PING":
            response = "OK\tpong=true";
            break;
          case "HEALTH":
            response = `OK\tstate=${nodeState}\tihsan=${ihsan}\tknows_me=${knowsMe.toFixed(4)}\tagents_registered=7`;
            break;
          case "START_SESSION":
            setSessionActive(true);
            setNodeState("Conversing");
            response = `OK\tsession_started=true\tuser=${parts[1] || "user_1"}`;
            break;
          case "END_SESSION":
            setSessionActive(false);
            setNodeState("Ready");
            const insights = Math.floor(Math.random() * 3) + 1;
            response = `OK\tsession_ended=true\tinsights_generated=${insights}`;
            break;
          case "RECEIVE": {
            const content = parts.slice(1).join("\t");
            const agents = 3 + Math.floor(Math.random() * 4);
            const fragments = 1 + Math.floor(Math.random() * 3);
            const conf = 0.7 + Math.random() * 0.25;
            const newKnows = Math.min(knowsMe + 0.003 + Math.random() * 0.008, 1);
            setKnowsMe(newKnows);

            const consultIds = new Set();
            const shuffled = [...PAT_AGENTS].sort(() => Math.random() - 0.5);
            for (let i = 0; i < agents && i < shuffled.length; i++) {
              consultIds.add(shuffled[i].id);
            }
            setConsultedAgents(consultIds);
            setTimeout(() => setConsultedAgents(new Set()), 3000);

            // Generate contextual response
            const responses = [
              `I notice you're interested in ${content.split(" ").slice(0, 3).join(" ")}. That aligns with patterns I've seen in our conversations.`,
              `Interesting perspective. My Scholar agent synthesized ${fragments} new knowledge fragments from this.`,
              `Processing through ${agents} agents. I'm beginning to understand your thinking patterns here.`,
              `This deepens what I know about you. Knowledge confidence is growing — ${(conf * 100).toFixed(1)}%.`,
              `My Oracle agent sees a pattern: this connects to topics you've explored before.`,
            ];
            const aiResponse = responses[Math.floor(Math.random() * responses.length)];

            setMessages((m) => [...m, {
              role: "assistant", content: aiResponse,
              meta: { agents, fragments, confidence: conf, knowsMe: newKnows }
            }]);

            // Extract simulated traits
            const words = content.toLowerCase().split(" ");
            if (words.some(w => ["rust", "code", "programming", "typescript", "python"].includes(w))) {
              setTraits(t => {
                if (!t.find(x => x.label === "expertise")) {
                  return [...t, { label: "expertise", value: "systems programming", confidence: conf }];
                }
                return t;
              });
            }
            if (words.some(w => ["design", "ui", "beautiful", "aesthetic"].includes(w))) {
              setTraits(t => {
                if (!t.find(x => x.value === "visual design")) {
                  return [...t, { label: "interest", value: "visual design", confidence: conf }];
                }
                return t;
              });
            }

            response = `OK\tconfidence=${conf.toFixed(4)}\tagents_consulted=${agents}\tfragments_extracted=${fragments}\tguardian_approved=true\tknows_me=${newKnows.toFixed(4)}`;
            break;
          }
          case "TEACH": {
            const kind = parts[1] || "preference";
            const content = parts[2] || "";
            const teachConf = 0.9 + Math.random() * 0.1;
            const newKnows = Math.min(knowsMe + 0.01 + Math.random() * 0.02, 1);
            setKnowsMe(newKnows);
            setTraits(t => [...t, { label: kind, value: content, confidence: teachConf }]);
            response = `OK\ttaught=${content}\tkind=${kind}\tconfidence=${Math.floor(teachConf * 10000)}`;
            break;
          }
          case "SYNTHESIZE": {
            const insights = Math.floor(Math.random() * 3);
            const newKnows = Math.min(knowsMe + 0.005, 1);
            setKnowsMe(newKnows);
            response = `OK\tinsights_generated=${insights}\tknows_me=${newKnows.toFixed(4)}`;
            break;
          }
          case "KNOWS_ME":
            response = `OK\tknows_me=${knowsMe.toFixed(4)}`;
            break;
          case "IHSAN": {
            const newScore = parseInt(parts[1]) || ihsan;
            setIhsan(newScore);
            response = `OK\tihsan=${newScore}\tprevious=${ihsan}`;
            break;
          }
          case "SHUTDOWN":
            setConnected(false);
            setNodeState("Dormant");
            setSessionActive(false);
            response = "OK\tshutdown=true";
            break;
          default:
            response = `ERR\tunknown_command=${verb}`;
        }
        addProtocol(cmd, response);
        resolve(response);
      }, delay);
    });
  }, [nodeState, ihsan, knowsMe, addProtocol]);

  // Boot sequence
  const bootNode = useCallback(async () => {
    setBootPhase(1);
    await new Promise(r => setTimeout(r, 400));
    setConnected(true);
    setBootPhase(2);

    await sendCommand("VERSION");
    setBootPhase(3);
    await new Promise(r => setTimeout(r, 200));

    await sendCommand("PING");
    setBootPhase(4);
    await new Promise(r => setTimeout(r, 200));

    await sendCommand("HEALTH");
    setNodeState("Ready");
    setBootPhase(5);
    setActiveAgents(new Set(PAT_AGENTS.map(a => a.id)));
    await new Promise(r => setTimeout(r, 200));

    await sendCommand("START_SESSION\tuser_alpha");
    setBootPhase(6);
  }, [sendCommand]);

  // Auto-scroll
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [protocolLog]);

  const handleSend = async () => {
    if (!chatInput.trim() || !connected) return;
    const msg = chatInput.trim();
    setChatInput("");
    setMessages(m => [...m, { role: "user", content: msg }]);
    await sendCommand(`RECEIVE\t${msg}`);
  };

  const handleTeach = async () => {
    if (!teachInput.trim() || !connected) return;
    const content = teachInput.trim();
    setTeachInput("");
    await sendCommand(`TEACH\t${teachKind}\t${content}\t9500\t${Date.now()}`);
  };

  const ihsanPct = ihsan / 10000;
  const ihsanColor = ihsanPct >= 0.95 ? PALETTE.green : ihsanPct >= 0.85 ? PALETTE.gold : PALETTE.red;

  // ─── STYLES ───
  const styles = {
    root: {
      width: "100%", height: "100vh", background: PALETTE.void,
      display: "flex", flexDirection: "column", overflow: "hidden",
      fontFamily: "'Cormorant Garamond', Georgia, serif",
      color: PALETTE.warmWhite,
      position: "relative",
    },
    header: {
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "12px 20px", borderBottom: `1px solid ${PALETTE.border}`,
      background: PALETTE.surface, position: "relative", zIndex: 2,
    },
    brand: {
      display: "flex", alignItems: "center", gap: "12px",
    },
    logo: {
      width: "32px", height: "32px", borderRadius: "8px",
      background: `linear-gradient(135deg, ${PALETTE.gold}, ${PALETTE.amber})`,
      display: "flex", alignItems: "center", justifyContent: "center",
      fontSize: "16px", fontWeight: 700, color: PALETTE.void,
      fontFamily: "'JetBrains Mono', monospace",
      boxShadow: `0 0 20px ${PALETTE.goldGlow}`,
    },
    title: {
      fontSize: "18px", fontWeight: 600, letterSpacing: "1px", color: PALETTE.lightGold,
    },
    subtitle: {
      fontSize: "10px", fontFamily: "'JetBrains Mono', monospace",
      color: PALETTE.dimText, letterSpacing: "2px", textTransform: "uppercase",
    },
    statusBar: {
      display: "flex", alignItems: "center", gap: "16px",
      fontFamily: "'JetBrains Mono', monospace", fontSize: "10px",
    },
    statusDot: (active) => ({
      width: "7px", height: "7px", borderRadius: "50%",
      background: active ? PALETTE.green : PALETTE.red,
      boxShadow: active ? `0 0 8px ${PALETTE.green}` : "none",
    }),
    main: {
      flex: 1, display: "flex", overflow: "hidden", position: "relative",
    },
    sidebar: {
      width: "260px", borderRight: `1px solid ${PALETTE.border}`,
      background: PALETTE.surface, display: "flex", flexDirection: "column",
      overflow: "hidden", position: "relative",
    },
    content: {
      flex: 1, display: "flex", flexDirection: "column", overflow: "hidden",
      position: "relative",
    },
    navTab: (active) => ({
      padding: "8px 14px", cursor: "pointer",
      fontFamily: "'JetBrains Mono', monospace",
      fontSize: "10px", letterSpacing: "1.5px", textTransform: "uppercase",
      color: active ? PALETTE.gold : PALETTE.dimText,
      borderBottom: active ? `2px solid ${PALETTE.gold}` : "2px solid transparent",
      transition: "all 0.3s ease",
      background: "transparent", border: "none",
    }),
    chatArea: {
      flex: 1, overflowY: "auto", padding: "16px 20px",
      display: "flex", flexDirection: "column", gap: "12px",
    },
    inputBar: {
      padding: "12px 16px", borderTop: `1px solid ${PALETTE.border}`,
      background: PALETTE.surface, display: "flex", gap: "8px",
    },
    input: {
      flex: 1, background: PALETTE.panel, border: `1px solid ${PALETTE.border}`,
      borderRadius: "8px", padding: "10px 14px", color: PALETTE.warmWhite,
      fontFamily: "'Cormorant Garamond', Georgia, serif", fontSize: "15px",
      outline: "none", transition: "border-color 0.3s",
    },
    sendBtn: {
      padding: "10px 20px", borderRadius: "8px", border: "none",
      background: `linear-gradient(135deg, ${PALETTE.gold}, ${PALETTE.amber})`,
      color: PALETTE.void, fontFamily: "'JetBrains Mono', monospace",
      fontSize: "11px", fontWeight: 700, letterSpacing: "1px",
      cursor: "pointer", textTransform: "uppercase",
      transition: "all 0.3s", opacity: connected ? 1 : 0.4,
    },
    msgBubble: (isUser) => ({
      maxWidth: "80%", padding: "10px 14px", borderRadius: "12px",
      alignSelf: isUser ? "flex-end" : "flex-start",
      background: isUser ? `${PALETTE.gold}15` : PALETTE.panel,
      border: `1px solid ${isUser ? PALETTE.gold + "30" : PALETTE.border}`,
      fontSize: "14px", lineHeight: 1.5,
    }),
    bootOverlay: {
      position: "absolute", inset: 0, zIndex: 100,
      background: PALETTE.void,
      display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center",
      gap: "24px", transition: "opacity 0.8s ease",
    },
  };

  // ─── BOOT SCREEN ───
  if (!connected && bootPhase === 0) {
    return (
      <div style={styles.root}>
        <div style={styles.bootOverlay}>
          <SeedOfLife size={220} opacity={0.15} />
          <div style={{
            width: "80px", height: "80px", borderRadius: "16px",
            background: `linear-gradient(135deg, ${PALETTE.gold}, ${PALETTE.amber})`,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: "36px", fontWeight: 700, color: PALETTE.void,
            fontFamily: "'JetBrains Mono', monospace",
            boxShadow: `0 0 60px ${PALETTE.goldGlow}`,
            position: "relative", zIndex: 2,
          }}>
            N0
          </div>
          <div style={{
            fontFamily: "'Cormorant Garamond', Georgia, serif",
            fontSize: "28px", fontWeight: 600, color: PALETTE.lightGold,
            letterSpacing: "2px", position: "relative", zIndex: 2,
          }}>
            Node0 Alpha-100
          </div>
          <div style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: "11px", color: PALETTE.dimText,
            letterSpacing: "3px", textTransform: "uppercase", position: "relative", zIndex: 2,
          }}>
            sovereign ai · personal genius · إحسان
          </div>
          <button
            onClick={bootNode}
            style={{
              marginTop: "24px", padding: "14px 40px",
              borderRadius: "10px", border: `1px solid ${PALETTE.gold}40`,
              background: `linear-gradient(135deg, ${PALETTE.gold}20, ${PALETTE.amber}20)`,
              color: PALETTE.gold, fontFamily: "'Cormorant Garamond', Georgia, serif",
              fontSize: "16px", fontWeight: 600, letterSpacing: "2px",
              cursor: "pointer", transition: "all 0.3s",
              position: "relative", zIndex: 2,
            }}
            onMouseOver={e => { e.target.style.background = `linear-gradient(135deg, ${PALETTE.gold}35, ${PALETTE.amber}35)`; e.target.style.boxShadow = `0 0 30px ${PALETTE.goldGlow}`; }}
            onMouseOut={e => { e.target.style.background = `linear-gradient(135deg, ${PALETTE.gold}20, ${PALETTE.amber}20)`; e.target.style.boxShadow = "none"; }}
          >
            Awaken Node
          </button>
          <div style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: "9px", color: PALETTE.dimText,
            position: "absolute", bottom: "20px",
            letterSpacing: "2px", opacity: 0.4,
          }}>
            BIZRA DISTRIBUTED AI · v0.1.0-GENESIS
          </div>
        </div>
      </div>
    );
  }

  // ─── BOOTING OVERLAY ───
  if (bootPhase > 0 && bootPhase < 6) {
    const phases = ["", "Spawning binary...", "Connecting protocol...", "Handshake...", "Health check...", "Starting session..."];
    return (
      <div style={styles.root}>
        <div style={{ ...styles.bootOverlay, opacity: 1 }}>
          <SeedOfLife size={180} opacity={0.1} />
          <div style={{
            width: "60px", height: "60px", borderRadius: "12px",
            background: `linear-gradient(135deg, ${PALETTE.gold}, ${PALETTE.amber})`,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: "24px", fontWeight: 700, color: PALETTE.void,
            fontFamily: "'JetBrains Mono', monospace",
            boxShadow: `0 0 40px ${PALETTE.goldGlow}`,
            position: "relative", zIndex: 2,
          }}>
            N0
          </div>
          <div style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: "12px", color: PALETTE.gold,
            letterSpacing: "2px", position: "relative", zIndex: 2,
          }}>
            {phases[bootPhase]}
          </div>
          <div style={{
            display: "flex", gap: "6px", position: "relative", zIndex: 2,
          }}>
            {[1,2,3,4,5].map(i => (
              <div key={i} style={{
                width: "8px", height: "8px", borderRadius: "50%",
                background: i <= bootPhase ? PALETTE.gold : PALETTE.border,
                boxShadow: i <= bootPhase ? `0 0 8px ${PALETTE.goldGlow}` : "none",
                transition: "all 0.3s ease",
              }} />
            ))}
          </div>
        </div>
      </div>
    );
  }

  // ─── MAIN DASHBOARD ───
  return (
    <div style={styles.root}>
      {/* Header */}
      <div style={styles.header}>
        <div style={styles.brand}>
          <div style={styles.logo}>N0</div>
          <div>
            <div style={styles.title}>Node0</div>
            <div style={styles.subtitle}>alpha-100 · sovereign instance</div>
          </div>
        </div>
        <div style={styles.statusBar}>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <div style={styles.statusDot(connected)} />
            <span style={{ color: connected ? PALETTE.green : PALETTE.red }}>
              {connected ? "LIVE" : "OFFLINE"}
            </span>
          </div>
          <span style={{ color: PALETTE.border }}>│</span>
          <span style={{ color: PALETTE.dimText }}>
            STATE: <span style={{ color: PALETTE.warmWhite }}>{nodeState}</span>
          </span>
          <span style={{ color: PALETTE.border }}>│</span>
          <span style={{ color: PALETTE.dimText }}>
            إحسان: <span style={{ color: ihsanColor }}>{(ihsanPct * 100).toFixed(1)}%</span>
          </span>
          <span style={{ color: PALETTE.border }}>│</span>
          <span style={{ color: PALETTE.dimText }}>
            CMDS: <span style={{ color: PALETTE.warmWhite }}>{commandsProcessed}</span>
          </span>
          {sessionActive && (
            <>
              <span style={{ color: PALETTE.border }}>│</span>
              <button
                onClick={() => sendCommand("END_SESSION")}
                style={{
                  background: "none", border: `1px solid ${PALETTE.red}40`,
                  borderRadius: "4px", padding: "2px 8px",
                  color: PALETTE.red, fontSize: "9px", cursor: "pointer",
                  fontFamily: "'JetBrains Mono', monospace", letterSpacing: "1px",
                }}
              >
                END SESSION
              </button>
            </>
          )}
        </div>
      </div>

      <div style={styles.main}>
        {/* ─── LEFT SIDEBAR ─── */}
        <div style={styles.sidebar}>
          <SeedOfLife size={260} opacity={0.04} />

          {/* Knows Me Score */}
          <div style={{ padding: "20px 16px 12px", textAlign: "center", position: "relative", zIndex: 1 }}>
            <KnowsMeOrb score={knowsMe} size={160} />
            <div style={{
              marginTop: "8px", fontFamily: "'JetBrains Mono', monospace",
              fontSize: "9px", color: PALETTE.dimText, letterSpacing: "2px",
            }}>
              KNOWLEDGE DEPTH
            </div>
          </div>

          {/* PAT Agents */}
          <div style={{
            padding: "8px 12px",
            borderTop: `1px solid ${PALETTE.border}`,
            flex: 1, overflowY: "auto",
          }}>
            <div style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: "9px", color: PALETTE.dimText, letterSpacing: "2px",
              textTransform: "uppercase", padding: "4px 0 8px",
            }}>
              Personal Agent Team
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
              {PAT_AGENTS.map(agent => (
                <AgentCard
                  key={agent.id}
                  agent={agent}
                  active={activeAgents.has(agent.id)}
                  consulted={consultedAgents.has(agent.id)}
                />
              ))}
            </div>
          </div>

          {/* Learned Traits */}
          {traits.length > 0 && (
            <div style={{
              padding: "10px 12px",
              borderTop: `1px solid ${PALETTE.border}`,
              maxHeight: "120px", overflowY: "auto",
            }}>
              <div style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: "9px", color: PALETTE.dimText, letterSpacing: "2px",
                textTransform: "uppercase", padding: "0 0 6px",
              }}>
                Learned Traits
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "4px" }}>
                {traits.map((t, i) => (
                  <TraitBadge key={i} {...t} />
                ))}
              </div>
            </div>
          )}

          {/* Quick Actions */}
          <div style={{
            padding: "10px 12px", borderTop: `1px solid ${PALETTE.border}`,
            display: "flex", gap: "4px", flexWrap: "wrap",
          }}>
            {[
              { label: "SYNTH", cmd: "SYNTHESIZE" },
              { label: "HEALTH", cmd: "HEALTH" },
              { label: "PING", cmd: "PING" },
            ].map(action => (
              <button
                key={action.label}
                onClick={() => sendCommand(action.cmd)}
                style={{
                  padding: "4px 10px", borderRadius: "4px",
                  border: `1px solid ${PALETTE.border}`,
                  background: PALETTE.panel, color: PALETTE.dimText,
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: "9px", letterSpacing: "1px", cursor: "pointer",
                  transition: "all 0.2s",
                }}
                onMouseOver={e => { e.target.style.borderColor = PALETTE.gold + "60"; e.target.style.color = PALETTE.gold; }}
                onMouseOut={e => { e.target.style.borderColor = PALETTE.border; e.target.style.color = PALETTE.dimText; }}
              >
                {action.label}
              </button>
            ))}
          </div>
        </div>

        {/* ─── MAIN CONTENT ─── */}
        <div style={styles.content}>
          <HexGrid width={800} height={600} cellSize={40} opacity={0.02} />

          {/* Tabs */}
          <div style={{
            display: "flex", borderBottom: `1px solid ${PALETTE.border}`,
            background: PALETTE.surface, position: "relative", zIndex: 1,
          }}>
            {[
              { id: "chat", label: "Converse" },
              { id: "teach", label: "Teach" },
              { id: "protocol", label: "Protocol" },
            ].map(tab => (
              <button key={tab.id} onClick={() => setView(tab.id)} style={styles.navTab(view === tab.id)}>
                {tab.label}
              </button>
            ))}
          </div>

          {/* Chat View */}
          {view === "chat" && (
            <>
              <div style={styles.chatArea}>
                {messages.length === 0 && (
                  <div style={{
                    flex: 1, display: "flex", flexDirection: "column",
                    alignItems: "center", justifyContent: "center",
                    gap: "12px", opacity: 0.4,
                  }}>
                    <div style={{ fontSize: "40px", color: PALETTE.gold }}>◎</div>
                    <div style={{ fontSize: "16px", color: PALETTE.dimText, textAlign: "center" }}>
                      Your sovereign AI node is listening.
                    </div>
                    <div style={{
                      fontFamily: "'JetBrains Mono', monospace",
                      fontSize: "10px", color: PALETTE.dimText,
                      letterSpacing: "2px",
                    }}>
                      Every message teaches me who you are.
                    </div>
                  </div>
                )}
                {messages.map((msg, i) => (
                  <div key={i}>
                    <div style={styles.msgBubble(msg.role === "user")}>
                      <div style={{ color: msg.role === "user" ? PALETTE.lightGold : PALETTE.warmWhite }}>
                        {msg.content}
                      </div>
                      {msg.meta && (
                        <div style={{
                          marginTop: "6px", fontFamily: "'JetBrains Mono', monospace",
                          fontSize: "9px", color: PALETTE.dimText, display: "flex", gap: "8px",
                        }}>
                          <span>⊛ {msg.meta.agents} agents</span>
                          <span>◈ {msg.meta.fragments} fragments</span>
                          <span>↑ {(msg.meta.confidence * 100).toFixed(0)}%</span>
                          <span style={{ color: PALETTE.gold }}>
                            ◎ {(msg.meta.knowsMe * 100).toFixed(1)}%
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                <div ref={chatEndRef} />
              </div>
              <div style={styles.inputBar}>
                <input
                  type="text"
                  value={chatInput}
                  onChange={e => setChatInput(e.target.value)}
                  onKeyDown={e => e.key === "Enter" && handleSend()}
                  onFocus={e => e.target.style.borderColor = PALETTE.gold + "60"}
                  onBlur={e => e.target.style.borderColor = PALETTE.border}
                  placeholder="Speak to your node..."
                  style={styles.input}
                />
                <button onClick={handleSend} style={styles.sendBtn} disabled={!connected}>
                  SEND
                </button>
              </div>
            </>
          )}

          {/* Teach View */}
          {view === "teach" && (
            <div style={{ flex: 1, padding: "24px", overflowY: "auto", position: "relative", zIndex: 1 }}>
              <div style={{
                fontSize: "20px", fontWeight: 600, color: PALETTE.lightGold,
                marginBottom: "4px",
              }}>
                Teach Your Node
              </div>
              <div style={{
                fontSize: "13px", color: PALETTE.dimText, marginBottom: "24px",
                lineHeight: 1.6,
              }}>
                Direct knowledge injection. Tell your AI what matters to you, what you know, how you think.
                Every teaching strengthens the bond between you and your sovereign intelligence.
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: "16px", maxWidth: "500px" }}>
                <div>
                  <div style={{
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: "10px", color: PALETTE.dimText,
                    letterSpacing: "1.5px", marginBottom: "6px", textTransform: "uppercase",
                  }}>
                    Knowledge Type
                  </div>
                  <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
                    {["preference", "expertise", "fact", "goal", "style", "emotion"].map(kind => (
                      <button
                        key={kind}
                        onClick={() => setTeachKind(kind)}
                        style={{
                          padding: "6px 14px", borderRadius: "6px",
                          border: `1px solid ${teachKind === kind ? PALETTE.gold + "60" : PALETTE.border}`,
                          background: teachKind === kind ? `${PALETTE.gold}15` : "transparent",
                          color: teachKind === kind ? PALETTE.gold : PALETTE.dimText,
                          fontFamily: "'JetBrains Mono', monospace",
                          fontSize: "10px", letterSpacing: "1px", cursor: "pointer",
                          textTransform: "uppercase", transition: "all 0.2s",
                        }}
                      >
                        {kind}
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <div style={{
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: "10px", color: PALETTE.dimText,
                    letterSpacing: "1.5px", marginBottom: "6px", textTransform: "uppercase",
                  }}>
                    Knowledge Content
                  </div>
                  <textarea
                    value={teachInput}
                    onChange={e => setTeachInput(e.target.value)}
                    onFocus={e => e.target.style.borderColor = PALETTE.gold + "60"}
                    onBlur={e => e.target.style.borderColor = PALETTE.border}
                    placeholder={`e.g., "I value clean architecture over quick hacks"`}
                    rows={3}
                    style={{
                      ...styles.input, width: "100%", resize: "vertical",
                      fontFamily: "'Cormorant Garamond', Georgia, serif", fontSize: "15px",
                    }}
                  />
                </div>
                <button
                  onClick={handleTeach}
                  disabled={!teachInput.trim() || !connected}
                  style={{
                    ...styles.sendBtn, alignSelf: "flex-start",
                    opacity: teachInput.trim() && connected ? 1 : 0.4,
                  }}
                >
                  TEACH NODE
                </button>
              </div>

              {/* Trait list */}
              {traits.length > 0 && (
                <div style={{ marginTop: "32px" }}>
                  <div style={{
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: "10px", color: PALETTE.dimText,
                    letterSpacing: "1.5px", marginBottom: "10px", textTransform: "uppercase",
                  }}>
                    Knowledge Graph · {traits.length} traits
                  </div>
                  <div style={{
                    display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
                    gap: "8px",
                  }}>
                    {traits.map((t, i) => (
                      <div key={i} style={{
                        padding: "10px 14px", borderRadius: "8px",
                        background: PALETTE.panel, border: `1px solid ${PALETTE.border}`,
                      }}>
                        <div style={{
                          fontFamily: "'JetBrains Mono', monospace",
                          fontSize: "9px", color: PALETTE.gold,
                          letterSpacing: "1.5px", textTransform: "uppercase",
                          marginBottom: "4px",
                        }}>
                          {t.label}
                        </div>
                        <div style={{ fontSize: "13px", color: PALETTE.warmWhite }}>
                          {t.value}
                        </div>
                        <div style={{
                          marginTop: "6px", height: "2px", borderRadius: "1px",
                          background: PALETTE.border,
                        }}>
                          <div style={{
                            width: `${t.confidence * 100}%`, height: "100%",
                            borderRadius: "1px", background: PALETTE.gold,
                            transition: "width 0.5s ease",
                          }} />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Protocol View */}
          {view === "protocol" && (
            <div style={{ flex: 1, display: "flex", flexDirection: "column", position: "relative", zIndex: 1 }}>
              <div style={{
                flex: 1, overflowY: "auto", background: PALETTE.panel,
              }}>
                {protocolLog.length === 0 && (
                  <div style={{
                    padding: "40px", textAlign: "center",
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: "11px", color: PALETTE.dimText,
                  }}>
                    Protocol log empty. Send commands to see raw protocol traffic.
                  </div>
                )}
                {protocolLog.map((entry, i) => (
                  <ProtocolLine key={i} {...entry} />
                ))}
                <div ref={logEndRef} />
              </div>
              {/* Raw command input */}
              <div style={styles.inputBar}>
                <input
                  type="text"
                  onKeyDown={e => {
                    if (e.key === "Enter" && e.target.value.trim()) {
                      sendCommand(e.target.value.trim());
                      e.target.value = "";
                    }
                  }}
                  onFocus={e => e.target.style.borderColor = PALETTE.gold + "60"}
                  onBlur={e => e.target.style.borderColor = PALETTE.border}
                  placeholder="Raw protocol command (e.g., PING, HEALTH, SYNTHESIZE)"
                  style={{ ...styles.input, fontFamily: "'JetBrains Mono', monospace", fontSize: "12px" }}
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
