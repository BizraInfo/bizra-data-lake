import { useState, useEffect, useRef, useCallback } from "react";

// ═══ SACRED PALETTE ═══
const C = {
  bg: "#050B14",
  bgCard: "#0A1628",
  gold: "#C9A962",
  goldDim: "#8A6B2E",
  goldLight: "#F9F1D8",
  goldGlow: "rgba(201,169,98,0.15)",
  goldGlow2: "rgba(201,169,98,0.06)",
  text: "#F8F6F1",
  textDim: "rgba(248,246,241,0.4)",
  textMid: "rgba(248,246,241,0.6)",
  border: "rgba(201,169,98,0.12)",
  borderFaint: "rgba(255,255,255,0.04)",
  green: "#22c55e",
  cyan: "#06b6d4",
  purple: "#a855f7",
  red: "#ef4444",
  amber: "#f59e0b",
  rose: "#f43f5e",
  orange: "#f97316",
  blue: "#3b82f6",
};

// ═══ PAT-7 AGENTS ═══
const AGENTS = {
  P1: { name: "Planner", call: "ATLAS", color: C.blue, icon: "◇",
    greeting: "Standing by for mission parameters.",
    idle: ["Analyzing priority queue...", "Three pending objectives identified.", "Shall I restructure your schedule?", "Your roadmap has a dependency conflict I can resolve."],
    working: ["Decomposing into subtasks...", "Dependency graph resolved.", "Execution order optimized.", "Critical path identified — 3 steps."],
  },
  P2: { name: "Researcher", call: "ORACLE", color: C.cyan, icon: "◈",
    greeting: "Knowledge systems online. What shall I find?",
    idle: ["I found something interesting in your domain...", "Three new papers match your interests.", "Your knowledge graph grew 12% this week.", "Shall I deep-dive on that topic from yesterday?"],
    working: ["Scanning knowledge base...", "Cross-referencing 47 sources.", "Signal-to-noise ratio: 0.94.", "Synthesis complete. Key findings extracted."],
  },
  P3: { name: "Coder", call: "FORGE", color: C.green, icon: "⬡",
    greeting: "Compiler ready. What are we building?",
    idle: ["Your test suite has 3 flaky tests I can fix.", "I spotted a refactoring opportunity.", "Build pipeline green. All 219 tests passing.", "Dependency update available — no breaking changes."],
    working: ["Generating implementation...", "Running test suite...", "All assertions pass.", "Code quality: Ihsan 0.97."],
  },
  P4: { name: "Evaluator", call: "JUDGE", color: C.amber, icon: "◆",
    greeting: "Quality gates armed. Show me what to assess.",
    idle: ["Your average Ihsan is trending up — 0.983.", "I've benchmarked 3 alternatives.", "Quality score: top 5% of all nodes.", "Recommending peer review for your latest reflex."],
    working: ["Running quality assessment...", "Shannon entropy: above threshold.", "Scoring against rubric...", "Verdict: exceeds constitutional floor."],
  },
  P5: { name: "Ethicist", call: "CROWN", color: C.rose, icon: "✦",
    greeting: "Constitutional watch active. All seven invariants holding.",
    idle: ["All invariants satisfied. System is constitutional.", "I-3 check: Gini at 0.31 — within bounds.", "No ethical flags in recent actions.", "The covenant holds. Integrity verified."],
    working: ["Scanning against I-1 through I-7...", "Shariah compliance: verified.", "No bias detected in output.", "Constitutional clearance granted."],
  },
  P6: { name: "Publisher", call: "HERALD", color: C.orange, icon: "▹",
    greeting: "Ready to deliver your message to the world.",
    idle: ["Your last report scored 4.8/5.0 readability.", "I've drafted three versions of your response.", "Format optimized for your audience.", "Feedback from last delivery was excellent."],
    working: ["Structuring output...", "Formatting for clarity...", "Final polish applied.", "Ready for delivery. Shall I publish?"],
  },
  P7: { name: "Integrator", call: "NEXUS", color: C.purple, icon: "⟡",
    greeting: "All agents reporting. Nexus is online.",
    idle: ["All seven agents nominal.", "Memory utilization: optimal.", "I've pre-loaded context from your last session.", "Cross-agent coordination score: 94%."],
    working: ["Routing to specialist...", "Context bridge established.", "Agent handoff complete.", "Aggregating results from all sources."],
  },
};

const SAT = [
  { id: "S1", name: "Sentinel", color: C.red },
  { id: "S2", name: "Oracle", color: C.gold },
  { id: "S3", name: "Ledger", color: C.amber },
  { id: "S4", name: "Conductor", color: C.blue },
  { id: "S5", name: "Ambassador", color: C.cyan },
];

const MISSIONS = [
  "Research the latest developments in sovereign AI architectures",
  "Build a testing framework for constitutional invariant verification",
  "Evaluate our deployment pipeline against production benchmarks",
  "Draft the quarterly progress report for stakeholders",
  "Plan the Alpha-100 rollout strategy and timeline",
  "Review the authentication module for security vulnerabilities",
];

// ═══ INIT STATE ═══
const initState = () => ({
  booted: false, phase: "idle",
  seed: 0, bloom: 0, rac: 0, vac: 0, level: 0,
  ihsan: 0, streak: 0, mye: 0, s1: 0, s2: 0,
  reflexes: 0, legendary: 0, epic: 0,
  messages: [], activeMission: null,
  agentStates: Object.fromEntries(Object.keys(AGENTS).map(k => [k, "idle"])),
  trustChecks: { node: null, ledger: null, token: null, supply: null, gate: null },
  ghostMessages: [],
});

// ═══ SACRED GEOMETRY SVG ═══
function SeedOfLife({ size = 80, opacity = 0.15, animate = false }) {
  const r = size * 0.2;
  const centers = [
    [0, 0], [0, -r], [r * 0.866, -r * 0.5], [r * 0.866, r * 0.5],
    [0, r], [-r * 0.866, r * 0.5], [-r * 0.866, -r * 0.5],
  ];
  return (
    <svg width={size} height={size} viewBox={`${-size/2} ${-size/2} ${size} ${size}`} style={{ opacity }}>
      {centers.map(([cx, cy], i) => (
        <circle key={i} cx={cx} cy={cy} r={r} fill="none" stroke={C.gold} strokeWidth="0.5"
          style={animate ? { animation: `seedPulse 3s ease-in-out ${i * 0.2}s infinite` } : {}} />
      ))}
    </svg>
  );
}

// ═══ TRUST PANEL ═══
function TrustPanel({ checks }) {
  const items = [
    { key: "node", label: "Node Health", icon: "●" },
    { key: "ledger", label: "SEL Integrity", icon: "◈" },
    { key: "token", label: "SEED Balance", icon: "◆" },
    { key: "supply", label: "Supply Cap", icon: "▣" },
    { key: "gate", label: "Ihsan Gate", icon: "✦" },
  ];
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
      {items.map(({ key, label, icon }) => {
        const v = checks[key];
        const color = v === true ? C.green : v === false ? C.red : C.textDim;
        return (
          <div key={key} style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "10px" }}>
            <span style={{ color, fontSize: "8px", transition: "color 0.5s" }}>{icon}</span>
            <span style={{ color: v === true ? C.textMid : C.textDim, flex: 1, letterSpacing: "0.5px" }}>{label}</span>
            <span style={{ color, fontSize: "9px", fontFamily: "monospace", letterSpacing: "1px" }}>
              {v === true ? "VERIFIED" : v === false ? "FAILED" : "—"}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ═══ AGENT RING ═══
function AgentRing({ agents, states, activeId }) {
  const entries = Object.entries(agents);
  const angleStep = (2 * Math.PI) / entries.length;
  const radius = 72;
  return (
    <div style={{ position: "relative", width: "180px", height: "180px", margin: "0 auto" }}>
      {/* Center seed */}
      <div style={{ position: "absolute", top: "50%", left: "50%", transform: "translate(-50%,-50%)" }}>
        <SeedOfLife size={50} opacity={0.25} animate />
      </div>
      {entries.map(([id, ag], i) => {
        const angle = angleStep * i - Math.PI / 2;
        const x = Math.cos(angle) * radius + 90;
        const y = Math.sin(angle) * radius + 90;
        const isActive = states[id] !== "idle";
        const isCurrent = id === activeId;
        return (
          <div key={id} style={{
            position: "absolute", left: x - 18, top: y - 18,
            width: "36px", height: "36px", borderRadius: "50%",
            display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
            background: isCurrent ? `${ag.color}25` : isActive ? `${ag.color}12` : "rgba(255,255,255,0.02)",
            border: `1px solid ${isCurrent ? ag.color : isActive ? ag.color + "40" : C.borderFaint}`,
            transition: "all 0.5s ease",
            boxShadow: isCurrent ? `0 0 20px ${ag.color}30, inset 0 0 10px ${ag.color}10` : "none",
          }}>
            <span style={{ fontSize: "12px", color: isActive ? ag.color : C.textDim, lineHeight: 1 }}>{ag.icon}</span>
            <span style={{ fontSize: "5px", color: isActive ? ag.color : "#374151", letterSpacing: "0.5px", marginTop: "2px" }}>{ag.call}</span>
          </div>
        );
      })}
    </div>
  );
}

// ═══ GHOST PANEL ═══
function GhostPanel({ messages }) {
  if (!messages.length) return null;
  return (
    <div style={{
      background: `linear-gradient(135deg, ${C.goldGlow2}, transparent)`,
      border: `1px solid ${C.border}`,
      borderRadius: "8px", padding: "10px 12px",
      marginBottom: "8px",
    }}>
      <div style={{ fontSize: "8px", color: C.gold, letterSpacing: "2px", marginBottom: "6px", textTransform: "uppercase" }}>
        Ghost Panel — Proactive Intelligence
      </div>
      {messages.slice(-3).map((m, i) => (
        <div key={i} style={{ display: "flex", gap: "8px", alignItems: "flex-start", marginBottom: "4px", opacity: 0.8 }}>
          <span style={{ color: AGENTS[m.agent]?.color || C.gold, fontSize: "8px", minWidth: "40px", textAlign: "right", fontWeight: "bold" }}>
            {AGENTS[m.agent]?.call || "SYS"}
          </span>
          <span style={{ color: C.textMid, fontSize: "10px", fontStyle: "italic", lineHeight: 1.4 }}>{m.text}</span>
        </div>
      ))}
    </div>
  );
}

// ═══ MAIN COMPONENT ═══
export default function JARVIS() {
  const [st, setSt] = useState(initState());
  const [input, setInput] = useState("");
  const [time, setTime] = useState(new Date());
  const [showTrust, setShowTrust] = useState(false);
  const [currentAgent, setCurrentAgent] = useState(null);
  const msgEnd = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    if (msgEnd.current) msgEnd.current.scrollIntoView({ behavior: "smooth" });
  }, [st.messages]);

  const addMsg = useCallback((agent, text, type = "agent") => {
    setSt(p => ({ ...p, messages: [...p.messages, { agent, text, type, ts: Date.now() }].slice(-80) }));
  }, []);

  const addGhost = useCallback((agent, text) => {
    setSt(p => ({ ...p, ghostMessages: [...p.ghostMessages, { agent, text, ts: Date.now() }].slice(-5) }));
  }, []);

  const delay = ms => new Promise(r => setTimeout(r, ms));

  // ═══ BOOT ═══
  const boot = useCallback(async () => {
    setSt(p => ({ ...p, phase: "booting" }));
    const steps = [
      ["SYS", "بسم الله الرحمن الرحيم", 600],
      ["SYS", "Initializing sovereign kernel...", 350],
      ["SYS", "Fixed-point arithmetic verified. Deterministic.", 250],
      ["SYS", "Constitutional invariants I-1 through I-7: LOADED", 250],
      ["SYS", "Covenant hash: 859649ea...verified ✓", 200],
      ["SYS", "Ed25519 identity generated.", 200],
      ["SYS", "Minting Personal Agentic Team...", 400],
    ];
    for (const [agent, text, ms] of steps) {
      addMsg(agent, text, "system");
      await delay(ms);
    }
    // Boot each agent
    const agentOrder = ["P7", "P1", "P2", "P3", "P4", "P5", "P6"];
    for (const id of agentOrder) {
      const ag = AGENTS[id];
      addMsg(id, `${ag.call} online. ${ag.greeting}`, "agent");
      setSt(p => ({ ...p, agentStates: { ...p.agentStates, [id]: "booting" } }));
      await delay(250);
      setSt(p => ({ ...p, agentStates: { ...p.agentStates, [id]: "idle" } }));
    }
    addMsg("SYS", "SAT-5 oversight deployed. You have zero control over validators.", "system");
    await delay(200);
    // Trust verification
    setSt(p => ({ ...p, trustChecks: { node: true, ledger: true, token: true, supply: true, gate: true } }));
    addMsg("SYS", "Trust Panel: all 5 checks VERIFIED ✓", "system");
    await delay(300);
    addMsg("P7", "Your sovereign AI team is ready. What shall we build today?", "greeting");
    setSt(p => ({ ...p, booted: true, phase: "ready" }));
    setTimeout(() => inputRef.current?.focus(), 100);
    // Ghost proactive after 6s
    setTimeout(() => {
      const picks = ["P1", "P2", "P3", "P4"];
      const id = picks[Math.floor(Math.random() * picks.length)];
      addGhost(id, AGENTS[id].idle[Math.floor(Math.random() * AGENTS[id].idle.length)]);
    }, 6000);
  }, [addMsg, addGhost]);

  // ═══ MISSION ═══
  const executeMission = useCallback(async (task) => {
    setSt(p => ({ ...p, phase: "mission", activeMission: task }));
    addMsg("P7", `Mission received. Analyzing: "${task.slice(0, 70)}..."`, "agent");
    setCurrentAgent("P7");
    await delay(600);
    // Route
    const kw = {
      P1: ["plan", "organize", "strategy", "roadmap", "schedule", "timeline"],
      P2: ["research", "find", "analyze", "study", "paper", "latest"],
      P3: ["code", "build", "test", "debug", "implement", "fix", "deploy", "framework"],
      P4: ["evaluate", "score", "assess", "review", "benchmark"],
      P5: ["check", "verify", "compliance", "constitution", "ethics", "security"],
      P6: ["write", "draft", "report", "document", "publish", "present"],
    };
    let best = "P2", bs = 0;
    for (const [a, ws] of Object.entries(kw)) {
      const s = ws.filter(w => task.toLowerCase().includes(w)).length;
      if (s > bs) { best = a; bs = s; }
    }
    const agent = AGENTS[best];
    addMsg("P7", `Routing to ${agent.call}. Best capability match.`, "agent");
    setSt(p => ({ ...p, agentStates: { ...p.agentStates, [best]: "active", P7: "routing" } }));
    setCurrentAgent(best);
    await delay(500);
    for (const msg of agent.working) {
      addMsg(best, msg, "working");
      await delay(600 + Math.random() * 400);
    }
    // P4 scores
    setSt(p => ({ ...p, agentStates: { ...p.agentStates, P4: "scoring" } }));
    setCurrentAgent("P4");
    addMsg("P4", "Quality assessment initiated.", "working");
    await delay(500);
    const ihsan = (0.95 + Math.random() * 0.04).toFixed(4);
    addMsg("P4", `Ihsan score: ${ihsan}. ${parseFloat(ihsan) >= 0.98 ? "Exceptional." : "Above constitutional floor."}`, "score");
    await delay(400);
    // P5 clears
    setSt(p => ({ ...p, agentStates: { ...p.agentStates, P5: "checking" } }));
    setCurrentAgent("P5");
    addMsg("P5", "Constitutional scan... All seven invariants hold. Cleared.", "clear");
    await delay(400);
    // Mint
    const isLegendary = parseFloat(ihsan) >= 0.98 && Math.random() > 0.5;
    const isEpic = !isLegendary && parseFloat(ihsan) >= 0.96;
    const drop = isLegendary ? "⚡ LEGENDARY" : isEpic ? "💜 EPIC" : "🔵 RARE";
    const mul = isLegendary ? 1.5 : isEpic ? 1.3 : 1.15;
    const seedEarned = (1.0 * parseFloat(ihsan) * mul).toFixed(3);
    const bloomEarned = (0.01 * parseFloat(ihsan)).toFixed(4);
    addMsg("SYS", `PoI receipt: ${drop} — +${seedEarned} SEED, +${bloomEarned} BLOOM`, "mint");
    await delay(300);
    addMsg("P6", "Results formatted and delivered. Receipt chained.", "agent");
    setCurrentAgent("P6");
    await delay(300);
    const newRac = st.rac + 1;
    const compiled = newRac > 0 && newRac % 5 === 0;
    addMsg("P7", `Mission complete. +${seedEarned} SEED earned.${compiled ? " Pattern compiled to reflex — 8× faster next time." : ""}`, "complete");
    setCurrentAgent(null);
    setSt(p => ({
      ...p, phase: "ready", activeMission: null,
      seed: p.seed + parseFloat(seedEarned), bloom: p.bloom + parseFloat(bloomEarned),
      rac: p.rac + 1, vac: p.vac + 1, streak: p.streak + 1, level: Math.floor((p.rac + 1) / 10),
      ihsan: parseFloat(ihsan), s2: p.s2 + 1, mye: p.s1 / Math.max(p.s1 + p.s2 + 1, 1),
      reflexes: p.reflexes + (compiled ? 1 : 0),
      legendary: p.legendary + (isLegendary ? 1 : 0), epic: p.epic + (isEpic ? 1 : 0),
      agentStates: Object.fromEntries(Object.keys(AGENTS).map(k => [k, "idle"])),
    }));
    setTimeout(() => {
      const picks = ["P2", "P1", "P4", "P3", "P7"];
      const followups = [
        "I noticed a related topic worth exploring next.",
        "Based on this, I've updated your priority queue.",
        "Your Ihsan average this session is exceptional.",
        "That pattern is close to compilation. Two more quality runs.",
        "All agents returning to standby. Ready for next directive.",
      ];
      const idx = Math.floor(Math.random() * picks.length);
      addGhost(picks[idx], followups[idx]);
    }, 3000);
  }, [addMsg, addGhost, st.rac, st.s1, st.s2]);

  const handleSubmit = () => {
    if (!input.trim() || st.phase === "mission") return;
    const task = input.trim();
    setInput("");
    addMsg("USER", task, "user");
    setTimeout(() => executeMission(task), 300);
  };

  const quickMission = (task) => {
    if (st.phase === "mission") return;
    addMsg("USER", task, "user");
    setTimeout(() => executeMission(task), 300);
  };

  const formatTime = (d) => d.toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const formatDate = (d) => d.toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" });

  const tierName = st.level < 2 ? "SEED" : st.level < 5 ? "NODE" : st.level < 10 ? "BUILDER" : st.level < 20 ? "VERIFIER" : "MENTOR";

  // ═══ PRE-BOOT SCREEN ═══
  if (!st.booted && st.phase !== "booting") {
    return (
      <div style={{
        minHeight: "100vh", background: C.bg,
        display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
        fontFamily: "'Playfair Display', 'Georgia', serif",
        position: "relative", overflow: "hidden",
      }}>
        {/* Background grid */}
        <div style={{
          position: "absolute", inset: 0,
          backgroundImage: `linear-gradient(rgba(201,169,98,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(201,169,98,0.03) 1px, transparent 1px)`,
          backgroundSize: "60px 60px",
          maskImage: "radial-gradient(circle at center, black 30%, transparent 80%)",
          WebkitMaskImage: "radial-gradient(circle at center, black 30%, transparent 80%)",
        }} />

        {/* Seed of Life */}
        <div style={{ marginBottom: "32px", position: "relative" }}>
          <div style={{
            position: "absolute", inset: "-20px", borderRadius: "50%",
            background: `radial-gradient(circle, ${C.goldGlow}, transparent 70%)`,
          }} />
          <SeedOfLife size={120} opacity={0.3} animate />
        </div>

        {/* Title */}
        <h1 style={{
          fontSize: "48px", fontWeight: 400, letterSpacing: "16px",
          background: `linear-gradient(180deg, ${C.goldLight}, ${C.gold}, ${C.goldDim})`,
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
          marginBottom: "8px", position: "relative",
        }}>
          BIZRA
        </h1>
        <div style={{
          fontFamily: "'Amiri', 'Georgia', serif", fontSize: "24px",
          color: `${C.gold}60`, marginBottom: "4px", letterSpacing: "2px",
        }}>
          البذرة
        </div>
        <div style={{
          fontFamily: "'Courier New', monospace", fontSize: "9px",
          color: C.textDim, letterSpacing: "4px", textTransform: "uppercase",
          marginBottom: "48px",
        }}>
          Sovereign Agent Operating System
        </div>

        {/* Initialize button */}
        <button onClick={boot} style={{
          background: "transparent",
          border: `1px solid ${C.gold}35`,
          color: C.gold,
          padding: "14px 56px",
          borderRadius: "2px",
          fontSize: "11px",
          letterSpacing: "6px",
          cursor: "pointer",
          fontFamily: "'Courier New', monospace",
          transition: "all 0.4s ease",
          textTransform: "uppercase",
          position: "relative",
        }}
          onMouseEnter={e => { e.target.style.background = `${C.gold}12`; e.target.style.borderColor = `${C.gold}60`; e.target.style.boxShadow = `0 0 40px ${C.goldGlow}, inset 0 0 20px ${C.goldGlow2}`; }}
          onMouseLeave={e => { e.target.style.background = "transparent"; e.target.style.borderColor = `${C.gold}35`; e.target.style.boxShadow = "none"; }}>
          Initialize
        </button>

        <div style={{ position: "absolute", bottom: "24px", fontSize: "8px", color: "#1a2540", letterSpacing: "3px", fontFamily: "monospace" }}>
          NODE0 · v2.0.0 · OMEGA
        </div>

        <style>{`
          @keyframes seedPulse {
            0%, 100% { opacity: 0.15; transform: scale(1); }
            50% { opacity: 0.35; transform: scale(1.02); }
          }
        `}</style>
      </div>
    );
  }

  // ═══ MAIN DASHBOARD ═══
  return (
    <div style={{
      minHeight: "100vh", background: C.bg, color: C.text,
      fontFamily: "'Courier New', monospace", fontSize: "11px",
      display: "flex", flexDirection: "column", position: "relative", overflow: "hidden",
    }}>
      <style>{`
        @keyframes seedPulse {
          0%, 100% { opacity: 0.15; transform: scale(1); }
          50% { opacity: 0.35; transform: scale(1.02); }
        }
        @keyframes barPulse {
          0%, 100% { opacity: 0.6; }
          50% { opacity: 1; }
        }
        @keyframes slideIn {
          from { opacity: 0; transform: translateY(4px); }
          to { opacity: 1; transform: translateY(0); }
        }
        input::placeholder { color: #374151; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: ${C.border}; border-radius: 2px; }
      `}</style>

      {/* Background grid */}
      <div style={{
        position: "fixed", inset: 0, pointerEvents: "none", zIndex: 0,
        backgroundImage: `linear-gradient(rgba(201,169,98,0.02) 1px, transparent 1px), linear-gradient(90deg, rgba(201,169,98,0.02) 1px, transparent 1px)`,
        backgroundSize: "60px 60px",
        maskImage: "radial-gradient(circle at center, black 40%, transparent 100%)",
        WebkitMaskImage: "radial-gradient(circle at center, black 40%, transparent 100%)",
      }} />

      {/* ═══ TOP BAR ═══ */}
      <div style={{
        display: "flex", justifyContent: "space-between", alignItems: "center",
        padding: "10px 20px", borderBottom: `1px solid ${C.border}`,
        background: `${C.bg}ee`, backdropFilter: "blur(8px)",
        position: "relative", zIndex: 10,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          <span style={{
            fontFamily: "'Playfair Display', serif", fontSize: "14px",
            color: C.gold, letterSpacing: "4px", fontWeight: 600,
          }}>BIZRA</span>
          <span style={{ color: "#1e293b", fontSize: "9px", letterSpacing: "2px" }}>NODE0</span>
          <div style={{
            display: "flex", alignItems: "center", gap: "5px",
            padding: "2px 10px", borderRadius: "2px",
            background: st.phase === "mission" ? `${C.amber}12` : `${C.green}10`,
            border: `1px solid ${st.phase === "mission" ? C.amber + "25" : C.green + "20"}`,
          }}>
            <div style={{
              width: "5px", height: "5px", borderRadius: "50%",
              background: st.phase === "mission" ? C.amber : C.green,
              animation: st.phase === "mission" ? "barPulse 1s ease-in-out infinite" : "none",
            }} />
            <span style={{
              fontSize: "8px", letterSpacing: "2px", textTransform: "uppercase",
              color: st.phase === "mission" ? C.amber : C.green,
            }}>
              {st.phase === "mission" ? "EXECUTING" : st.phase === "booting" ? "BOOTING" : "READY"}
            </span>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "20px", fontSize: "9px" }}>
          <span style={{ color: C.green }}>{st.seed.toFixed(2)} <span style={{ color: "#374151" }}>SEED</span></span>
          <span style={{ color: C.purple }}>{st.bloom.toFixed(3)} <span style={{ color: "#374151" }}>BLOOM</span></span>
          <span style={{ color: C.blue }}>Lv.{st.level} <span style={{ color: "#374151" }}>{tierName}</span></span>
          <div style={{ width: "1px", height: "14px", background: C.borderFaint }} />
          <span style={{ color: C.textDim, fontVariantNumeric: "tabular-nums" }}>{formatDate(time)}</span>
          <span style={{ color: C.gold, fontVariantNumeric: "tabular-nums" }}>{formatTime(time)}</span>
        </div>
      </div>

      {/* ═══ AGENT STATUS BAR ═══ */}
      <div style={{
        display: "flex", alignItems: "center", gap: "3px",
        padding: "6px 20px", borderBottom: `1px solid ${C.borderFaint}`,
        background: `rgba(5,11,20,0.6)`, zIndex: 10,
      }}>
        {Object.entries(AGENTS).map(([id, ag]) => {
          const active = st.agentStates[id] !== "idle";
          const isCurrent = id === currentAgent;
          return (
            <div key={id} style={{
              flex: 1, padding: "5px 4px", borderRadius: "3px", textAlign: "center",
              background: isCurrent ? `${ag.color}18` : active ? `${ag.color}08` : "transparent",
              border: `1px solid ${isCurrent ? ag.color + "40" : active ? ag.color + "20" : C.borderFaint}`,
              transition: "all 0.4s ease",
              boxShadow: isCurrent ? `0 0 12px ${ag.color}15` : "none",
            }}>
              <div style={{
                fontSize: "8px", fontWeight: isCurrent ? "bold" : "normal",
                color: active ? ag.color : "#4B5563", letterSpacing: "1px",
                transition: "color 0.3s",
              }}>
                {ag.icon} {ag.call}
              </div>
            </div>
          );
        })}
        <div style={{ width: "1px", height: "20px", background: C.borderFaint, margin: "0 6px" }} />
        {/* SAT indicators */}
        <div style={{ display: "flex", gap: "6px", alignItems: "center", padding: "0 4px" }}>
          <span style={{ fontSize: "7px", color: "#374151", letterSpacing: "1px" }}>SAT</span>
          {SAT.map(s => (
            <div key={s.id} title={s.name} style={{
              width: "7px", height: "7px", borderRadius: "50%",
              background: `${s.color}50`,
              boxShadow: `0 0 6px ${s.color}25`,
            }} />
          ))}
        </div>

        <div style={{ marginLeft: "auto", display: "flex", gap: "8px" }}>
          <button onClick={() => setShowTrust(!showTrust)} style={{
            background: showTrust ? `${C.gold}12` : "transparent",
            border: `1px solid ${showTrust ? C.gold + "30" : C.borderFaint}`,
            color: showTrust ? C.gold : "#4B5563",
            padding: "2px 10px", borderRadius: "2px", fontSize: "7px",
            cursor: "pointer", fontFamily: "monospace", letterSpacing: "1px",
            transition: "all 0.3s",
          }}>
            TRUST
          </button>
        </div>
      </div>

      {/* ═══ MAIN CONTENT ═══ */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden", position: "relative", zIndex: 5 }}>

        {/* Left sidebar — Agent Ring + Trust */}
        <div style={{
          width: "220px", padding: "16px", borderRight: `1px solid ${C.borderFaint}`,
          display: "flex", flexDirection: "column", gap: "16px",
          background: "rgba(5,11,20,0.4)",
          overflowY: "auto",
        }}>
          <AgentRing agents={AGENTS} states={st.agentStates} activeId={currentAgent} />

          {/* Stats */}
          <div style={{
            background: "rgba(255,255,255,0.02)", borderRadius: "6px",
            border: `1px solid ${C.borderFaint}`, padding: "10px 12px",
          }}>
            <div style={{ fontSize: "7px", color: C.textDim, letterSpacing: "2px", marginBottom: "8px", textTransform: "uppercase" }}>
              Session Metrics
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px" }}>
              {[
                ["Ihsan", st.ihsan.toFixed(4), C.gold],
                ["Streak", st.streak, C.green],
                ["Reflexes", `${st.reflexes}⚡`, C.cyan],
                ["RAC", st.rac, C.blue],
                ["Legendary", st.legendary, C.amber],
                ["Epic", st.epic, C.purple],
              ].map(([label, val, color]) => (
                <div key={label} style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                  <span style={{ fontSize: "8px", color: C.textDim }}>{label}</span>
                  <span style={{ fontSize: "10px", color, fontVariantNumeric: "tabular-nums" }}>{val}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Trust Panel */}
          {showTrust && (
            <div style={{
              background: "rgba(255,255,255,0.02)", borderRadius: "6px",
              border: `1px solid ${C.border}`, padding: "10px 12px",
              animation: "slideIn 0.3s ease",
            }}>
              <div style={{ fontSize: "7px", color: C.gold, letterSpacing: "2px", marginBottom: "8px", textTransform: "uppercase" }}>
                Trust Verification
              </div>
              <TrustPanel checks={st.trustChecks} />
            </div>
          )}
        </div>

        {/* ═══ CENTER — FEED ═══ */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
          {/* Ghost Panel */}
          <div style={{ padding: "8px 16px 0" }}>
            <GhostPanel messages={st.ghostMessages} />
          </div>

          {/* Message Feed */}
          <div style={{ flex: 1, overflowY: "auto", padding: "8px 16px" }}>
            {st.messages.map((msg, i) => {
              const isUser = msg.type === "user";
              const isSystem = msg.type === "system";
              const isMint = msg.type === "mint";
              const isComplete = msg.type === "complete";
              const isGreeting = msg.type === "greeting";
              const agCol = isUser ? C.gold : isSystem ? "#4B5563" : isMint ? C.green : AGENTS[msg.agent]?.color || C.gold;
              const label = isUser ? "YOU" : isSystem ? "SYS" : AGENTS[msg.agent]?.call || msg.agent;
              return (
                <div key={i} style={{
                  marginBottom: "2px", padding: "3px 0",
                  display: "flex", gap: "10px", alignItems: "flex-start",
                  animation: "slideIn 0.2s ease",
                  opacity: isSystem ? 0.5 : 1,
                }}>
                  <span style={{
                    color: agCol, fontWeight: "bold", minWidth: "52px",
                    fontSize: "9px", textAlign: "right",
                    opacity: isSystem ? 0.7 : 1,
                  }}>
                    {label}
                  </span>
                  <span style={{
                    color: isUser ? C.text : isMint ? C.green : isComplete ? C.gold : isGreeting ? C.goldLight : "#9CA3AF",
                    fontSize: isUser ? "11px" : isGreeting ? "11px" : "10px",
                    lineHeight: 1.5,
                    fontFamily: isGreeting ? "'Playfair Display', serif" : "inherit",
                    letterSpacing: isGreeting ? "0.5px" : "0",
                  }}>
                    {isMint ? `► ${msg.text}` : isComplete ? `✓ ${msg.text}` : msg.text}
                  </span>
                </div>
              );
            })}
            <div ref={msgEnd} />
          </div>

          {/* Quick Missions */}
          {st.phase === "ready" && (
            <div style={{
              padding: "6px 16px", display: "flex", gap: "5px", flexWrap: "wrap",
              borderTop: `1px solid ${C.borderFaint}`,
            }}>
              {MISSIONS.slice(0, 4).map((m, i) => (
                <button key={i} onClick={() => quickMission(m)} style={{
                  background: "rgba(255,255,255,0.02)",
                  border: `1px solid ${C.borderFaint}`,
                  color: "#4B5563", padding: "4px 10px", borderRadius: "2px",
                  fontSize: "8px", cursor: "pointer", fontFamily: "monospace",
                  transition: "all 0.3s", maxWidth: "220px",
                  overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                }}
                  onMouseEnter={e => { e.target.style.borderColor = `${C.gold}30`; e.target.style.color = C.gold; e.target.style.background = `${C.goldGlow2}`; }}
                  onMouseLeave={e => { e.target.style.borderColor = C.borderFaint; e.target.style.color = "#4B5563"; e.target.style.background = "rgba(255,255,255,0.02)"; }}>
                  {m.slice(0, 45)}...
                </button>
              ))}
            </div>
          )}

          {/* Input */}
          <div style={{
            padding: "10px 16px",
            borderTop: `1px solid ${C.border}`,
            display: "flex", gap: "10px", alignItems: "center",
            background: `rgba(10,22,40,0.5)`,
          }}>
            <span style={{
              color: st.phase === "mission" ? C.amber : C.gold,
              fontSize: "12px", transition: "color 0.3s",
            }}>▸</span>
            <input
              ref={inputRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === "Enter" && handleSubmit()}
              placeholder={st.phase === "mission" ? "Mission in progress..." : "Speak your mission..."}
              disabled={st.phase === "mission"}
              style={{
                flex: 1, background: "transparent", border: "none",
                color: C.text, fontSize: "12px", fontFamily: "monospace",
                outline: "none", letterSpacing: "0.5px",
              }}
            />
            {input.trim() && st.phase !== "mission" && (
              <button onClick={handleSubmit} style={{
                background: `${C.gold}15`, border: `1px solid ${C.gold}30`,
                color: C.gold, padding: "4px 16px", borderRadius: "2px",
                fontSize: "9px", cursor: "pointer", fontFamily: "monospace",
                letterSpacing: "2px", transition: "all 0.3s",
              }}>
                EXECUTE
              </button>
            )}
          </div>
        </div>
      </div>

      {/* ═══ BOTTOM STATUS ═══ */}
      <div style={{
        padding: "5px 20px", borderTop: `1px solid ${C.borderFaint}`,
        display: "flex", justifyContent: "space-between",
        fontSize: "7px", color: "#1e293b", letterSpacing: "1px",
        background: `${C.bg}`, zIndex: 10,
      }}>
        <span>TIER: {tierName} · IHSAN: {st.ihsan.toFixed(4)} · STREAK: {st.streak} · MYE: {(st.mye * 100).toFixed(0)}%</span>
        <span style={{ fontFamily: "'Amiri', serif", fontSize: "9px", color: `${C.gold}20` }}>بذرة واحدة تصنع غابة</span>
        <span>PAT-7 SOVEREIGN · SAT-5 OVERSIGHT · 15 ALGORITHMS · 7 INVARIANTS</span>
      </div>
    </div>
  );
}
