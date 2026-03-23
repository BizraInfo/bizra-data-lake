import { useState, useEffect, useRef, useCallback, useMemo } from "react";

/*
  BIZRA SOVEREIGN MISSION COCKPIT v1.0
  ════════════════════════════════════════════════════════════════
  
  4-Scene Parallel Architecture (from v1 Product Decision):
  Scene 1: Morning Brief / Ghost Panel          [TOP HERO]
  Scene 2: Trust Panel (5 cryptographic checks)  [RIGHT RAIL]
  Scene 3: PAT-7 Agent Dashboard                 [CENTER]
  Scene 4: Living Memory                         [EXPANDABLE DRAWER]
  
  7 Golden Gems Implemented:
  1. Integration over invention — wires existing backend schemas
  2. Self-harness bridge — ghost_ws.py OverlayEvent JSON schema
  3. Sacred geometry = live telemetry — 7 circles = 7 agents
  4. Hash table navigation — all scenes coexist at different Z-levels
  5. Boot IS onboarding IS product — geometry becomes cockpit
  6. Python warmth / Rust precision — two visual temperatures
  7. Flower blooms with economy — petal fill = session value
  
  Backend Schema (ghost_ws.py OverlayEvent):
  { action_label, intent_summary, confidence, ihsan_precheck,
    ihsan_score, source_agent, timestamp }
  
  Standing on Giants:
  Boyd (OODA), Shannon (SNR), Norman (invisible design),
  Al-Ghazali (Ihsān), Engelbart (augmentation), Deming (PDCA)
*/

// ═══ DESIGN TOKENS — Extracted from 29-file corpus ═══
const P = {
  bg: "#050B14", bgCard: "#0A1628", bgPanel: "#0D1B2F",
  gold: "#C9A962", goldDim: "#8A6B2E", goldLight: "#F9F1D8", goldGlow: "rgba(201,169,98,0.12)",
  text: "#F8F6F1", textMid: "rgba(248,246,241,0.6)", textDim: "rgba(248,246,241,0.35)",
  border: "rgba(201,169,98,0.12)", borderFaint: "rgba(255,255,255,0.04)",
  // Agent colors — warm palette for PAT side
  blue: "#3b82f6", cyan: "#06b6d4", green: "#22c55e", amber: "#f59e0b",
  rose: "#f43f5e", orange: "#f97316", purple: "#a855f7",
  // SAT side — cool/steel palette  
  steel: "#64748b", steelDim: "#334155", steelBorder: "rgba(100,116,139,0.2)",
  red: "#ef4444", verified: "#22c55e",
};

// ═══ PAT-7 AGENTS ═══
const AGENTS = {
  P1: { name: "Planner", call: "ATLAS", color: P.blue, icon: "◇", angle: -90,
    idle: ["Analyzing priority queue...", "Three pending objectives identified.", "Your roadmap has a dependency conflict I can resolve."],
    working: ["Decomposing into subtasks...", "Dependency graph resolved.", "Execution order optimized.", "Critical path identified."] },
  P2: { name: "Researcher", call: "ORACLE", color: P.cyan, icon: "◈", angle: -38.6,
    idle: ["I found something in your domain...", "Three new papers match your interests.", "Knowledge graph grew 12% this week."],
    working: ["Scanning knowledge base...", "Cross-referencing 47 sources.", "SNR: 0.94.", "Key findings extracted."] },
  P3: { name: "Coder", call: "FORGE", color: P.green, icon: "⬡", angle: 12.9,
    idle: ["Your test suite has 3 flaky tests I can fix.", "Build pipeline green. All tests passing.", "Refactoring opportunity spotted."],
    working: ["Generating implementation...", "Running test suite...", "All assertions pass.", "Code quality: Ihsan 0.97."] },
  P4: { name: "Evaluator", call: "JUDGE", color: P.amber, icon: "◆", angle: 64.3,
    idle: ["Average Ihsan trending up — 0.983.", "Quality score: top 5% of all nodes.", "Benchmarked 3 alternatives."],
    working: ["Quality assessment...", "Shannon entropy: above threshold.", "Scoring against rubric...", "Exceeds constitutional floor."] },
  P5: { name: "Ethicist", call: "CROWN", color: P.rose, icon: "✦", angle: 115.7,
    idle: ["All invariants satisfied.", "I-3 Gini at 0.31 — within bounds.", "Covenant holds. Integrity verified."],
    working: ["Scanning I-1 through I-7...", "Shariah compliance: verified.", "No bias detected.", "Constitutional clearance granted."] },
  P6: { name: "Publisher", call: "HERALD", color: P.orange, icon: "▹", angle: 167.1,
    idle: ["Last report scored 4.8/5.0 readability.", "Three versions drafted.", "Format optimized for audience."],
    working: ["Structuring output...", "Formatting for clarity...", "Final polish applied.", "Ready for delivery."] },
  P7: { name: "Integrator", call: "NEXUS", color: P.purple, icon: "⟡", angle: 218.6,
    idle: ["All seven agents nominal.", "Context pre-loaded from last session.", "Coordination score: 94%."],
    working: ["Routing to specialist...", "Context bridge established.", "Agent handoff complete.", "Aggregating all sources."] },
};

const SAT = [
  { id: "S1", name: "Sentinel", role: "WATCHING", color: P.red },
  { id: "S2", name: "Oracle", role: "SCORING", color: P.gold },
  { id: "S3", name: "Ledger", role: "RECORDING", color: P.amber },
  { id: "S4", name: "Conductor", role: "ROUTING", color: P.blue },
  { id: "S5", name: "Ambassador", role: "LISTENING", color: P.cyan },
];

// ═══ GHOST PANEL DEMO DATA (matches ghost_ws.py OverlayEvent schema) ═══
const GHOST_EVENTS = [
  { action_label: "Open weekly review template", intent_summary: "Review overdue by 3 days. Recent idle detected.", confidence: 0.91, ihsan_precheck: "pass", ihsan_score: 0.97, source_agent: "P1", ts: Date.now() - 3600000 },
  { action_label: "Run CI stabilization on 3 flaky tests", intent_summary: "Test failures detected in last 2 commits. Pattern matches known fix.", confidence: 0.88, ihsan_precheck: "pass", ihsan_score: 0.95, source_agent: "P3", ts: Date.now() - 1800000 },
  { action_label: "Draft Enforceable Spine v1.2 amendment", intent_summary: "Two inconsistencies identified between Spine and codebase.", confidence: 0.72, ihsan_precheck: "blocked", ihsan_score: 0.62, source_agent: "P5", ts: Date.now() - 900000 },
];

const QUICK_MISSIONS = [
  "Research latest sovereign AI architectures",
  "Build constitutional invariant test framework",
  "Evaluate deployment pipeline benchmarks",
  "Draft quarterly progress report",
];

const MEMORY_FRAGMENTS = [
  { type: "reflex", text: "CI stabilization pattern: isolate → reproduce → fix → gate", age: "2h ago", score: 0.94 },
  { type: "knowledge", text: "Phase 81 Omega: 471,917 LOC, 11,135 tests, SNR 0.958", age: "1d ago", score: 0.99 },
  { type: "episode", text: "Mint Court rejected founder work at SNR 0.577 — governance works", age: "3d ago", score: 0.97 },
  { type: "promoted", text: "Python/Rust boundary IS the PAT/SAT constitutional gate", age: "5d ago", score: 0.98 },
];

// ═══ INIT STATE ═══
const init = () => ({
  phase: "idle", booted: false,
  seed: 0, bloom: 0, ihsan: 0, streak: 0, rac: 0, level: 0, reflexes: 0, legendary: 0,
  messages: [], ghostEvents: [],
  agentStates: Object.fromEntries(Object.keys(AGENTS).map(k => [k, "idle"])),
  currentAgent: null,
  trust: { node: null, ledger: null, token: null, supply: null, gate: null },
  memoryOpen: false, briefExpanded: true, petalFill: 0,
});

// ═══ SEED OF LIFE — Sacred Geometry as Live Telemetry (Gem 3) ═══
function SeedOfLifeTelemetry({ agentStates, currentAgent, petalFill }) {
  const R = 22;
  const agents = Object.entries(AGENTS);
  const positions = [
    [0, 0], [0, -R], [R * 0.866, -R * 0.5], [R * 0.866, R * 0.5],
    [0, R], [-R * 0.866, R * 0.5], [-R * 0.866, -R * 0.5],
  ];
  const agentMap = ["P5", "P1", "P2", "P3", "P4", "P6", "P7"]; // center=CROWN, then clockwise

  return (
    <svg width="130" height="130" viewBox="-44 -44 88 88" style={{ display: "block", margin: "0 auto" }}>
      <defs>
        <linearGradient id="gGold" x1="0%" y1="100%" x2="100%" y2="0%">
          <stop offset="0%" stopColor={P.goldDim} />
          <stop offset="100%" stopColor={P.goldLight} />
        </linearGradient>
      </defs>
      {/* Outer federation ring (L3 — dormant) */}
      <circle cx="0" cy="0" r="42" fill="none" stroke={P.borderFaint} strokeWidth="0.5" strokeDasharray="3 3" />
      {/* 7 circles mapped to agents */}
      {positions.map(([cx, cy], i) => {
        const agId = agentMap[i];
        const ag = AGENTS[agId];
        const active = agentStates[agId] !== "idle";
        const isCurrent = agId === currentAgent;
        return (
          <g key={i}>
            <circle cx={cx} cy={cy} r={R} fill={isCurrent ? `${ag.color}18` : active ? `${ag.color}08` : "none"}
              stroke={isCurrent ? ag.color : active ? `${ag.color}50` : `${P.gold}20`}
              strokeWidth={isCurrent ? 1.2 : 0.5}
              style={{ transition: "all 0.6s ease" }} />
            {(active || isCurrent) && (
              <text x={cx} y={cy + 1} textAnchor="middle" dominantBaseline="central"
                style={{ fontSize: "5px", fill: ag.color, fontFamily: "monospace", letterSpacing: "0.5px", transition: "fill 0.3s" }}>
                {ag.call}
              </text>
            )}
          </g>
        );
      })}
      {/* Flower petals — fill with economic activity (Gem 7) */}
      {[0, 60, 120, 180, 240, 300].map((angle, i) => {
        const rad = (angle * Math.PI) / 180;
        const x1 = 0, y1 = -R;
        const fill = Math.min(petalFill / 6, 1);
        return (
          <path key={`petal-${i}`}
            d={`M0 ${-R} Q${R * 0.4} ${-R * 0.4} 0 0 Q${-R * 0.4} ${-R * 0.4} 0 ${-R}`}
            transform={`rotate(${angle})`}
            fill={`${P.gold}${Math.round(fill * 30 + 5).toString(16).padStart(2, "0")}`}
            stroke={`${P.gold}${fill > 0.1 ? "40" : "15"}`}
            strokeWidth="0.5"
            style={{ transition: "fill 1s ease, stroke 1s ease" }} />
        );
      })}
      {/* Center nuqta — constitutional core */}
      <rect x="-2.5" y="-2.5" width="5" height="5" rx="1" transform="rotate(45)"
        fill={`url(#gGold)`} style={{ opacity: agentStates.P5 !== "idle" ? 1 : 0.5, transition: "opacity 0.5s" }} />
    </svg>
  );
}

// ═══ TRUST PANEL — Right Rail (Gem 6: Rust precision / cool steel) ═══
function TrustRail({ trust, ihsan }) {
  const checks = [
    { key: "node", label: "Node health", icon: "●" },
    { key: "ledger", label: "SEL integrity", icon: "◈" },
    { key: "token", label: "SEED balance", icon: "◆" },
    { key: "supply", label: "Supply cap", icon: "▣" },
    { key: "gate", label: "Ihsan gate", icon: "✦" },
  ];
  return (
    <div style={{
      width: "180px", minWidth: "180px", padding: "12px",
      borderLeft: `1px solid ${P.steelBorder}`,
      background: `linear-gradient(180deg, rgba(13,27,47,0.6), rgba(5,11,20,0.8))`,
      display: "flex", flexDirection: "column", gap: "12px", overflowY: "auto",
    }}>
      <div style={{ fontSize: "8px", color: P.steel, letterSpacing: "2px", textTransform: "uppercase" }}>
        Trust verification
      </div>
      {checks.map(({ key, label, icon }) => {
        const v = trust[key];
        const col = v === true ? P.verified : v === false ? P.red : P.steelDim;
        return (
          <div key={key} style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{ color: col, fontSize: "8px", transition: "color 0.5s" }}>{icon}</span>
            <span style={{ color: v === true ? P.textMid : P.textDim, fontSize: "9px", flex: 1, letterSpacing: "0.3px" }}>{label}</span>
            <span style={{ color: col, fontSize: "8px", fontFamily: "'DM Mono', monospace", letterSpacing: "1px" }}>
              {v === true ? "OK" : v === false ? "FAIL" : "—"}
            </span>
          </div>
        );
      })}
      {/* Ihsan gauge */}
      <div style={{ marginTop: "4px", padding: "8px", borderRadius: "4px", background: `${P.gold}08`, border: `1px solid ${P.border}` }}>
        <div style={{ fontSize: "7px", color: P.gold, letterSpacing: "2px", textTransform: "uppercase", marginBottom: "6px" }}>
          Ihsan composite
        </div>
        <div style={{ fontSize: "20px", fontFamily: "'DM Mono', monospace", color: ihsan >= 0.95 ? P.gold : P.amber, textAlign: "center", fontVariantNumeric: "tabular-nums" }}>
          {ihsan.toFixed(4)}
        </div>
        <div style={{ height: "2px", background: P.steelDim, borderRadius: "1px", marginTop: "6px", overflow: "hidden" }}>
          <div style={{ height: "100%", width: `${Math.min(ihsan * 100, 100)}%`, background: ihsan >= 0.95 ? P.gold : P.amber, transition: "width 0.8s ease", borderRadius: "1px" }} />
        </div>
      </div>
      {/* SAT-5 status */}
      <div style={{ marginTop: "auto" }}>
        <div style={{ fontSize: "7px", color: P.steelDim, letterSpacing: "2px", textTransform: "uppercase", marginBottom: "6px" }}>
          SAT-5 oversight
        </div>
        {SAT.map(s => (
          <div key={s.id} style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "3px" }}>
            <div style={{ width: "5px", height: "5px", borderRadius: "50%", background: `${s.color}50`, boxShadow: `0 0 4px ${s.color}25` }} />
            <span style={{ fontSize: "8px", color: P.steelDim, flex: 1 }}>{s.name}</span>
            <span style={{ fontSize: "7px", color: P.steelDim, fontFamily: "monospace" }}>{s.role}</span>
          </div>
        ))}
      </div>
      {/* Covenant hash */}
      <div style={{ fontSize: "7px", fontFamily: "monospace", color: `${P.gold}30`, wordBreak: "break-all", lineHeight: 1.4 }}>
        859649ea...verified
      </div>
    </div>
  );
}

// ═══ GHOST PANEL — Scene 1: Morning Brief (Gem 2) ═══
function GhostBrief({ events, expanded, onToggle }) {
  if (!events.length) return null;
  const passed = events.filter(e => e.ihsan_precheck === "pass");
  const blocked = events.filter(e => e.ihsan_precheck !== "pass");
  return (
    <div style={{
      background: `linear-gradient(135deg, ${P.goldGlow}, transparent)`,
      border: `1px solid ${P.border}`, borderRadius: "6px",
      padding: expanded ? "12px" : "8px 12px",
      transition: "all 0.3s ease", cursor: "pointer",
    }}>
      <div onClick={onToggle} style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={{ fontSize: "8px", color: P.gold, letterSpacing: "2px", textTransform: "uppercase" }}>
            Morning brief
          </span>
          <span style={{ fontSize: "9px", color: P.textDim }}>
            {passed.length} actionable · {blocked.length} blocked
          </span>
        </div>
        <span style={{ color: P.textDim, fontSize: "10px", transform: expanded ? "rotate(180deg)" : "none", transition: "transform 0.2s" }}>
          ▾
        </span>
      </div>
      {expanded && (
        <div style={{ marginTop: "10px", display: "flex", flexDirection: "column", gap: "6px" }}>
          {events.map((ev, i) => {
            const ag = AGENTS[ev.source_agent];
            const blocked = ev.ihsan_precheck !== "pass";
            return (
              <div key={i} style={{
                display: "flex", gap: "10px", alignItems: "flex-start",
                padding: "8px 10px", borderRadius: "4px",
                background: blocked ? "rgba(239,68,68,0.04)" : "rgba(201,169,98,0.04)",
                border: `1px solid ${blocked ? "rgba(239,68,68,0.12)" : P.border}`,
                opacity: blocked ? 0.55 : 1,
              }}>
                <div style={{ minWidth: "40px", textAlign: "right" }}>
                  <div style={{ fontSize: "8px", color: ag?.color || P.gold, fontWeight: "bold", letterSpacing: "0.5px" }}>
                    {ag?.call || "SYS"}
                  </div>
                  <div style={{ fontSize: "7px", color: P.textDim, marginTop: "2px" }}>
                    {(ev.confidence * 100).toFixed(0)}%
                  </div>
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: "10px", color: blocked ? P.textDim : P.text, lineHeight: 1.4 }}>
                    {ev.action_label}
                  </div>
                  <div style={{ fontSize: "9px", color: P.textDim, marginTop: "2px", lineHeight: 1.3 }}>
                    {ev.intent_summary}
                  </div>
                </div>
                <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: "3px" }}>
                  <span style={{
                    fontSize: "7px", padding: "1px 6px", borderRadius: "2px", letterSpacing: "1px",
                    background: blocked ? "rgba(239,68,68,0.1)" : "rgba(34,197,94,0.1)",
                    color: blocked ? P.red : P.verified, fontFamily: "monospace",
                  }}>
                    {blocked ? "BLOCKED" : "PASS"}
                  </span>
                  <span style={{ fontSize: "8px", color: P.textDim, fontFamily: "monospace" }}>
                    {ev.ihsan_score.toFixed(2)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ═══ MEMORY DRAWER — Scene 4 (Gem 4: expandable, hash table nav) ═══
function MemoryDrawer({ open, onClose }) {
  if (!open) return null;
  const typeColors = { reflex: P.cyan, knowledge: P.gold, episode: P.purple, promoted: P.green };
  return (
    <div style={{
      position: "absolute", right: 0, top: 0, bottom: 0, width: "260px", zIndex: 20,
      background: `${P.bgCard}f0`, backdropFilter: "blur(12px)",
      borderLeft: `1px solid ${P.border}`, padding: "16px",
      display: "flex", flexDirection: "column", gap: "10px",
      animation: "slideInRight 0.25s ease",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontSize: "8px", color: P.gold, letterSpacing: "2px", textTransform: "uppercase" }}>Living memory</span>
        <button onClick={onClose} style={{ background: "none", border: "none", color: P.textDim, cursor: "pointer", fontSize: "14px", fontFamily: "monospace" }}>×</button>
      </div>
      {MEMORY_FRAGMENTS.map((m, i) => (
        <div key={i} style={{
          padding: "8px", borderRadius: "4px",
          background: `${typeColors[m.type]}06`,
          border: `1px solid ${typeColors[m.type]}15`,
        }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
            <span style={{ fontSize: "7px", color: typeColors[m.type], letterSpacing: "1px", textTransform: "uppercase" }}>{m.type}</span>
            <span style={{ fontSize: "7px", color: P.textDim }}>{m.age}</span>
          </div>
          <div style={{ fontSize: "9px", color: P.textMid, lineHeight: 1.4 }}>{m.text}</div>
          <div style={{ fontSize: "7px", color: P.textDim, marginTop: "3px", fontFamily: "monospace" }}>score: {m.score.toFixed(2)}</div>
        </div>
      ))}
    </div>
  );
}

// ═══ MAIN COCKPIT ═══
export default function SovereignCockpit() {
  const [st, setSt] = useState(init());
  const [input, setInput] = useState("");
  const [time, setTime] = useState(new Date());
  const msgEnd = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => { const t = setInterval(() => setTime(new Date()), 1000); return () => clearInterval(t); }, []);
  useEffect(() => { msgEnd.current?.scrollIntoView({ behavior: "smooth" }); }, [st.messages]);

  const addMsg = useCallback((agent, text, type = "agent") => {
    setSt(p => ({ ...p, messages: [...p.messages, { agent, text, type, ts: Date.now() }].slice(-80) }));
  }, []);
  const delay = ms => new Promise(r => setTimeout(r, ms));

  const tierName = st.level < 2 ? "SEED" : st.level < 5 ? "NODE" : st.level < 10 ? "BUILDER" : st.level < 20 ? "VERIFIER" : "MENTOR";
  const formatTime = d => d.toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });

  // ═══ BOOT — Geometry becomes telemetry (Gem 5) ═══
  const boot = useCallback(async () => {
    setSt(p => ({ ...p, phase: "booting" }));
    const sysSteps = [
      "بسم الله الرحمن الرحيم",
      "Initializing sovereign kernel...",
      "Fixed-point arithmetic verified. Deterministic.",
      "Constitutional invariants I-1 through I-7: LOADED",
      "Covenant hash: 859649ea...verified",
      "Ed25519 identity generated.",
    ];
    for (const text of sysSteps) { addMsg("SYS", text, "system"); await delay(250); }
    addMsg("SYS", "Minting Personal Agentic Team...", "system");
    await delay(300);
    for (const id of ["P7", "P1", "P2", "P3", "P4", "P5", "P6"]) {
      const ag = AGENTS[id];
      setSt(p => ({ ...p, agentStates: { ...p.agentStates, [id]: "booting" }, currentAgent: id }));
      addMsg(id, `${ag.call} online.`, "agent");
      await delay(180);
      setSt(p => ({ ...p, agentStates: { ...p.agentStates, [id]: "idle" } }));
    }
    addMsg("SYS", "SAT-5 deployed. Zero operator control over validators.", "system");
    await delay(200);
    // Trust verification sweep
    for (const key of ["node", "ledger", "token", "supply", "gate"]) {
      setSt(p => ({ ...p, trust: { ...p.trust, [key]: true } }));
      await delay(120);
    }
    addMsg("SYS", "Trust Panel: 5/5 checks VERIFIED", "system");
    await delay(200);
    // Load Ghost Panel events (would be from ws://127.0.0.1:9743/ws/ghost in production)
    setSt(p => ({ ...p, ghostEvents: GHOST_EVENTS }));
    addMsg("P7", "Sovereign team online. Morning brief loaded. What shall we build?", "greeting");
    setSt(p => ({ ...p, booted: true, phase: "ready", currentAgent: null }));
    setTimeout(() => inputRef.current?.focus(), 100);
  }, [addMsg]);

  // ═══ MISSION EXECUTION ═══
  const executeMission = useCallback(async (task) => {
    setSt(p => ({ ...p, phase: "mission" }));
    addMsg("P7", `Mission received: "${task.slice(0, 65)}..."`, "agent");
    setSt(p => ({ ...p, currentAgent: "P7" }));
    await delay(500);
    const kw = { P1: ["plan","strategy","roadmap","schedule","timeline"], P2: ["research","find","analyze","study","latest"], P3: ["code","build","test","debug","implement","framework"], P4: ["evaluate","score","assess","review","benchmark"], P5: ["check","verify","compliance","constitution","security"], P6: ["write","draft","report","document","publish"] };
    let best = "P2", bs = 0;
    for (const [a, ws] of Object.entries(kw)) { const s = ws.filter(w => task.toLowerCase().includes(w)).length; if (s > bs) { best = a; bs = s; } }
    const agent = AGENTS[best];
    addMsg("P7", `Routing to ${agent.call}.`, "agent");
    setSt(p => ({ ...p, agentStates: { ...p.agentStates, [best]: "active", P7: "routing" }, currentAgent: best }));
    await delay(400);
    for (const msg of agent.working) { addMsg(best, msg, "working"); await delay(500 + Math.random() * 300); }
    // P4 scores
    setSt(p => ({ ...p, agentStates: { ...p.agentStates, P4: "scoring" }, currentAgent: "P4" }));
    await delay(400);
    const ihsan = (0.95 + Math.random() * 0.04).toFixed(4);
    addMsg("P4", `Ihsan: ${ihsan}. ${parseFloat(ihsan) >= 0.98 ? "Exceptional." : "Above floor."}`, "score");
    await delay(300);
    // P5 constitutional
    setSt(p => ({ ...p, agentStates: { ...p.agentStates, P5: "checking" }, currentAgent: "P5" }));
    addMsg("P5", "All invariants hold. Cleared.", "clear");
    await delay(300);
    // Mint
    const isLeg = parseFloat(ihsan) >= 0.98 && Math.random() > 0.5;
    const isEpic = !isLeg && parseFloat(ihsan) >= 0.96;
    const drop = isLeg ? "LEGENDARY" : isEpic ? "EPIC" : "RARE";
    const mul = isLeg ? 1.5 : isEpic ? 1.3 : 1.15;
    const seedE = (parseFloat(ihsan) * mul).toFixed(3);
    const bloomE = (0.01 * parseFloat(ihsan)).toFixed(4);
    addMsg("SYS", `PoI: ${drop} — +${seedE} SEED, +${bloomE} BLOOM`, "mint");
    setSt(p => ({ ...p, currentAgent: "P6" }));
    addMsg("P6", "Receipt chained. Delivered.", "agent");
    await delay(200);
    const newRac = st.rac + 1;
    const compiled = newRac > 0 && newRac % 5 === 0;
    addMsg("P7", `Complete. +${seedE} SEED.${compiled ? " Reflex compiled — 8x faster." : ""}`, "complete");
    setSt(p => ({
      ...p, phase: "ready", currentAgent: null,
      seed: p.seed + parseFloat(seedE), bloom: p.bloom + parseFloat(bloomE),
      rac: p.rac + 1, streak: p.streak + 1, level: Math.floor((p.rac + 1) / 10),
      ihsan: parseFloat(ihsan), reflexes: p.reflexes + (compiled ? 1 : 0),
      legendary: p.legendary + (isLeg ? 1 : 0),
      petalFill: p.petalFill + parseFloat(bloomE) * 100,
      agentStates: Object.fromEntries(Object.keys(AGENTS).map(k => [k, "idle"])),
    }));
  }, [addMsg, st.rac]);

  const handleSubmit = () => {
    if (!input.trim() || st.phase === "mission") return;
    const task = input.trim(); setInput("");
    addMsg("USER", task, "user");
    setTimeout(() => executeMission(task), 300);
  };

  // ═══ PRE-BOOT ═══
  if (!st.booted && st.phase !== "booting") {
    return (
      <div style={{
        minHeight: "100vh", background: P.bg, display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center", position: "relative", overflow: "hidden",
      }}>
        <style>{`
          @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;500;600&family=DM+Mono:wght@300;400&family=Amiri:wght@400;700&display=swap');
          @keyframes seedFloat { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-4px); } }
          @keyframes gridFade { 0% { opacity: 0; } 100% { opacity: 1; } }
        `}</style>
        <div style={{ position: "absolute", inset: 0, backgroundImage: `linear-gradient(rgba(201,169,98,0.025) 1px, transparent 1px), linear-gradient(90deg, rgba(201,169,98,0.025) 1px, transparent 1px)`, backgroundSize: "56px 56px", maskImage: "radial-gradient(circle, black 25%, transparent 75%)", WebkitMaskImage: "radial-gradient(circle, black 25%, transparent 75%)", animation: "gridFade 2s ease" }} />
        <div style={{ animation: "seedFloat 4s ease-in-out infinite", position: "relative" }}>
          <div style={{ position: "absolute", inset: "-30px", borderRadius: "50%", background: `radial-gradient(circle, ${P.goldGlow}, transparent 70%)` }} />
          <SeedOfLifeTelemetry agentStates={Object.fromEntries(Object.keys(AGENTS).map(k => [k, "idle"]))} currentAgent={null} petalFill={0} />
        </div>
        <h1 style={{ fontFamily: "'Cormorant Garamond', serif", fontSize: "52px", fontWeight: 300, letterSpacing: "14px", background: `linear-gradient(180deg, ${P.goldLight}, ${P.gold}, ${P.goldDim})`, WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", margin: "20px 0 6px", position: "relative" }}>BIZRA</h1>
        <div style={{ fontFamily: "'Amiri', serif", fontSize: "22px", color: `${P.gold}50`, letterSpacing: "2px" }}>البذرة</div>
        <div style={{ fontFamily: "'DM Mono', monospace", fontSize: "9px", color: P.textDim, letterSpacing: "4px", textTransform: "uppercase", margin: "6px 0 40px" }}>Sovereign Agent Operating System</div>
        <button onClick={boot} style={{ background: "transparent", border: `1px solid ${P.gold}30`, color: P.gold, padding: "12px 48px", borderRadius: "2px", fontSize: "10px", letterSpacing: "5px", cursor: "pointer", fontFamily: "'DM Mono', monospace", transition: "all 0.4s", textTransform: "uppercase" }}
          onMouseEnter={e => { e.target.style.background = `${P.gold}0a`; e.target.style.borderColor = `${P.gold}50`; e.target.style.boxShadow = `0 0 30px ${P.goldGlow}`; }}
          onMouseLeave={e => { e.target.style.background = "transparent"; e.target.style.borderColor = `${P.gold}30`; e.target.style.boxShadow = "none"; }}>
          Initialize
        </button>
        <div style={{ position: "absolute", bottom: "20px", fontSize: "7px", color: `${P.gold}15`, letterSpacing: "3px", fontFamily: "monospace" }}>NODE0 · v2.0.0 · OMEGA</div>
      </div>
    );
  }

  // ═══ MAIN COCKPIT RENDER ═══
  const agCol = id => AGENTS[id]?.color || P.gold;
  return (
    <div style={{
      minHeight: "100vh", background: P.bg, color: P.text,
      fontFamily: "'DM Mono', monospace", fontSize: "11px",
      display: "flex", flexDirection: "column", position: "relative", overflow: "hidden",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;500;600&family=DM+Mono:wght@300;400&family=Amiri:wght@400;700&display=swap');
        @keyframes slideIn { from { opacity:0; transform:translateY(3px); } to { opacity:1; transform:translateY(0); } }
        @keyframes slideInRight { from { opacity:0; transform:translateX(20px); } to { opacity:1; transform:translateX(0); } }
        @keyframes barPulse { 0%,100% { opacity:0.6; } 50% { opacity:1; } }
        input::placeholder { color: #334155; }
        ::-webkit-scrollbar { width: 3px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: ${P.border}; border-radius: 2px; }
      `}</style>
      {/* Grid bg */}
      <div style={{ position: "fixed", inset: 0, pointerEvents: "none", zIndex: 0, backgroundImage: `linear-gradient(rgba(201,169,98,0.015) 1px, transparent 1px), linear-gradient(90deg, rgba(201,169,98,0.015) 1px, transparent 1px)`, backgroundSize: "56px 56px", maskImage: "radial-gradient(circle at center, black 40%, transparent 100%)", WebkitMaskImage: "radial-gradient(circle at center, black 40%, transparent 100%)" }} />

      {/* ═══ TOP BAR ═══ */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 16px", borderBottom: `1px solid ${P.border}`, background: `${P.bg}ee`, backdropFilter: "blur(6px)", zIndex: 10 }}>
        <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
          <span style={{ fontFamily: "'Cormorant Garamond', serif", fontSize: "15px", color: P.gold, letterSpacing: "4px", fontWeight: 500 }}>BIZRA</span>
          <span style={{ color: "#1a2540", fontSize: "8px", letterSpacing: "2px" }}>NODE0</span>
          <div style={{ display: "flex", alignItems: "center", gap: "4px", padding: "2px 8px", borderRadius: "2px", background: st.phase === "mission" ? `${P.amber}10` : `${P.verified}08`, border: `1px solid ${st.phase === "mission" ? P.amber + "20" : P.verified + "15"}` }}>
            <div style={{ width: "4px", height: "4px", borderRadius: "50%", background: st.phase === "mission" ? P.amber : P.verified, animation: st.phase === "mission" ? "barPulse 1s ease infinite" : "none" }} />
            <span style={{ fontSize: "7px", letterSpacing: "2px", color: st.phase === "mission" ? P.amber : P.verified, textTransform: "uppercase" }}>
              {st.phase === "mission" ? "EXECUTING" : st.phase === "booting" ? "BOOTING" : "READY"}
            </span>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "16px", fontSize: "9px" }}>
          <span style={{ color: P.verified }}>{st.seed.toFixed(2)} <span style={{ color: "#1e293b" }}>SEED</span></span>
          <span style={{ color: P.purple }}>{st.bloom.toFixed(3)} <span style={{ color: "#1e293b" }}>BLOOM</span></span>
          <span style={{ color: P.blue }}>Lv.{st.level} <span style={{ color: "#1e293b" }}>{tierName}</span></span>
          <div style={{ width: "1px", height: "12px", background: P.borderFaint }} />
          <span style={{ color: P.gold, fontVariantNumeric: "tabular-nums" }}>{formatTime(time)}</span>
          <button onClick={() => setSt(p => ({ ...p, memoryOpen: !p.memoryOpen }))}
            style={{ background: st.memoryOpen ? `${P.purple}12` : "transparent", border: `1px solid ${st.memoryOpen ? P.purple + "30" : P.borderFaint}`, color: st.memoryOpen ? P.purple : P.steelDim, padding: "2px 8px", borderRadius: "2px", fontSize: "7px", cursor: "pointer", fontFamily: "monospace", letterSpacing: "1px", transition: "all 0.3s" }}>
            MEMORY
          </button>
        </div>
      </div>

      {/* ═══ AGENT STATUS BAR ═══ */}
      <div style={{ display: "flex", alignItems: "center", gap: "2px", padding: "4px 16px", borderBottom: `1px solid ${P.borderFaint}`, zIndex: 10 }}>
        {Object.entries(AGENTS).map(([id, ag]) => {
          const active = st.agentStates[id] !== "idle";
          const isCurrent = id === st.currentAgent;
          return (
            <div key={id} style={{ flex: 1, padding: "3px 2px", borderRadius: "2px", textAlign: "center", background: isCurrent ? `${ag.color}15` : active ? `${ag.color}06` : "transparent", border: `1px solid ${isCurrent ? ag.color + "35" : active ? ag.color + "15" : P.borderFaint}`, transition: "all 0.4s", boxShadow: isCurrent ? `0 0 8px ${ag.color}12` : "none" }}>
              <div style={{ fontSize: "7px", color: active ? ag.color : "#374151", letterSpacing: "0.5px", fontWeight: isCurrent ? "bold" : "normal" }}>
                {ag.icon} {ag.call}
              </div>
            </div>
          );
        })}
      </div>

      {/* ═══ MAIN BODY — Hash Table Layout (Gem 4) ═══ */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden", position: "relative", zIndex: 5 }}>

        {/* LEFT: Scene 3 center — Seed of Life + Feed */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>

          {/* Scene 1: Morning Brief (Gem 2 — top hero) */}
          <div style={{ padding: "8px 14px 0" }}>
            <GhostBrief events={st.ghostEvents} expanded={st.briefExpanded}
              onToggle={() => setSt(p => ({ ...p, briefExpanded: !p.briefExpanded }))} />
          </div>

          {/* Sacred Geometry Telemetry + Stats */}
          <div style={{ display: "flex", alignItems: "center", gap: "12px", padding: "8px 14px" }}>
            <SeedOfLifeTelemetry agentStates={st.agentStates} currentAgent={st.currentAgent} petalFill={st.petalFill} />
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "4px 12px", fontSize: "8px" }}>
              {[
                ["Ihsan", st.ihsan.toFixed(4), P.gold],
                ["Streak", st.streak, P.verified],
                ["Reflexes", `${st.reflexes}`, P.cyan],
                ["RAC", st.rac, P.blue],
                ["Legendary", st.legendary, P.amber],
                ["Level", st.level, P.purple],
              ].map(([l, v, c]) => (
                <div key={l} style={{ display: "flex", justifyContent: "space-between", gap: "8px" }}>
                  <span style={{ color: P.textDim }}>{l}</span>
                  <span style={{ color: c, fontVariantNumeric: "tabular-nums" }}>{v}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Message Feed — Scene 3: PAT-7 Dashboard */}
          <div style={{ flex: 1, overflowY: "auto", padding: "4px 14px" }}>
            {st.messages.map((msg, i) => {
              const isUser = msg.type === "user";
              const isSystem = msg.type === "system";
              const isMint = msg.type === "mint";
              const isComplete = msg.type === "complete";
              const isGreeting = msg.type === "greeting";
              const col = isUser ? P.gold : isSystem ? "#374151" : isMint ? P.verified : agCol(msg.agent);
              const label = isUser ? "YOU" : isSystem ? "SYS" : AGENTS[msg.agent]?.call || msg.agent;
              return (
                <div key={i} style={{ marginBottom: "2px", padding: "2px 0", display: "flex", gap: "8px", alignItems: "flex-start", animation: "slideIn 0.15s ease", opacity: isSystem ? 0.45 : 1 }}>
                  <span style={{ color: col, fontWeight: "bold", minWidth: "44px", fontSize: "8px", textAlign: "right" }}>{label}</span>
                  <span style={{ color: isUser ? P.text : isMint ? P.verified : isComplete ? P.gold : isGreeting ? P.goldLight : "#7c8594", fontSize: isGreeting ? "11px" : "10px", lineHeight: 1.5, fontFamily: isGreeting ? "'Cormorant Garamond', serif" : "inherit", letterSpacing: isGreeting ? "0.5px" : "0" }}>
                    {isMint ? `► ${msg.text}` : isComplete ? `✓ ${msg.text}` : msg.text}
                  </span>
                </div>
              );
            })}
            <div ref={msgEnd} />
          </div>

          {/* Quick Missions */}
          {st.phase === "ready" && (
            <div style={{ padding: "4px 14px", display: "flex", gap: "4px", flexWrap: "wrap", borderTop: `1px solid ${P.borderFaint}` }}>
              {QUICK_MISSIONS.map((m, i) => (
                <button key={i} onClick={() => { addMsg("USER", m, "user"); setTimeout(() => executeMission(m), 300); }}
                  style={{ background: `rgba(255,255,255,0.015)`, border: `1px solid ${P.borderFaint}`, color: "#4B5563", padding: "3px 8px", borderRadius: "2px", fontSize: "7px", cursor: "pointer", fontFamily: "monospace", transition: "all 0.3s", maxWidth: "200px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
                  onMouseEnter={e => { e.target.style.borderColor = `${P.gold}25`; e.target.style.color = P.gold; }}
                  onMouseLeave={e => { e.target.style.borderColor = P.borderFaint; e.target.style.color = "#4B5563"; }}>
                  {m}
                </button>
              ))}
            </div>
          )}

          {/* Input — Operator Rail */}
          <div style={{ padding: "8px 14px", borderTop: `1px solid ${P.border}`, display: "flex", gap: "8px", alignItems: "center", background: "rgba(10,22,40,0.4)" }}>
            <span style={{ color: st.phase === "mission" ? P.amber : P.gold, fontSize: "11px" }}>▸</span>
            <input ref={inputRef} value={input} onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === "Enter" && handleSubmit()}
              placeholder={st.phase === "mission" ? "Mission in progress..." : "Speak your mission..."}
              disabled={st.phase === "mission"}
              style={{ flex: 1, background: "transparent", border: "none", color: P.text, fontSize: "11px", fontFamily: "'DM Mono', monospace", outline: "none", letterSpacing: "0.3px" }} />
            {input.trim() && st.phase !== "mission" && (
              <button onClick={handleSubmit} style={{ background: `${P.gold}0a`, border: `1px solid ${P.gold}25`, color: P.gold, padding: "3px 14px", borderRadius: "2px", fontSize: "8px", cursor: "pointer", fontFamily: "monospace", letterSpacing: "2px" }}>
                EXECUTE
              </button>
            )}
          </div>
        </div>

        {/* RIGHT: Scene 2 — Trust Rail (Gem 6: Rust precision) */}
        <TrustRail trust={st.trust} ihsan={st.ihsan} />

        {/* Scene 4: Memory Drawer (Gem 4: Z-index overlay) */}
        <MemoryDrawer open={st.memoryOpen} onClose={() => setSt(p => ({ ...p, memoryOpen: false }))} />
      </div>

      {/* ═══ BOTTOM STATUS ═══ */}
      <div style={{ padding: "4px 16px", borderTop: `1px solid ${P.borderFaint}`, display: "flex", justifyContent: "space-between", fontSize: "7px", color: "#1a2540", letterSpacing: "0.8px", zIndex: 10 }}>
        <span>{tierName} · IHSAN {st.ihsan.toFixed(4)} · STREAK {st.streak} · RAC {st.rac}</span>
        <span style={{ fontFamily: "'Amiri', serif", fontSize: "9px", color: `${P.gold}18` }}>بذرة واحدة تصنع غابة</span>
        <span>PAT-7 · SAT-5 · 15 ALGORITHMS · 7 INVARIANTS</span>
      </div>
    </div>
  );
}
