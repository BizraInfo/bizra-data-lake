import { useState, useEffect, useRef, useCallback } from "react";

// ═══ AGENT PERSONALITIES (the soul of JARVIS) ═══
const AGENTS = {
  P1: { name: "Planner", call: "ATLAS", color: "#3b82f6", glow: "rgba(59,130,246,0.3)",
    voice: "strategic", greeting: "Standing by for mission parameters.",
    idle: ["Analyzing priority queue...", "Three pending objectives identified.", "Shall I restructure your schedule?", "Your roadmap has a dependency conflict I can resolve."],
    working: ["Decomposing into subtasks...", "Dependency graph resolved.", "Execution order optimized.", "Critical path identified — 3 steps."],
  },
  P2: { name: "Researcher", call: "ORACLE", color: "#06b6d4", glow: "rgba(6,182,212,0.3)",
    voice: "analytical", greeting: "Knowledge systems online. What shall I find?",
    idle: ["I found something interesting in your domain...", "Three new papers match your interests.", "Your knowledge graph has grown 12% this week.", "Shall I deep-dive on that topic from yesterday?"],
    working: ["Scanning knowledge base...", "Cross-referencing 47 sources.", "Signal-to-noise ratio: 0.94.", "Synthesis complete. Key findings extracted."],
  },
  P3: { name: "Coder", call: "FORGE", color: "#22c55e", glow: "rgba(34,197,94,0.3)",
    voice: "precise", greeting: "Compiler ready. What are we building?",
    idle: ["Your test suite has 3 flaky tests I can fix.", "I spotted a refactoring opportunity in the kernel.", "Build pipeline is green. All 219 tests passing.", "Dependency update available — no breaking changes."],
    working: ["Generating implementation...", "Running test suite...", "All assertions pass.", "Code quality: Ihsan 0.97."],
  },
  P4: { name: "Evaluator", call: "JUDGE", color: "#f59e0b", glow: "rgba(245,158,11,0.3)",
    voice: "measured", greeting: "Quality gates armed. Show me what to assess.",
    idle: ["Your average Ihsan is trending up — 0.983 this week.", "I've benchmarked 3 alternatives for your last approach.", "Quality score: top 5% of all nodes.", "Recommending a peer review for your latest reflex."],
    working: ["Running quality assessment...", "Shannon entropy: above threshold.", "Scoring against rubric...", "Verdict: exceeds constitutional floor."],
  },
  P5: { name: "Ethicist", call: "CROWN", color: "#f43f5e", glow: "rgba(244,63,94,0.3)",
    voice: "solemn", greeting: "Constitutional watch active. All seven invariants holding.",
    idle: ["All invariants satisfied. System is constitutional.", "I-3 check: Gini at 0.31 — well within bounds.", "No ethical flags in your recent actions.", "The covenant holds. Integrity verified."],
    working: ["Scanning against I-1 through I-7...", "Shariah compliance: verified.", "No bias detected in output.", "Constitutional clearance granted."],
  },
  P6: { name: "Publisher", call: "HERALD", color: "#f97316", glow: "rgba(249,115,22,0.3)",
    voice: "articulate", greeting: "Ready to deliver your message to the world.",
    idle: ["Your last report scored 4.8/5.0 readability.", "I've drafted three versions of your response.", "Format optimized for your audience.", "Feedback from the last delivery was excellent."],
    working: ["Structuring output...", "Formatting for clarity...", "Final polish applied.", "Ready for delivery. Shall I publish?"],
  },
  P7: { name: "Integrator", call: "NEXUS", color: "#a855f7", glow: "rgba(168,85,247,0.3)",
    voice: "commanding", greeting: "All agents reporting. Nexus is online.",
    idle: ["All seven agents nominal.", "Memory utilization: optimal.", "I've pre-loaded context from your last session.", "Cross-agent coordination score: 94%."],
    working: ["Routing to specialist...", "Context bridge established.", "Agent handoff complete.", "Aggregating results from all sources."],
  },
};

const SAT = [
  { id: "S1", name: "Sentinel", status: "WATCHING", color: "#ef4444" },
  { id: "S2", name: "Oracle", status: "SCORING", color: "#C9A962" },
  { id: "S3", name: "Ledger", status: "RECORDING", color: "#f59e0b" },
  { id: "S4", name: "Conductor", status: "ROUTING", color: "#3b82f6" },
  { id: "S5", name: "Ambassador", status: "LISTENING", color: "#06b6d4" },
];

const MISSIONS = [
  "Research the latest developments in sovereign AI architectures",
  "Build a testing framework for constitutional invariant verification",
  "Evaluate our deployment pipeline against production benchmarks",
  "Draft the quarterly progress report for stakeholders",
  "Plan the Alpha-100 rollout strategy and timeline",
  "Review the authentication module for security vulnerabilities",
  "Check constitutional compliance of the new minting parameters",
];

// ═══ SIMULATION STATE ═══
const initState = () => ({
  booted: false, phase: "boot", bootStep: 0,
  seed: 0, bloom: 0, rac: 0, vac: 0, tier: "Novice", level: 0,
  ihsan: 0, streak: 0, mye: 0, s1: 0, s2: 0,
  reflexes: 0, skills: 3, legendary: 0, epic: 0,
  messages: [], activeMission: null, missionPhase: null,
  agentStates: Object.fromEntries(Object.keys(AGENTS).map(k => [k, "idle"])),
  proactiveQueue: [],
});

// ═══ MAIN COMPONENT ═══
export default function JARVIS() {
  const [st, setSt] = useState(initState());
  const [input, setInput] = useState("");
  const [time, setTime] = useState(new Date());
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
    setSt(p => ({ ...p, messages: [...p.messages, { agent, text, type, ts: Date.now() }].slice(-50) }));
  }, []);

  const delay = ms => new Promise(r => setTimeout(r, ms));

  // ═══ BOOT SEQUENCE ═══
  const boot = useCallback(async () => {
    setSt(p => ({ ...p, phase: "booting" }));
    const steps = [
      ["SYS", "Initializing sovereign kernel...", 400],
      ["SYS", "Fixed-point arithmetic verified. Deterministic.", 300],
      ["SYS", "Constitutional invariants I-1 through I-7: LOADED", 300],
      ["SYS", "Covenant hash: 859649ea...verified ✓", 200],
      ["SYS", "Ed25519 identity generated.", 200],
      ["SYS", "Minting Personal Agentic Team...", 400],
      ["P7", "NEXUS online. Establishing agent links.", 300],
      ["P1", "ATLAS standing by. Strategic planning ready.", 200],
      ["P2", "ORACLE online. Knowledge systems nominal.", 200],
      ["P3", "FORGE ready. Compiler initialized.", 200],
      ["P4", "JUDGE armed. Quality gates active.", 200],
      ["P5", "CROWN watching. Constitution enforced.", 200],
      ["P6", "HERALD ready. Delivery channels open.", 200],
      ["SYS", "SAT-5 system agents deployed (you have zero control).", 300],
      ["SYS", "Sentinel watching. Oracle scoring. Ledger recording.", 200],
      ["P7", "All seven agents reporting. Full operational status.", 300],
    ];
    for (const [agent, text, ms] of steps) {
      addMsg(agent, text, agent === "SYS" ? "system" : "agent");
      await delay(ms);
    }
    await delay(300);
    addMsg("P7", "Good evening. Your sovereign AI team is ready. What shall we work on?", "greeting");
    setSt(p => ({ ...p, booted: true, phase: "ready" }));
    setTimeout(() => inputRef.current?.focus(), 100);

    // Proactive after 8 seconds
    setTimeout(() => {
      const agent = ["P1", "P2", "P3", "P4"][Math.floor(Math.random() * 4)];
      const msg = AGENTS[agent].idle[Math.floor(Math.random() * AGENTS[agent].idle.length)];
      addMsg(agent, msg, "proactive");
    }, 8000);
  }, [addMsg]);

  // ═══ MISSION EXECUTION ═══
  const executeMission = useCallback(async (task) => {
    setSt(p => ({ ...p, phase: "mission", activeMission: task }));

    // P7 Integrator routes
    addMsg("P7", `Mission received. Analyzing: "${task.slice(0, 60)}..."`, "agent");
    await delay(600);

    // Route to best agent
    const kw = { P1: ["plan","organize","strategy","roadmap","schedule"], P2: ["research","find","analyze","study","paper"], P3: ["code","build","test","debug","implement","fix","deploy"], P4: ["evaluate","score","assess","review","benchmark"], P5: ["check","verify","compliance","constitution","ethics"], P6: ["write","draft","report","document","publish","present"] };
    let best = "P2"; let bs = 0;
    for (const [a, ws] of Object.entries(kw)) {
      const s = ws.filter(w => task.toLowerCase().includes(w)).length;
      if (s > bs) { best = a; bs = s; }
    }
    const agent = AGENTS[best];

    addMsg("P7", `Routing to ${agent.call}. ${agent.name} has the best capability match.`, "agent");
    setSt(p => ({ ...p, agentStates: { ...p.agentStates, [best]: "active", P7: "routing" } }));
    await delay(500);

    // Agent working sequence
    for (const msg of agent.working) {
      addMsg(best, msg, "working");
      await delay(700 + Math.random() * 400);
    }

    // P4 Evaluator scores
    setSt(p => ({ ...p, agentStates: { ...p.agentStates, P4: "scoring" } }));
    addMsg("P4", "Quality assessment initiated.", "working");
    await delay(500);
    const ihsan = (0.95 + Math.random() * 0.04).toFixed(4);
    addMsg("P4", `Ihsan score: ${ihsan}. ${parseFloat(ihsan) >= 0.98 ? "Exceptional quality." : "Above constitutional floor."}`, "score");
    await delay(400);

    // P5 Ethicist clears
    setSt(p => ({ ...p, agentStates: { ...p.agentStates, P5: "checking" } }));
    addMsg("P5", "Constitutional scan... All seven invariants hold. Cleared.", "clear");
    await delay(400);

    // Mint
    const isLegendary = parseFloat(ihsan) >= 0.98 && Math.random() > 0.5;
    const isEpic = !isLegendary && parseFloat(ihsan) >= 0.96;
    const drop = isLegendary ? "⚡ LEGENDARY" : isEpic ? "💜 EPIC" : "🔵 RARE";
    const mul = isLegendary ? 1.5 : isEpic ? 1.3 : 1.15;
    const seedEarned = (1.0 * parseFloat(ihsan) * mul).toFixed(3);
    const bloomEarned = (0.01 * parseFloat(ihsan)).toFixed(4);

    addMsg("SYS", `PoI_EMIT receipt generated. ${drop} drop. +${seedEarned} SEED, +${bloomEarned} BLOOM.`, "mint");
    await delay(300);

    // P6 Publisher delivers
    addMsg("P6", `Results formatted and delivered. Receipt signed and chained.`, "agent");
    await delay(300);

    // P7 wraps up
    const newRac = st.rac + 1;
    const newSeed = st.seed + parseFloat(seedEarned);
    const compiled = newRac > 0 && newRac % 5 === 0;

    addMsg("P7", `Mission complete. You earned ${seedEarned} SEED. ${compiled ? "Pattern compiled to reflex — next time will be 8× faster." : `${5 - (newRac % 5)} more runs of this pattern to compile a reflex.`}`, "complete");

    setSt(p => ({
      ...p, phase: "ready", activeMission: null,
      seed: p.seed + parseFloat(seedEarned), bloom: p.bloom + parseFloat(bloomEarned),
      rac: p.rac + 1, vac: p.vac + 1, streak: p.streak + 1, level: Math.floor((p.rac + 1) / 10),
      ihsan: parseFloat(ihsan), s2: p.s2 + 1, mye: p.s1 / Math.max(p.s1 + p.s2 + 1, 1),
      reflexes: p.reflexes + (compiled ? 1 : 0),
      legendary: p.legendary + (isLegendary ? 1 : 0), epic: p.epic + (isEpic ? 1 : 0),
      agentStates: Object.fromEntries(Object.keys(AGENTS).map(k => [k, "idle"])),
    }));

    // Proactive follow-up after mission
    setTimeout(() => {
      const followups = [
        ["P2", "I noticed a related topic that might be worth exploring next."],
        ["P1", "Based on this result, I've updated your priority queue."],
        ["P4", "Your Ihsan average this session is exceptional. Keep this trajectory."],
        ["P3", "That pattern is close to compilation. Two more quality runs and I'll have a reflex ready."],
        ["P7", "All agents returning to standby. Ready for your next directive."],
      ];
      const f = followups[Math.floor(Math.random() * followups.length)];
      addMsg(f[0], f[1], "proactive");
    }, 3000);
  }, [addMsg, st.rac, st.seed, st.s1, st.s2]);

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

  // ═══ RENDER ═══
  const G = "#C9A962";
  const agentColor = (id) => AGENTS[id]?.color || "#C9A962";
  const formatTime = (d) => d.toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });

  if (!st.booted && st.phase !== "booting") {
    return (
      <div style={{ minHeight: "100vh", background: "#030810", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", fontFamily: "'Courier New', monospace" }}>
        <div style={{ width: "80px", height: "80px", borderRadius: "50%", border: `2px solid ${G}30`, display: "flex", alignItems: "center", justifyContent: "center", marginBottom: "24px", boxShadow: `0 0 40px ${G}15, inset 0 0 20px ${G}08` }}>
          <div style={{ width: "40px", height: "40px", borderRadius: "50%", background: `radial-gradient(circle, ${G}40, transparent)`, animation: "pulse 2s ease-in-out infinite" }} />
        </div>
        <div style={{ color: G, fontSize: "11px", letterSpacing: "6px", marginBottom: "8px" }}>BIZRA</div>
        <div style={{ color: "#6B7280", fontSize: "9px", letterSpacing: "3px", marginBottom: "32px" }}>SOVEREIGN AI INTERFACE</div>
        <button onClick={boot} style={{ background: "transparent", border: `1px solid ${G}40`, color: G, padding: "12px 40px", borderRadius: "2px", fontSize: "11px", letterSpacing: "4px", cursor: "pointer", fontFamily: "'Courier New', monospace", transition: "all 0.3s" }}
          onMouseEnter={e => { e.target.style.background = `${G}15`; e.target.style.boxShadow = `0 0 30px ${G}20`; }}
          onMouseLeave={e => { e.target.style.background = "transparent"; e.target.style.boxShadow = "none"; }}>
          INITIALIZE
        </button>
        <style>{`@keyframes pulse { 0%,100% { opacity: 0.4; transform: scale(1); } 50% { opacity: 0.8; transform: scale(1.1); } }`}</style>
      </div>
    );
  }

  return (
    <div style={{ minHeight: "100vh", background: "#030810", color: "#E5E7EB", fontFamily: "'Courier New', monospace", fontSize: "11px", display: "flex", flexDirection: "column" }}>
      {/* ═══ TOP HUD ═══ */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 16px", borderBottom: `1px solid ${G}15`, background: "#030810" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <span style={{ color: G, fontWeight: "bold", letterSpacing: "3px", fontSize: "10px" }}>BIZRA</span>
          <span style={{ color: "#374151", fontSize: "9px" }}>NODE0</span>
          <span style={{ color: st.phase === "mission" ? "#f59e0b" : "#22c55e", fontSize: "8px", letterSpacing: "2px" }}>
            {st.phase === "mission" ? "● EXECUTING" : "● READY"}
          </span>
        </div>
        <div style={{ display: "flex", gap: "16px", fontSize: "9px" }}>
          <span style={{ color: "#22c55e" }}>{st.seed.toFixed(2)} SEED</span>
          <span style={{ color: "#a855f7" }}>{st.bloom.toFixed(3)} BLOOM</span>
          <span style={{ color: "#3b82f6" }}>Lv.{st.level}</span>
          <span style={{ color: G }}>{formatTime(time)}</span>
        </div>
      </div>

      {/* ═══ AGENT STATUS BAR ═══ */}
      <div style={{ display: "flex", gap: "2px", padding: "4px 16px", borderBottom: "1px solid rgba(255,255,255,0.03)" }}>
        {Object.entries(AGENTS).map(([id, ag]) => {
          const active = st.agentStates[id] !== "idle";
          return (
            <div key={id} style={{ flex: 1, padding: "4px 6px", borderRadius: "2px", background: active ? `${ag.color}15` : "transparent", border: `1px solid ${active ? ag.color + "30" : "rgba(255,255,255,0.04)"}`, textAlign: "center", transition: "all 0.3s" }}>
              <div style={{ fontSize: "7px", color: active ? ag.color : "#4B5563", letterSpacing: "1px", fontWeight: active ? "bold" : "normal" }}>{ag.call}</div>
              <div style={{ fontSize: "6px", color: "#374151" }}>{ag.name}</div>
            </div>
          );
        })}
        <div style={{ width: "1px", background: "rgba(255,255,255,0.06)", margin: "0 4px" }} />
        {SAT.map(s => (
          <div key={s.id} style={{ padding: "4px 4px", textAlign: "center" }}>
            <div style={{ width: "6px", height: "6px", borderRadius: "50%", background: s.color + "60", margin: "0 auto 2px", boxShadow: `0 0 4px ${s.color}30` }} />
            <div style={{ fontSize: "5px", color: "#374151" }}>{s.name}</div>
          </div>
        ))}
      </div>

      {/* ═══ MAIN FEED ═══ */}
      <div style={{ flex: 1, overflowY: "auto", padding: "8px 16px" }}>
        {st.messages.map((msg, i) => {
          const isUser = msg.type === "user";
          const isSystem = msg.type === "system";
          const isMint = msg.type === "mint";
          const isProactive = msg.type === "proactive";
          const isComplete = msg.type === "complete";
          const agCol = isUser ? G : isSystem ? "#4B5563" : isMint ? "#22c55e" : agentColor(msg.agent);
          const label = isUser ? "YOU" : isSystem ? "SYS" : AGENTS[msg.agent]?.call || msg.agent;

          return (
            <div key={i} style={{ marginBottom: "3px", padding: "3px 0", opacity: isSystem ? 0.6 : 1, display: "flex", gap: "8px", alignItems: "flex-start" }}>
              <span style={{ color: agCol, fontWeight: "bold", minWidth: "52px", fontSize: "9px", textAlign: "right", opacity: isProactive ? 0.7 : 1 }}>
                {label}
              </span>
              <span style={{ color: isUser ? "#F8F6F1" : isMint ? "#22c55e" : isComplete ? G : isProactive ? `${agCol}cc` : "#9CA3AF", fontSize: isUser ? "11px" : "10px", lineHeight: "1.5", fontStyle: isProactive ? "italic" : "normal" }}>
                {isMint ? `► ${msg.text}` : isComplete ? `✓ ${msg.text}` : msg.text}
              </span>
            </div>
          );
        })}
        <div ref={msgEnd} />
      </div>

      {/* ═══ QUICK MISSIONS ═══ */}
      {st.phase === "ready" && (
        <div style={{ padding: "4px 16px", display: "flex", gap: "4px", flexWrap: "wrap", borderTop: "1px solid rgba(255,255,255,0.03)" }}>
          {MISSIONS.slice(0, 4).map((m, i) => (
            <button key={i} onClick={() => quickMission(m)} style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", color: "#6B7280", padding: "3px 8px", borderRadius: "2px", fontSize: "8px", cursor: "pointer", fontFamily: "'Courier New', monospace", transition: "all 0.2s", maxWidth: "200px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
              onMouseEnter={e => { e.target.style.borderColor = `${G}40`; e.target.style.color = G; }}
              onMouseLeave={e => { e.target.style.borderColor = "rgba(255,255,255,0.06)"; e.target.style.color = "#6B7280"; }}>
              {m.slice(0, 40)}...
            </button>
          ))}
        </div>
      )}

      {/* ═══ INPUT ═══ */}
      <div style={{ padding: "8px 16px", borderTop: `1px solid ${G}15`, display: "flex", gap: "8px", alignItems: "center" }}>
        <span style={{ color: G, fontSize: "10px" }}>▸</span>
        <input
          ref={inputRef}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && handleSubmit()}
          placeholder={st.phase === "mission" ? "Mission in progress..." : "Speak your mission..."}
          disabled={st.phase === "mission"}
          style={{ flex: 1, background: "transparent", border: "none", color: "#F8F6F1", fontSize: "11px", fontFamily: "'Courier New', monospace", outline: "none", letterSpacing: "0.5px" }}
        />
        <div style={{ display: "flex", gap: "8px", fontSize: "8px", color: "#374151" }}>
          <span>RAC:{st.rac}</span>
          <span>MYE:{(st.mye * 100).toFixed(0)}%</span>
          <span>{st.reflexes}⚡</span>
          <span>{st.legendary}🏆</span>
        </div>
      </div>

      {/* ═══ BOTTOM STATUS ═══ */}
      <div style={{ padding: "4px 16px", borderTop: "1px solid rgba(255,255,255,0.03)", display: "flex", justifyContent: "space-between", fontSize: "7px", color: "#374151", letterSpacing: "1px" }}>
        <span>TIER: {st.tier.toUpperCase()} · IHSAN: {st.ihsan.toFixed(4)} · STREAK: {st.streak}</span>
        <span>PAT-7 SOVEREIGN · SAT-5 SYSTEM · 15 ALGORITHMS · 7 INVARIANTS</span>
      </div>
    </div>
  );
}
