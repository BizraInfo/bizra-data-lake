import { useState, useEffect, useRef, useCallback } from "react";

/*
  ═══════════════════════════════════════════════════════════════
  BIZRA — The Front Door
  ═══════════════════════════════════════════════════════════════
  
  "The distance between a working system and a product is not
  proportional to the system's complexity. It's proportional
  to the quality of the front door." — Gem 11
  
  Behind this cursor: 768K LOC, 26 crates, 12,644 tests.
  In front of this cursor: "Good morning, Mumo."
  
  Design: TUI-soul desktop. Terminal precision meets sacred
  geometry. The operator types بذرة › and the system responds
  with everything it knows, everything it verified, and
  everything it anticipates.
  
  ═══════════════════════════════════════════════════════════════
*/

// ═══ PALETTE ═══
const $ = {
  void: "#020408", deep: "#060D18", ink: "#0B1524", well: "#101D30",
  gold: "#C9A962", goldSoft: "#D4BC82", goldDim: "#7A6332", goldGhost: "rgba(201,169,98,0.07)",
  goldLine: "rgba(201,169,98,0.12)", goldPulse: "rgba(201,169,98,0.25)",
  bone: "#F0EDE4", ash: "rgba(240,237,228,0.55)", ghost: "rgba(240,237,228,0.28)", ember: "rgba(240,237,228,0.12)",
  pass: "#3DD68C", fail: "#F0544F", warn: "#F0A03C",
  p1: "#5B8DEF", p2: "#3CC8CF", p3: "#3DD68C", p4: "#E8B73A", p5: "#E85D6F", p6: "#E8883A", p7: "#A36BDF",
};

// ═══ PAT-7 ═══
const A = {
  P1: { n: "ATLAS", r: "Planner", c: $.p1, idle: ["Strategic planning ready.", "Priority queue analyzed — 3 pending.", "Roadmap dependency resolved."], work: ["Decomposing subtasks...", "Critical path: 3 steps.", "Execution order locked."] },
  P2: { n: "ORACLE", r: "Researcher", c: $.p2, idle: ["Knowledge systems nominal.", "3 papers match your domain.", "Graph grew 12% this week."], work: ["Scanning 47 sources...", "SNR: 0.94.", "Key findings extracted."] },
  P3: { n: "FORGE", r: "Coder", c: $.p3, idle: ["Build pipeline green.", "Refactoring opportunity spotted.", "All 219 tests passing."], work: ["Generating implementation...", "Test suite running...", "Ihsan 0.97. Clean."] },
  P4: { n: "JUDGE", r: "Evaluator", c: $.p4, idle: ["Quality trending up — 0.983.", "Top 5% of all nodes.", "3 alternatives benchmarked."], work: ["Scoring against rubric...", "Shannon entropy: above threshold.", "Verdict: constitutional."] },
  P5: { n: "CROWN", r: "Ethicist", c: $.p5, idle: ["All 7 invariants hold.", "Gini 0.31 — within bounds.", "Covenant verified."], work: ["Scanning I-1 through I-7...", "Shariah compliance: verified.", "Clearance granted."] },
  P6: { n: "HERALD", r: "Publisher", c: $.p6, idle: ["Readability 4.8/5.0.", "Three drafts prepared.", "Format optimized."], work: ["Structuring output...", "Polish applied.", "Ready for delivery."] },
  P7: { n: "NEXUS", r: "Integrator", c: $.p7, idle: ["All seven nominal.", "Context pre-loaded.", "Coordination: 94%."], work: ["Routing to specialist...", "Context bridge established.", "Results aggregated."] },
};

const SAT = ["Sentinel", "Oracle", "Ledger", "Conductor", "Ambassador"];

// ═══ GHOST EVENTS (ghost_ws.py OverlayEvent schema) ═══
const GHOSTS = [
  { label: "Open weekly review template", why: "Review overdue by 3 days. Recent idle pattern.", conf: 0.91, ihsan: "pass", score: 0.97, from: "P1" },
  { label: "Run CI stabilization on 3 flaky tests", why: "Failures in last 2 commits. Known fix pattern.", conf: 0.88, ihsan: "pass", score: 0.95, from: "P3" },
  { label: "Draft Spine v1.2 amendment", why: "Two inconsistencies between Spine and codebase.", conf: 0.72, ihsan: "blocked", score: 0.62, from: "P5" },
];

const MEMORIES = [
  { kind: "reflex", text: "CI pattern: isolate → reproduce → fix → gate", age: "2h" },
  { kind: "knowledge", text: "Phase 81: 471,917 LOC, 11,135 tests, SNR 0.958", age: "1d" },
  { kind: "episode", text: "Mint Court rejected at SNR 0.577 — governance works", age: "3d" },
];

// ═══ STATE ═══
const boot = () => ({
  up: false, phase: "dark",
  seed: 0, bloom: 0, ihsan: 0, streak: 0, rac: 0, lv: 0, rfx: 0, leg: 0,
  feed: [], agents: Object.fromEntries(Object.keys(A).map(k => [k, "idle"])),
  focus: null, trust: {}, ghosts: [], mem: false, brief: true, petal: 0,
});

// ═══ SEED OF LIFE ═══
function Seed({ agents, focus, petal }) {
  const R = 18, map = ["P5","P1","P2","P3","P4","P6","P7"];
  const pts = [[0,0],[0,-R],[R*.866,-R*.5],[R*.866,R*.5],[0,R],[-R*.866,R*.5],[-R*.866,-R*.5]];
  return (
    <svg width="110" height="110" viewBox="-36 -36 72 72" style={{ display: "block" }}>
      <circle cx="0" cy="0" r="34" fill="none" stroke={$.ember} strokeWidth=".4" strokeDasharray="2 2"/>
      {[0,60,120,180,240,300].map((a,i) => {
        const f = Math.min(petal/6,1);
        return <path key={`p${i}`} d={`M0 ${-R} Q${R*.35} ${-R*.35} 0 0 Q${-R*.35} ${-R*.35} 0 ${-R}`}
          transform={`rotate(${a})`} fill={`rgba(201,169,98,${(f*.18+.02).toFixed(2)})`}
          stroke={`rgba(201,169,98,${(f*.3+.08).toFixed(2)})`} strokeWidth=".4"
          style={{ transition: "all 1.2s ease" }}/>;
      })}
      {pts.map(([x,y],i) => {
        const id = map[i], ag = A[id], on = agents[id] !== "idle", hot = id === focus;
        return <g key={i}>
          <circle cx={x} cy={y} r={R} fill={hot ? `${ag.c}15` : on ? `${ag.c}08` : "none"}
            stroke={hot ? ag.c : on ? `${ag.c}40` : `${$.gold}18`} strokeWidth={hot ? 1 : .4}
            style={{ transition: "all .5s" }}/>
          {(on||hot) && <text x={x} y={y+1} textAnchor="middle" dominantBaseline="central"
            style={{ fontSize: "4.5px", fill: ag.c, fontFamily: "monospace", letterSpacing: ".3px" }}>{ag.n}</text>}
        </g>;
      })}
      <rect x="-2" y="-2" width="4" height="4" rx=".8" transform="rotate(45)"
        fill={$.gold} opacity={agents.P5 !== "idle" ? .9 : .35} style={{ transition: "opacity .5s" }}/>
    </svg>
  );
}

// ═══ TRUST RAIL ═══
function Trust({ trust, ihsan }) {
  const checks = [
    { k: "node", l: "Node health", i: "●" },
    { k: "ledger", l: "SEL chain", i: "◈" },
    { k: "token", l: "SEED balance", i: "◆" },
    { k: "supply", l: "Supply cap", i: "▣" },
    { k: "gate", l: "Ihsan gate", i: "✦" },
  ];
  return (
    <div style={{ width: "160px", minWidth: "160px", padding: "10px", borderLeft: `1px solid ${$.ember}`, background: `linear-gradient(180deg, ${$.ink}80, ${$.void}90)`, display: "flex", flexDirection: "column", gap: "10px", overflowY: "auto" }}>
      <div style={{ fontSize: "7px", color: $.ghost, letterSpacing: "2px" }}>TRUST VERIFICATION</div>
      {checks.map(({ k, l, i }) => {
        const v = trust[k]; const c = v === true ? $.pass : v === false ? $.fail : $.ember;
        return <div key={k} style={{ display: "flex", alignItems: "center", gap: "5px", fontSize: "9px" }}>
          <span style={{ color: c, fontSize: "7px", transition: "color .5s" }}>{i}</span>
          <span style={{ color: v ? $.ash : $.ghost, flex: 1, letterSpacing: ".3px" }}>{l}</span>
          <span style={{ color: c, fontSize: "7px", fontFamily: "monospace", letterSpacing: "1px" }}>{v === true ? "OK" : "—"}</span>
        </div>;
      })}
      <div style={{ marginTop: "4px", padding: "8px", borderRadius: "3px", background: $.goldGhost, border: `1px solid ${$.goldLine}` }}>
        <div style={{ fontSize: "6px", color: $.gold, letterSpacing: "2px", marginBottom: "4px" }}>IHSAN COMPOSITE</div>
        <div style={{ fontSize: "18px", fontFamily: "monospace", color: ihsan >= .95 ? $.gold : $.warn, textAlign: "center", fontVariantNumeric: "tabular-nums" }}>{ihsan.toFixed(4)}</div>
        <div style={{ height: "2px", background: $.ember, borderRadius: "1px", marginTop: "5px", overflow: "hidden" }}>
          <div style={{ height: "100%", width: `${Math.min(ihsan*100,100)}%`, background: ihsan >= .95 ? $.gold : $.warn, transition: "width .8s", borderRadius: "1px" }}/>
        </div>
      </div>
      <div style={{ marginTop: "auto" }}>
        <div style={{ fontSize: "6px", color: $.ghost, letterSpacing: "2px", marginBottom: "5px" }}>SAT-5 OVERSIGHT</div>
        <div style={{ display: "flex", gap: "5px", flexWrap: "wrap" }}>
          {SAT.map((s,i) => <div key={i} style={{ fontSize: "7px", color: $.ghost, display: "flex", alignItems: "center", gap: "3px" }}>
            <div style={{ width: "4px", height: "4px", borderRadius: "50%", background: `${[$.fail,$.gold,$.warn,$.p1,$.p2][i]}40` }}/>{s}
          </div>)}
        </div>
      </div>
      <div style={{ fontSize: "6px", fontFamily: "monospace", color: `${$.gold}20`, wordBreak: "break-all", lineHeight: 1.3 }}>859649ea...verified</div>
    </div>
  );
}

// ═══ GHOST BRIEF ═══
function Brief({ events, open, toggle }) {
  if (!events.length) return null;
  const ok = events.filter(e => e.ihsan === "pass").length;
  return (
    <div style={{ background: $.goldGhost, border: `1px solid ${$.goldLine}`, borderRadius: "4px", padding: open ? "10px" : "7px 10px", transition: "all .3s" }}>
      <div onClick={toggle} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", cursor: "pointer" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={{ fontSize: "7px", color: $.gold, letterSpacing: "2px" }}>MORNING BRIEF</span>
          <span style={{ fontSize: "8px", color: $.ash }}>{ok} actionable · {events.length - ok} blocked</span>
        </div>
        <span style={{ color: $.ghost, fontSize: "9px", transform: open ? "rotate(180deg)" : "none", transition: "transform .2s" }}>▾</span>
      </div>
      {open && <div style={{ marginTop: "8px", display: "flex", flexDirection: "column", gap: "5px" }}>
        {events.map((e, i) => {
          const ag = A[e.from]; const no = e.ihsan !== "pass";
          return <div key={i} style={{ display: "flex", gap: "8px", padding: "6px 8px", borderRadius: "3px", background: no ? "rgba(240,84,79,.03)" : `${$.gold}05`, border: `1px solid ${no ? "rgba(240,84,79,.1)" : $.goldLine}`, opacity: no ? .5 : 1 }}>
            <div style={{ minWidth: "36px", textAlign: "right" }}>
              <div style={{ fontSize: "7px", color: ag?.c || $.gold, fontWeight: "bold", letterSpacing: ".3px" }}>{ag?.n}</div>
              <div style={{ fontSize: "7px", color: $.ghost, marginTop: "1px" }}>{(e.conf*100)|0}%</div>
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: "9px", color: no ? $.ghost : $.bone, lineHeight: 1.4 }}>{e.label}</div>
              <div style={{ fontSize: "8px", color: $.ghost, marginTop: "1px" }}>{e.why}</div>
            </div>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: "2px" }}>
              <span style={{ fontSize: "6px", padding: "1px 5px", borderRadius: "2px", letterSpacing: "1px", fontFamily: "monospace", background: no ? "rgba(240,84,79,.08)" : "rgba(61,214,140,.08)", color: no ? $.fail : $.pass }}>{no ? "BLOCKED" : "PASS"}</span>
              <span style={{ fontSize: "7px", color: $.ghost, fontFamily: "monospace" }}>{e.score.toFixed(2)}</span>
            </div>
          </div>;
        })}
      </div>}
    </div>
  );
}

// ═══ MEMORY DRAWER ═══
function Mem({ open, close }) {
  if (!open) return null;
  const kc = { reflex: $.p2, knowledge: $.gold, episode: $.p7 };
  return (
    <div style={{ position: "absolute", right: 0, top: 0, bottom: 0, width: "230px", zIndex: 30, background: `${$.ink}f5`, backdropFilter: "blur(10px)", borderLeft: `1px solid ${$.goldLine}`, padding: "12px", display: "flex", flexDirection: "column", gap: "8px", animation: "slideR .2s ease" }}>
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <span style={{ fontSize: "7px", color: $.gold, letterSpacing: "2px" }}>LIVING MEMORY</span>
        <button onClick={close} style={{ background: "none", border: "none", color: $.ghost, cursor: "pointer", fontSize: "12px", fontFamily: "monospace", lineHeight: 1 }}>×</button>
      </div>
      {MEMORIES.map((m, i) => <div key={i} style={{ padding: "7px", borderRadius: "3px", background: `${kc[m.kind]}06`, border: `1px solid ${kc[m.kind]}12` }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "3px" }}>
          <span style={{ fontSize: "6px", color: kc[m.kind], letterSpacing: "1px", textTransform: "uppercase" }}>{m.kind}</span>
          <span style={{ fontSize: "6px", color: $.ghost }}>{m.age}</span>
        </div>
        <div style={{ fontSize: "8px", color: $.ash, lineHeight: 1.4 }}>{m.text}</div>
      </div>)}
    </div>
  );
}

// ═══ MAIN ═══
export default function BIZRA() {
  const [s, set] = useState(boot());
  const [inp, setInp] = useState("");
  const [t, setT] = useState(new Date());
  const end = useRef(null);
  const ref = useRef(null);

  useEffect(() => { const i = setInterval(() => setT(new Date()), 1000); return () => clearInterval(i); }, []);
  useEffect(() => { end.current?.scrollIntoView({ behavior: "smooth" }); }, [s.feed]);

  const msg = useCallback((from, text, kind = "agent") => {
    set(p => ({ ...p, feed: [...p.feed, { from, text, kind, ts: Date.now() }].slice(-100) }));
  }, []);
  const wait = ms => new Promise(r => setTimeout(r, ms));

  // ═══ BOOT — Geometry becomes telemetry ═══
  const ignite = useCallback(async () => {
    set(p => ({ ...p, phase: "booting" }));
    const sys = [
      "بسم الله الرحمن الرحيم",
      "Sovereign kernel initializing...",
      "Fixed-point arithmetic: deterministic.",
      "Invariants I-1 through I-7: loaded.",
      "Covenant 859649ea: verified.",
      "Ed25519 identity: generated.",
      "Minting Personal Agentic Team...",
    ];
    for (const t of sys) { msg("SYS", t, "sys"); await wait(200); }
    for (const id of ["P7","P1","P2","P3","P4","P5","P6"]) {
      set(p => ({ ...p, agents: { ...p.agents, [id]: "boot" }, focus: id }));
      msg(id, `${A[id].n} online.`, "agent");
      await wait(150);
      set(p => ({ ...p, agents: { ...p.agents, [id]: "idle" } }));
    }
    msg("SYS", "SAT-5 deployed. Zero operator control.", "sys");
    await wait(150);
    for (const k of ["node","ledger","token","supply","gate"])
      { set(p => ({ ...p, trust: { ...p.trust, [k]: true } })); await wait(80); }
    msg("SYS", "Trust: 5/5 verified.", "sys");
    await wait(200);
    set(p => ({ ...p, ghosts: GHOSTS }));
    // The morning briefing — the single most important UX moment (SAPE 0.98)
    msg("P7", "Good morning, Mumo.", "greet");
    await wait(400);
    msg("P7", "3 suggestions in your brief. 2 actionable, 1 blocked by Ihsan gate. Ready when you are.", "agent");
    set(p => ({ ...p, up: true, phase: "ready", focus: null }));
    setTimeout(() => ref.current?.focus(), 100);
  }, [msg]);

  // ═══ MISSION ═══
  const exec = useCallback(async (task) => {
    set(p => ({ ...p, phase: "mission" }));
    msg("P7", `Mission: "${task.slice(0, 60)}..."`, "agent");
    set(p => ({ ...p, focus: "P7" }));
    await wait(450);
    const kw = { P1:["plan","strategy","roadmap","schedule"], P2:["research","find","analyze","latest"], P3:["code","build","test","debug","implement"], P4:["evaluate","score","review","benchmark"], P5:["check","verify","compliance","security"], P6:["write","draft","report","publish"] };
    let best = "P2", bs = 0;
    for (const [a,ws] of Object.entries(kw)) { const sc = ws.filter(w => task.toLowerCase().includes(w)).length; if (sc > bs) { best = a; bs = sc; } }
    const ag = A[best];
    msg("P7", `Routing → ${ag.n}.`, "agent");
    set(p => ({ ...p, agents: { ...p.agents, [best]: "active", P7: "route" }, focus: best }));
    await wait(350);
    for (const m of ag.work) { msg(best, m, "work"); await wait(400 + Math.random()*250); }
    set(p => ({ ...p, agents: { ...p.agents, P4: "score" }, focus: "P4" }));
    await wait(350);
    const ih = (0.95 + Math.random() * .04).toFixed(4);
    msg("P4", `Ihsan: ${ih}.${parseFloat(ih) >= .98 ? " Exceptional." : ""}`, "score");
    await wait(250);
    set(p => ({ ...p, agents: { ...p.agents, P5: "check" }, focus: "P5" }));
    msg("P5", "All invariants hold. Cleared.", "clear");
    await wait(250);
    const leg = parseFloat(ih) >= .98 && Math.random() > .5;
    const epic = !leg && parseFloat(ih) >= .96;
    const drop = leg ? "LEGENDARY" : epic ? "EPIC" : "RARE";
    const mul = leg ? 1.5 : epic ? 1.3 : 1.15;
    const se = (parseFloat(ih)*mul).toFixed(3);
    const be = (0.01*parseFloat(ih)).toFixed(4);
    msg("SYS", `PoI: ${drop} — +${se} SEED, +${be} BLOOM`, "mint");
    set(p => ({ ...p, focus: "P6" }));
    msg("P6", "Receipt chained.", "agent");
    await wait(200);
    const nr = s.rac + 1; const compiled = nr > 0 && nr % 5 === 0;
    msg("P7", `Complete. +${se} SEED.${compiled ? " Reflex compiled — 8× faster." : ""}`, "done");
    set(p => ({
      ...p, phase: "ready", focus: null,
      seed: p.seed + parseFloat(se), bloom: p.bloom + parseFloat(be),
      rac: p.rac+1, streak: p.streak+1, lv: Math.floor((p.rac+1)/10),
      ihsan: parseFloat(ih), rfx: p.rfx+(compiled?1:0), leg: p.leg+(leg?1:0),
      petal: p.petal + parseFloat(be)*100,
      agents: Object.fromEntries(Object.keys(A).map(k=>[k,"idle"])),
    }));
  }, [msg, s.rac]);

  const go = () => {
    if (!inp.trim() || s.phase === "mission") return;
    const task = inp.trim(); setInp("");
    msg("USER", task, "user");
    setTimeout(() => exec(task), 250);
  };

  const tier = s.lv < 2 ? "SEED" : s.lv < 5 ? "NODE" : s.lv < 10 ? "BUILDER" : "VERIFIER";
  const hms = t.toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });

  // ═══ THE DARK — before ignition ═══
  if (!s.up && s.phase !== "booting") return (
    <div style={{ minHeight: "100vh", background: $.void, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", position: "relative", overflow: "hidden" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,300;6..72,500&family=IBM+Plex+Mono:wght@300;400&family=Amiri:wght@400;700&display=swap');
        @keyframes breathe { 0%,100% { transform: translateY(0); opacity: .8; } 50% { transform: translateY(-3px); opacity: 1; } }
        @keyframes gridIn { from { opacity: 0; } to { opacity: 1; } }
      `}</style>
      <div style={{ position: "absolute", inset: 0, backgroundImage: `linear-gradient(${$.goldGhost} 1px, transparent 1px), linear-gradient(90deg, ${$.goldGhost} 1px, transparent 1px)`, backgroundSize: "48px 48px", maskImage: "radial-gradient(circle, black 20%, transparent 70%)", WebkitMaskImage: "radial-gradient(circle, black 20%, transparent 70%)", animation: "gridIn 3s ease" }}/>
      <div style={{ animation: "breathe 5s ease-in-out infinite", position: "relative" }}>
        <div style={{ position: "absolute", inset: "-24px", borderRadius: "50%", background: `radial-gradient(circle, ${$.goldGhost}, transparent 70%)` }}/>
        <Seed agents={Object.fromEntries(Object.keys(A).map(k=>[k,"idle"]))} focus={null} petal={0}/>
      </div>
      <h1 style={{ fontFamily: "'Newsreader', serif", fontSize: "44px", fontWeight: 300, letterSpacing: "12px", background: `linear-gradient(180deg, ${$.goldSoft}, ${$.gold}, ${$.goldDim})`, WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", margin: "18px 0 4px" }}>BIZRA</h1>
      <div style={{ fontFamily: "'Amiri', serif", fontSize: "20px", color: `${$.gold}45`, letterSpacing: "1px" }}>البذرة</div>
      <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "8px", color: $.ghost, letterSpacing: "3px", margin: "4px 0 36px" }}>SOVEREIGN AGENT OPERATING SYSTEM</div>
      <button onClick={ignite} style={{ background: "transparent", border: `1px solid ${$.gold}28`, color: $.gold, padding: "10px 44px", borderRadius: "2px", fontSize: "9px", letterSpacing: "4px", cursor: "pointer", fontFamily: "'IBM Plex Mono', monospace", transition: "all .4s" }}
        onMouseEnter={e => { e.target.style.background = `${$.gold}08`; e.target.style.borderColor = `${$.gold}45`; e.target.style.boxShadow = `0 0 28px ${$.goldGhost}`; }}
        onMouseLeave={e => { e.target.style.background = "transparent"; e.target.style.borderColor = `${$.gold}28`; e.target.style.boxShadow = "none"; }}>
        INITIALIZE
      </button>
      <div style={{ position: "absolute", bottom: "16px", fontSize: "6px", color: `${$.gold}12`, letterSpacing: "2px", fontFamily: "monospace" }}>NODE0 · OMEGA · 768K LOC · 12,644 TESTS</div>
    </div>
  );

  // ═══ THE COCKPIT ═══
  return (
    <div style={{ minHeight: "100vh", background: $.void, color: $.bone, fontFamily: "'IBM Plex Mono', monospace", fontSize: "10px", display: "flex", flexDirection: "column", position: "relative", overflow: "hidden" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,300;6..72,500&family=IBM+Plex+Mono:wght@300;400&family=Amiri:wght@400;700&display=swap');
        @keyframes slideIn { from { opacity:0; transform:translateY(2px); } to { opacity:1; transform:translateY(0); } }
        @keyframes slideR { from { opacity:0; transform:translateX(16px); } to { opacity:1; transform:translateX(0); } }
        @keyframes pulse { 0%,100% { opacity:.5; } 50% { opacity:1; } }
        @keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0; } }
        input::placeholder { color: ${$.ghost}; }
        ::-webkit-scrollbar { width: 2px; }
        ::-webkit-scrollbar-thumb { background: ${$.goldLine}; border-radius: 1px; }
        ::-webkit-scrollbar-track { background: transparent; }
      `}</style>
      <div style={{ position: "fixed", inset: 0, pointerEvents: "none", zIndex: 0, backgroundImage: `linear-gradient(${$.goldGhost} 1px, transparent 1px), linear-gradient(90deg, ${$.goldGhost} 1px, transparent 1px)`, backgroundSize: "48px 48px", maskImage: "radial-gradient(circle at center, black 35%, transparent 100%)", WebkitMaskImage: "radial-gradient(circle at center, black 35%, transparent 100%)" }}/>

      {/* ═══ TOP BAR ═══ */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "6px 14px", borderBottom: `1px solid ${$.goldLine}`, background: `${$.void}ee`, backdropFilter: "blur(4px)", zIndex: 10 }}>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <span style={{ fontFamily: "'Newsreader', serif", fontSize: "13px", color: $.gold, letterSpacing: "3px", fontWeight: 500 }}>BIZRA</span>
          <span style={{ color: `${$.gold}18`, fontSize: "7px", letterSpacing: "1px" }}>NODE0</span>
          <div style={{ display: "flex", alignItems: "center", gap: "3px", padding: "1px 7px", borderRadius: "1px", background: s.phase === "mission" ? `${$.warn}0a` : `${$.pass}06`, border: `1px solid ${s.phase === "mission" ? $.warn+"18" : $.pass+"12"}` }}>
            <div style={{ width: "3px", height: "3px", borderRadius: "50%", background: s.phase === "mission" ? $.warn : $.pass, animation: s.phase === "mission" ? "pulse 1s ease infinite" : "none" }}/>
            <span style={{ fontSize: "6px", letterSpacing: "1.5px", color: s.phase === "mission" ? $.warn : $.pass }}>{s.phase === "mission" ? "EXECUTING" : "READY"}</span>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "14px", fontSize: "8px" }}>
          <span style={{ color: $.pass }}>{s.seed.toFixed(2)} <span style={{ color: $.ember }}>SEED</span></span>
          <span style={{ color: $.p7 }}>{s.bloom.toFixed(3)} <span style={{ color: $.ember }}>BLOOM</span></span>
          <span style={{ color: $.p1 }}>Lv.{s.lv} <span style={{ color: $.ember }}>{tier}</span></span>
          <div style={{ width: "1px", height: "10px", background: $.ember }}/>
          <span style={{ color: $.gold, fontVariantNumeric: "tabular-nums" }}>{hms}</span>
          <button onClick={() => set(p => ({ ...p, mem: !p.mem }))} style={{ background: s.mem ? `${$.p7}0a` : "transparent", border: `1px solid ${s.mem ? $.p7+"25" : $.ember}`, color: s.mem ? $.p7 : $.ghost, padding: "1px 6px", borderRadius: "1px", fontSize: "6px", cursor: "pointer", fontFamily: "monospace", letterSpacing: "1px", transition: "all .3s" }}>MEM</button>
        </div>
      </div>

      {/* ═══ AGENT BAR ═══ */}
      <div style={{ display: "flex", gap: "1px", padding: "3px 14px", borderBottom: `1px solid ${$.ember}`, zIndex: 10 }}>
        {Object.entries(A).map(([id, ag]) => {
          const on = s.agents[id] !== "idle", hot = id === s.focus;
          return <div key={id} style={{ flex: 1, padding: "2px 1px", borderRadius: "1px", textAlign: "center", background: hot ? `${ag.c}12` : "transparent", border: `1px solid ${hot ? ag.c+"30" : $.ember}`, transition: "all .3s", boxShadow: hot ? `0 0 6px ${ag.c}0a` : "none" }}>
            <div style={{ fontSize: "6px", color: on ? ag.c : $.ghost, letterSpacing: ".3px", fontWeight: hot ? "bold" : "normal" }}>{ag.n}</div>
          </div>;
        })}
      </div>

      {/* ═══ BODY ═══ */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden", position: "relative", zIndex: 5 }}>
        {/* CENTER */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
          {/* Scene 1: Brief */}
          <div style={{ padding: "6px 12px 0" }}>
            <Brief events={s.ghosts} open={s.brief} toggle={() => set(p => ({ ...p, brief: !p.brief }))}/>
          </div>
          {/* Telemetry + Stats */}
          <div style={{ display: "flex", alignItems: "center", gap: "10px", padding: "6px 12px" }}>
            <Seed agents={s.agents} focus={s.focus} petal={s.petal}/>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "2px 10px", fontSize: "7px" }}>
              {[["Ihsan", s.ihsan.toFixed(4), $.gold], ["Streak", s.streak, $.pass], ["Reflexes", s.rfx, $.p2], ["RAC", s.rac, $.p1], ["Legendary", s.leg, $.warn], ["Level", s.lv, $.p7]].map(([l,v,c]) =>
                <div key={l} style={{ display: "flex", justifyContent: "space-between", gap: "6px" }}>
                  <span style={{ color: $.ghost }}>{l}</span>
                  <span style={{ color: c, fontVariantNumeric: "tabular-nums" }}>{v}</span>
                </div>
              )}
            </div>
          </div>
          {/* Feed */}
          <div style={{ flex: 1, overflowY: "auto", padding: "2px 12px" }}>
            {s.feed.map((m, i) => {
              const u = m.kind === "user", sy = m.kind === "sys", mt = m.kind === "mint", dn = m.kind === "done", gr = m.kind === "greet";
              const c = u ? $.gold : sy ? $.ghost : mt ? $.pass : A[m.from]?.c || $.gold;
              const lab = u ? "YOU" : sy ? "SYS" : A[m.from]?.n || m.from;
              return <div key={i} style={{ marginBottom: "1px", padding: "2px 0", display: "flex", gap: "7px", alignItems: "flex-start", animation: "slideIn .12s ease", opacity: sy ? .4 : 1 }}>
                <span style={{ color: c, fontWeight: "bold", minWidth: "40px", fontSize: "7px", textAlign: "right" }}>{lab}</span>
                <span style={{
                  color: u ? $.bone : mt ? $.pass : dn ? $.gold : gr ? $.goldSoft : $.ash,
                  fontSize: gr ? "11px" : "9px", lineHeight: 1.5,
                  fontFamily: gr ? "'Newsreader', serif" : "inherit",
                  letterSpacing: gr ? ".3px" : "0",
                }}>
                  {mt ? `► ${m.text}` : dn ? `✓ ${m.text}` : m.text}
                </span>
              </div>;
            })}
            <div ref={end}/>
          </div>
          {/* Quick missions */}
          {s.phase === "ready" && <div style={{ padding: "3px 12px", display: "flex", gap: "3px", flexWrap: "wrap", borderTop: `1px solid ${$.ember}` }}>
            {["Research sovereign AI architectures", "Build invariant test framework", "Evaluate deployment pipeline", "Draft progress report"].map((m,i) =>
              <button key={i} onClick={() => { msg("USER", m, "user"); setTimeout(() => exec(m), 250); }} style={{ background: `${$.bone}02`, border: `1px solid ${$.ember}`, color: $.ghost, padding: "2px 7px", borderRadius: "1px", fontSize: "7px", cursor: "pointer", fontFamily: "monospace", transition: "all .3s", maxWidth: "180px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
                onMouseEnter={e => { e.target.style.borderColor = $.goldLine; e.target.style.color = $.gold; }}
                onMouseLeave={e => { e.target.style.borderColor = $.ember; e.target.style.color = $.ghost; }}>
                {m}
              </button>
            )}
          </div>}
          {/* ═══ THE PROMPT — بذرة › — the 4-character brand (SAPE 0.99) ═══ */}
          <div style={{ padding: "7px 12px", borderTop: `1px solid ${$.goldLine}`, display: "flex", gap: "7px", alignItems: "center", background: `${$.deep}80` }}>
            <span style={{ fontFamily: "'Amiri', serif", color: s.phase === "mission" ? $.warn : $.gold, fontSize: "12px", direction: "rtl" }}>بذرة</span>
            <span style={{ color: s.phase === "mission" ? $.warn : $.gold, fontSize: "10px" }}>›</span>
            <input ref={ref} value={inp} onChange={e => setInp(e.target.value)}
              onKeyDown={e => e.key === "Enter" && go()}
              placeholder={s.phase === "mission" ? "mission in progress..." : "speak your mission..."}
              disabled={s.phase === "mission"}
              style={{ flex: 1, background: "transparent", border: "none", color: $.bone, fontSize: "10px", fontFamily: "'IBM Plex Mono', monospace", outline: "none", letterSpacing: ".2px" }}/>
            {inp.trim() && s.phase !== "mission" && (
              <button onClick={go} style={{ background: `${$.gold}08`, border: `1px solid ${$.gold}20`, color: $.gold, padding: "2px 12px", borderRadius: "1px", fontSize: "7px", cursor: "pointer", fontFamily: "monospace", letterSpacing: "2px" }}>EXECUTE</button>
            )}
          </div>
        </div>
        {/* Scene 2: Trust Rail */}
        <Trust trust={s.trust} ihsan={s.ihsan}/>
        {/* Scene 4: Memory */}
        <Mem open={s.mem} close={() => set(p => ({ ...p, mem: false }))}/>
      </div>

      {/* ═══ BOTTOM ═══ */}
      <div style={{ padding: "3px 14px", borderTop: `1px solid ${$.ember}`, display: "flex", justifyContent: "space-between", fontSize: "6px", color: `${$.gold}15`, letterSpacing: ".6px", zIndex: 10 }}>
        <span>{tier} · IHSAN {s.ihsan.toFixed(4)} · STREAK {s.streak} · RAC {s.rac}</span>
        <span style={{ fontFamily: "'Amiri', serif", fontSize: "8px" }}>بذرة واحدة تصنع غابة</span>
        <span>PAT-7 · SAT-5 · 7 INVARIANTS · 768K LOC</span>
      </div>
    </div>
  );
}
