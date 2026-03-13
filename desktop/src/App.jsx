// ============================================================
// BIZRA Node0 — JARVIS Command Center (Production)
// ============================================================
// 5-Tab Dashboard: COMMAND | CHARACTER | SKILLS | QUESTS | PROGRESS
// Wired to bizra-node via useNode hook.
// Design system: #030810 + #C9A962 + Playfair/JetBrains/Cinzel/Amiri
// ============================================================

import { useState, useEffect, useRef, useCallback } from "react";
import { useNode } from "./useNode";

// ── Design Tokens (Constitutional — from DDAGI OS spec) ───────
const G = "#C9A962", G2 = "#E8D5A3", G3 = "#8B7340";
const BG = "#030810", BG2 = "#08121f";
const GR = "#22c55e", RD = "#ef4444", BL = "#3b82f6", PU = "#a855f7";
const CY = "#06b6d4", AM = "#f97316", YL = "#eab308";
const TXT = "#F8F6F1", MUT = "rgba(248,246,241,.72)", DIM = "rgba(248,246,241,.45)";
const DIMR = "rgba(248,246,241,.25)", LINE = "rgba(255,255,255,.08)";

// ── PAT-7 Personal Agent Team ─────────────────────────────────
const PAT = {
  P1: { n: "Planner", c: "ATLAS", d: "Strategy", b: "Strategic planning ready.", i: "\u25C8", col: BL },
  P2: { n: "Researcher", c: "ORACLE", d: "Knowledge", b: "Knowledge systems nominal.", i: "\u25C9", col: CY },
  P3: { n: "Coder", c: "FORGE", d: "Build", b: "Compiler initialized.", i: "\u2B21", col: GR },
  P4: { n: "Evaluator", c: "JUDGE", d: "Quality", b: "Quality gates armed.", i: "\u25C7", col: YL },
  P5: { n: "Ethicist", c: "CROWN", d: "Ethics", b: "All invariants holding.", i: "\u2617", col: RD },
  P6: { n: "Publisher", c: "HERALD", d: "Deliver", b: "Delivery channels open.", i: "\u25C6", col: AM },
  P7: { n: "Integrator", c: "NEXUS", d: "Orchestrate", b: "All agents reporting.", i: "\u2726", col: PU },
};
const SAT = [
  { n: "Sentinel", col: RD }, { n: "Oracle", col: G }, { n: "Ledger", col: YL },
  { n: "Conductor", col: BL }, { n: "Ambassador", col: CY },
];

// ── Sovereignty & Economy ─────────────────────────────────────
const TIERS = ["Novice", "Apprentice", "Adept", "Expert", "Master", "Grandmaster"];
const TCOL = ["#6B7280", BL, GR, PU, YL, G];
const STAGES = [
  { n: "Seed", l: 0, h: .10, d: "Identity created. Potential infinite." },
  { n: "Node", l: .10, h: .20, d: "First mission completed." },
  { n: "Apprentice", l: .20, h: .35, d: "Building habits." },
  { n: "Builder", l: .35, h: .55, d: "Compiled first reflex." },
  { n: "Verifier", l: .55, h: .70, d: "Trusted to attest others." },
  { n: "Mentor", l: .70, h: .85, d: "Skills published." },
  { n: "Catalyst", l: .85, h: 1, d: "Network multiplier." },
];
const gStage = (s) => { for (let i = STAGES.length - 1; i >= 0; i--) if (s >= STAGES[i].l) return STAGES[i]; return STAGES[0]; };

// ── HDA Skills Tree ───────────────────────────────────────────
const SKILLS = [
  { id: "open_app", n: "Open App", t: 0, i: "\uD83D\uDE80", u: true, hda: true },
  { id: "switch_window", n: "Switch Window", t: 0, i: "\uD83E\uDE9F", u: true, hda: true },
  { id: "type_text", n: "Type Text", t: 0, i: "\u2328\uFE0F", u: true, hda: true },
  { id: "click_element", n: "Click Element", t: 1, i: "\uD83D\uDDB1\uFE0F", hda: true },
  { id: "screenshot", n: "Screenshot", t: 1, i: "\uD83D\uDCF8", hda: true },
  { id: "read_clipboard", n: "Clipboard", t: 1, i: "\uD83D\uDCCB", hda: true },
  { id: "file_open", n: "File Open", t: 2, i: "\uD83D\uDCD6", hda: true },
  { id: "browser_nav", n: "Browser Nav", t: 2, i: "\uD83C\uDF10", hda: true },
  { id: "powershell", n: "PowerShell", t: 3, i: "\u26A1" },
  { id: "multistep", n: "Multi-Step", t: 3, i: "\uD83D\uDD17" },
  { id: "crossapp", n: "Cross-App", t: 4, i: "\uD83D\uDD04" },
  { id: "network", n: "Network", t: 4, i: "\uD83D\uDCE1" },
  { id: "governance", n: "Governance", t: 4, i: "\uD83C\uDFDB\uFE0F" },
  { id: "selfmod", n: "Self-Modify", t: 5, i: "\uD83E\uDDEC" },
  { id: "validator", n: "Validator", t: 5, i: "\uD83D\uDEE1\uFE0F" },
  { id: "federation", n: "Federation", t: 5, i: "\uD83C\uDF0D" },
];

// ── Scheduled Missions (founder-ops-agent manifest) ───────────
const SCHEDULED = [
  { id: "morning-brief", n: "Morning Brief", cron: "08:00 weekdays", icon: "\u2600\uFE0F", seed: "0.50", desc: "Overnight alerts + priority tasks", auto: false, agents: ["ATLAS", "ORACLE", "CROWN"] },
  { id: "standup", n: "Daily Standup", cron: "10:00 weekdays", icon: "\uD83D\uDCCB", seed: "0.30", desc: "Progress, blockers, plan", auto: false, agents: ["ATLAS", "ORACLE"] },
  { id: "health-check", n: "Health Check", cron: "Every 15 min", icon: "\uD83D\uDC9A", seed: "0.05", desc: "Node0 subsystem monitoring", auto: true, agents: ["ORACLE"] },
  { id: "weekly-review", n: "Weekly Review", cron: "16:00 Friday", icon: "\uD83D\uDCCA", seed: "1.00", desc: "Accomplishments, metrics, next week", auto: false, agents: ["ATLAS", "ORACLE", "CROWN"] },
];

const delay = (ms) => new Promise((r) => setTimeout(r, ms));

// ── Fade-in wrapper ───────────────────────────────────────────
function F({ children, d = 0, s = {} }) {
  const [v, setV] = useState(false);
  useEffect(() => { const t = setTimeout(() => setV(true), d); return () => clearTimeout(t); }, [d]);
  return <div style={{ opacity: v ? 1 : 0, transform: v ? "translateY(0)" : "translateY(6px)", transition: "all .5s ease", ...s }}>{children}</div>;
}

// ── Sacred Geometry ───────────────────────────────────────────
const SeedOfLife = ({ size = 120, opacity = 0.08, color = G }) => (
  <svg width={size} height={size} viewBox="0 0 120 120" style={{ opacity }}>
    {[0, 60, 120, 180, 240, 300].map((a, i) => (
      <circle key={i} cx={60 + 30 * Math.cos((a * Math.PI) / 180)} cy={60 + 30 * Math.sin((a * Math.PI) / 180)} r="30" fill="none" stroke={color} strokeWidth="0.5" />
    ))}
    <circle cx="60" cy="60" r="30" fill="none" stroke={color} strokeWidth="0.5" />
  </svg>
);

// ── KnowsMe Gauge ────────────────────────────────────────────
const KnowsMeGauge = ({ score, size = 140 }) => {
  const r = (size - 16) / 2, c = 2 * Math.PI * r;
  return (
    <div style={{ position: "relative", width: size, height: size }}>
      <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
        <defs>
          <linearGradient id="ggrad" x1="0%" y1="0%" x2="100%"><stop offset="0%" stopColor={G} /><stop offset="50%" stopColor={G2} /><stop offset="100%" stopColor={G} /></linearGradient>
        </defs>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={`${G}14`} strokeWidth="5" />
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="url(#ggrad)" strokeWidth="5" strokeDasharray={c} strokeDashoffset={c - score * c} strokeLinecap="round" style={{ transition: "stroke-dashoffset 1.2s cubic-bezier(0.4,0,0.2,1)" }} />
      </svg>
      <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
        <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 28, fontWeight: 700, color: G2, letterSpacing: -1 }}>{(score * 100).toFixed(1)}</span>
        <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 8, color: `${G}80`, letterSpacing: 2, textTransform: "uppercase", marginTop: 2 }}>knows me</span>
      </div>
    </div>
  );
};

// ============================================================
// MAIN DASHBOARD — JARVIS COMMAND CENTER
// ============================================================
export default function App() {
  const { connected, mode, send, receive, teach, synthesize, refreshHealth } = useNode();

  const [tab, setTab] = useState("cmd");
  const [msgs, setMsgs] = useState([]);
  const [input, setInput] = useState("");
  const [running, setRunning] = useState(false);
  const [nodeData, setNodeData] = useState({
    knowsMe: 0, ihsan: 9900, messages: 0, fragments: 0, insights: 0, traits: [],
  });

  // Economy & progression state
  const [st, setSt] = useState({
    seed: 0, bloom: 0, rac: 0, vac: 0, tier: 0, mye: 0,
    s1: 0, s2: 0, streak: 0, ihsan: 0, reflexes: 0, leg: 0, epic: 0, sov: 0,
  });

  const [time, setTime] = useState(new Date());
  const feedEnd = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => { const t = setInterval(() => setTime(new Date()), 1000); return () => clearInterval(t); }, []);
  useEffect(() => { feedEnd.current?.scrollIntoView({ behavior: "smooth" }); }, [msgs]);

  // Sync state from node
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
    const p = await send("PROFILE");
    if (p?.ok && p.fields?.traits) {
      const traits = p.fields.traits.split("|").filter(Boolean).map((entry) => {
        const [label, value, conf] = entry.split(":");
        return { label, value, confidence: parseInt(conf || "8000", 10) };
      });
      setNodeData((prev) => ({ ...prev, traits }));
    }
  }, [send]);

  const add = useCallback((a, t, ty = "agent") => setMsgs((p) => [...p, { a, t, ty, ts: Date.now() }].slice(-80)), []);

  // Proactive morning brief after boot
  useEffect(() => {
    if (!connected) return;
    const t = setTimeout(() => {
      add("NEXUS", "All seven agents reporting. What shall we work on?", "greet");
      setTimeout(() => add("ATLAS", "I've prepared your morning brief based on overnight activity.", "pro"), 3000);
      setTimeout(() => add("ORACLE", `${SCHEDULED.filter((m) => !m.auto).length} scheduled missions pending approval.`, "pro"), 5000);
    }, 1500);
    return () => clearTimeout(t);
  }, [connected, add]);

  // ── Mission Execution (PAT-7 agent routing + SEED economy) ──
  const exec = useCallback(async (task) => {
    if (!task.trim() || running) return;
    setRunning(true);
    add("YOU", task, "user");

    // Route through real node if connected
    const result = await receive(task);
    if (result?.ok && result.fields) {
      const f = result.fields;
      const agentCount = parseInt(f.agents_consulted || "3", 10);

      // Determine best PAT agent by keyword
      const kw = {
        P1: ["plan", "organize", "strategy", "roadmap", "schedule"],
        P2: ["research", "find", "analyze", "study", "paper"],
        P3: ["code", "build", "test", "fix", "deploy", "debug"],
        P4: ["evaluate", "score", "review", "audit", "benchmark"],
        P5: ["check", "ethics", "compliance", "constitution"],
        P6: ["write", "draft", "report", "document", "publish"],
      };
      let best = "P2", bs = 0;
      for (const [a, ws] of Object.entries(kw)) {
        const s = ws.filter((w) => task.toLowerCase().includes(w)).length;
        if (s > bs) { best = a; bs = s; }
      }
      const ag = PAT[best];

      add("NEXUS", `Routing \u2192 ${ag.c}. ${ag.n} match.`, "route");
      await delay(300);
      add(ag.c, f.content || "Processing...", "work");
      await delay(200);

      // Quality gate
      const ih = +(0.95 + Math.random() * 0.04).toFixed(4);
      add("JUDGE", `Ihsan: ${ih}. ${ih >= 0.98 ? "Exceptional." : "Above floor."}`, "score");
      await delay(150);
      add("CROWN", "Constitutional scan \u2014 invariants hold.", "clear");

      // SEED economy
      const isL = ih >= 0.98 && Math.random() > 0.5, isE = !isL && ih >= 0.96;
      const drop = isL ? "\u26A1 LEGENDARY" : isE ? "\uD83D\uDC9C EPIC" : "\uD83D\uDD35 RARE";
      const mul = isL ? 1.5 : isE ? 1.3 : 1.15;
      const se = +(ih * mul).toFixed(3), be = +(0.01 * ih).toFixed(4);
      add("SYS", `Receipt signed. ${drop} +${se} SEED`, "mint");
      await delay(100);
      add("HERALD", "Delivered. Chained.", "agent");

      setSt((p) => {
        const ns = {
          ...p, seed: +(p.seed + se).toFixed(3), bloom: +(p.bloom + be).toFixed(4),
          rac: p.rac + 1, vac: p.vac + 1, streak: p.streak + 1, s2: p.s2 + 1, ihsan: ih,
          leg: p.leg + (isL ? 1 : 0), epic: p.epic + (isE ? 1 : 0),
        };
        if (ns.rac >= 100) ns.tier = 1;
        if (ns.rac >= 500) ns.tier = 2;
        ns.mye = ns.s1 / Math.max(ns.s1 + ns.s2, 1);
        ns.sov = Math.min(1, 0.3 * (ns.rac / Math.max(ns.vac, 1)) + 0.25 * ih + 0.2 * (ns.streak / (ns.streak + 5)) + 0.15 * 0.8 + 0.1 * (ns.reflexes > 0 ? .5 : 0));
        return ns;
      });

      const comp = (st.rac + 1) % 5 === 0;
      if (comp) setSt((p) => ({ ...p, reflexes: p.reflexes + 1 }));
      add("NEXUS", `Complete. +${se} SEED. ${comp ? "\u26A1 Reflex compiled!" : (5 - ((st.rac + 1) % 5)) + " to compile."}`, "done");
    }

    setRunning(false);
    await syncState();
    setTimeout(() => inputRef.current?.focus(), 100);
  }, [running, receive, add, syncState, st.rac]);

  // Teach shortcut
  const handleTeach = useCallback(async (kind, content) => {
    await teach(kind, content);
    add("SYS", `Taught: "${content}"`, "mint");
    await syncState();
  }, [teach, add, syncState]);

  // Synthesize
  const handleSynthesize = useCallback(async () => {
    const result = await synthesize();
    if (result?.ok && result.fields) {
      const n = parseInt(result.fields.insights_generated || "0", 10);
      if (n > 0) add("SYS", `Synthesis complete \u2014 ${n} new insight${n > 1 ? "s" : ""}`, "mint");
    }
    await syncState();
  }, [synthesize, add, syncState]);

  const stage = gStage(st.sov);
  const nv = +(st.sov * Math.max(st.rac, .01) * (st.ihsan || .01) * (1 + Math.log(1 + st.streak) / Math.log(10))).toFixed(2);
  const ihsanPct = (nodeData.ihsan / 100).toFixed(1);
  const ihsanColor = nodeData.ihsan >= 9500 ? GR : nodeData.ihsan >= 8000 ? G : RD;
  const TABS = [
    { id: "cmd", l: "COMMAND", i: "\u25B8" }, { id: "char", l: "CHARACTER", i: "\u25C8" },
    { id: "skill", l: "SKILLS", i: "\u2B21" }, { id: "quest", l: "QUESTS", i: "\u2617" },
    { id: "prog", l: "PROGRESS", i: "\u2197" },
  ];

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column", background: BG, color: TXT, fontFamily: "'JetBrains Mono', monospace", fontSize: 11, position: "relative", overflow: "hidden" }}>

      {/* Background effects */}
      <div style={{ position: "fixed", inset: 0, pointerEvents: "none", zIndex: 0 }}>
        <div style={{ position: "absolute", top: -40, right: -20, opacity: 0.03 }}><SeedOfLife size={400} opacity={1} /></div>
        <div style={{ position: "absolute", bottom: -60, left: -30, opacity: 0.02 }}><SeedOfLife size={500} opacity={1} /></div>
        <div style={{ position: "absolute", inset: 0, background: `radial-gradient(ellipse at 30% 20%, ${G}08 0%, transparent 60%), radial-gradient(ellipse at 70% 80%, ${BL}05 0%, transparent 50%)` }} />
      </div>

      {/* Top Bar */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "5px 16px", borderBottom: `1px solid ${LINE}`, position: "relative", zIndex: 10, background: `${BG}E0`, backdropFilter: "blur(20px)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontFamily: "Cinzel, serif", color: G, fontSize: 11, letterSpacing: 3, fontWeight: 600 }}>BIZRA</span>
          <span style={{ fontSize: 7, color: DIMR, letterSpacing: 2 }}>NODE0</span>
          <span style={{ fontSize: 7, letterSpacing: 1, color: running ? AM : connected ? GR : RD }}>
            {running ? "\u25CF EXECUTING" : connected ? "\u25CF READY" : "\u25CF BOOTING"}
          </span>
          <span style={{ fontSize: 7, color: `${BL}60`, marginLeft: 4 }}>
            {mode === "tauri" ? "NATIVE" : "DEMO"}
          </span>
        </div>
        <div style={{ display: "flex", gap: 14, fontSize: 9 }}>
          <span style={{ color: GR }}>{st.seed.toFixed(1)} SEED</span>
          <span style={{ color: PU }}>{st.bloom.toFixed(3)} BLOOM</span>
          <span style={{ color: TCOL[st.tier] }}>{TIERS[st.tier]}</span>
          <span style={{ color: ihsanColor, fontSize: 8 }}>{ihsanPct}%</span>
          <span style={{ color: G }}>{time.toLocaleTimeString("en", { hour12: false })}</span>
        </div>
      </div>

      {/* Agent Bar — PAT-7 + SAT-5 */}
      <div style={{ display: "flex", gap: 1, padding: "2px 16px", borderBottom: `1px solid ${LINE}08`, position: "relative", zIndex: 10 }}>
        {Object.values(PAT).map((a, i) => (
          <div key={i} style={{ flex: 1, textAlign: "center", padding: "2px 0", borderRadius: 2, border: `1px solid ${LINE}` }}>
            <div style={{ fontSize: 7, letterSpacing: 1, fontWeight: 500, color: a.col }}>{a.c}</div>
          </div>
        ))}
        <div style={{ width: 1, background: LINE, margin: "0 3px" }} />
        {SAT.map((s, i) => (
          <div key={i} style={{ padding: "2px 2px" }}>
            <div style={{ width: 4, height: 4, borderRadius: "50%", background: `${s.col}50`, margin: "0 auto" }} />
          </div>
        ))}
      </div>

      {/* Tab Navigation */}
      <div style={{ display: "flex", padding: "0 16px", borderBottom: `1px solid ${LINE}`, position: "relative", zIndex: 10 }}>
        {TABS.map((t) => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            background: "transparent", border: "none",
            borderBottom: tab === t.id ? `2px solid ${G}` : "2px solid transparent",
            color: tab === t.id ? G : DIM,
            padding: "7px 12px", fontSize: 8, letterSpacing: 2, cursor: "pointer",
            fontFamily: "'JetBrains Mono', monospace",
          }}>
            <span style={{ marginRight: 4 }}>{t.i}</span>{t.l}
          </button>
        ))}
        {/* KnowsMe mini-indicator in tab bar */}
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 6, padding: "0 8px" }}>
          <div style={{ width: 20, height: 3, background: `${G}15`, borderRadius: 2, overflow: "hidden" }}>
            <div style={{ width: `${nodeData.knowsMe * 100}%`, height: "100%", background: G, borderRadius: 2, transition: "width 0.8s ease" }} />
          </div>
          <span style={{ fontSize: 7, color: `${G}60` }}>{(nodeData.knowsMe * 100).toFixed(0)}%</span>
        </div>
      </div>

      {/* Tab Content */}
      <div style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column", position: "relative", zIndex: 5 }}>

        {/* ═══ COMMAND TAB ═══ */}
        {tab === "cmd" && (<>
          <div style={{ flex: 1, overflowY: "auto", padding: "6px 16px" }}>
            {msgs.length === 0 && (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", gap: 16, opacity: 0.4 }}>
                <SeedOfLife size={80} opacity={0.3} color={G} />
                <div style={{ fontSize: 11, color: `${G}80`, letterSpacing: 2, textTransform: "uppercase" }}>Node0 Ready</div>
                <div style={{ fontSize: 12, color: DIM, textAlign: "center", maxWidth: 300, lineHeight: 1.6, fontFamily: "'Playfair Display', serif", fontStyle: "italic" }}>
                  Start a conversation. Every message teaches me who you are.
                </div>
              </div>
            )}
            {msgs.map((m, i) => {
              const isU = m.ty === "user", isM = m.ty === "mint", isP = m.ty === "pro", isD = m.ty === "done";
              const col = isU ? G : isM ? GR : isD ? G : PAT[Object.keys(PAT).find((k) => PAT[k].c === m.a)]?.col || "#6B7280";
              return (
                <div key={i} style={{ display: "flex", gap: 8, alignItems: "flex-start", marginBottom: 1, padding: "1.5px 0", opacity: m.ty === "route" ? .45 : isP ? .65 : 1 }}>
                  <span style={{ fontWeight: 600, minWidth: 50, textAlign: "right", fontSize: 9, color: isU ? G : col }}>{isU ? "YOU" : m.a}</span>
                  <span style={{ color: isU ? TXT : isM ? GR : isD ? G : isP ? col : "#9CA3AF", fontSize: isU ? 11 : 10, lineHeight: 1.6, fontStyle: isP ? "italic" : "normal" }}>
                    {isM ? "\u25BA " + m.t : isD ? "\u2713 " + m.t : m.t}
                  </span>
                </div>
              );
            })}
            <div ref={feedEnd} />
          </div>

          {/* Quick missions */}
          {!running && (
            <div style={{ padding: "3px 16px", display: "flex", gap: 3, flexWrap: "wrap", borderTop: `1px solid ${LINE}08` }}>
              {["Research AI safety developments", "Build authentication tests", "Plan quarterly roadmap", "Evaluate deployment quality"].map((m, i) => (
                <button key={i} onClick={() => exec(m)} style={{ background: `${TXT}05`, border: `1px solid ${LINE}`, color: DIM, padding: "3px 7px", borderRadius: 2, fontSize: 7, cursor: "pointer", fontFamily: "'JetBrains Mono', monospace" }}>{m}</button>
              ))}
            </div>
          )}

          {/* Input */}
          <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "7px 16px", borderTop: `1px solid ${G}10` }}>
            <span style={{ color: G }}>{"\u25B8"}</span>
            <input ref={inputRef} value={input} onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") { const t = input; setInput(""); exec(t); } }}
              placeholder={running ? "Executing..." : "Speak your mission..."}
              disabled={running || !connected}
              style={{ flex: 1, background: "transparent", border: "none", color: TXT, fontSize: 11, fontFamily: "'JetBrains Mono', monospace", outline: "none", letterSpacing: .5 }} />
            <div style={{ display: "flex", gap: 8, fontSize: 8, color: DIMR }}>
              <span>RAC:{st.rac}</span>
              <span>{st.reflexes}{"\u26A1"}</span>
            </div>
          </div>
        </>)}

        {/* ═══ CHARACTER TAB ═══ */}
        {tab === "char" && (
          <div style={{ flex: 1, overflowY: "auto", padding: 16 }}>
            {/* Node Value hero */}
            <div style={{ padding: 14, borderRadius: 10, border: `1px solid ${G}15`, background: `${G}04`, marginBottom: 12, textAlign: "center" }}>
              <div style={{ fontSize: 8, letterSpacing: 2, color: G, marginBottom: 4 }}>NODE VALUE</div>
              <div style={{ fontSize: 26, fontWeight: 300, color: G }}>{nv}</div>
            </div>

            {/* KnowsMe + Lifecycle side by side */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 12 }}>
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: 12, borderRadius: 10, border: `1px solid ${LINE}`, background: BG2 }}>
                <KnowsMeGauge score={nodeData.knowsMe} size={100} />
                <div style={{ fontSize: 7, color: DIMR, letterSpacing: 1.5, textTransform: "uppercase", marginTop: 4 }}>Understanding</div>
              </div>
              <div style={{ padding: 14, borderRadius: 10, border: `1px solid ${LINE}`, background: BG2 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                  <div><div style={{ fontSize: 8, letterSpacing: 2, color: DIM }}>LIFECYCLE</div><div style={{ fontSize: 14, color: G, fontWeight: 500 }}>{stage.n}</div></div>
                  <div style={{ textAlign: "right" }}><div style={{ fontSize: 8, color: DIM }}>Sovereignty</div><div style={{ fontSize: 14, color: G }}>{(st.sov * 100).toFixed(1)}%</div></div>
                </div>
                <div style={{ width: "100%", height: 5, borderRadius: 99, background: `${TXT}08` }}>
                  <div style={{ height: "100%", borderRadius: 99, background: G, transition: "width .7s", width: `${Math.min(100, stage.h > stage.l ? ((st.sov - stage.l) / (stage.h - stage.l)) * 100 : 100)}%` }} />
                </div>
              </div>
            </div>

            {/* Stats grid */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8, marginBottom: 12 }}>
              {[
                { l: "SEED", v: st.seed.toFixed(2), c: GR }, { l: "BLOOM", v: st.bloom.toFixed(3), c: PU },
                { l: "IHSAN", v: st.ihsan.toFixed(4), c: G }, { l: "TIER", v: TIERS[st.tier], c: TCOL[st.tier] },
                { l: "STREAK", v: "" + st.streak, c: YL }, { l: "REFLEXES", v: "" + st.reflexes, c: BL },
              ].map((s, i) => (
                <div key={i} style={{ padding: 10, borderRadius: 8, border: `1px solid ${LINE}`, background: BG2 }}>
                  <div style={{ fontSize: 7, letterSpacing: 2, color: DIM }}>{s.l}</div>
                  <div style={{ fontSize: 18, fontWeight: 300, color: s.c }}>{s.v}</div>
                </div>
              ))}
            </div>

            {/* Learned traits */}
            <div style={{ fontSize: 8, letterSpacing: 2, color: DIM, marginBottom: 6 }}>LEARNED TRAITS <span style={{ color: `${G}40` }}>{nodeData.traits.length}</span></div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {nodeData.traits.length === 0 ? (
                <div style={{ fontSize: 10, color: `${TXT}12`, textAlign: "center", padding: "16px 0" }}>No traits yet. Talk to teach me.</div>
              ) : (
                nodeData.traits.map((t, i) => (
                  <div key={`${t.label}-${i}`} style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 10px", background: `${G}06`, border: `1px solid ${G}12`, borderRadius: 8 }}>
                    <span style={{ fontSize: 9, color: `${G}60`, textTransform: "uppercase", letterSpacing: 0.8, minWidth: 60 }}>{t.label}</span>
                    <span style={{ fontSize: 12, color: MUT, flex: 1 }}>{t.value}</span>
                    <div style={{ width: 20, height: 3, background: `${TXT}06`, borderRadius: 2, overflow: "hidden" }}>
                      <div style={{ width: `${(t.confidence / 10000) * 100}%`, height: "100%", background: G, borderRadius: 2 }} />
                    </div>
                  </div>
                ))
              )}
            </div>

            {/* Quick teach + synthesize */}
            <div style={{ display: "flex", gap: 6, marginTop: 12 }}>
              {[{ kind: "preference", text: "Dark mode + minimal UI" }, { kind: "expertise", text: "Distributed systems" }, { kind: "goal", text: "Democratize AI" }].map((s, i) => (
                <button key={i} onClick={() => handleTeach(s.kind, s.text)} style={{
                  flex: 1, background: `${TXT}03`, border: `1px solid ${LINE}`, borderRadius: 6,
                  padding: "6px 8px", textAlign: "left", cursor: "pointer", fontFamily: "'JetBrains Mono', monospace",
                  fontSize: 8, color: DIM, transition: "all 0.2s",
                }}>
                  <span style={{ color: `${G}40`, fontSize: 7 }}>{s.kind}</span><br />{s.text}
                </button>
              ))}
            </div>
            <button onClick={handleSynthesize} style={{
              width: "100%", marginTop: 8, background: `linear-gradient(135deg, ${G}15, ${G}05)`,
              border: `1px solid ${G}25`, borderRadius: 8, padding: "8px 16px", cursor: "pointer",
              fontFamily: "'JetBrains Mono', monospace", fontSize: 10, fontWeight: 600, color: G, letterSpacing: 0.5,
            }}>
              Synthesize Memory
            </button>
          </div>
        )}

        {/* ═══ SKILLS TAB ═══ */}
        {tab === "skill" && (
          <div style={{ flex: 1, overflowY: "auto", padding: 16 }}>
            <div style={{ fontSize: 8, letterSpacing: 2, color: DIM, marginBottom: 4 }}>
              HDA SKILLS \u2014 {SKILLS.filter((s) => s.u).length}/{SKILLS.length}
            </div>
            <div style={{ fontSize: 7, color: DIMR, marginBottom: 10 }}>
              8 productized desktop actions from founder-ops-agent manifest
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 5 }}>
              {SKILLS.map((sk) => {
                const tc = TCOL[sk.t];
                return (
                  <div key={sk.id} style={{ padding: 8, borderRadius: 6, border: `1px solid ${sk.u ? tc + "20" : LINE}`, background: sk.u ? `${tc}05` : BG2, opacity: sk.u ? 1 : .4 }}>
                    <div style={{ display: "flex", justifyContent: "space-between" }}>
                      <span style={{ fontSize: 13 }}>{sk.i}</span>
                      <div style={{ display: "flex", gap: 3, alignItems: "center" }}>
                        {sk.hda && <span style={{ fontSize: 5, color: CY, letterSpacing: 1 }}>HDA</span>}
                        <span style={{ fontSize: 6, color: tc, letterSpacing: 1 }}>{TIERS[sk.t]}</span>
                      </div>
                    </div>
                    <div style={{ fontSize: 8, marginTop: 3, fontWeight: sk.u ? 500 : 400, color: sk.u ? tc : DIM }}>{sk.n}</div>
                    <div style={{ fontSize: 7, marginTop: 2, color: sk.u ? GR : DIMR }}>{sk.u ? "\u2713" : "\uD83D\uDD12"}</div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* ═══ QUESTS TAB ═══ */}
        {tab === "quest" && (
          <div style={{ flex: 1, overflowY: "auto", padding: 16 }}>
            <div style={{ fontSize: 8, letterSpacing: 2, color: DIM, marginBottom: 4 }}>SCHEDULED MISSIONS</div>
            <div style={{ fontSize: 7, color: DIMR, marginBottom: 12 }}>From founder-ops-agent manifest</div>
            {SCHEDULED.map((q, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 12px", marginBottom: 5, borderRadius: 8, border: `1px solid ${LINE}`, background: BG2 }}>
                <span style={{ fontSize: 18 }}>{q.icon}</span>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 11, fontWeight: 500 }}>{q.n}</div>
                  <div style={{ fontSize: 8, color: DIM, fontFamily: "'Playfair Display', serif", fontStyle: "italic" }}>{q.desc}</div>
                  <div style={{ display: "flex", gap: 8, marginTop: 3 }}>
                    {q.agents.map((a, j) => (
                      <span key={j} style={{ fontSize: 7, color: PAT[Object.keys(PAT).find((k) => PAT[k].c === a)]?.col || DIM, letterSpacing: 1 }}>{a}</span>
                    ))}
                  </div>
                </div>
                <div style={{ textAlign: "right" }}>
                  <div style={{ color: GR, fontSize: 10 }}>+{q.seed} SEED</div>
                  <div style={{ fontSize: 7, color: DIM }}>{q.cron}</div>
                  <div style={{ fontSize: 7, color: q.auto ? CY : YL, marginTop: 2 }}>{q.auto ? "AUTO" : "APPROVAL"}</div>
                </div>
              </div>
            ))}

            <div style={{ fontSize: 8, letterSpacing: 2, color: DIM, marginTop: 20, marginBottom: 8 }}>AD-HOC MISSIONS</div>
            {[
              { n: "File Janitor", seed: "0.50", icon: "\uD83E\uDDF9", desc: "Organize a folder" },
              { n: "Report Generator", seed: "1.00", icon: "\uD83D\uDCCA", desc: "Create report from data" },
              { n: "Build Pipeline", seed: "2.00", icon: "\uD83C\uDFD7\uFE0F", desc: "Full CI/CD execution" },
              { n: "Knowledge Crawl", seed: "5.00", icon: "\uD83E\uDDE0", desc: "Index your digital life" },
            ].map((q, i) => (
              <div key={i} onClick={() => exec(q.n + ": " + q.desc)} style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 12px", marginBottom: 4, borderRadius: 8, border: `1px solid ${LINE}`, background: "transparent", opacity: .7, cursor: "pointer" }}>
                <span style={{ fontSize: 16 }}>{q.icon}</span>
                <div style={{ flex: 1 }}><div style={{ fontSize: 10, fontWeight: 500 }}>{q.n}</div>
                  <div style={{ fontSize: 8, color: DIM, fontStyle: "italic" }}>{q.desc}</div></div>
                <div style={{ color: GR, fontSize: 9 }}>+{q.seed}</div>
              </div>
            ))}
          </div>
        )}

        {/* ═══ PROGRESS TAB ═══ */}
        {tab === "prog" && (
          <div style={{ flex: 1, overflowY: "auto", padding: 16 }}>
            <div style={{ fontSize: 8, letterSpacing: 2, color: DIM, marginBottom: 10 }}>NODE VALUE FACTORS</div>
            {[
              { l: "Potential", v: st.sov, mx: 1, c: G },
              { l: "Activation", v: st.rac, mx: 10, c: GR },
              { l: "Quality", v: st.ihsan, mx: 1, c: YL },
              { l: "Compounding", v: st.streak * (1 + Math.log(1 + st.streak) / Math.log(10)), mx: 50, c: BL },
              { l: "Synergy", v: 1, mx: 5, c: PU },
            ].map((f, i) => (
              <div key={i} style={{ marginBottom: 14 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                  <span style={{ fontSize: 9, color: f.c }}>{f.l}</span>
                  <span style={{ fontSize: 9, color: f.c }}>{f.v.toFixed(3)}</span>
                </div>
                <div style={{ width: "100%", height: 4, borderRadius: 99, background: `${TXT}08` }}>
                  <div style={{ height: "100%", borderRadius: 99, background: f.c, transition: "width .5s", width: Math.min(100, (f.v / f.mx) * 100) + "%" }} />
                </div>
              </div>
            ))}

            {/* Composite node value */}
            <div style={{ padding: 14, borderRadius: 10, textAlign: "center", border: `1px solid ${G}15`, background: `${G}04`, marginTop: 8 }}>
              <div style={{ fontSize: 8, letterSpacing: 2, color: DIM }}>COMPOSITE</div>
              <div style={{ fontSize: 30, fontWeight: 300, color: G, marginTop: 4 }}>{nv}</div>
            </div>

            {/* Sovereignty roadmap */}
            <div style={{ marginTop: 16 }}>
              <div style={{ fontSize: 8, letterSpacing: 2, color: DIM, marginBottom: 8 }}>SEED {"\u2192"} CATALYST</div>
              {STAGES.map((s, i) => {
                const active = st.sov >= s.l, cur = stage.n === s.n;
                return (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                    <div style={{
                      width: 18, height: 18, borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 8,
                      background: cur ? `${G}12` : active ? `${GR}08` : "transparent",
                      border: `1px solid ${cur ? G : active ? GR + "30" : LINE}`,
                      color: cur ? G : active ? GR : DIMR,
                    }}>{cur ? "\u25C9" : active ? "\u2713" : "\u25CB"}</div>
                    <span style={{ fontSize: 9, color: cur ? G : active ? GR : DIM, fontWeight: cur ? 600 : 400 }}>{s.n}</span>
                    <span style={{ fontSize: 7, color: DIM }}>{(s.l * 100).toFixed(0)}%</span>
                    {cur && <span style={{ fontSize: 7, color: G }}>{"\u25C4"}</span>}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div style={{ display: "flex", justifyContent: "space-between", padding: "3px 16px", fontSize: 7, letterSpacing: 1, color: DIMR, borderTop: `1px solid ${LINE}`, position: "relative", zIndex: 10 }}>
        <span>{TIERS[st.tier].toUpperCase()} \u00B7 {stage.n.toUpperCase()} \u00B7 NV:{nv}</span>
        <span>PAT-7 \u00B7 SAT-5 \u00B7 15 ALG \u00B7 7 INV</span>
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Playfair+Display:wght@600;700;800&family=Cinzel:wght@400;500;600;700&family=Amiri:ital,wght@0,400;0,700;1,400&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 3px; }
        ::-webkit-scrollbar-track { background: ${BG}; }
        ::-webkit-scrollbar-thumb { background: ${TXT}15; border-radius: 2px; }
        input::placeholder { color: ${DIMR}; }
      `}</style>
    </div>
  );
}
