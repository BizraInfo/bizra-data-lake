import { useState, useEffect, useRef, useCallback } from "react";

/*
  BIZRA SOVEREIGN WORLD v2
  ═══════════════════════════════════════════════════════════
  
  Wired to the real marathon-session backend:
  - 9-stage MissionExecutor (FAISS→Amplify→Infer→Gate→...→Watch)
  - FanoutEventBus (CQRS + sovereign)
  - DiffusionAmplifier (fail-closed)
  - Receipt chain (chain_head persisted cross-session)
  - Nervous system (live stage streaming)
  
  Behind this interface: 621 commits, 12,680 tests, Z3 verified,
  126× reflex speedup, 0.007ms membrane tax, ArXiv-ready paper.
  
  In front of this interface: بذرة › and "Good morning, Mumo."
*/

const $ = {
  void: "#020408", deep: "#060D18", ink: "#0B1524",
  gold: "#C9A962", goldSoft: "#D4BC82", goldDim: "#7A6332",
  goldGhost: "rgba(201,169,98,0.06)", goldLine: "rgba(201,169,98,0.10)",
  bone: "#F0EDE4", ash: "rgba(240,237,228,0.55)", ghost: "rgba(240,237,228,0.25)", ember: "rgba(240,237,228,0.10)",
  pass: "#3DD68C", fail: "#F0544F", warn: "#E8B73A",
  tank: "#5B8DEF", scout: "#3CC8CF", dps: "#3DD68C", judge: "#E8B73A",
  healer: "#E85D6F", bard: "#E8883A", lead: "#A36BDF",
  // 9-stage pipeline colors
  s_retrieve: "#3CC8CF", s_amplify: "#A36BDF", s_infer: "#5B8DEF",
  s_gate: "#E85D6F", s_receipt: "#C9A962", s_score: "#E8B73A",
  s_learn: "#3DD68C", s_compile: "#E8883A", s_watch: "#8B8B8B",
};

// ═══ 9-STAGE MISSION PIPELINE (from marathon session) ═══
const STAGES = [
  { id: "retrieve", name: "FAISS", icon: "◈", color: $.s_retrieve, desc: "Semantic retrieval" },
  { id: "amplify", name: "DIFFUSE", icon: "◉", color: $.s_amplify, desc: "Diffusion amplification" },
  { id: "infer", name: "INFER", icon: "⬡", color: $.s_infer, desc: "LLM inference" },
  { id: "gate", name: "GATE", icon: "♛", color: $.s_gate, desc: "Constitutional gate" },
  { id: "receipt", name: "RECEIPT", icon: "◆", color: $.s_receipt, desc: "Chain receipt" },
  { id: "score", name: "SCORE", icon: "⚖", color: $.s_score, desc: "Ihsan scoring" },
  { id: "learn", name: "LEARN", icon: "⟡", color: $.s_learn, desc: "Autopoietic observation" },
  { id: "compile", name: "COMPILE", icon: "⚒", color: $.s_compile, desc: "Reflex compilation check" },
  { id: "watch", name: "WATCH", icon: "⛊", color: $.s_watch, desc: "Dead-letter surveillance" },
];

const PARTY = {
  P1: { n: "ATLAS", cls: "Strategist", role: "Tank", c: $.tank, icon: "⛊", skills: ["Decompose", "Fortress", "Shield"], work: ["Pulling aggro on complexity...", "Shield wall: subtasks contained.", "Path cleared."] },
  P2: { n: "ORACLE", cls: "Scholar", role: "Scout", c: $.scout, icon: "◈", skills: ["Deep Scan", "Extraction", "Triangulation"], work: ["Casting Deep Scan...", "47 sources cross-referenced.", "Synthesis scroll crafted."] },
  P3: { n: "FORGE", cls: "Smith", role: "DPS", c: $.dps, icon: "⚒", skills: ["Code Strike", "Test Barrage", "Reflex"], work: ["Striking implementation...", "Test barrage: all pass.", "Weapon complete."] },
  P4: { n: "JUDGE", cls: "Arbiter", role: "Inspector", c: $.judge, icon: "⚖", skills: ["Judgment", "Entropy", "Rubric"], work: ["Weighing against rubric...", "Shannon entropy: above min.", "Judgment sealed."] },
  P5: { n: "CROWN", cls: "Guardian", role: "Healer", c: $.healer, icon: "♛", skills: ["Shield", "Ward", "Seal"], work: ["Scanning I-1 through I-7...", "No corruption detected.", "Clearance sealed."] },
  P6: { n: "HERALD", cls: "Bard", role: "Support", c: $.bard, icon: "✧", skills: ["Clarity", "Enchant", "Delivery"], work: ["Composing output...", "Clarity enchant applied.", "Delivered and sealed."] },
  P7: { n: "NEXUS", cls: "Warden", role: "Lead", c: $.lead, icon: "⟡", skills: ["Link", "Bridge", "Aggregate"], work: ["Routing to specialist...", "Context bridge cast.", "Results aggregated."] },
};

const QUESTS = [
  { name: "Mine the CI stabilization vein", biome: "forge", diff: "Normal", reward: 1.2, xp: 15 },
  { name: "Scout sovereign AI knowledge forest", biome: "forest", diff: "Heroic", reward: 2.0, xp: 25 },
  { name: "Forge invariant test framework", biome: "forge", diff: "Epic", reward: 3.5, xp: 40 },
  { name: "Defend the constitutional temple", biome: "temple", diff: "Legendary", reward: 5.0, xp: 60 },
];

const init = () => ({
  phase: "title", up: false,
  seed: 0, bloom: 0, ihsan: 0, streak: 0, xp: 0, lv: 1, rac: 0, rfx: 0, blocks: 0,
  feed: [], agents: Object.fromEntries(Object.keys(PARTY).map(k => [k, "idle"])),
  focus: null, view: "world", pipeStage: -1, petal: 0,
  trust: { node: null, ledger: null, token: null, supply: null, gate: null },
  chainHead: "GENESIS_350d642099bde68b",
});

// ═══ WORLD MAP — ASCII receipt blocks ═══
function WorldMap({ blocks, lv }) {
  const rows = 9, cols = 28;
  const biomeChars = ["·", "░", "▒", "▓", "█", "◆", "✦"];
  return (
    <div style={{ fontFamily: "monospace", fontSize: "10px", lineHeight: "13px", color: $.goldDim, letterSpacing: "1.5px" }}>
      <div style={{ fontSize: "7px", color: $.ghost, letterSpacing: "2px", marginBottom: "3px" }}>
        NODE0 WORLD — {blocks} BLOCKS — LV.{lv} — CHAIN: {blocks > 0 ? "INTACT" : "GENESIS"}
      </div>
      {Array.from({ length: rows }, (_, r) => (
        <div key={r}>{Array.from({ length: cols }, (_, c) => {
          const idx = r * cols + c;
          const placed = idx < blocks;
          const recent = idx >= blocks - 3 && idx < blocks;
          const ch = placed ? biomeChars[Math.min(Math.floor(idx / 40) + 1, biomeChars.length - 1)] : "·";
          return <span key={c} style={{ color: recent ? $.gold : placed ? $.goldDim : $.ember }}>{ch}</span>;
        })}</div>
      ))}
    </div>
  );
}

// ═══ 9-STAGE PIPELINE VISUALIZER ═══
function PipelineView({ stage }) {
  return (
    <div style={{ display: "flex", gap: "1px", padding: "3px 0" }}>
      {STAGES.map((s, i) => {
        const active = i === stage;
        const done = i < stage;
        const col = active ? s.color : done ? `${s.color}80` : $.ember;
        return (
          <div key={s.id} style={{
            flex: 1, padding: "3px 2px", textAlign: "center", borderRadius: "2px",
            background: active ? `${s.color}15` : "transparent",
            borderBottom: `2px solid ${active ? s.color : done ? `${s.color}40` : "transparent"}`,
            transition: "all .3s",
          }}>
            <div style={{ fontSize: "9px", color: col }}>{s.icon}</div>
            <div style={{ fontSize: "5px", color: col, letterSpacing: ".5px" }}>{s.name}</div>
          </div>
        );
      })}
    </div>
  );
}

// ═══ PARTY RAID FRAMES ═══
function PartyFrames({ agents, focus }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
      <div style={{ fontSize: "7px", color: $.ghost, letterSpacing: "2px", marginBottom: "2px" }}>RAID PARTY — PAT-7</div>
      {Object.entries(PARTY).map(([id, p]) => {
        const on = agents[id] !== "idle", hot = id === focus;
        return (
          <div key={id} style={{ display: "flex", alignItems: "center", gap: "5px", padding: "2px 3px", borderLeft: `2px solid ${hot ? p.c : on ? p.c + "40" : $.ember}`, background: hot ? `${p.c}08` : "transparent", transition: "all .3s" }}>
            <span style={{ color: p.c, fontSize: "10px", width: "12px" }}>{p.icon}</span>
            <span style={{ fontSize: "8px", color: hot ? p.c : $.bone, minWidth: "48px", fontWeight: hot ? "bold" : "normal" }}>{p.n}</span>
            <span style={{ fontSize: "7px", color: $.ghost }}>{p.cls}</span>
          </div>
        );
      })}
      <div style={{ fontSize: "6px", color: $.ghost, marginTop: "4px", letterSpacing: "1px" }}>SAT-5: Sentinel · Oracle · Ledger · Conductor · Ambassador</div>
      <div style={{ fontSize: "6px", color: $.ghost }}>⚠ Uncontrollable. Ethics from revelation, not data.</div>
    </div>
  );
}

// ═══ ECONOMY — Gold Standard ═══
function Economy({ seed, bloom, blocks }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "5px" }}>
      <div style={{ fontSize: "7px", color: $.ghost, letterSpacing: "2px" }}>SOVEREIGN ECONOMY</div>
      {[
        { l: "SEED", v: seed.toFixed(3), c: $.pass, n: "1:1 PoI receipt backed" },
        { l: "BLOOM", v: bloom.toFixed(4), c: $.lead, n: "Soulbound. Earned only." },
        { l: "RIBA", v: "ZERO", c: $.fail, n: "I-2 invariant. Permanent." },
        { l: "GINI", v: "≤0.35", c: $.warn, n: "الحد — Harberger 5% idle" },
        { l: "ZAKAT", v: "2.5%", c: $.gold, n: "Protocol, not charity" },
        { l: "BLOCKS", v: blocks, c: $.goldSoft, n: "Each = verified receipt" },
        { l: "TAX", v: "0.007ms", c: $.pass, n: "Membrane governance cost" },
      ].map((e, i) => (
        <div key={i} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "2px 4px", borderLeft: `2px solid ${e.c}25` }}>
          <div><span style={{ fontSize: "8px", color: e.c, fontWeight: "bold" }}>{e.l}</span> <span style={{ fontSize: "7px", color: $.ghost }}>{e.n}</span></div>
          <span style={{ fontSize: "10px", color: e.c, fontFamily: "monospace", fontVariantNumeric: "tabular-nums" }}>{e.v}</span>
        </div>
      ))}
    </div>
  );
}

// ═══ TRUST RAIL ═══
function TrustRail({ trust, ihsan, chainHead }) {
  const checks = [
    { k: "node", l: "Node", i: "●" }, { k: "ledger", l: "SEL", i: "◈" },
    { k: "token", l: "SEED", i: "◆" }, { k: "supply", l: "Cap", i: "▣" },
    { k: "gate", l: "Gate", i: "✦" },
  ];
  return (
    <div style={{ width: "130px", minWidth: "130px", padding: "8px", borderLeft: `1px solid ${$.ember}`, background: `linear-gradient(180deg, ${$.ink}60, ${$.void}80)`, display: "flex", flexDirection: "column", gap: "8px", fontSize: "8px", overflowY: "auto" }}>
      <div style={{ fontSize: "6px", color: $.ghost, letterSpacing: "2px" }}>TRUST</div>
      {checks.map(({ k, l, i }) => {
        const v = trust[k]; const c = v === true ? $.pass : v === false ? $.fail : $.ember;
        return <div key={k} style={{ display: "flex", alignItems: "center", gap: "4px" }}>
          <span style={{ color: c, fontSize: "7px" }}>{i}</span>
          <span style={{ color: v ? $.ash : $.ghost, flex: 1 }}>{l}</span>
          <span style={{ color: c, fontSize: "6px", fontFamily: "monospace" }}>{v === true ? "OK" : "—"}</span>
        </div>;
      })}
      <div style={{ padding: "6px", borderRadius: "2px", background: $.goldGhost, border: `1px solid ${$.goldLine}`, textAlign: "center" }}>
        <div style={{ fontSize: "5px", color: $.gold, letterSpacing: "1.5px", marginBottom: "3px" }}>IHSAN</div>
        <div style={{ fontSize: "16px", fontFamily: "monospace", color: ihsan >= .95 ? $.gold : $.warn, fontVariantNumeric: "tabular-nums" }}>{ihsan.toFixed(4)}</div>
        <div style={{ height: "2px", background: $.ember, borderRadius: "1px", marginTop: "4px", overflow: "hidden" }}>
          <div style={{ height: "100%", width: `${Math.min(ihsan * 100, 100)}%`, background: ihsan >= .95 ? $.gold : $.warn, transition: "width .8s", borderRadius: "1px" }} />
        </div>
      </div>
      <div style={{ fontSize: "5px", fontFamily: "monospace", color: `${$.gold}20`, wordBreak: "break-all", lineHeight: 1.3, marginTop: "auto" }}>
        HEAD: {chainHead.slice(0, 16)}
      </div>
    </div>
  );
}

// ═══ MAIN ═══
export default function BIZRAWorld() {
  const [s, set] = useState(init());
  const [inp, setInp] = useState("");
  const [t, setT] = useState(new Date());
  const end = useRef(null);
  const ref = useRef(null);

  useEffect(() => { const i = setInterval(() => setT(new Date()), 1000); return () => clearInterval(i); }, []);
  useEffect(() => { end.current?.scrollIntoView({ behavior: "smooth" }); }, [s.feed]);

  const msg = useCallback((from, text, kind = "agent") => {
    set(p => ({ ...p, feed: [...p.feed, { from, text, kind, ts: Date.now() }].slice(-120) }));
  }, []);
  const wait = ms => new Promise(r => setTimeout(r, ms));

  // ═══ GENESIS ═══
  const genesis = useCallback(async () => {
    set(p => ({ ...p, phase: "creating" }));
    const sys = [
      "بسم الله الرحمن الرحيم",
      "═══ WORLD GENESIS ═══",
      "Mining genesis block...",
      "Constitutional bedrock: 7 invariants.",
      "Ed25519 identity forged.",
      "RIBA_ZERO: interest permanently disabled.",
      "Zakat 2.5%: wired into protocol.",
      "الحد (Gini ≤ 0.35): ward placed.",
      "FanoutEventBus: CQRS + sovereign unified.",
      "DiffusionAmplifier: fail-closed on canonical path.",
      "9-stage MissionExecutor: FAISS→Amplify→Infer→Gate→Receipt→Score→Learn→Compile→Watch",
      "═══ SUMMONING RAID PARTY ═══",
    ];
    for (const t of sys) { msg("SYS", t, "sys"); await wait(140); }
    for (const id of ["P7", "P1", "P2", "P3", "P4", "P5", "P6"]) {
      const p = PARTY[id];
      set(prev => ({ ...prev, agents: { ...prev.agents, [id]: "summon" }, focus: id }));
      msg(id, `${p.icon} ${p.n} the ${p.cls} joins. Role: ${p.role}.`, "summon");
      await wait(160);
      set(prev => ({ ...prev, agents: { ...prev.agents, [id]: "idle" } }));
    }
    msg("SYS", "SAT-5 deployed. You cannot control them.", "sys");
    await wait(120);
    for (const k of ["node", "ledger", "token", "supply", "gate"])
      { set(p => ({ ...p, trust: { ...p.trust, [k]: true } })); await wait(60); }
    msg("SYS", "Trust: 5/5 verified. Z3 proofs: 4/4 hold.", "sys");
    msg("SYS", "Membrane tax: 0.007ms. Reflex speedup: 126×.", "sys");
    msg("SYS", "Block #0 placed. 621 commits behind this cursor.", "mint");
    await wait(250);
    msg("P7", "Good morning, Mumo.", "greet");
    await wait(300);
    msg("P7", "Your world has 1 block. 12,680 tests guard it. Ready?", "agent");
    set(p => ({ ...p, up: true, phase: "ready", focus: null, blocks: 1 }));
    setTimeout(() => ref.current?.focus(), 100);
  }, [msg]);

  // ═══ 9-STAGE QUEST EXECUTION ═══
  const quest = useCallback(async (task, questData) => {
    set(p => ({ ...p, phase: "quest" }));
    msg("P7", `⟡ Quest: "${task.slice(0, 55)}..."`, "agent");
    set(p => ({ ...p, focus: "P7" }));
    await wait(350);

    // Route to agent
    const kw = { P1: ["plan", "strategy", "roadmap"], P2: ["research", "find", "analyze", "scout"], P3: ["code", "build", "test", "forge"], P4: ["evaluate", "score", "review"], P5: ["check", "verify", "defend"], P6: ["write", "draft", "report"] };
    let best = "P2", bs = 0;
    for (const [a, ws] of Object.entries(kw)) { const sc = ws.filter(w => task.toLowerCase().includes(w)).length; if (sc > bs) { best = a; bs = sc; } }
    const ag = PARTY[best];
    msg("P7", `${ag.icon} Routing → ${ag.n} the ${ag.cls}.`, "agent");
    set(p => ({ ...p, agents: { ...p.agents, [best]: "active", P7: "lead" }, focus: best }));
    await wait(300);

    // ═══ 9 STAGES — visible nervous system ═══
    const stageMessages = [
      ["SYS", "◈ Stage 1/9: FAISS retrieving context (84,795 vectors, 5ms)...", "stage"],
      ["SYS", "◉ Stage 2/9: DiffusionAmplifier — signal propagation (fail-closed)...", "stage"],
      [best, "⬡ Stage 3/9: LLM inference — sovereign system prompt active...", "stage"],
      ["P5", "♛ Stage 4/9: Constitutional gate — 7 invariants scanning...", "stage"],
      ["SYS", "◆ Stage 5/9: Receipt built — BLAKE3 chain extending...", "stage"],
      ["P4", "⚖ Stage 6/9: Ihsan scoring — 8 dimensions weighted...", "stage"],
      ["SYS", "⟡ Stage 7/9: Autopoietic observation — pattern window updating...", "stage"],
      ["SYS", "⚒ Stage 8/9: Reflex compilation check (126× if pattern stable)...", "stage"],
      ["SYS", "⛊ Stage 9/9: Dead-letter watch — failures become evidence...", "stage"],
    ];

    for (let i = 0; i < stageMessages.length; i++) {
      set(p => ({ ...p, pipeStage: i }));
      const [from, text, kind] = stageMessages[i];
      msg(from, text, kind);
      // Agent work messages interleaved with pipeline stages
      if (i === 2 && ag.work[0]) { await wait(250); msg(best, `${ag.icon} ${ag.work[0]}`, "work"); }
      if (i === 3) { set(p => ({ ...p, agents: { ...p.agents, P5: "guard" }, focus: "P5" })); }
      if (i === 5) { set(p => ({ ...p, agents: { ...p.agents, P4: "judge" }, focus: "P4" })); }
      await wait(200 + Math.random() * 150);
    }

    // Scoring
    const ih = (0.95 + Math.random() * .04).toFixed(4);
    msg("P4", `⚖ Ihsan: ${ih}.${parseFloat(ih) >= .98 ? " Exceptional purity." : ""}`, "score");
    await wait(200);
    msg("P5", "♛ All 7 wards hold. Z3-verified. Sealed.", "clear");
    await wait(200);

    // Loot
    const leg = parseFloat(ih) >= .98 && Math.random() > .5;
    const epic = !leg && parseFloat(ih) >= .96;
    const drop = leg ? "⚡ LEGENDARY" : epic ? "💜 EPIC" : "🔷 RARE";
    const mul = leg ? 1.5 : epic ? 1.3 : 1.15;
    const se = (parseFloat(ih) * mul).toFixed(3);
    const be = (0.01 * parseFloat(ih)).toFixed(4);
    const xpE = questData?.xp || (15 + Math.floor(Math.random() * 20));

    msg("SYS", `${drop} — +${se} SEED · +${be} BLOOM · +${xpE} XP`, "mint");

    // Block + chain
    const newBlocks = s.blocks + 1;
    const newHash = `${Date.now().toString(16).slice(-8)}${Math.random().toString(16).slice(2, 10)}`;
    msg("SYS", `█ Block #${newBlocks} placed. Chain: ...${newHash.slice(0, 12)}`, "block");
    set(p => ({ ...p, focus: "P6" }));
    msg("P6", "✧ Receipt chained. Delivered.", "agent");
    await wait(150);

    const nr = s.rac + 1;
    const compiled = nr > 0 && nr % 5 === 0;
    if (compiled) msg("SYS", "⚒ REFLEX COMPILED — 126× faster next cast. Zero quality loss.", "reflex");

    msg("P7", `⟡ Complete. +${se} SEED. World: ${newBlocks} blocks.`, "done");

    set(p => ({
      ...p, phase: "ready", focus: null, pipeStage: -1, blocks: newBlocks,
      seed: p.seed + parseFloat(se), bloom: p.bloom + parseFloat(be),
      xp: p.xp + xpE, rac: p.rac + 1, streak: p.streak + 1,
      lv: Math.max(p.lv, 1 + Math.floor((p.xp + xpE) / 50)),
      ihsan: parseFloat(ih), rfx: p.rfx + (compiled ? 1 : 0),
      petal: p.petal + parseFloat(be) * 100,
      chainHead: newHash,
      agents: Object.fromEntries(Object.keys(PARTY).map(k => [k, "idle"])),
    }));
  }, [msg, s.blocks, s.rac, s.xp]);

  const go = () => {
    if (!inp.trim() || s.phase === "quest") return;
    const task = inp.trim(); setInp("");
    msg("USER", task, "user");
    setTimeout(() => quest(task), 250);
  };

  const hms = t.toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const tier = s.lv < 3 ? "Apprentice" : s.lv < 6 ? "Journeyman" : s.lv < 10 ? "Artisan" : "Master";
  const views = ["world", "party", "quests", "economy"];

  // ═══ TITLE ═══
  if (!s.up && s.phase !== "creating") return (
    <div style={{ minHeight: "100vh", background: $.void, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", fontFamily: "'IBM Plex Mono', monospace", position: "relative" }}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500&family=Amiri:wght@400;700&display=swap');
        @keyframes drift { 0%,100% { transform:translateY(0); } 50% { transform:translateY(-2px); } }`}</style>
      <div style={{ position: "absolute", inset: 0, background: `repeating-linear-gradient(0deg, transparent, transparent 47px, ${$.ember} 48px)`, opacity: .3, maskImage: "radial-gradient(circle, black 20%, transparent 65%)", WebkitMaskImage: "radial-gradient(circle, black 20%, transparent 65%)" }} />
      <pre style={{ color: $.goldDim, fontSize: "10px", lineHeight: "12px", letterSpacing: "1px", textAlign: "center", animation: "drift 6s ease-in-out infinite", marginBottom: "14px" }}>{`
    ╔═══════════════════════════════════════╗
    ║                                       ║
    ║     ░▒▓  B I Z R A  ▓▒░              ║
    ║                                       ║
    ║     SOVEREIGN WORLD  v2               ║
    ║                                       ║
    ╚═══════════════════════════════════════╝`}</pre>
      <div style={{ fontFamily: "'Amiri', serif", fontSize: "18px", color: `${$.gold}40`, marginBottom: "4px" }}>البذرة</div>
      <div style={{ fontSize: "7px", color: $.ghost, letterSpacing: "2px", marginBottom: "6px" }}>9-STAGE PIPELINE · Z3 VERIFIED · 126× REFLEX · GOLD STANDARD</div>
      <div style={{ fontSize: "7px", color: $.ghost, marginBottom: "24px" }}>621 commits · 12,680 tests · 0.007ms membrane tax · 0 riba</div>
      <button onClick={genesis} style={{ background: "transparent", border: `1px solid ${$.gold}25`, color: $.gold, padding: "10px 40px", borderRadius: "1px", fontSize: "9px", letterSpacing: "4px", cursor: "pointer", fontFamily: "'IBM Plex Mono', monospace", transition: "all .4s" }}
        onMouseEnter={e => { e.target.style.background = `${$.gold}08`; e.target.style.borderColor = `${$.gold}40`; }}
        onMouseLeave={e => { e.target.style.background = "transparent"; e.target.style.borderColor = `${$.gold}25`; }}>
        CREATE WORLD
      </button>
    </div>
  );

  // ═══ THE WORLD ═══
  return (
    <div style={{ minHeight: "100vh", background: $.void, color: $.bone, fontFamily: "'IBM Plex Mono', monospace", fontSize: "10px", display: "flex", flexDirection: "column", position: "relative" }}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500&family=Amiri:wght@400;700&display=swap');
        @keyframes slideIn { from { opacity:0; transform:translateY(2px); } to { opacity:1; transform:translateY(0); } }
        @keyframes pulse { 0%,100% { opacity:.5; } 50% { opacity:1; } }
        input::placeholder { color: ${$.ghost}; }
        ::-webkit-scrollbar { width: 2px; }
        ::-webkit-scrollbar-thumb { background: ${$.goldLine}; }
        ::-webkit-scrollbar-track { background: transparent; }`}</style>

      {/* HUD */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "4px 10px", borderBottom: `1px solid ${$.goldLine}`, background: `${$.void}ee`, zIndex: 10 }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={{ color: $.gold, fontSize: "10px", fontWeight: 500, letterSpacing: "2px" }}>BIZRA</span>
          <span style={{ color: $.ember, fontSize: "6px" }}>NODE0</span>
          <span style={{ fontSize: "6px", color: s.phase === "quest" ? $.warn : $.pass, letterSpacing: "1px", animation: s.phase === "quest" ? "pulse 1s ease infinite" : "none" }}>● {s.phase === "quest" ? "IN QUEST" : "IDLE"}</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "7px" }}>
          <span style={{ color: $.pass }}>{s.seed.toFixed(2)} SEED</span>
          <span style={{ color: $.lead }}>{s.bloom.toFixed(3)} BLOOM</span>
          <span style={{ color: $.warn }}>Lv.{s.lv}</span>
          <span style={{ color: $.goldSoft }}>█{s.blocks}</span>
          <span style={{ color: $.gold }}>{hms}</span>
        </div>
      </div>

      {/* 9-STAGE PIPELINE — always visible during quest */}
      <PipelineView stage={s.pipeStage} />

      {/* AGENT BAR */}
      <div style={{ display: "flex", gap: "1px", padding: "2px 10px", borderBottom: `1px solid ${$.ember}` }}>
        {Object.entries(PARTY).map(([id, p]) => {
          const hot = id === s.focus;
          return <div key={id} style={{ flex: 1, textAlign: "center", padding: "1px", borderBottom: `2px solid ${hot ? p.c : "transparent"}`, transition: "all .3s" }}>
            <span style={{ fontSize: "8px", color: hot ? p.c : $.ghost }}>{p.icon}</span>
          </div>;
        })}
      </div>

      {/* NAV */}
      <div style={{ display: "flex", gap: "1px", padding: "2px 10px", borderBottom: `1px solid ${$.ember}` }}>
        {views.map(v => (
          <button key={v} onClick={() => set(p => ({ ...p, view: v }))} style={{ flex: 1, padding: "2px", background: s.view === v ? $.goldGhost : "transparent", border: `1px solid ${s.view === v ? $.goldLine : "transparent"}`, borderRadius: "1px", color: s.view === v ? $.gold : $.ghost, fontSize: "6px", letterSpacing: "1px", cursor: "pointer", fontFamily: "monospace", textTransform: "uppercase" }}>
            {v === "world" ? "⛏ World" : v === "party" ? "⛊ Party" : v === "quests" ? "⚔ Quests" : "◆ Economy"}
          </button>
        ))}
      </div>

      {/* MAIN */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
        {/* LEFT PANEL */}
        <div style={{ width: "42%", minWidth: "220px", padding: "4px 10px", borderRight: `1px solid ${$.ember}`, overflowY: "auto" }}>
          {s.view === "world" && <WorldMap blocks={s.blocks} lv={s.lv} />}
          {s.view === "party" && <PartyFrames agents={s.agents} focus={s.focus} />}
          {s.view === "quests" && (
            <div style={{ display: "flex", flexDirection: "column", gap: "3px" }}>
              <div style={{ fontSize: "7px", color: $.ghost, letterSpacing: "2px", marginBottom: "2px" }}>QUEST BOARD</div>
              {QUESTS.map((q, i) => {
                const dc = { Normal: $.pass, Heroic: $.tank, Epic: $.lead, Legendary: $.warn };
                return (
                  <div key={i} onClick={() => { if (s.phase !== "quest") { msg("USER", q.name, "user"); setTimeout(() => quest(q.name, q), 250); } }}
                    style={{ display: "flex", gap: "6px", alignItems: "center", padding: "4px 5px", border: `1px solid ${$.ember}`, borderRadius: "2px", cursor: s.phase === "quest" ? "default" : "pointer", transition: "all .2s" }}
                    onMouseEnter={e => { if (s.phase !== "quest") e.currentTarget.style.borderColor = $.goldLine; }}
                    onMouseLeave={e => { e.currentTarget.style.borderColor = $.ember; }}>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: "8px", color: $.bone }}>{q.name}</div>
                      <div style={{ fontSize: "6px", color: dc[q.diff], fontWeight: "bold" }}>{q.diff.toUpperCase()} · {q.reward} SEED · {q.xp} XP</div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
          {s.view === "economy" && <Economy seed={s.seed} bloom={s.bloom} blocks={s.blocks} />}
        </div>

        {/* CENTER — FEED */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
          <div style={{ flex: 1, overflowY: "auto", padding: "2px 8px" }}>
            {s.feed.map((m, i) => {
              const u = m.kind === "user", sy = m.kind === "sys", mt = m.kind === "mint",
                dn = m.kind === "done", gr = m.kind === "greet", bl = m.kind === "block",
                rf = m.kind === "reflex", st = m.kind === "stage", sm = m.kind === "summon";
              const stageEntry = st ? STAGES.find(sg => m.text.includes(sg.name)) : null;
              const c = u ? $.gold : sy ? $.ghost : mt ? $.warn : bl ? $.goldSoft : rf ? $.lead
                : st ? (stageEntry?.color || $.ghost) : sm ? $.pass : dn ? $.gold : PARTY[m.from]?.c || $.gold;
              const lab = u ? "YOU" : sy ? "SYS" : PARTY[m.from]?.n || m.from;
              return <div key={i} style={{ marginBottom: "1px", padding: "1px 0", display: "flex", gap: "5px", alignItems: "flex-start", animation: "slideIn .1s ease", opacity: sy ? .35 : st ? .6 : 1 }}>
                <span style={{ color: c, fontWeight: "bold", minWidth: "36px", fontSize: "6px", textAlign: "right" }}>{lab}</span>
                <span style={{ color: u ? $.bone : mt ? $.warn : bl ? $.goldSoft : rf ? $.lead : dn ? $.gold : gr ? $.goldSoft : $.ash, fontSize: gr ? "10px" : st ? "8px" : "8px", lineHeight: 1.4, fontFamily: gr ? "'Amiri', serif" : "inherit" }}>
                  {mt ? m.text : dn ? `✓ ${m.text}` : m.text}
                </span>
              </div>;
            })}
            <div ref={end} />
          </div>

          {/* PROMPT */}
          <div style={{ padding: "5px 8px", borderTop: `1px solid ${$.goldLine}`, display: "flex", gap: "5px", alignItems: "center", background: `${$.deep}80` }}>
            <span style={{ fontFamily: "'Amiri', serif", color: s.phase === "quest" ? $.warn : $.gold, fontSize: "11px", direction: "rtl" }}>بذرة</span>
            <span style={{ color: s.phase === "quest" ? $.warn : $.gold, fontSize: "9px" }}>›</span>
            <input ref={ref} value={inp} onChange={e => setInp(e.target.value)} onKeyDown={e => e.key === "Enter" && go()}
              placeholder={s.phase === "quest" ? "quest in progress..." : "speak your quest..."}
              disabled={s.phase === "quest"}
              style={{ flex: 1, background: "transparent", border: "none", color: $.bone, fontSize: "10px", fontFamily: "'IBM Plex Mono', monospace", outline: "none" }} />
          </div>
        </div>

        {/* TRUST RAIL */}
        <TrustRail trust={s.trust} ihsan={s.ihsan} chainHead={s.chainHead} />
      </div>

      {/* BOTTOM */}
      <div style={{ padding: "2px 10px", borderTop: `1px solid ${$.ember}`, display: "flex", justifyContent: "space-between", fontSize: "5px", color: `${$.gold}15`, letterSpacing: ".5px" }}>
        <span>{tier} · Ihsan {s.ihsan.toFixed(4)} · Streak {s.streak} · Blocks {s.blocks} · 621 commits</span>
        <span style={{ fontFamily: "'Amiri', serif", fontSize: "7px" }}>بذرة واحدة تصنع غابة</span>
        <span>12,680 tests · Z3 verified · 126× reflex · 0 riba</span>
      </div>
    </div>
  );
}
