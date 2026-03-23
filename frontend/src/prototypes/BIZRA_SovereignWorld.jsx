import { useState, useEffect, useRef, useCallback } from "react";

/*
  ═══════════════════════════════════════════════════════════════
  BIZRA SOVEREIGN WORLD
  ═══════════════════════════════════════════════════════════════
  
  Minecraft: Every receipt is a BLOCK. Your node is your WORLD.
  You mine compute, craft missions, build knowledge structures.
  The world grows visibly. Biomes = skill domains.
  
  Warcraft: PAT-7 is your RAID PARTY. Agents have CLASS ROLES.
  Missions are QUESTS with loot drops. Talent trees unlock.
  Guilds form through federation. BLOOM is soulbound reputation.
  
  Islamic Gold Standard: Every SEED is backed by PoI receipt.
  No fiat. No riba. Zakat 2.5% is protocol, not charity.
  URP = بيت المال. Gini ≤ 0.35 = الحد. The economy cannot
  become what it was built to replace.
  
  TUI: Dwarf Fortress depth. Bloomberg density. بذرة › prompt.
  
  Standing on: Notch (procedural worlds), Blizzard (raid design),
  Al-Ghazali (Ihsan), Ibn Khaldun (economic cycles),
  Al-Khwarizmi (algorithms), Tufte (data density in text)
  ═══════════════════════════════════════════════════════════════
*/

const $ = {
  void: "#020408", deep: "#060D18", ink: "#0B1524",
  gold: "#C9A962", goldSoft: "#D4BC82", goldDim: "#7A6332",
  goldGhost: "rgba(201,169,98,0.06)", goldLine: "rgba(201,169,98,0.10)",
  bone: "#F0EDE4", ash: "rgba(240,237,228,0.55)", ghost: "rgba(240,237,228,0.25)", ember: "rgba(240,237,228,0.10)",
  pass: "#3DD68C", fail: "#F0544F", warn: "#E8B73A",
  // Class colors — Warcraft-inspired role palette
  tank: "#5B8DEF",     // ATLAS — Strategist/Tank
  healer: "#E85D6F",   // CROWN — Ethicist/Healer (prevents harm)
  dps: "#3DD68C",      // FORGE — Builder/DPS (damage = output)
  scout: "#3CC8CF",    // ORACLE — Researcher/Scout
  judge: "#E8B73A",    // JUDGE — Evaluator/Arbiter
  bard: "#E8883A",     // HERALD — Publisher/Bard
  lead: "#A36BDF",     // NEXUS — Integrator/Raid Lead
  // Biome colors — Minecraft-inspired domain zones
  mine: "#8B6914",     // Compute mining
  forest: "#2D6B3F",   // Knowledge growth
  forge: "#8B3A14",    // Code crafting
  ocean: "#14528B",    // Research depths
  temple: "#6B4B8B",   // Constitutional sacred space
};

// ═══ RAID PARTY — PAT-7 as Warcraft Classes ═══
const PARTY = {
  P1: { name: "ATLAS", class: "Strategist", role: "Tank", c: $.tank, icon: "⛊",
    rank: "Commander", hp: 100, mp: 80,
    skills: ["Strategic Decomposition", "Priority Fortress", "Roadmap Shield"],
    idle: ["Scouting the objective map...", "Three quest markers identified.", "Dependency wall breached."],
    work: ["Pulling aggro on complexity...", "Shield wall: subtasks contained.", "Path cleared — 3 moves to objective."] },
  P2: { name: "ORACLE", class: "Scholar", role: "Scout", c: $.scout, icon: "◈",
    rank: "Seer", hp: 60, mp: 100,
    skills: ["Deep Scan", "Knowledge Extraction", "Source Triangulation"],
    idle: ["Scanning the knowledge biome...", "3 rare scrolls detected nearby.", "Graph grew — new connections forged."],
    work: ["Casting Deep Scan...", "47 sources cross-referenced.", "SNR 0.94 — signal locked.", "Synthesis scroll crafted."] },
  P3: { name: "FORGE", class: "Smith", role: "DPS", c: $.dps, icon: "⚒",
    rank: "Artisan", hp: 80, mp: 70,
    skills: ["Code Strike", "Test Barrage", "Reflex Compile"],
    idle: ["Forge is hot. Ready to craft.", "3 flaky ingots need resmelting.", "All 219 blades holding edge."],
    work: ["Striking implementation...", "Test barrage: all pass.", "Ihsan-tempered: 0.97 purity.", "Weapon complete."] },
  P4: { name: "JUDGE", class: "Arbiter", role: "Inspector", c: $.judge, icon: "⚖",
    rank: "Magistrate", hp: 70, mp: 90,
    skills: ["Quality Judgment", "Entropy Measure", "Rubric Binding"],
    idle: ["Court is in session.", "Quality trending — 0.983 average.", "3 challengers benchmarked."],
    work: ["Weighing against constitutional rubric...", "Shannon entropy: above minimum.", "Ruling: exceeds floor.", "Judgment sealed."] },
  P5: { name: "CROWN", class: "Guardian", role: "Healer", c: $.healer, icon: "♛",
    rank: "Sovereign", hp: 90, mp: 95,
    skills: ["Constitutional Shield", "Invariant Ward", "Covenant Seal"],
    idle: ["All 7 wards active.", "Gini ward: 0.31 — holding.", "Covenant unbroken."],
    work: ["Scanning I-1 through I-7...", "Shariah ward: intact.", "No corruption detected.", "Constitutional clearance sealed."] },
  P6: { name: "HERALD", class: "Bard", role: "Support", c: $.bard, icon: "✧",
    rank: "Voice", hp: 65, mp: 85,
    skills: ["Clarity Song", "Format Enchant", "Delivery Strike"],
    idle: ["Readability: 4.8/5.0.", "Three compositions drafted.", "Audience resonance: optimal."],
    work: ["Composing output...", "Clarity enchant applied.", "Format: battle-ready.", "Delivered and sealed."] },
  P7: { name: "NEXUS", class: "Warden", role: "Raid Lead", c: $.lead, icon: "⟡",
    rank: "Archon", hp: 85, mp: 100,
    skills: ["Party Link", "Context Bridge", "Aggregate Command"],
    idle: ["All seven souls linked.", "Context pre-loaded.", "Coordination: 94%."],
    work: ["Routing to specialist...", "Context bridge cast.", "Handoff complete.", "All results aggregated."] },
};

// ═══ SYSTEM AGENTS — SAT-5 (NPCs you don't control) ═══
const NPCS = [
  { id: "S1", name: "Sentinel", role: "Dungeon Guard", icon: "⚔" },
  { id: "S2", name: "Oracle", role: "Scoring NPC", icon: "☉" },
  { id: "S3", name: "Ledger", role: "Block Scribe", icon: "📜" },
  { id: "S4", name: "Conductor", role: "Route Master", icon: "⇶" },
  { id: "S5", name: "Ambassador", role: "Federation Herald", icon: "⚜" },
];

// ═══ SKILL TREES (Talent Trees — unlock with BLOOM) ═══
const TALENTS = [
  { tier: 0, name: "Apprentice Strike", cost: 0, unlocked: true, desc: "Basic mission execution" },
  { tier: 0, name: "Recall I", cost: 0, unlocked: true, desc: "Access recent memory" },
  { tier: 0, name: "Kernel Sight", cost: 0, unlocked: true, desc: "View system status" },
  { tier: 1, name: "Reflex I", cost: 5, unlocked: false, desc: "Compile first S1 pattern (8× speed)" },
  { tier: 1, name: "Deep Scan", cost: 5, unlocked: false, desc: "FAISS semantic search" },
  { tier: 1, name: "Proof Reading", cost: 5, unlocked: false, desc: "View receipt chain" },
  { tier: 2, name: "Multi-Agent Quest", cost: 15, unlocked: false, desc: "Route to 2+ agents per mission" },
  { tier: 2, name: "Knowledge Forge", cost: 15, unlocked: false, desc: "Promote episodic → semantic" },
  { tier: 3, name: "Constitutional Sight", cost: 30, unlocked: false, desc: "See Ihsan breakdown by dimension" },
  { tier: 3, name: "Legendary Craft", cost: 30, unlocked: false, desc: "Chance for legendary drops increases" },
  { tier: 4, name: "Guild Charter", cost: 50, unlocked: false, desc: "Initiate node federation" },
  { tier: 4, name: "Waqf Endowment", cost: 50, unlocked: false, desc: "Permanently endow community resource" },
];

// ═══ BIOMES (Minecraft-inspired domain zones) ═══
const BIOMES = {
  mine: { name: "The Compute Mines", icon: "⛏", color: $.mine, desc: "Raw processing. CI pipelines. Test runs." },
  forest: { name: "Knowledge Forest", icon: "🌳", color: $.forest, desc: "Research. Papers. Graph growth." },
  forge: { name: "Code Forge", icon: "🔥", color: $.forge, desc: "Implementation. Build. Deploy." },
  ocean: { name: "Research Depths", icon: "🌊", color: $.ocean, desc: "Deep analysis. Cross-referencing." },
  temple: { name: "Constitutional Temple", icon: "🕌", color: $.temple, desc: "Governance. Ethics. Invariants." },
};

// ═══ QUEST LOG ═══
const QUESTS = [
  { name: "Mine the CI Stabilization Vein", biome: "mine", difficulty: "Normal", reward: "1.2 SEED", xp: 15, agents: ["P3"] },
  { name: "Scout the Sovereign AI Knowledge Forest", biome: "forest", difficulty: "Heroic", reward: "2.0 SEED", xp: 25, agents: ["P2", "P1"] },
  { name: "Forge the Invariant Test Framework", biome: "forge", difficulty: "Epic", reward: "3.5 SEED", xp: 40, agents: ["P3", "P4"] },
  { name: "Defend the Constitutional Temple", biome: "temple", difficulty: "Legendary", reward: "5.0 SEED", xp: 60, agents: ["P5", "P4", "P7"] },
  { name: "Chart the Deployment Ocean", biome: "ocean", difficulty: "Normal", reward: "1.5 SEED", xp: 20, agents: ["P1", "P6"] },
];

// ═══ ECONOMY (Islamic Gold Standard) ═══
const ECONOMY = {
  seed: { name: "SEED", type: "Utility", backing: "PoI receipt (1:1)", transferable: true, zakat: "2.5% annual" },
  bloom: { name: "BLOOM", type: "Soulbound", backing: "Quality-weighted work", transferable: false, zakat: "N/A (non-transferable)" },
  urp: { name: "بيت المال", type: "Community Pool", backing: "50% founder revenue oath", source: "Voluntary, not protocol tax" },
  gini: { limit: 0.35, name: "الحد", mechanism: "Harberger 5% on idle + causal drag above threshold" },
};

// ═══ STATE ═══
const init = () => ({
  phase: "title", up: false,
  seed: 0, bloom: 0, ihsan: 0, streak: 0, xp: 0, lv: 1, rac: 0, rfx: 0, leg: 0,
  blocks: 0, // Minecraft: every receipt = a block placed
  feed: [], agents: Object.fromEntries(Object.keys(PARTY).map(k => [k, "idle"])),
  focus: null, view: "world", // world | party | talents | quests | economy | map
  petal: 0,
});

// ═══ WORLD MAP — ASCII Art ═══
function WorldMap({ blocks, lv }) {
  const rows = 7, cols = 24;
  const grid = Array.from({ length: rows }, (_, r) =>
    Array.from({ length: cols }, (_, c) => {
      const idx = r * cols + c;
      if (idx < blocks) {
        const types = ["░", "▒", "▓", "█", "◆", "◈", "✦"];
        return types[Math.min(Math.floor(Math.random() * 3 + (lv > 3 ? 1 : 0)), types.length - 1)];
      }
      return "·";
    })
  );
  return (
    <div style={{ fontFamily: "monospace", fontSize: "11px", lineHeight: "14px", color: $.goldDim, letterSpacing: "2px", padding: "4px 0" }}>
      <div style={{ fontSize: "7px", color: $.ghost, letterSpacing: "2px", marginBottom: "4px" }}>NODE0 WORLD — {blocks} BLOCKS PLACED</div>
      {grid.map((row, r) => (
        <div key={r}>{row.map((ch, c) => {
          const idx = r * cols + c;
          const placed = idx < blocks;
          const recent = idx >= blocks - 3 && idx < blocks;
          return <span key={c} style={{ color: recent ? $.gold : placed ? $.goldDim : $.ember }}>{ch}</span>;
        })}</div>
      ))}
    </div>
  );
}

// ═══ PARTY VIEW — Warcraft-style raid frames ═══
function PartyView({ agents, focus }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "3px", padding: "4px 0" }}>
      <div style={{ fontSize: "7px", color: $.ghost, letterSpacing: "2px", marginBottom: "2px" }}>RAID PARTY — PAT-7</div>
      {Object.entries(PARTY).map(([id, p]) => {
        const on = agents[id] !== "idle", hot = id === focus;
        const hpPct = p.hp; const mpPct = p.mp;
        return (
          <div key={id} style={{ display: "flex", alignItems: "center", gap: "6px", padding: "2px 4px", background: hot ? `${p.c}10` : "transparent", borderLeft: `2px solid ${hot ? p.c : on ? p.c + "40" : $.ember}`, transition: "all .3s" }}>
            <span style={{ color: p.c, fontSize: "11px", width: "14px" }}>{p.icon}</span>
            <div style={{ minWidth: "56px" }}>
              <div style={{ fontSize: "9px", color: hot ? p.c : $.bone, fontWeight: hot ? "bold" : "normal" }}>{p.name}</div>
              <div style={{ fontSize: "7px", color: $.ghost }}>{p.class} · {p.role}</div>
            </div>
            <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "1px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                <span style={{ fontSize: "6px", color: $.pass, width: "14px" }}>HP</span>
                <div style={{ flex: 1, height: "3px", background: $.ember, borderRadius: "1px", overflow: "hidden" }}>
                  <div style={{ width: `${hpPct}%`, height: "100%", background: $.pass, borderRadius: "1px" }} />
                </div>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                <span style={{ fontSize: "6px", color: $.tank, width: "14px" }}>MP</span>
                <div style={{ flex: 1, height: "3px", background: $.ember, borderRadius: "1px", overflow: "hidden" }}>
                  <div style={{ width: `${mpPct}%`, height: "100%", background: $.tank, borderRadius: "1px" }} />
                </div>
              </div>
            </div>
            <div style={{ fontSize: "7px", color: $.ghost, minWidth: "50px", textAlign: "right" }}>{p.rank}</div>
          </div>
        );
      })}
      <div style={{ marginTop: "4px", fontSize: "7px", color: $.ghost, letterSpacing: "2px" }}>SAT-5 DUNGEON NPCS (uncontrollable)</div>
      <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
        {NPCS.map(n => <span key={n.id} style={{ fontSize: "7px", color: $.ghost }}>
          {n.icon} {n.name} <span style={{ color: $.ember }}>({n.role})</span>
        </span>)}
      </div>
    </div>
  );
}

// ═══ TALENT TREE VIEW ═══
function TalentView({ bloom, xp }) {
  const tiers = [0, 1, 2, 3, 4];
  return (
    <div style={{ padding: "4px 0" }}>
      <div style={{ fontSize: "7px", color: $.ghost, letterSpacing: "2px", marginBottom: "4px" }}>TALENT TREE — BLOOM: {bloom.toFixed(3)} · XP: {xp}</div>
      {tiers.map(tier => (
        <div key={tier} style={{ marginBottom: "4px" }}>
          <div style={{ fontSize: "7px", color: $.goldDim, marginBottom: "2px" }}>Tier {tier} {tier === 0 ? "(Apprentice)" : tier === 1 ? "(Journeyman)" : tier === 2 ? "(Artisan)" : tier === 3 ? "(Master)" : "(Grandmaster)"}</div>
          <div style={{ display: "flex", gap: "4px", flexWrap: "wrap" }}>
            {TALENTS.filter(t => t.tier === tier).map((t, i) => (
              <div key={i} style={{ padding: "3px 6px", borderRadius: "2px", border: `1px solid ${t.unlocked ? $.pass + "30" : $.ember}`, background: t.unlocked ? `${$.pass}08` : "transparent", fontSize: "8px" }}>
                <span style={{ color: t.unlocked ? $.pass : $.ghost }}>{t.unlocked ? "✓" : "○"} {t.name}</span>
                {!t.unlocked && <span style={{ color: $.goldDim, fontSize: "7px" }}> ({t.cost} BLOOM)</span>}
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

// ═══ QUEST LOG VIEW ═══
function QuestView({ onSelect }) {
  const dc = { Normal: $.pass, Heroic: $.tank, Epic: $.lead, Legendary: $.warn };
  return (
    <div style={{ padding: "4px 0" }}>
      <div style={{ fontSize: "7px", color: $.ghost, letterSpacing: "2px", marginBottom: "4px" }}>QUEST BOARD — Available Missions</div>
      {QUESTS.map((q, i) => {
        const biome = BIOMES[q.biome];
        return (
          <div key={i} onClick={() => onSelect(q)} style={{ display: "flex", gap: "8px", alignItems: "center", padding: "5px 6px", marginBottom: "2px", borderRadius: "2px", border: `1px solid ${$.ember}`, cursor: "pointer", transition: "all .2s" }}
            onMouseEnter={e => { e.currentTarget.style.borderColor = $.goldLine; e.currentTarget.style.background = $.goldGhost; }}
            onMouseLeave={e => { e.currentTarget.style.borderColor = $.ember; e.currentTarget.style.background = "transparent"; }}>
            <span style={{ fontSize: "12px" }}>{biome.icon}</span>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: "9px", color: $.bone }}>{q.name}</div>
              <div style={{ fontSize: "7px", color: $.ghost }}>{biome.name} · Agents: {q.agents.map(a => PARTY[a].name).join(", ")}</div>
            </div>
            <div style={{ textAlign: "right" }}>
              <div style={{ fontSize: "7px", color: dc[q.difficulty], fontWeight: "bold", letterSpacing: ".5px" }}>{q.difficulty.toUpperCase()}</div>
              <div style={{ fontSize: "7px", color: $.goldDim }}>{q.reward} · {q.xp} XP</div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ═══ ECONOMY VIEW — Islamic Gold Standard ═══
function EconomyView({ seed, bloom, blocks }) {
  return (
    <div style={{ padding: "4px 0" }}>
      <div style={{ fontSize: "7px", color: $.ghost, letterSpacing: "2px", marginBottom: "6px" }}>SOVEREIGN ECONOMY — GOLD STANDARD</div>
      <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
        {[
          { label: "SEED (Utility)", value: `${seed.toFixed(3)}`, backing: "1 SEED = 1 PoI receipt (proof-backed)", color: $.pass, note: "Transferable · Zakat 2.5% annual" },
          { label: "BLOOM (Soulbound)", value: `${bloom.toFixed(4)}`, backing: "Quality-weighted work history", color: $.lead, note: "Non-transferable · Earned only" },
          { label: "بيت المال (URP)", value: "50% founder oath", backing: "Founder/Foundation revenue — not user tax", color: $.gold, note: "Community pool · Voluntary" },
          { label: "الحد (Adl Limit)", value: "Gini ≤ 0.35", backing: "Harberger 5% idle + causal drag", color: $.warn, note: "Anti-plutocracy invariant" },
          { label: "رِبَا (Interest)", value: "ZERO", backing: "I-2 invariant — constitutional prohibition", color: $.fail, note: "Cannot be overridden by any agent" },
          { label: "Blocks Placed", value: blocks, backing: "Each receipt = 1 block in your world", color: $.goldSoft, note: "World grows with every verified action" },
        ].map((e, i) => (
          <div key={i} style={{ display: "flex", gap: "8px", alignItems: "flex-start", padding: "4px 6px", borderLeft: `2px solid ${e.color}30`, background: `${e.color}05` }}>
            <div style={{ minWidth: "120px" }}>
              <div style={{ fontSize: "8px", color: e.color, fontWeight: "bold" }}>{e.label}</div>
              <div style={{ fontSize: "7px", color: $.ghost }}>{e.note}</div>
            </div>
            <div style={{ flex: 1, textAlign: "right" }}>
              <div style={{ fontSize: "11px", color: e.color, fontFamily: "monospace", fontVariantNumeric: "tabular-nums" }}>{e.value}</div>
              <div style={{ fontSize: "7px", color: $.ghost }}>{e.backing}</div>
            </div>
          </div>
        ))}
      </div>
      <div style={{ marginTop: "8px", padding: "6px", border: `1px solid ${$.goldLine}`, borderRadius: "3px", background: $.goldGhost }}>
        <div style={{ fontSize: "7px", color: $.gold, letterSpacing: "1px", marginBottom: "3px" }}>CONSTITUTIONAL GUARANTEE</div>
        <div style={{ fontSize: "8px", color: $.ash, lineHeight: 1.5 }}>
          Every SEED in circulation is backed by a verified Proof-of-Ihsan receipt. No token exists without work. No work exists without constitutional clearance. The economy cannot inflate, cannot charge interest, and cannot concentrate beyond Gini 0.35. This is not policy — it is compiled into the type system.
        </div>
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
    set(p => ({ ...p, feed: [...p.feed, { from, text, kind, ts: Date.now() }].slice(-100) }));
  }, []);
  const wait = ms => new Promise(r => setTimeout(r, ms));

  // ═══ GENESIS — World Creation ═══
  const genesis = useCallback(async () => {
    set(p => ({ ...p, phase: "creating" }));
    const steps = [
      ["SYS", "بسم الله الرحمن الرحيم", "sys"],
      ["SYS", "═══ WORLD GENESIS ═══", "sys"],
      ["SYS", "Mining genesis block...", "sys"],
      ["SYS", "Constitutional bedrock: 7 invariants placed.", "sys"],
      ["SYS", "Sovereign identity forged (Ed25519).", "sys"],
      ["SYS", "Economic foundation: gold standard activated.", "sys"],
      ["SYS", "RIBA_ZERO: interest permanently disabled.", "sys"],
      ["SYS", "Zakat 2.5%: wired into protocol.", "sys"],
      ["SYS", "الحد (Gini ≤ 0.35): anti-accumulation ward placed.", "sys"],
      ["SYS", "بيت المال (URP): community treasury initialized.", "sys"],
      ["SYS", "═══ SUMMONING RAID PARTY ═══", "sys"],
    ];
    for (const [f, t, k] of steps) { msg(f, t, k); await wait(180); }
    for (const id of ["P7","P1","P2","P3","P4","P5","P6"]) {
      const p = PARTY[id];
      set(prev => ({ ...prev, agents: { ...prev.agents, [id]: "summon" }, focus: id }));
      msg(id, `${p.icon} ${p.name} the ${p.class} joins the party. Role: ${p.role}.`, "summon");
      await wait(200);
      set(prev => ({ ...prev, agents: { ...prev.agents, [id]: "idle" } }));
    }
    msg("SYS", "SAT-5 dungeon NPCs deployed. You cannot control them.", "sys");
    await wait(150);
    msg("SYS", "═══ TRUST VERIFICATION ═══", "sys");
    msg("SYS", "Node ● | Ledger ◈ | Token ◆ | Supply ▣ | Gate ✦ — ALL VERIFIED", "sys");
    await wait(200);
    msg("SYS", `Block #0 placed. Your world begins.`, "mint");
    await wait(300);
    msg("P7", "Good morning, Mumo.", "greet");
    await wait(300);
    msg("P7", "Your world has 1 block. Every quest you complete places more. The map grows with your work. Ready?", "agent");
    set(p => ({ ...p, up: true, phase: "ready", focus: null, blocks: 1 }));
    setTimeout(() => ref.current?.focus(), 100);
  }, [msg]);

  // ═══ QUEST EXECUTION ═══
  const quest = useCallback(async (task, questData) => {
    set(p => ({ ...p, phase: "quest" }));
    const biome = questData ? BIOMES[questData.biome] : BIOMES.forest;
    msg("P7", `⟡ Quest accepted: "${task.slice(0, 55)}..."`, "agent");
    msg("SYS", `Entering ${biome.icon} ${biome.name}...`, "sys");
    set(p => ({ ...p, focus: "P7" }));
    await wait(400);
    // Route
    const kw = { P1:["plan","strategy","roadmap"], P2:["research","find","analyze","scout"], P3:["code","build","test","forge","craft"], P4:["evaluate","score","review","judge"], P5:["check","verify","defend","guard"], P6:["write","draft","report","publish"] };
    let best = "P2", bs = 0;
    for (const [a,ws] of Object.entries(kw)) { const sc = ws.filter(w => task.toLowerCase().includes(w)).length; if (sc > bs) { best = a; bs = sc; } }
    if (questData?.agents?.[0]) best = questData.agents[0];
    const ag = PARTY[best];
    msg("P7", `${ag.icon} Routing to ${ag.name} the ${ag.class}.`, "agent");
    set(p => ({ ...p, agents: { ...p.agents, [best]: "active", P7: "lead" }, focus: best }));
    await wait(350);
    for (const m of ag.work) { msg(best, `${ag.icon} ${m}`, "work"); await wait(400 + Math.random()*200); }
    // Judge scores
    set(p => ({ ...p, agents: { ...p.agents, P4: "judge" }, focus: "P4" }));
    await wait(300);
    const ih = (0.95 + Math.random() * .04).toFixed(4);
    msg("P4", `⚖ Quality judgment: Ihsan ${ih}.${parseFloat(ih) >= .98 ? " Exceptional purity." : " Above constitutional floor."}`, "score");
    await wait(250);
    // Crown clears
    set(p => ({ ...p, agents: { ...p.agents, P5: "guard" }, focus: "P5" }));
    msg("P5", "♛ All 7 wards hold. Constitutional clearance sealed.", "clear");
    await wait(250);
    // Loot drop
    const leg = parseFloat(ih) >= .98 && Math.random() > .5;
    const epic = !leg && parseFloat(ih) >= .96;
    const drop = leg ? "⚡ LEGENDARY" : epic ? "💜 EPIC" : "🔷 RARE";
    const mul = leg ? 1.5 : epic ? 1.3 : 1.15;
    const se = (parseFloat(ih)*mul).toFixed(3);
    const be = (0.01*parseFloat(ih)).toFixed(4);
    const xpEarned = questData?.xp || (15 + Math.floor(Math.random() * 20));
    msg("SYS", `${drop} LOOT — +${se} SEED · +${be} BLOOM · +${xpEarned} XP`, "mint");
    // Block placed
    const newBlocks = s.blocks + 1;
    msg("SYS", `█ Block #${newBlocks} placed. World grows.`, "block");
    set(p => ({ ...p, focus: "P6" }));
    msg("P6", "✧ Receipt sealed and chained. Delivered.", "agent");
    await wait(200);
    const nr = s.rac + 1; const compiled = nr > 0 && nr % 5 === 0;
    if (compiled) msg("SYS", "⚡ REFLEX COMPILED — Pattern learned. 8× faster next cast.", "reflex");
    msg("P7", `⟡ Quest complete. +${se} SEED. World: ${newBlocks} blocks.${compiled ? " New reflex unlocked." : ""}`, "done");
    set(p => ({
      ...p, phase: "ready", focus: null, blocks: newBlocks,
      seed: p.seed + parseFloat(se), bloom: p.bloom + parseFloat(be),
      xp: p.xp + xpEarned, rac: p.rac+1, streak: p.streak+1,
      lv: Math.max(p.lv, 1 + Math.floor((p.xp + xpEarned) / 50)),
      ihsan: parseFloat(ih), rfx: p.rfx+(compiled?1:0), leg: p.leg+(leg?1:0),
      petal: p.petal + parseFloat(be)*100,
      agents: Object.fromEntries(Object.keys(PARTY).map(k=>[k,"idle"])),
    }));
  }, [msg, s.blocks, s.rac, s.xp]);

  const go = () => {
    if (!inp.trim() || s.phase === "quest") return;
    const task = inp.trim(); setInp("");
    msg("USER", task, "user");
    setTimeout(() => quest(task), 250);
  };

  const selectQuest = (q) => {
    if (s.phase === "quest") return;
    msg("USER", q.name, "user");
    setTimeout(() => quest(q.name, q), 250);
  };

  const hms = t.toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const tierName = s.lv < 3 ? "Apprentice" : s.lv < 6 ? "Journeyman" : s.lv < 10 ? "Artisan" : s.lv < 15 ? "Master" : "Grandmaster";
  const views = ["world", "party", "talents", "quests", "economy"];

  // ═══ TITLE SCREEN — World creation ═══
  if (!s.up && s.phase !== "creating") return (
    <div style={{ minHeight: "100vh", background: $.void, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", fontFamily: "'IBM Plex Mono', monospace", position: "relative", overflow: "hidden" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500&family=Amiri:wght@400;700&display=swap');
        @keyframes blink { 0%,49% { opacity:1; } 50%,100% { opacity:0; } }
        @keyframes drift { 0%,100% { transform:translateY(0); } 50% { transform:translateY(-2px); } }
      `}</style>
      <div style={{ position: "absolute", inset: 0, background: `repeating-linear-gradient(0deg, transparent, transparent 47px, ${$.ember} 48px)`, opacity: .3, maskImage: "radial-gradient(circle, black 20%, transparent 65%)", WebkitMaskImage: "radial-gradient(circle, black 20%, transparent 65%)" }}/>
      <pre style={{ color: $.goldDim, fontSize: "10px", lineHeight: "12px", letterSpacing: "1px", textAlign: "center", animation: "drift 6s ease-in-out infinite", marginBottom: "16px" }}>{`
    ╔═══════════════════════════════════════╗
    ║                                       ║
    ║     ░▒▓  B I Z R A  ▓▒░              ║
    ║                                       ║
    ║     SOVEREIGN WORLD                   ║
    ║                                       ║
    ╚═══════════════════════════════════════╝`}</pre>
      <div style={{ fontFamily: "'Amiri', serif", fontSize: "18px", color: `${$.gold}40`, letterSpacing: "1px", marginBottom: "4px" }}>البذرة</div>
      <div style={{ fontSize: "7px", color: $.ghost, letterSpacing: "3px", marginBottom: "28px" }}>MINECRAFT × WARCRAFT × GOLD STANDARD × TUI</div>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "6px", fontSize: "8px", color: $.ghost, marginBottom: "24px" }}>
        <span>⛏ Mine compute · ⚒ Craft missions · 🌳 Grow knowledge</span>
        <span>⛊ Raid with 7 agents · ⚖ Constitutional loot · 📜 Receipt blocks</span>
        <span>Every SEED backed by proof · Every block a verified action</span>
      </div>
      <button onClick={genesis} style={{ background: "transparent", border: `1px solid ${$.gold}25`, color: $.gold, padding: "10px 40px", borderRadius: "1px", fontSize: "9px", letterSpacing: "4px", cursor: "pointer", fontFamily: "'IBM Plex Mono', monospace", transition: "all .4s" }}
        onMouseEnter={e => { e.target.style.background = `${$.gold}08`; e.target.style.borderColor = `${$.gold}40`; }}
        onMouseLeave={e => { e.target.style.background = "transparent"; e.target.style.borderColor = `${$.gold}25`; }}>
        CREATE WORLD
      </button>
      <div style={{ position: "absolute", bottom: "14px", fontSize: "6px", color: `${$.gold}10`, letterSpacing: "2px" }}>NODE0 · 768K LOC · 12,644 TESTS · 0 RIBA</div>
    </div>
  );

  // ═══ THE WORLD ═══
  return (
    <div style={{ minHeight: "100vh", background: $.void, color: $.bone, fontFamily: "'IBM Plex Mono', monospace", fontSize: "10px", display: "flex", flexDirection: "column", position: "relative" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500&family=Amiri:wght@400;700&display=swap');
        @keyframes slideIn { from { opacity:0; transform:translateY(2px); } to { opacity:1; transform:translateY(0); } }
        @keyframes pulse { 0%,100% { opacity:.5; } 50% { opacity:1; } }
        input::placeholder { color: ${$.ghost}; }
        ::-webkit-scrollbar { width: 2px; }
        ::-webkit-scrollbar-thumb { background: ${$.goldLine}; }
        ::-webkit-scrollbar-track { background: transparent; }
      `}</style>

      {/* HUD — Top bar */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "4px 12px", borderBottom: `1px solid ${$.goldLine}`, background: `${$.void}ee`, zIndex: 10 }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <span style={{ color: $.gold, fontSize: "10px", fontWeight: 500, letterSpacing: "2px" }}>BIZRA</span>
          <span style={{ color: $.ember, fontSize: "7px" }}>NODE0</span>
          <span style={{ fontSize: "7px", color: s.phase === "quest" ? $.warn : $.pass, letterSpacing: "1px" }}>
            ● {s.phase === "quest" ? "IN QUEST" : "IDLE"}
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "10px", fontSize: "7px" }}>
          <span style={{ color: $.pass }}>{s.seed.toFixed(2)} SEED</span>
          <span style={{ color: $.lead }}>{s.bloom.toFixed(3)} BLOOM</span>
          <span style={{ color: $.warn }}>Lv.{s.lv} {tierName}</span>
          <span style={{ color: $.tank }}>█{s.blocks}</span>
          <span style={{ color: $.gold }}>{hms}</span>
        </div>
      </div>

      {/* Agent bar */}
      <div style={{ display: "flex", gap: "1px", padding: "2px 12px", borderBottom: `1px solid ${$.ember}` }}>
        {Object.entries(PARTY).map(([id, p]) => {
          const hot = id === s.focus;
          return <div key={id} style={{ flex: 1, textAlign: "center", padding: "1px", borderBottom: `2px solid ${hot ? p.c : "transparent"}`, transition: "all .3s" }}>
            <span style={{ fontSize: "9px", color: hot ? p.c : $.ghost }}>{p.icon}</span>
          </div>;
        })}
      </div>

      {/* NAV TABS */}
      <div style={{ display: "flex", gap: "1px", padding: "2px 12px", borderBottom: `1px solid ${$.ember}` }}>
        {views.map(v => (
          <button key={v} onClick={() => set(p => ({ ...p, view: v }))}
            style={{ flex: 1, padding: "3px", background: s.view === v ? $.goldGhost : "transparent", border: `1px solid ${s.view === v ? $.goldLine : "transparent"}`, borderRadius: "1px", color: s.view === v ? $.gold : $.ghost, fontSize: "7px", letterSpacing: "1px", cursor: "pointer", fontFamily: "monospace", textTransform: "uppercase", transition: "all .2s" }}>
            {v === "world" ? "⛏ World" : v === "party" ? "⛊ Party" : v === "talents" ? "✦ Talents" : v === "quests" ? "⚔ Quests" : "◆ Economy"}
          </button>
        ))}
      </div>

      {/* MAIN — split: content left, feed right */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
        {/* LEFT: View Panel */}
        <div style={{ width: "45%", minWidth: "250px", padding: "4px 12px", borderRight: `1px solid ${$.ember}`, overflowY: "auto" }}>
          {s.view === "world" && <WorldMap blocks={s.blocks} lv={s.lv} />}
          {s.view === "party" && <PartyView agents={s.agents} focus={s.focus} />}
          {s.view === "talents" && <TalentView bloom={s.bloom} xp={s.xp} />}
          {s.view === "quests" && <QuestView onSelect={selectQuest} />}
          {s.view === "economy" && <EconomyView seed={s.seed} bloom={s.bloom} blocks={s.blocks} />}
        </div>

        {/* RIGHT: Mission Feed */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
          <div style={{ flex: 1, overflowY: "auto", padding: "2px 10px" }}>
            {s.feed.map((m, i) => {
              const u = m.kind === "user", sy = m.kind === "sys", mt = m.kind === "mint", dn = m.kind === "done", gr = m.kind === "greet", bl = m.kind === "block", rf = m.kind === "reflex", sm = m.kind === "summon";
              const c = u ? $.gold : sy ? $.ghost : mt ? $.warn : bl ? $.goldSoft : rf ? $.lead : sm ? $.pass : dn ? $.gold : PARTY[m.from]?.c || $.gold;
              const lab = u ? "YOU" : sy ? "SYS" : PARTY[m.from]?.name || m.from;
              return <div key={i} style={{ marginBottom: "1px", padding: "1px 0", display: "flex", gap: "6px", alignItems: "flex-start", animation: "slideIn .1s ease", opacity: sy ? .4 : 1 }}>
                <span style={{ color: c, fontWeight: "bold", minWidth: "38px", fontSize: "7px", textAlign: "right" }}>{lab}</span>
                <span style={{ color: u ? $.bone : mt ? $.warn : bl ? $.goldSoft : rf ? $.lead : dn ? $.gold : gr ? $.goldSoft : $.ash, fontSize: gr ? "10px" : "8px", lineHeight: 1.5, fontFamily: gr ? "'Amiri', serif" : "inherit" }}>
                  {mt ? `${m.text}` : dn ? `✓ ${m.text}` : bl ? `${m.text}` : rf ? `${m.text}` : m.text}
                </span>
              </div>;
            })}
            <div ref={end}/>
          </div>

          {/* PROMPT */}
          <div style={{ padding: "6px 10px", borderTop: `1px solid ${$.goldLine}`, display: "flex", gap: "6px", alignItems: "center", background: `${$.deep}80` }}>
            <span style={{ fontFamily: "'Amiri', serif", color: s.phase === "quest" ? $.warn : $.gold, fontSize: "11px", direction: "rtl" }}>بذرة</span>
            <span style={{ color: s.phase === "quest" ? $.warn : $.gold, fontSize: "9px" }}>›</span>
            <input ref={ref} value={inp} onChange={e => setInp(e.target.value)}
              onKeyDown={e => e.key === "Enter" && go()}
              placeholder={s.phase === "quest" ? "quest in progress..." : "speak your quest..."}
              disabled={s.phase === "quest"}
              style={{ flex: 1, background: "transparent", border: "none", color: $.bone, fontSize: "10px", fontFamily: "'IBM Plex Mono', monospace", outline: "none" }}/>
          </div>
        </div>
      </div>

      {/* BOTTOM */}
      <div style={{ padding: "2px 12px", borderTop: `1px solid ${$.ember}`, display: "flex", justifyContent: "space-between", fontSize: "6px", color: `${$.gold}15`, letterSpacing: ".5px" }}>
        <span>{tierName} · Ihsan {s.ihsan.toFixed(4)} · Streak {s.streak} · Blocks {s.blocks}</span>
        <span style={{ fontFamily: "'Amiri', serif", fontSize: "7px" }}>بذرة واحدة تصنع غابة</span>
        <span>PAT-7 · SAT-5 · 0 RIBA · GOLD STANDARD</span>
      </div>
    </div>
  );
}
