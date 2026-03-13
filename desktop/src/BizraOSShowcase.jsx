import { useState, useEffect, useRef, useCallback } from "react";

const G = "#C9A962";
const G2 = "#E8D5A3";
const G3 = "#8B7340";
const BG = "#030810";
const BG2 = "#08121f";
const GR = "#22c55e";
const RD = "#ef4444";
const BL = "#3b82f6";
const PU = "#a855f7";
const CY = "#06b6d4";
const AM = "#f97316";
const YL = "#eab308";
const RS = "#f43f5e";
const TXT = "#F8F6F1";
const MUT = "rgba(248,246,241,.72)";
const DIM = "rgba(248,246,241,.45)";
const DIMR = "rgba(248,246,241,.25)";
const LINE = "rgba(255,255,255,.08)";

const PAT = {
  P1: { n: "Planner", c: "ATLAS", d: "Strategy", b: "Strategic planning ready.", i: "◈", col: BL },
  P2: { n: "Researcher", c: "ORACLE", d: "Knowledge", b: "Knowledge systems nominal.", i: "◉", col: CY },
  P3: { n: "Coder", c: "FORGE", d: "Build", b: "Compiler initialized.", i: "⬡", col: GR },
  P4: { n: "Evaluator", c: "JUDGE", d: "Quality", b: "Quality gates armed.", i: "◇", col: YL },
  P5: { n: "Ethicist", c: "CROWN", d: "Ethics", b: "All invariants holding.", i: "☗", col: RD },
  P6: { n: "Publisher", c: "HERALD", d: "Deliver", b: "Delivery channels open.", i: "◆", col: AM },
  P7: { n: "Integrator", c: "NEXUS", d: "Orchestrate", b: "All agents reporting.", i: "✦", col: PU },
};

const SAT = [
  { n: "Sentinel", col: RD },
  { n: "Oracle", col: G },
  { n: "Ledger", col: YL },
  { n: "Conductor", col: BL },
  { n: "Ambassador", col: CY },
];

const TIERS = ["Novice", "Apprentice", "Adept", "Expert", "Master", "Grandmaster"];
const TCOL = ["#6B7280", BL, GR, PU, YL, G];

const STAGES = [
  { n: "Seed", l: 0, h: 0.10, d: "Identity created. Potential infinite." },
  { n: "Node", l: 0.10, h: 0.20, d: "First mission completed." },
  { n: "Apprentice", l: 0.20, h: 0.35, d: "Building habits." },
  { n: "Builder", l: 0.35, h: 0.55, d: "Compiled first reflex." },
  { n: "Verifier", l: 0.55, h: 0.70, d: "Trusted to attest others." },
  { n: "Mentor", l: 0.70, h: 0.85, d: "Skills published." },
  { n: "Catalyst", l: 0.85, h: 1, d: "Network multiplier." },
];

const SKILLS = [
  { id: "open_app", n: "Open App", t: 0, i: "🚀", u: true, hda: true },
  { id: "switch_window", n: "Switch Window", t: 0, i: "🪟", u: true, hda: true },
  { id: "type_text", n: "Type Text", t: 0, i: "⌨️", u: true, hda: true },
  { id: "click_element", n: "Click Element", t: 1, i: "🖱️", hda: true },
  { id: "screenshot", n: "Screenshot", t: 1, i: "📸", hda: true },
  { id: "read_clipboard", n: "Clipboard", t: 1, i: "📋", hda: true },
  { id: "file_open", n: "File Open", t: 2, i: "📖", hda: true },
  { id: "browser_nav", n: "Browser Nav", t: 2, i: "🌐", hda: true },
  { id: "powershell", n: "PowerShell", t: 3, i: "⚡" },
  { id: "multistep", n: "Multi-Step", t: 3, i: "🔗" },
  { id: "crossapp", n: "Cross-App", t: 4, i: "🔄" },
  { id: "network", n: "Network", t: 4, i: "📡" },
  { id: "governance", n: "Governance", t: 4, i: "🏛️" },
  { id: "selfmod", n: "Self-Modify", t: 5, i: "🧬" },
  { id: "validator", n: "Validator", t: 5, i: "🛡️" },
  { id: "federation", n: "Federation", t: 5, i: "🌍" },
];

const SCHEDULED = [
  { id: "morning-brief", n: "Morning Brief", cron: "08:00 weekdays", icon: "☀️", seed: "0.50", desc: "Overnight alerts + priority tasks", auto: false, agents: ["ATLAS", "ORACLE", "CROWN"] },
  { id: "standup", n: "Daily Standup", cron: "10:00 weekdays", icon: "📋", seed: "0.30", desc: "Progress, blockers, plan", auto: false, agents: ["ATLAS", "ORACLE"] },
  { id: "health-check", n: "Health Check", cron: "Every 15 min", icon: "💚", seed: "0.05", desc: "Node0 subsystem monitoring", auto: true, agents: ["ORACLE"] },
  { id: "weekly-review", n: "Weekly Review", cron: "16:00 Friday", icon: "📊", seed: "1.00", desc: "Accomplishments, metrics, next week", auto: false, agents: ["ATLAS", "ORACLE", "CROWN"] },
];

const TEACH_QUESTIONS = [
  { id: "work_schedule", prompt: "What's your typical work schedule?", type: "text", default: "8:00-18:00", icon: "🕐" },
  {
    id: "primary_tools",
    prompt: "Which apps do you use most?",
    type: "multi",
    opts: ["VS Code", "Chrome", "Slack", "Terminal", "Notion", "Figma", "Excel"],
    icon: "🛠️",
  },
  {
    id: "communication_pref",
    prompt: "How should I communicate with you?",
    type: "single",
    opts: ["Concise bullet points", "Detailed explanations", "Only when critical"],
    default: "Concise bullet points",
    icon: "💬",
  },
  {
    id: "priority_domains",
    prompt: "What are your top priority domains?",
    type: "multi",
    opts: ["Engineering", "Business strategy", "Marketing", "Operations", "Research"],
    icon: "🎯",
  },
  {
    id: "autonomy",
    prompt: "How much autonomy should I have?",
    type: "single",
    opts: ["Ask before every action", "Auto low-risk, ask high-risk", "Full autonomous within budget"],
    default: "Auto low-risk, ask high-risk",
    icon: "🤖",
  },
];

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function getStage(score) {
  for (let i = STAGES.length - 1; i >= 0; i -= 1) {
    if (score >= STAGES[i].l) {
      return STAGES[i];
    }
  }
  return STAGES[0];
}

function useViewport() {
  const getWidth = () => (typeof window === "undefined" ? 1280 : window.innerWidth);
  const [width, setWidth] = useState(getWidth);

  useEffect(() => {
    const onResize = () => setWidth(getWidth());
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  return {
    width,
    isMobile: width < 768,
    isTablet: width >= 768 && width < 1100,
  };
}

function FadeIn({ children, d = 0, s = {} }) {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setVisible(true), d);
    return () => clearTimeout(timer);
  }, [d]);

  return (
    <div
      style={{
        opacity: visible ? 1 : 0,
        transform: visible ? "translateY(0)" : "translateY(10px)",
        transition: "all .6s ease",
        ...s,
      }}
    >
      {children}
    </div>
  );
}

function TrustSite({ onEnter, onReadSeed, viewport }) {
  const [hovered, setHovered] = useState(false);
  const [scrollY, setScrollY] = useState(0);
  const ref = useRef(null);
  const statsColumns = viewport.isMobile ? "1fr" : viewport.isTablet ? "repeat(2, 1fr)" : "repeat(4, 1fr)";
  const invariantColumns = viewport.isMobile ? "1fr" : viewport.isTablet ? "repeat(2, 1fr)" : "repeat(5, 1fr)";
  const layerColumns = viewport.isMobile ? "1fr" : "32px 1fr 200px 80px";

  useEffect(() => {
    const node = ref.current;
    if (!node) {
      return undefined;
    }
    const onScroll = () => setScrollY(node.scrollTop || 0);
    node.addEventListener("scroll", onScroll);
    return () => node.removeEventListener("scroll", onScroll);
  }, []);

  const layers = [
    { n: "Human Seed", c: "الرسالة + البذرة", t: "—", col: G },
    { n: "Sovereign Node", c: "identity_genesis.py", t: "332", col: BL },
    { n: "Agentic Dev", c: "mission_pipeline.py", t: "151", col: GR },
    { n: "Verification", c: "evidence_receipt.py", t: "50+", col: YL },
    { n: "Learning", c: "seed_engine.py", t: "46", col: CY },
    { n: "Economic", c: "algorithms.py", t: "100+", col: PU },
    { n: "Civilizational", c: "federation/", t: "60+", col: RS },
  ];

  return (
    <div ref={ref} style={{ height: "100vh", overflow: "auto", background: BG, color: TXT, fontFamily: "Inter, system-ui, sans-serif" }}>
      <div
        style={{
          position: "sticky",
          top: 0,
          zIndex: 50,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: 16,
          padding: viewport.isMobile ? "12px 20px" : "12px 32px",
          background: scrollY > 50 ? "rgba(3,8,16,.92)" : "transparent",
          backdropFilter: scrollY > 50 ? "blur(20px)" : "none",
          borderBottom: scrollY > 50 ? `1px solid ${LINE}` : "none",
          transition: "all .4s",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ fontFamily: "Cinzel, serif", color: G, fontSize: 14, fontWeight: 600, letterSpacing: 4 }}>BIZRA</span>
          <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 8, color: DIMR, letterSpacing: 3 }}>DDAGI OS</span>
        </div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "flex-end" }}>
          <button
            onClick={onReadSeed}
            style={{
              background: "transparent",
              border: `1px solid ${LINE}`,
              color: MUT,
              padding: viewport.isMobile ? "8px 12px" : "8px 16px",
              borderRadius: 4,
              fontSize: 10,
              fontFamily: "'JetBrains Mono', monospace",
              letterSpacing: 1.5,
              cursor: "pointer",
              whiteSpace: "nowrap",
            }}
          >
            READ THE SEED
          </button>
          <button
            onClick={onEnter}
            style={{
              background: `${G}12`,
              border: `1px solid ${G}40`,
              color: G,
              padding: viewport.isMobile ? "8px 14px" : "8px 20px",
              borderRadius: 4,
              fontSize: 11,
              fontFamily: "'JetBrains Mono', monospace",
              letterSpacing: 2,
              cursor: "pointer",
              whiteSpace: "nowrap",
            }}
          >
            INITIALIZE NODE
          </button>
        </div>
      </div>

      <div
        style={{
          position: "relative",
          padding: viewport.isMobile ? "72px 20px 56px" : "100px 48px 80px",
          background:
            "radial-gradient(circle at 15% 15%, rgba(201,169,98,.1), transparent 35%), radial-gradient(circle at 85% 20%, rgba(59,130,246,.08), transparent 30%), linear-gradient(180deg, #07111d, #030810)",
        }}
      >
        <div
          style={{
            position: "absolute",
            inset: 0,
            backgroundImage:
              "linear-gradient(rgba(255,255,255,.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.03) 1px, transparent 1px)",
            backgroundSize: "42px 42px",
            maskImage: "linear-gradient(180deg, rgba(0,0,0,.6), transparent)",
          }}
        />
        <div style={{ position: "relative", maxWidth: 1100, margin: "0 auto" }}>
          <FadeIn d={200}>
            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: G, letterSpacing: 3 }}>DISTRIBUTED DECENTRALIZED AGI OPERATING SYSTEM</div>
          </FadeIn>
          <FadeIn d={500}>
            <h1
              style={{
                fontFamily: "'Playfair Display', serif",
                fontSize: viewport.isMobile ? 36 : viewport.isTablet ? 46 : 56,
                lineHeight: 0.96,
                margin: "20px 0",
                maxWidth: 800,
                fontWeight: 700,
              }}
            >
              From human need
              <br />
              to sovereign intelligence.
            </h1>
          </FadeIn>
          <FadeIn d={800}>
            <p style={{ color: MUT, fontSize: viewport.isMobile ? 15 : 17, maxWidth: 700, lineHeight: 1.7, margin: 0 }}>
              BIZRA turns every human into a sovereign node, every node into a living seed, and every verified act of growth into shared intelligence, capability, and value.
            </p>
          </FadeIn>
          <FadeIn d={1100}>
            <button
              onClick={onEnter}
              onMouseEnter={() => setHovered(true)}
              onMouseLeave={() => setHovered(false)}
              style={{
                marginTop: 32,
                background: hovered ? G : "transparent",
                color: hovered ? BG : G,
                border: `1.5px solid ${G}`,
                padding: viewport.isMobile ? "14px 22px" : "14px 36px",
                borderRadius: 6,
                fontSize: 12,
                fontFamily: "'JetBrains Mono', monospace",
                letterSpacing: 3,
                cursor: "pointer",
                transition: "all .3s",
              }}
            >
              BEGIN YOUR JOURNEY
            </button>
          </FadeIn>
          <FadeIn d={1400}>
            <div style={{ display: "grid", gridTemplateColumns: statsColumns, gap: 16, marginTop: 48 }}>
              {[
                { v: "8,237", l: "Tests Passing" },
                { v: "22", l: "Rust Crates" },
                { v: "31+", l: "Days Live" },
                { v: "0.95+", l: "Ihsan Floor" },
              ].map((metric) => (
                <div key={metric.l} style={{ padding: "16px 18px", borderRadius: 16, background: "rgba(255,255,255,.025)", border: `1px solid ${LINE}`, backdropFilter: "blur(12px)" }}>
                  <div style={{ fontSize: 28, fontWeight: 800, letterSpacing: -1, color: G2 }}>{metric.v}</div>
                  <div style={{ fontSize: 11, color: DIM, letterSpacing: 1.2, textTransform: "uppercase", marginTop: 4 }}>{metric.l}</div>
                </div>
              ))}
            </div>
          </FadeIn>
        </div>
      </div>

      <div style={{ maxWidth: 1100, margin: "0 auto", padding: viewport.isMobile ? "48px 20px" : "64px 48px" }}>
        <div style={{ borderLeft: `3px solid ${G}`, padding: "20px 24px", background: `${G}0A`, borderRadius: "0 16px 16px 0", marginBottom: 48 }}>
          <div style={{ fontFamily: "Amiri, serif", fontSize: 16, color: `${G}60`, direction: "rtl", marginBottom: 8 }}>بسم الله الرحمن الرحيم</div>
          <div style={{ fontSize: viewport.isMobile ? 16 : 18, lineHeight: 1.6 }}>
            "Every human is a node, and every node is a seed, and every seed has infinite potential."
          </div>
          <div style={{ fontSize: 12, color: DIM, marginTop: 8, fontFamily: "'JetBrains Mono', monospace" }}>— البذرة, Ramadan 2023</div>
        </div>

        <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: G, letterSpacing: 3, marginBottom: 8 }}>FIVE NON-NEGOTIABLE INVARIANTS</div>
        <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: viewport.isMobile ? 28 : 32, margin: "0 0 24px" }}>Machine-enforced. No exceptions.</h2>
        <div style={{ display: "grid", gridTemplateColumns: invariantColumns, gap: 12, marginBottom: 48 }}>
          {[
            { id: "I-1", n: "Excellence", v: "Ihsan ≥ 0.95", c: G },
            { id: "I-2", n: "Signal", v: "SNR ≥ 0.85", c: BL },
            { id: "I-3", n: "Justice", v: "Gini ≤ 0.35", c: GR },
            { id: "I-4", n: "Sovereignty", v: "Keys LOCAL", c: PU },
            { id: "I-5", n: "Proof", v: "Hash-chained", c: CY },
          ].map((item) => (
            <div key={item.id} style={{ padding: 16, borderRadius: 16, background: "rgba(255,255,255,.025)", border: `1px solid ${LINE}` }}>
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: item.c, letterSpacing: 2, marginBottom: 8 }}>{item.id}</div>
              <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>{item.n}</div>
              <div style={{ fontSize: 11, color: MUT, fontFamily: "'JetBrains Mono', monospace" }}>{item.v}</div>
            </div>
          ))}
        </div>

        <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: G, letterSpacing: 3, marginBottom: 8 }}>SEVEN-LAYER DDAGI STACK</div>
        <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: viewport.isMobile ? 28 : 32, margin: "0 0 24px" }}>Every layer has code. Every layer has tests.</h2>
        <div style={{ display: "flex", flexDirection: "column", gap: 6, marginBottom: 48 }}>
          {layers.map((layer, index) => (
            <div
              key={layer.n}
              style={{
                display: "grid",
                gridTemplateColumns: layerColumns,
                gap: viewport.isMobile ? 6 : 16,
                alignItems: viewport.isMobile ? "flex-start" : "center",
                padding: "12px 16px",
                borderRadius: 12,
                background: "rgba(255,255,255,.02)",
                border: `1px solid ${LINE}`,
              }}
            >
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: layer.col, fontWeight: 600 }}>L{index}</div>
              <div style={{ fontSize: 14, fontWeight: 500 }}>{layer.n}</div>
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: DIM }}>{layer.c}</div>
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: GR, textAlign: viewport.isMobile ? "left" : "right" }}>{layer.t}</div>
            </div>
          ))}
        </div>

        <div style={{ textAlign: "center", padding: "48px 0" }}>
          <div style={{ fontFamily: "Amiri, serif", fontSize: 20, color: `${G}50`, direction: "rtl", marginBottom: 16 }}>كل بذرة تحمل في داخلها مخطط غابة بأكملها</div>
          <button onClick={onEnter} style={{ background: G, color: BG, border: "none", padding: viewport.isMobile ? "14px 28px" : "16px 48px", borderRadius: 6, fontSize: 13, fontFamily: "'JetBrains Mono', monospace", letterSpacing: 3, cursor: "pointer", fontWeight: 600 }}>
            BECOME A NODE
          </button>
          <div style={{ marginTop: 14 }}>
            <button onClick={onReadSeed} style={{ background: "transparent", border: "none", color: G, fontFamily: "'JetBrains Mono', monospace", fontSize: 10, letterSpacing: 2, cursor: "pointer" }}>
              READ THE CONSTITUTIONAL SEED
            </button>
          </div>
          <div style={{ marginTop: 12, fontSize: 11, color: DIM }}>Zero cloud. Zero cost. Your keys. Your sovereignty.</div>
        </div>
      </div>

      <div style={{ borderTop: `1px solid ${LINE}`, padding: viewport.isMobile ? "18px 20px" : "24px 48px", display: "flex", flexDirection: viewport.isMobile ? "column" : "row", gap: 8, justifyContent: "space-between", fontSize: 11, color: DIM }}>
        <span style={{ fontFamily: "Cinzel, serif", letterSpacing: 3, color: G3 }}>BIZRA</span>
        <span style={{ fontFamily: "Amiri, serif" }}>بسم الله الرحمن الرحيم · Dubai</span>
        <span style={{ fontFamily: "'JetBrains Mono', monospace" }}>v0.3.0-GENESIS</span>
      </div>
    </div>
  );
}

function Splash({ onStart }) {
  const [hovered, setHovered] = useState(false);

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", background: BG, fontFamily: "'JetBrains Mono', monospace", position: "relative", overflow: "hidden" }}>
      <div style={{ position: "absolute", width: 400, height: 400, borderRadius: "50%", opacity: 0.06, background: `radial-gradient(circle, ${G}, transparent)`, top: "15%", left: "25%", filter: "blur(60px)" }} />
      <FadeIn d={300}>
        <div style={{ width: 100, height: 100, borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", border: `1.5px solid ${G}18`, boxShadow: `0 0 80px ${G}06`, marginBottom: 32 }}>
          <div style={{ width: 48, height: 48, borderRadius: "50%", background: `radial-gradient(circle, ${G}25, transparent)`, animation: "pulse 3s ease-in-out infinite" }} />
        </div>
      </FadeIn>
      <FadeIn d={700}><div style={{ fontFamily: "Cinzel, serif", color: G, fontSize: 16, letterSpacing: 6, fontWeight: 600 }}>BIZRA</div></FadeIn>
      <FadeIn d={1000}><div style={{ fontSize: 8, color: DIMR, letterSpacing: 4, marginTop: 4 }}>SOVEREIGN AI OPERATING SYSTEM</div></FadeIn>
      <FadeIn d={1400}>
        <div style={{ marginTop: 28, textAlign: "center", padding: "0 24px" }}>
          <div style={{ fontFamily: "Amiri, serif", fontSize: 16, color: `${G}35`, direction: "rtl", marginBottom: 8 }}>بسم الله الرحمن الرحيم</div>
          <div style={{ color: `${TXT}55`, fontSize: 12, lineHeight: 1.9, fontFamily: "'Playfair Display', serif", fontStyle: "italic", maxWidth: 320 }}>
            Every human is a node. Every node is a seed.
            <br />
            Every seed has infinite potential.
          </div>
        </div>
      </FadeIn>
      <FadeIn d={2000}>
        <button
          onClick={onStart}
          onMouseEnter={() => setHovered(true)}
          onMouseLeave={() => setHovered(false)}
          style={{
            marginTop: 36,
            background: hovered ? `${G}0A` : "transparent",
            border: `1px solid ${hovered ? `${G}50` : `${G}20`}`,
            color: G,
            padding: "14px 44px",
            borderRadius: 2,
            fontSize: 10,
            letterSpacing: 5,
            cursor: "pointer",
            fontFamily: "'JetBrains Mono', monospace",
            transition: "all .4s",
            boxShadow: hovered ? `0 0 50px ${G}0A` : "none",
          }}
        >
          INITIALIZE NODE
        </button>
      </FadeIn>
    </div>
  );
}

function Genesis({ onDone }) {
  const [name, setName] = useState("");
  const [phase, setPhase] = useState("input");
  const [lines, setLines] = useState([]);
  const inputRef = useRef(null);

  useEffect(() => {
    const timer = setTimeout(() => inputRef.current?.focus(), 500);
    return () => clearTimeout(timer);
  }, []);

  const go = async () => {
    if (!name.trim()) {
      return;
    }
    setPhase("genesis");
    const bytes = new Uint8Array(16);
    crypto.getRandomValues(bytes);
    const id = Array.from(bytes)
      .map((byte) => byte.toString(16).padStart(2, "0"))
      .join("");
    const steps = [
      ["Generating Ed25519 sovereign keypair...", 400],
      [`Node ID: ${id.slice(0, 20)}...`, 300],
      ["Deriving 12 agent child keys (HD-Ed25519)...", 500],
      ["Loading constitution v5.0.0-GENESIS...", 300],
      ["Covenant: 859649ea...verified ✓", 400],
      ["7 constitutional rights bound.", 300],
      [`Genesis complete. Welcome, ${name.trim()}.`, 500],
    ];

    for (const [text, ms] of steps) {
      await delay(ms);
      setLines((prev) => [...prev, text]);
    }

    await delay(600);
    onDone(name.trim(), id);
  };

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", background: BG, fontFamily: "'JetBrains Mono', monospace", padding: "0 24px" }}>
      <FadeIn d={200}><div style={{ fontFamily: "Cinzel, serif", color: G, fontSize: 11, letterSpacing: 5, marginBottom: 24 }}>IDENTITY GENESIS</div></FadeIn>
      {phase === "input" && (
        <FadeIn d={400}>
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 20, width: "100%", maxWidth: 420 }}>
            <div style={{ color: DIM, fontSize: 12, fontFamily: "'Playfair Display', serif", fontStyle: "italic", textAlign: "center" }}>What shall the network know you as?</div>
            <div style={{ display: "flex", alignItems: "center", gap: 8, width: "100%" }}>
              <span style={{ color: G }}>▸</span>
              <input
                ref={inputRef}
                value={name}
                onChange={(event) => setName(event.target.value)}
                onKeyDown={(event) => event.key === "Enter" && go()}
                placeholder="Your sovereign name"
                style={{ background: "transparent", border: "none", borderBottom: `1px solid ${G}25`, color: TXT, fontSize: 14, fontFamily: "'JetBrains Mono', monospace", padding: "8px 0", width: "100%", outline: "none", letterSpacing: 1 }}
              />
            </div>
            <button onClick={go} disabled={!name.trim()} style={{ marginTop: 4, background: "transparent", border: `1px solid ${name.trim() ? `${G}35` : LINE}`, color: name.trim() ? G : `${TXT}20`, padding: "10px 32px", borderRadius: 2, fontSize: 9, letterSpacing: 4, fontFamily: "'JetBrains Mono', monospace", cursor: name.trim() ? "pointer" : "default" }}>
              GENERATE IDENTITY
            </button>
          </div>
        </FadeIn>
      )}
      {phase !== "input" && (
        <div style={{ maxWidth: 440, width: "100%" }}>
          {lines.map((line, index) => (
            <FadeIn key={`${line}-${index}`} d={index * 60}>
              <div style={{ padding: "3px 0", fontSize: 10, color: line.includes("✓") ? GR : line.includes("Welcome") ? G : "#9CA3AF" }}>
                {line.includes("Welcome") ? <span style={{ fontWeight: 500 }}>{line}</span> : <><span style={{ color: GR, marginRight: 8 }}>✓</span>{line}</>}
              </div>
            </FadeIn>
          ))}
        </div>
      )}
    </div>
  );
}

function TeachSteps({ onDone }) {
  const [step, setStep] = useState(0);
  const [answers, setAnswers] = useState({});
  const [textVal, setTextVal] = useState("");
  const [selected, setSelected] = useState([]);
  const question = TEACH_QUESTIONS[step];
  const total = TEACH_QUESTIONS.length;

  const next = () => {
    const nextAnswers = { ...answers };
    if (question.type === "text") {
      nextAnswers[question.id] = textVal || question.default;
    } else if (question.type === "single") {
      nextAnswers[question.id] = selected[0] || question.default;
    } else {
      nextAnswers[question.id] = selected.length ? selected : [];
    }

    setAnswers(nextAnswers);
    setSelected([]);
    setTextVal("");

    if (step < total - 1) {
      setStep(step + 1);
      return;
    }

    onDone(nextAnswers);
  };

  const toggleOpt = (option) => {
    if (question.type === "single") {
      setSelected([option]);
      return;
    }
    setSelected((prev) => (prev.includes(option) ? prev.filter((entry) => entry !== option) : [...prev, option]));
  };

  const canNext = question.type === "text" || selected.length > 0;

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", background: BG, fontFamily: "'JetBrains Mono', monospace", padding: "24px" }}>
      <FadeIn d={100}>
        <div style={{ display: "flex", gap: 6, marginBottom: 32, flexWrap: "wrap", justifyContent: "center" }}>
          {TEACH_QUESTIONS.map((item, index) => (
            <div key={item.id} style={{ width: index === step ? 32 : 20, height: 3, borderRadius: 99, background: index < step ? GR : index === step ? G : `${TXT}15`, transition: "all .4s" }} />
          ))}
        </div>
      </FadeIn>
      <FadeIn d={200}><div style={{ fontFamily: "Cinzel, serif", color: G, fontSize: 9, letterSpacing: 4, marginBottom: 4 }}>TEACH · STEP {step + 1}/{total}</div></FadeIn>
      <FadeIn d={300} key={question.id}>
        <div style={{ textAlign: "center", maxWidth: 420 }}>
          <div style={{ fontSize: 32, marginBottom: 16 }}>{question.icon}</div>
          <div style={{ fontSize: 14, color: TXT, marginBottom: 6, fontFamily: "'Playfair Display', serif" }}>{question.prompt}</div>
          <div style={{ fontSize: 8, color: DIMR, marginBottom: 24 }}>This configures your PAT-7 agent team</div>

          {question.type === "text" && (
            <input
              value={textVal}
              onChange={(event) => setTextVal(event.target.value)}
              onKeyDown={(event) => event.key === "Enter" && next()}
              placeholder={question.default}
              autoFocus
              style={{ background: "transparent", border: "none", borderBottom: `1px solid ${G}25`, color: TXT, fontSize: 14, fontFamily: "'JetBrains Mono', monospace", padding: "8px 0", width: "100%", outline: "none", textAlign: "center" }}
            />
          )}

          {(question.type === "single" || question.type === "multi") && (
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {question.opts.map((option) => {
                const isSelected = selected.includes(option);
                return (
                  <button
                    key={option}
                    onClick={() => toggleOpt(option)}
                    style={{
                      padding: "10px 16px",
                      borderRadius: 6,
                      background: isSelected ? `${G}12` : "transparent",
                      border: `1px solid ${isSelected ? `${G}40` : LINE}`,
                      color: isSelected ? G : `${TXT}80`,
                      fontSize: 12,
                      fontFamily: "'JetBrains Mono', monospace",
                      cursor: "pointer",
                      transition: "all .2s",
                      textAlign: "left",
                      display: "flex",
                      alignItems: "center",
                      gap: 10,
                    }}
                  >
                    <div style={{ width: 16, height: 16, borderRadius: question.type === "single" ? "50%" : 4, border: `1.5px solid ${isSelected ? G : LINE}`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                      {isSelected && <div style={{ width: 8, height: 8, borderRadius: question.type === "single" ? "50%" : 2, background: G }} />}
                    </div>
                    {option}
                  </button>
                );
              })}
            </div>
          )}

          <button onClick={next} disabled={question.type !== "text" && !canNext} style={{ marginTop: 24, background: canNext || question.type === "text" ? `${G}15` : "transparent", border: `1px solid ${canNext || question.type === "text" ? `${G}40` : LINE}`, color: canNext || question.type === "text" ? G : `${TXT}20`, padding: "10px 36px", borderRadius: 4, fontSize: 10, letterSpacing: 3, fontFamily: "'JetBrains Mono', monospace", cursor: canNext || question.type === "text" ? "pointer" : "default", transition: "all .3s" }}>
            {step === total - 1 ? "CONFIGURE AGENTS" : "NEXT →"}
          </button>
        </div>
      </FadeIn>
    </div>
  );
}

function Assembly({ userName, config, onDone }) {
  const [booted, setBooted] = useState([]);
  const [sat, setSat] = useState(false);
  const [done, setDone] = useState(false);
  const [configLines, setConfigLines] = useState([]);

  useEffect(() => {
    let cancelled = false;

    const run = async () => {
      const pendingLines = [];
      if (config.work_schedule) {
        pendingLines.push(`Schedule: ${config.work_schedule}`);
      }
      if (config.primary_tools?.length) {
        pendingLines.push(`Tools: ${config.primary_tools.join(", ")}`);
      }
      if (config.communication_pref) {
        pendingLines.push(`Comms: ${config.communication_pref}`);
      }
      if (config.priority_domains?.length) {
        pendingLines.push(`Domains: ${config.priority_domains.join(", ")}`);
      }
      if (config.autonomy) {
        pendingLines.push(`Autonomy: ${config.autonomy}`);
      }

      for (const line of pendingLines) {
        await delay(220);
        if (cancelled) {
          return;
        }
        setConfigLines((prev) => [...prev, line]);
      }

      await delay(320);
      for (const id of Object.keys(PAT)) {
        await delay(260);
        if (cancelled) {
          return;
        }
        setBooted((prev) => [...prev, id]);
      }

      await delay(300);
      if (cancelled) {
        return;
      }
      setSat(true);
      await delay(500);
      if (cancelled) {
        return;
      }
      setDone(true);
      await delay(500);
      if (!cancelled) {
        onDone();
      }
    };

    run();
    return () => {
      cancelled = true;
    };
  }, [config, onDone]);

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 20, background: BG, fontFamily: "'JetBrains Mono', monospace", padding: "24px" }}>
      <FadeIn d={100}><div style={{ textAlign: "center" }}><div style={{ fontFamily: "Cinzel, serif", color: G, fontSize: 10, letterSpacing: 5 }}>ASSEMBLING YOUR TEAM</div></div></FadeIn>
      {configLines.length > 0 && (
        <div style={{ minWidth: 0, width: "100%", maxWidth: 420 }}>
          {configLines.map((line, index) => (
            <FadeIn key={`${line}-${index}`} d={index * 50}><div style={{ fontSize: 9, color: CY, padding: "2px 0" }}><span style={{ color: GR, marginRight: 8 }}>⚙</span>{line}</div></FadeIn>
          ))}
        </div>
      )}
      <div style={{ display: "flex", flexDirection: "column", gap: 4, width: "100%", maxWidth: 440 }}>
        {Object.entries(PAT).map(([id, agent], index) => {
          const on = booted.includes(id);
          return (
            <FadeIn key={id} d={100 + index * 60}>
              <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 14px", borderRadius: 6, background: on ? `${agent.col}06` : "transparent", border: `1px solid ${on ? `${agent.col}18` : LINE}`, transition: "all .5s" }}>
                <div style={{ width: 24, height: 24, borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, border: `1px solid ${on ? `${agent.col}30` : LINE}`, color: on ? agent.col : `${TXT}15`, transition: "all .5s" }}>{agent.i}</div>
                <div style={{ flex: 1 }}>
                  <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                    <span style={{ fontSize: 9, fontWeight: 600, letterSpacing: 2, color: on ? agent.col : `${TXT}15`, transition: "color .5s" }}>{agent.c}</span>
                    <span style={{ fontSize: 9, color: on ? "#9CA3AF" : `${TXT}15`, transition: "color .5s" }}>{agent.n}</span>
                  </div>
                  <div style={{ fontSize: 7, marginTop: 1, color: on ? DIM : `${TXT}10`, fontFamily: "'Playfair Display', serif", fontStyle: "italic", transition: "color .5s" }}>{on ? agent.b : "..."}</div>
                </div>
                <div style={{ width: 6, height: 6, borderRadius: "50%", background: on ? agent.col : `${TXT}10`, boxShadow: on ? `0 0 5px ${agent.col}35` : "none", transition: "all .5s" }} />
              </div>
            </FadeIn>
          );
        })}
      </div>
      {sat && (
        <FadeIn>
          <div style={{ padding: "8px 16px", borderRadius: 6, border: `1px solid ${PU}12`, background: `${PU}04` }}>
            <div style={{ fontSize: 7, letterSpacing: 2, color: PU, marginBottom: 4 }}>SAT-5 — ZERO USER CONTROL</div>
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap", justifyContent: "center" }}>
              {SAT.map((sentinel) => (
                <div key={sentinel.n} style={{ textAlign: "center" }}>
                  <div style={{ width: 4, height: 4, borderRadius: "50%", background: sentinel.col, margin: "0 auto 2px", boxShadow: `0 0 3px ${sentinel.col}35` }} />
                  <div style={{ fontSize: 6, color: DIM }}>{sentinel.n}</div>
                </div>
              ))}
            </div>
          </div>
        </FadeIn>
      )}
      {done && <FadeIn><div style={{ color: G, fontSize: 11, fontFamily: "'Playfair Display', serif", fontStyle: "italic", textAlign: "center" }}>Your sovereign AI team is configured and assembled, {userName}.</div></FadeIn>}
    </div>
  );
}

function Dashboard({ userName, config, viewport }) {
  const [tab, setTab] = useState("cmd");
  const commStyle = config?.communication_pref || "Concise bullet points";
  const greeting = commStyle.includes("critical")
    ? `${userName}. Systems nominal.`
    : commStyle.includes("Detailed")
      ? `Good evening, ${userName}. All seven agents are online and reporting nominal status. Your schedule is loaded, domains are configured, and I'm ready for your first mission.`
      : `Good evening, ${userName}. All agents reporting. What shall we work on?`;
  const [messages, setMessages] = useState([{ a: "NEXUS", t: greeting, ty: "greet", ts: Date.now() }]);
  const [input, setInput] = useState("");
  const [running, setRunning] = useState(false);
  const [state, setState] = useState({ seed: 0, bloom: 0, rac: 0, vac: 0, tier: 0, mye: 0, s1: 0, s2: 0, streak: 0, ihsan: 0, reflexes: 0, leg: 0, epic: 0, sov: 0 });
  const [time, setTime] = useState(new Date());
  const feedEndRef = useRef(null);
  const inputRef = useRef(null);
  const tabs = [
    { id: "cmd", l: "COMMAND", i: "▸" },
    { id: "char", l: "CHARACTER", i: "◈" },
    { id: "skill", l: "SKILLS", i: "⬡" },
    { id: "quest", l: "QUESTS", i: "☗" },
    { id: "prog", l: "PROGRESS", i: "↗" },
  ];

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    feedEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const add = useCallback((agent, text, ty = "agent") => {
    setMessages((prev) => [...prev, { a: agent, t: text, ty, ts: Date.now() }].slice(-80));
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => {
      add("ATLAS", "I've prepared your morning brief based on overnight activity.", "pro");
      setTimeout(() => add("ORACLE", `Priority domains loaded: ${(config?.priority_domains || ["Engineering"]).join(", ")}.`, "pro"), 2000);
      setTimeout(() => add("NEXUS", `${SCHEDULED.filter((mission) => !mission.auto).length} scheduled missions pending approval.`, "pro"), 3500);
    }, 6000);
    return () => clearTimeout(timer);
  }, [add, config]);

  const exec = useCallback(async (task) => {
    if (!task.trim() || running) {
      return;
    }

    setRunning(true);
    add("YOU", task, "user");
    await delay(300);
    add("NEXUS", `Analyzing: "${task.slice(0, 50)}${task.length > 50 ? "..." : ""}"`, "work");
    await delay(500);

    const keywords = {
      P1: ["plan", "organize", "strategy", "roadmap", "schedule"],
      P2: ["research", "find", "analyze", "study", "paper"],
      P3: ["code", "build", "test", "fix", "deploy", "debug"],
      P4: ["evaluate", "score", "review", "audit", "benchmark"],
      P5: ["check", "ethics", "compliance", "constitution"],
      P6: ["write", "draft", "report", "document", "publish"],
    };

    let best = "P2";
    let bestScore = 0;
    for (const [agentId, words] of Object.entries(keywords)) {
      const score = words.filter((word) => task.toLowerCase().includes(word)).length;
      if (score > bestScore) {
        best = agentId;
        bestScore = score;
      }
    }

    const agent = PAT[best];
    add("NEXUS", `Routing → ${agent.c}. ${agent.n} match.`, "route");
    await delay(400);
    for (const step of ["Scanning...", "Processing...", "Synthesizing...", "Quality check..."]) {
      add(agent.c, step, "work");
      await delay(400 + Math.random() * 300);
    }

    add("JUDGE", "Quality assessment.", "work");
    await delay(300);
    const ihsan = Number((0.95 + Math.random() * 0.04).toFixed(4));
    add("JUDGE", `Ihsan: ${ihsan}. ${ihsan >= 0.98 ? "Exceptional." : "Above floor."}`, "score");
    await delay(200);
    add("CROWN", "Constitutional scan — invariants hold.", "clear");
    await delay(200);

    const isLegendary = ihsan >= 0.98 && Math.random() > 0.5;
    const isEpic = !isLegendary && ihsan >= 0.96;
    const drop = isLegendary ? "⚡ LEGENDARY" : isEpic ? "💜 EPIC" : "🔵 RARE";
    const multiplier = isLegendary ? 1.5 : isEpic ? 1.3 : 1.15;
    const seedEarned = Number((ihsan * multiplier).toFixed(3));
    const bloomEarned = Number((0.01 * ihsan).toFixed(4));

    add("SYS", `Receipt signed. ${drop} +${seedEarned} SEED`, "mint");
    await delay(150);
    add("HERALD", "Delivered. Chained.", "agent");

    setState((prev) => {
      const next = {
        ...prev,
        seed: Number((prev.seed + seedEarned).toFixed(3)),
        bloom: Number((prev.bloom + bloomEarned).toFixed(4)),
        rac: prev.rac + 1,
        vac: prev.vac + 1,
        streak: prev.streak + 1,
        s2: prev.s2 + 1,
        ihsan,
        leg: prev.leg + (isLegendary ? 1 : 0),
        epic: prev.epic + (isEpic ? 1 : 0),
      };
      if (next.rac >= 100) {
        next.tier = 1;
      }
      if (next.rac >= 500) {
        next.tier = 2;
      }
      next.mye = next.s1 / Math.max(next.s1 + next.s2, 1);
      next.sov = Math.min(1, 0.3 * (next.rac / Math.max(next.vac, 1)) + 0.25 * ihsan + 0.2 * (next.streak / (next.streak + 5)) + 0.15 * 0.8 + 0.1 * (next.reflexes > 0 ? 0.5 : 0));
      return next;
    });

    const compileReflex = (state.rac + 1) % 5 === 0;
    if (compileReflex) {
      setState((prev) => ({ ...prev, reflexes: prev.reflexes + 1 }));
    }

    add("NEXUS", `Complete. +${seedEarned} SEED. ${compileReflex ? "⚡ Reflex compiled!" : `${5 - ((state.rac + 1) % 5)} to compile.`}`, "done");
    setRunning(false);
    setTimeout(() => inputRef.current?.focus(), 100);
    setTimeout(() => {
      const prompts = [[agent.c, "Follow-up available."], ["ORACLE", "Related pattern found."], ["JUDGE", `Ihsan: ${ihsan}.`], ["ATLAS", "Queue updated."]];
      const [promptAgent, promptText] = prompts[Math.floor(Math.random() * prompts.length)];
      add(promptAgent, promptText, "pro");
    }, 3500);
  }, [add, running, state.rac]);

  const stage = getStage(state.sov);
  const nodeValue = Number((state.sov * Math.max(state.rac, 0.01) * (state.ihsan || 0.01) * (1 + Math.log(1 + state.streak) / Math.log(10))).toFixed(2));
  const skillColumns = viewport.isMobile ? "1fr 1fr" : viewport.isTablet ? "1fr 1fr 1fr" : "1fr 1fr 1fr";
  const metricColumns = viewport.isMobile ? "1fr" : "1fr 1fr";

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column", background: BG, color: TXT, fontFamily: "'JetBrains Mono', monospace", fontSize: 11 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: viewport.isMobile ? "flex-start" : "center", flexDirection: viewport.isMobile ? "column" : "row", gap: 8, padding: "8px 16px", borderBottom: `1px solid ${LINE}` }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <span style={{ fontFamily: "Cinzel, serif", color: G, fontSize: 11, letterSpacing: 3, fontWeight: 600 }}>BIZRA</span>
          <span style={{ fontSize: 7, color: DIMR, letterSpacing: 2 }}>NODE0</span>
          <span style={{ fontSize: 7, letterSpacing: 1, color: running ? AM : GR }}>{running ? "● EXECUTING" : "● READY"}</span>
        </div>
        <div style={{ display: "flex", gap: 14, fontSize: 9, flexWrap: "wrap" }}>
          <span style={{ color: GR }}>{state.seed.toFixed(1)} SEED</span>
          <span style={{ color: PU }}>{state.bloom.toFixed(3)} BLOOM</span>
          <span style={{ color: TCOL[state.tier] }}>{TIERS[state.tier]}</span>
          <span style={{ color: G }}>{time.toLocaleTimeString("en", { hour12: false })}</span>
        </div>
      </div>

      <div style={{ display: "flex", gap: 1, padding: "4px 16px", borderBottom: `1px solid ${LINE}08`, overflowX: "auto" }}>
        {Object.values(PAT).map((agent) => (
          <div key={agent.c} style={{ flex: 1, minWidth: 86, textAlign: "center", padding: "2px 0", borderRadius: 2, border: `1px solid ${LINE}` }}>
            <div style={{ fontSize: 7, letterSpacing: 1, fontWeight: 500, color: agent.col }}>{agent.c}</div>
          </div>
        ))}
        <div style={{ width: 1, background: LINE, margin: "0 3px" }} />
        {SAT.map((sentinel) => (
          <div key={sentinel.n} style={{ padding: "2px 2px" }}><div style={{ width: 4, height: 4, borderRadius: "50%", background: `${sentinel.col}50`, margin: "0 auto" }} /></div>
        ))}
      </div>

      <div style={{ display: "flex", padding: "0 16px", borderBottom: `1px solid ${LINE}`, overflowX: "auto" }}>
        {tabs.map((item) => (
          <button key={item.id} onClick={() => setTab(item.id)} style={{ background: "transparent", border: "none", borderBottom: tab === item.id ? `2px solid ${G}` : "2px solid transparent", color: tab === item.id ? G : DIM, padding: "7px 12px", fontSize: 8, letterSpacing: 2, cursor: "pointer", fontFamily: "'JetBrains Mono', monospace", whiteSpace: "nowrap" }}>
            <span style={{ marginRight: 4 }}>{item.i}</span>
            {item.l}
          </button>
        ))}
      </div>

      <div style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }}>
        {tab === "cmd" && (
          <>
            <div style={{ flex: 1, overflowY: "auto", padding: "6px 16px" }}>
              {messages.map((message, index) => {
                const isUser = message.ty === "user";
                const isMint = message.ty === "mint";
                const isProactive = message.ty === "pro";
                const isDone = message.ty === "done";
                const color = isUser ? G : isMint ? GR : isDone ? G : PAT[message.a]?.col || "#6B7280";
                return (
                  <div key={`${message.ts}-${index}`} style={{ display: "flex", gap: 8, alignItems: "flex-start", marginBottom: 1, padding: "1.5px 0", opacity: message.ty === "route" ? 0.45 : isProactive ? 0.65 : 1 }}>
                    <span style={{ fontWeight: 600, minWidth: viewport.isMobile ? 44 : 50, textAlign: "right", fontSize: 9, color }}>{isUser ? "YOU" : message.a}</span>
                    <span style={{ color: isUser ? TXT : isMint ? GR : isDone ? G : isProactive ? color : "#9CA3AF", fontSize: isUser ? 11 : 10, lineHeight: 1.6, fontStyle: isProactive ? "italic" : "normal" }}>{isMint ? `► ${message.t}` : isDone ? `✓ ${message.t}` : message.t}</span>
                  </div>
                );
              })}
              <div ref={feedEndRef} />
            </div>
            {!running && (
              <div style={{ padding: "6px 16px", display: "flex", gap: 6, flexWrap: "wrap", borderTop: `1px solid ${LINE}08` }}>
                {["Research AI safety developments", "Build authentication tests", "Plan quarterly roadmap", "Evaluate deployment quality"].map((mission) => (
                  <button key={mission} onClick={() => exec(mission)} style={{ background: `${TXT}05`, border: `1px solid ${LINE}`, color: DIM, padding: "4px 8px", borderRadius: 2, fontSize: 7, cursor: "pointer", fontFamily: "'JetBrains Mono', monospace" }}>{mission}</button>
                ))}
              </div>
            )}
            <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 16px", borderTop: `1px solid ${G}10` }}>
              <span style={{ color: G }}>▸</span>
              <input
                ref={inputRef}
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    const text = input;
                    setInput("");
                    exec(text);
                  }
                }}
                placeholder={running ? "Executing..." : "Speak your mission..."}
                disabled={running}
                style={{ flex: 1, background: "transparent", border: "none", color: TXT, fontSize: 11, fontFamily: "'JetBrains Mono', monospace", outline: "none", letterSpacing: 0.5 }}
              />
              <div style={{ display: "flex", gap: 8, fontSize: 8, color: DIMR, flexShrink: 0 }}>
                <span>RAC:{state.rac}</span>
                <span>{state.reflexes}⚡</span>
              </div>
            </div>
          </>
        )}

        {tab === "char" && (
          <div style={{ flex: 1, overflowY: "auto", padding: 16 }}>
            <div style={{ padding: 14, borderRadius: 10, border: `1px solid ${G}15`, background: `${G}04`, marginBottom: 12 }}>
              <div style={{ fontSize: 8, letterSpacing: 2, color: G, marginBottom: 4 }}>NODE VALUE</div>
              <div style={{ fontSize: 26, fontWeight: 300, color: G }}>{nodeValue}</div>
            </div>
            <div style={{ padding: 14, borderRadius: 10, border: `1px solid ${LINE}`, background: BG2, marginBottom: 12 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8, gap: 12 }}>
                <div>
                  <div style={{ fontSize: 8, letterSpacing: 2, color: DIM }}>LIFECYCLE</div>
                  <div style={{ fontSize: 14, color: G, fontWeight: 500 }}>{stage.n}</div>
                </div>
                <div style={{ textAlign: "right" }}>
                  <div style={{ fontSize: 8, color: DIM }}>Sovereignty</div>
                  <div style={{ fontSize: 14, color: G }}>{(state.sov * 100).toFixed(1)}%</div>
                </div>
              </div>
              <div style={{ width: "100%", height: 5, borderRadius: 99, background: `${TXT}08` }}>
                <div style={{ height: "100%", borderRadius: 99, background: G, transition: "width .7s", width: `${Math.min(100, stage.h > stage.l ? ((state.sov - stage.l) / (stage.h - stage.l)) * 100 : 100)}%` }} />
              </div>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: metricColumns, gap: 8 }}>
              {[
                { l: "SEED", v: state.seed.toFixed(2), c: GR },
                { l: "BLOOM", v: state.bloom.toFixed(3), c: PU },
                { l: "IHSAN", v: state.ihsan.toFixed(4), c: G },
                { l: "TIER", v: TIERS[state.tier], c: TCOL[state.tier] },
                { l: "MYELINATION", v: `${(state.mye * 100).toFixed(0)}%`, c: BL },
                { l: "STREAK", v: `${state.streak}`, c: YL },
              ].map((metric) => (
                <div key={metric.l} style={{ padding: 10, borderRadius: 8, border: `1px solid ${LINE}`, background: BG2 }}>
                  <div style={{ fontSize: 7, letterSpacing: 2, color: DIM }}>{metric.l}</div>
                  <div style={{ fontSize: 18, fontWeight: 300, color: metric.c }}>{metric.v}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {tab === "skill" && (
          <div style={{ flex: 1, overflowY: "auto", padding: 16 }}>
            <div style={{ fontSize: 8, letterSpacing: 2, color: DIM, marginBottom: 4 }}>HDA SKILLS — {SKILLS.filter((skill) => skill.u).length}/{SKILLS.length}</div>
            <div style={{ fontSize: 7, color: DIMR, marginBottom: 10 }}>8 productized desktop actions from founder-ops-agent manifest</div>
            <div style={{ display: "grid", gridTemplateColumns: skillColumns, gap: 5 }}>
              {SKILLS.map((skill) => {
                const tierColor = TCOL[skill.t];
                return (
                  <div key={skill.id} style={{ padding: 8, borderRadius: 6, border: `1px solid ${skill.u ? `${tierColor}20` : LINE}`, background: skill.u ? `${tierColor}05` : BG2, opacity: skill.u ? 1 : 0.4 }}>
                    <div style={{ display: "flex", justifyContent: "space-between" }}>
                      <span style={{ fontSize: 13 }}>{skill.i}</span>
                      <div style={{ display: "flex", gap: 3, alignItems: "center", flexWrap: "wrap", justifyContent: "flex-end" }}>
                        {skill.hda && <span style={{ fontSize: 5, color: CY, letterSpacing: 1 }}>HDA</span>}
                        <span style={{ fontSize: 6, color: tierColor, letterSpacing: 1 }}>{TIERS[skill.t]}</span>
                      </div>
                    </div>
                    <div style={{ fontSize: 8, marginTop: 3, fontWeight: skill.u ? 500 : 400, color: skill.u ? tierColor : DIM }}>{skill.n}</div>
                    <div style={{ fontSize: 7, marginTop: 2, color: skill.u ? GR : DIMR }}>{skill.u ? "✓" : "🔒"}</div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {tab === "quest" && (
          <div style={{ flex: 1, overflowY: "auto", padding: 16 }}>
            <div style={{ fontSize: 8, letterSpacing: 2, color: DIM, marginBottom: 4 }}>SCHEDULED MISSIONS</div>
            <div style={{ fontSize: 7, color: DIMR, marginBottom: 12 }}>From founder-ops-agent manifest · {config?.work_schedule || "8:00-18:00"}</div>
            {SCHEDULED.map((mission) => (
              <div key={mission.id} style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 12px", marginBottom: 5, borderRadius: 8, border: `1px solid ${LINE}`, background: BG2 }}>
                <span style={{ fontSize: 18 }}>{mission.icon}</span>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 11, fontWeight: 500 }}>{mission.n}</div>
                  <div style={{ fontSize: 8, color: DIM, fontFamily: "'Playfair Display', serif", fontStyle: "italic" }}>{mission.desc}</div>
                  <div style={{ display: "flex", gap: 8, marginTop: 3, flexWrap: "wrap" }}>
                    {mission.agents.map((agentCode) => (
                      <span key={agentCode} style={{ fontSize: 7, color: PAT[Object.keys(PAT).find((key) => PAT[key].c === agentCode)]?.col || DIM, letterSpacing: 1 }}>{agentCode}</span>
                    ))}
                  </div>
                </div>
                <div style={{ textAlign: "right" }}>
                  <div style={{ color: GR, fontSize: 10 }}>+{mission.seed} SEED</div>
                  <div style={{ fontSize: 7, color: DIM, fontFamily: "'JetBrains Mono', monospace" }}>{mission.cron}</div>
                  <div style={{ fontSize: 7, color: mission.auto ? CY : YL, marginTop: 2 }}>{mission.auto ? "AUTO" : "APPROVAL"}</div>
                </div>
              </div>
            ))}
            <div style={{ fontSize: 8, letterSpacing: 2, color: DIM, marginTop: 20, marginBottom: 8 }}>AD-HOC MISSIONS</div>
            {[
              { n: "File Janitor", seed: "0.50", icon: "🧹", desc: "Organize a folder" },
              { n: "Report Generator", seed: "1.00", icon: "📊", desc: "Create report from data" },
              { n: "Build Pipeline", seed: "2.00", icon: "🏗️", desc: "Full CI/CD execution" },
              { n: "Knowledge Crawl", seed: "5.00", icon: "🧠", desc: "Index your digital life" },
            ].map((mission) => (
              <div key={mission.n} style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 12px", marginBottom: 4, borderRadius: 8, border: `1px solid ${LINE}`, background: "transparent", opacity: 0.7 }}>
                <span style={{ fontSize: 16 }}>{mission.icon}</span>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 10, fontWeight: 500 }}>{mission.n}</div>
                  <div style={{ fontSize: 8, color: DIM, fontStyle: "italic" }}>{mission.desc}</div>
                </div>
                <div style={{ color: GR, fontSize: 9 }}>+{mission.seed}</div>
              </div>
            ))}
          </div>
        )}

        {tab === "prog" && (
          <div style={{ flex: 1, overflowY: "auto", padding: 16 }}>
            <div style={{ fontSize: 8, letterSpacing: 2, color: DIM, marginBottom: 10 }}>NODE VALUE FACTORS</div>
            {[
              { l: "Potential", v: state.sov, mx: 1, c: G },
              { l: "Activation", v: state.rac, mx: 10, c: GR },
              { l: "Quality", v: state.ihsan, mx: 1, c: YL },
              { l: "Compounding", v: state.streak * (1 + Math.log(1 + state.streak) / Math.log(10)), mx: 50, c: BL },
              { l: "Synergy", v: 1, mx: 5, c: PU },
            ].map((factor) => (
              <div key={factor.l} style={{ marginBottom: 14 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                  <span style={{ fontSize: 9, color: factor.c }}>{factor.l}</span>
                  <span style={{ fontSize: 9, color: factor.c }}>{factor.v.toFixed(3)}</span>
                </div>
                <div style={{ width: "100%", height: 4, borderRadius: 99, background: `${TXT}08` }}>
                  <div style={{ height: "100%", borderRadius: 99, background: factor.c, transition: "width .5s", width: `${Math.min(100, (factor.v / factor.mx) * 100)}%` }} />
                </div>
              </div>
            ))}
            <div style={{ padding: 14, borderRadius: 10, textAlign: "center", border: `1px solid ${G}15`, background: `${G}04`, marginTop: 8 }}>
              <div style={{ fontSize: 8, letterSpacing: 2, color: DIM }}>COMPOSITE</div>
              <div style={{ fontSize: 30, fontWeight: 300, color: G, marginTop: 4 }}>{nodeValue}</div>
            </div>
            <div style={{ marginTop: 16 }}>
              <div style={{ fontSize: 8, letterSpacing: 2, color: DIM, marginBottom: 8 }}>SEED → CATALYST</div>
              {STAGES.map((item) => {
                const active = state.sov >= item.l;
                const current = stage.n === item.n;
                return (
                  <div key={item.n} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                    <div style={{ width: 18, height: 18, borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 8, background: current ? `${G}12` : active ? `${GR}08` : "transparent", border: `1px solid ${current ? G : active ? `${GR}30` : LINE}`, color: current ? G : active ? GR : DIMR }}>{current ? "◉" : active ? "✓" : "○"}</div>
                    <span style={{ fontSize: 9, color: current ? G : active ? GR : DIM, fontWeight: current ? 600 : 400 }}>{item.n}</span>
                    <span style={{ fontSize: 7, color: DIM }}>{(item.l * 100).toFixed(0)}%</span>
                    {current && <span style={{ fontSize: 7, color: G }}>◄</span>}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", gap: 8, flexDirection: viewport.isMobile ? "column" : "row", padding: "6px 16px", fontSize: 7, letterSpacing: 1, color: DIMR, borderTop: `1px solid ${LINE}` }}>
        <span>{userName.toUpperCase()} · {TIERS[state.tier].toUpperCase()} · {stage.n.toUpperCase()}</span>
        <span>PAT-7 · SAT-5 · 15 ALG · 7 INV · {config?.autonomy?.includes("Full") ? "FULL AUTO" : config?.autonomy?.includes("Ask") ? "MANUAL" : "SEMI-AUTO"}</span>
      </div>
    </div>
  );
}

export default function BizraOSShowcase() {
  const [phase, setPhase] = useState("trust");
  const [userName, setUserName] = useState("");
  const [config, setConfig] = useState({});
  const viewport = useViewport();

  return (
    <div style={{ minHeight: "100vh", background: BG }}>
      <style>{`
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: ${BG}; margin: 0; }
        ::-webkit-scrollbar { width: 3px; }
        ::-webkit-scrollbar-track { background: ${BG}; }
        ::-webkit-scrollbar-thumb { background: ${TXT}15; border-radius: 2px; }
        input::placeholder { color: ${DIMR}; }
        a { color: inherit; text-decoration: none; }
        @keyframes pulse {
          0%, 100% { opacity: .25; transform: scale(1); }
          50% { opacity: .5; transform: scale(1.06); }
        }
      `}</style>
      {phase === "trust" && <TrustSite viewport={viewport} onEnter={() => setPhase("splash")} onReadSeed={() => { window.location.hash = "#/seed"; }} />}
      {phase === "splash" && <Splash onStart={() => setPhase("genesis")} />}
      {phase === "genesis" && <Genesis onDone={(name) => { setUserName(name); setPhase("teach"); }} />}
      {phase === "teach" && <TeachSteps onDone={(nextConfig) => { setConfig(nextConfig); setPhase("assembly"); }} />}
      {phase === "assembly" && <Assembly userName={userName} config={config} onDone={() => setPhase("dashboard")} />}
      {phase === "dashboard" && <Dashboard userName={userName} config={config} viewport={viewport} />}
    </div>
  );
}
