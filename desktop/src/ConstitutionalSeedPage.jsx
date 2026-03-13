import React from "react";

const BG = "#030810";
const BG2 = "#08121f";
const TXT = "#F8F6F1";
const MUT = "rgba(248,246,241,.72)";
const DIM = "rgba(248,246,241,.45)";
const LINE = "rgba(255,255,255,.08)";
const G = "#C9A962";
const BL = "#3b82f6";
const GR = "#22c55e";
const RD = "#ef4444";
const PU = "#a855f7";

const FACTS = [
  {
    title: "Fact 1 — Life",
    subtitle: "You begin. This is certain.",
    body: [
      "Every human being is born. This event is not a theory, not a social construct, not an interpretation. It is the precondition for all other experience.",
      "It carries with it an obligation: to live with excellence, because the gift of beginning demands nothing less.",
    ],
    formula: "P(alive) = 1.0 — axiomatic, non-negotiable, universal",
    color: G,
  },
  {
    title: "Fact 2 — Death",
    subtitle: "You end. This is certain.",
    body: [
      "Every human being dies. This boundary is absolute. It establishes finitude — the constraint that gives all action between birth and death its weight.",
      "You cannot extract value from time you will not live. No economic or technological system can override that boundary.",
    ],
    formula: "P(mortal) = 1.0 — axiomatic, non-negotiable, universal",
    color: BL,
  },
  {
    title: "Fact 3 — The Rule",
    subtitle: "Between life and death, everything is right with a chance to be wrong, and wrong with a chance to be right.",
    body: [
      "No position held between point A and point B is absolutely certain or absolutely impossible. Every word, meaning, theory, belief, and relationship lives in the open interval between zero and one.",
      "This is not relativism. It is the structural property of existence between two certainties.",
    ],
    formula: "∀ claim C where C ∉ {Life, Death}: 0 < P(C) < 1",
    color: PU,
  },
];

const EPOCHS = [
  {
    title: "Epoch I — The Constitution",
    source: "Human agreement",
    description: "Sacred texts, laws, social contracts, spoken oaths. Communities gathered and declared what they held to be true.",
    vulnerability: "Those who controlled the text could rewrite it. The form survived while the substance rotted.",
    status: "Corrupted",
    color: DIM,
  },
  {
    title: "Epoch II — The Algorithm",
    source: "Engagement metrics",
    description: "Truth became what the feed surfaces, what generates clicks, reactions, outrage, and addiction.",
    vulnerability: "ظنّ turned speculation into felt certainty. ربا normalized extraction from futures not yet lived.",
    status: "Killing us slowly",
    color: RD,
  },
  {
    title: "Epoch III — Verified Truth",
    source: "Cryptographic proof",
    description: "Evidence-backed claims, signed with sovereign identity, passed through constitutional gates, and recorded in immutable ledgers.",
    vulnerability: "Its defense is explicit: ZANN_ZERO, RIBA_ZERO, IHSAN_FLOOR.",
    status: "Genesis",
    color: GR,
  },
];

const INVARIANTS = [
  {
    title: "ZANN_ZERO",
    arabic: "لا ظنّ",
    maps: "Maps to Fact 3 — The Rule",
    kills: "Silent Killer 1 — assumption as truth",
    text: "No unverified claim passes the gate. Speculation is marked as speculation. Hallucination is structurally impossible.",
    color: BL,
  },
  {
    title: "RIBA_ZERO",
    arabic: "لا ربا",
    maps: "Maps to Fact 2 — Death",
    kills: "Silent Killer 2 — extraction as wealth",
    text: "Debt, interest, attention extraction, and environmental borrowing violate the boundary established by mortality.",
    color: RD,
  },
  {
    title: "IHSAN_FLOOR",
    arabic: "إحسان ≥ 0.90",
    maps: "Maps to Fact 1 — Life",
    kills: "Constitutional rot",
    text: "The floor is 0.90 because below that threshold the work is not worthy of the gift of beginning. The system would rather go silent than go corrupt.",
    color: G,
  },
];

const gateCode = `pub fn evaluate(
    &mut self,
    payload: &[u8],
    ihsan_score: f64,
    has_evidence: bool,
    contains_riba: bool,
) -> GateReceipt {
    if contains_riba {
        return self.reject(KernelInvariant::RibaZero);
    }
    if !has_evidence {
        return self.reject(KernelInvariant::ZannZero);
    }
    let health = self.watchdog.record(ihsan_score);
    if health == HealthStatus::Degraded {
        return self.reject(KernelInvariant::IhsanFloor);
    }
    self.sign_receipt(payload)
}`;

const genesisJson = `{
  "version": "1.0.0",
  "height": 0,
  "invariants": {
    "riba_zero": true,
    "zann_zero": true,
    "ihsan_floor": 0.90,
    "origin": "البذرة (The Seed) — Ramadan 2023"
  },
  "message": "Epoch 3 gives us truth by verification. This block is its first heartbeat."
}`;

function Section({ kicker, title, children }) {
  return (
    <section style={{ maxWidth: 1080, margin: "0 auto", padding: "0 24px 56px" }}>
      {kicker && <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: G, letterSpacing: 3, marginBottom: 10 }}>{kicker}</div>}
      <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: 34, lineHeight: 1.05, marginBottom: 18 }}>{title}</h2>
      {children}
    </section>
  );
}

function Card({ children, style = {} }) {
  return <div style={{ padding: 20, borderRadius: 18, background: "rgba(255,255,255,.025)", border: `1px solid ${LINE}`, ...style }}>{children}</div>;
}

export default function ConstitutionalSeedPage() {
  return (
    <div style={{ minHeight: "100vh", background: BG, color: TXT, fontFamily: "Inter, system-ui, sans-serif" }}>
      <div style={{ position: "sticky", top: 0, zIndex: 20, display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, padding: "12px 24px", background: "rgba(3,8,16,.92)", backdropFilter: "blur(20px)", borderBottom: `1px solid ${LINE}` }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
          <span style={{ fontFamily: "Cinzel, serif", color: G, fontSize: 14, letterSpacing: 4 }}>البذرة</span>
          <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 8, color: DIM, letterSpacing: 2 }}>THE CONSTITUTIONAL SEED</span>
        </div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "flex-end" }}>
          <button onClick={() => { window.location.hash = "#/landing"; }} style={{ background: "transparent", border: `1px solid ${LINE}`, color: MUT, padding: "8px 14px", borderRadius: 6, fontFamily: "'JetBrains Mono', monospace", fontSize: 10, letterSpacing: 1.5, cursor: "pointer" }}>LANDING</button>
          <button onClick={() => { window.location.hash = "#/app"; }} style={{ background: `${G}12`, border: `1px solid ${G}40`, color: G, padding: "8px 14px", borderRadius: 6, fontFamily: "'JetBrains Mono', monospace", fontSize: 10, letterSpacing: 1.5, cursor: "pointer" }}>COMMAND CENTER</button>
        </div>
      </div>

      <div style={{ position: "relative", padding: "72px 24px 48px", background: "radial-gradient(circle at 20% 20%, rgba(201,169,98,.12), transparent 32%), radial-gradient(circle at 80% 15%, rgba(59,130,246,.08), transparent 28%), linear-gradient(180deg, #07111d, #030810)" }}>
        <div style={{ maxWidth: 1080, margin: "0 auto" }}>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: G, letterSpacing: 3, marginBottom: 10 }}>CONSTITUTIONAL · IMMUTABLE · VERSION 2.0.0</div>
          <h1 style={{ fontFamily: "'Playfair Display', serif", fontSize: 58, lineHeight: 0.95, maxWidth: 860, marginBottom: 20 }}>The Constitutional Seed</h1>
          <div style={{ fontFamily: "Amiri, serif", fontSize: 24, color: `${G}80`, direction: "rtl", marginBottom: 20 }}>بسم الله الرحمن الرحيم</div>
          <p style={{ maxWidth: 820, color: MUT, fontSize: 18, lineHeight: 1.8, marginBottom: 24 }}>
            A formal foundation for the Third Epoch of Human Truth. Not truth by decree. Not truth by engagement. Truth by verification.
          </p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 12 }}>
            {[
              ["Origin", "Ramadan 2023 — البذرة"],
              ["Formalized", "February 2026"],
              ["System", "BIZRA — Blockchain-Integrated Zero-Knowledge Recursive Agents"],
              ["Status", "Awaiting Block 0 Creation"],
            ].map(([label, value]) => (
              <Card key={label} style={{ padding: 16 }}>
                <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: DIM, letterSpacing: 2, marginBottom: 8 }}>{label.toUpperCase()}</div>
                <div style={{ fontSize: 14, color: TXT }}>{value}</div>
              </Card>
            ))}
          </div>
        </div>
      </div>

      <Section kicker="§ 0 — PREAMBLE" title="On the Origin of This Document">
        <Card>
          <p style={{ color: MUT, lineHeight: 1.8, marginBottom: 16 }}>This document was not written in a laboratory. It was discovered in the dark, during the hardest fight a human being can face — the fight with oneself.</p>
          <p style={{ color: MUT, lineHeight: 1.8, marginBottom: 16 }}>In Ramadan 2023, a man tried to understand why his marriage was breaking. The fight was about assumptions. Two people, both certain, both building on sand. Neither could prove who was right because neither had examined what “right” meant.</p>
          <p style={{ color: MUT, lineHeight: 1.8, marginBottom: 16 }}>Instead of insisting on his position, he turned the question inward: not “is she wrong?” but “how do I know I am right? What can I actually stand on? What is a fact?”</p>
          <p style={{ color: MUT, lineHeight: 1.8 }}>He found three things that do not move. Everything in this document follows from them.</p>
        </Card>
      </Section>

      <Section kicker="§ 1 — THE THREE FACTS" title="Axioms of the Constitutional Seed">
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 14 }}>
          {FACTS.map((fact) => (
            <Card key={fact.title}>
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: fact.color, letterSpacing: 2, marginBottom: 8 }}>{fact.title}</div>
              <h3 style={{ fontFamily: "'Playfair Display', serif", fontSize: 24, lineHeight: 1.15, marginBottom: 12 }}>{fact.subtitle}</h3>
              {fact.body.map((paragraph) => (
                <p key={paragraph} style={{ color: MUT, lineHeight: 1.75, marginBottom: 12 }}>{paragraph}</p>
              ))}
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: fact.color, paddingTop: 6 }}>{fact.formula}</div>
            </Card>
          ))}
        </div>
        <div style={{ marginTop: 16, padding: 18, borderRadius: 18, background: `${G}08`, border: `1px solid ${G}18` }}>
          <div style={{ fontFamily: "Amiri, serif", fontSize: 18, color: `${G}70`, direction: "rtl", marginBottom: 8 }}>القاعدة الثالثة</div>
          <div style={{ fontSize: 16, lineHeight: 1.8, color: TXT }}>“My word is right with a chance to be wrong, and my word can be wrong with a chance to be right.”</div>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: DIM, marginTop: 8 }}>— The First Architect, on the Third Fact</div>
        </div>
      </Section>

      <Section kicker="§ 2 — THE THREE EPOCHS OF HUMAN TRUTH" title="How Civilizations Decide What Is Real">
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 14 }}>
          {EPOCHS.map((epoch) => (
            <Card key={epoch.title}>
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: epoch.color, letterSpacing: 2, marginBottom: 8 }}>{epoch.title}</div>
              <div style={{ fontSize: 14, color: G, marginBottom: 10 }}>{epoch.source}</div>
              <p style={{ color: MUT, lineHeight: 1.75, marginBottom: 12 }}>{epoch.description}</p>
              <p style={{ color: DIM, lineHeight: 1.65, marginBottom: 14 }}>Vulnerability: {epoch.vulnerability}</p>
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: epoch.color }}>{epoch.status.toUpperCase()}</div>
            </Card>
          ))}
        </div>
      </Section>

      <Section kicker="§ 3 — THE THREE KERNEL INVARIANTS" title="Constitutional Constraints of the Third Epoch">
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 14 }}>
          {INVARIANTS.map((invariant) => (
            <Card key={invariant.title}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 10, marginBottom: 10, flexWrap: "wrap" }}>
                <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: invariant.color, letterSpacing: 2 }}>{invariant.title}</div>
                <div style={{ fontFamily: "Amiri, serif", fontSize: 18, color: `${invariant.color}CC` }}>{invariant.arabic}</div>
              </div>
              <div style={{ fontSize: 12, color: G, marginBottom: 8 }}>{invariant.maps}</div>
              <p style={{ color: MUT, lineHeight: 1.75, marginBottom: 12 }}>{invariant.text}</p>
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: invariant.color }}>{invariant.kills}</div>
            </Card>
          ))}
        </div>
      </Section>

      <Section kicker="§ 4 — IMPLEMENTATION" title="The Sovereign Kernel">
        <Card style={{ marginBottom: 14 }}>
          <p style={{ color: MUT, lineHeight: 1.8, marginBottom: 14 }}>The Three Facts and Three Invariants are constitutional. They are compiled into the binary, enforced at runtime, and signed by sovereign identity. The GateChain bridges the probabilistic layer and the deterministic layer.</p>
          <pre style={{ margin: 0, padding: 18, borderRadius: 14, overflowX: "auto", background: BG2, border: `1px solid ${LINE}`, color: "#d8e1f1", fontFamily: "'JetBrains Mono', monospace", fontSize: 11, lineHeight: 1.6 }}>{gateCode}</pre>
        </Card>
        <Card>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: G, letterSpacing: 2, marginBottom: 10 }}>BLOCK 0 — GENESIS</div>
          <pre style={{ margin: 0, padding: 18, borderRadius: 14, overflowX: "auto", background: BG2, border: `1px solid ${LINE}`, color: "#d8e1f1", fontFamily: "'JetBrains Mono', monospace", fontSize: 11, lineHeight: 1.6 }}>{genesisJson}</pre>
        </Card>
      </Section>

      <Section kicker="§ 5 — ON THE NAME" title="بذرة — Seed">
        <Card>
          <p style={{ color: MUT, lineHeight: 1.8, marginBottom: 14 }}>Epoch 1 was a tree. Great, but vulnerable to internal rot. Epoch 2 is a fire. Bright, addictive, and consuming. Epoch 3 is a seed. Small, almost invisible, carrying the DNA of what has not yet fully emerged.</p>
          <p style={{ color: MUT, lineHeight: 1.8, marginBottom: 14 }}>You cannot corrupt the seed because it has not yet grown. You cannot extract from it because it has not yet produced fruit. All it contains is three instructions: do not assume, do not extract, do not drop below excellence.</p>
          <p style={{ color: MUT, lineHeight: 1.8, marginBottom: 14 }}>The soil is the first hundred users. The water is verified interaction. The light is the open network.</p>
          <div style={{ padding: 18, borderRadius: 16, background: `${G}08`, border: `1px solid ${G}18` }}>
            <div style={{ fontFamily: "Amiri, serif", fontSize: 22, color: `${G}80`, direction: "rtl", marginBottom: 8 }}>كل بذرة تحمل في داخلها مخطط غابة بأكملها</div>
            <div style={{ color: TXT, fontSize: 16, lineHeight: 1.7 }}>In a world that lost the meaning of the word, where assumption became the source of truth and debt became the source of wealth — a seed of hope.</div>
          </div>
        </Card>
      </Section>

      <Section kicker="FIRST ARCHITECT" title="Block 0 Awaits Genesis">
        <Card style={{ textAlign: "center" }}>
          <div style={{ fontFamily: "'Playfair Display', serif", fontSize: 28, marginBottom: 8 }}>Mumo</div>
          <div style={{ color: G, marginBottom: 12 }}>Founder of BIZRA — بذرة</div>
          <div style={{ color: MUT, lineHeight: 1.8, maxWidth: 720, margin: "0 auto 18px" }}>This is Block 0. This is the first heartbeat of the Third Epoch. Truth regains its weight not by power or popularity, but by proof.</div>
          <div style={{ display: "flex", gap: 10, justifyContent: "center", flexWrap: "wrap" }}>
            <button onClick={() => { window.location.hash = "#/landing"; }} style={{ background: `${G}12`, border: `1px solid ${G}40`, color: G, padding: "12px 18px", borderRadius: 8, fontFamily: "'JetBrains Mono', monospace", fontSize: 10, letterSpacing: 1.5, cursor: "pointer" }}>RETURN TO LANDING</button>
            <button onClick={() => { window.location.hash = "#/app"; }} style={{ background: "transparent", border: `1px solid ${LINE}`, color: MUT, padding: "12px 18px", borderRadius: 8, fontFamily: "'JetBrains Mono', monospace", fontSize: 10, letterSpacing: 1.5, cursor: "pointer" }}>ENTER COMMAND CENTER</button>
          </div>
        </Card>
      </Section>
    </div>
  );
}
