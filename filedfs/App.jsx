// ============================================================
// BIZRA Node0 — Alpha-100 Dashboard (Production)
// ============================================================
// Wired to bizra-node via useNode hook.
// In Tauri: invoke → Rust → Node.execute()
// In browser: simulated bridge (demo mode)
// ============================================================

import { useState, useEffect, useRef, useCallback } from "react";
import { useNode } from "./useNode";
import OnboardingFlow from "./onboarding/OnboardingFlow";
import LandingDemo from "./LandingDemo";

// ── Sacred Geometry ─────────────────────────────────────────

const SeedOfLife = ({ size = 120, opacity = 0.08, color = "#D4A547" }) => (
  <svg width={size} height={size} viewBox="0 0 120 120" style={{ opacity }}>
    {[0, 60, 120, 180, 240, 300].map((a, i) => (
      <circle key={i} cx={60 + 30 * Math.cos((a * Math.PI) / 180)} cy={60 + 30 * Math.sin((a * Math.PI) / 180)} r="30" fill="none" stroke={color} strokeWidth="0.5" />
    ))}
    <circle cx="60" cy="60" r="30" fill="none" stroke={color} strokeWidth="0.5" />
  </svg>
);

// ── Score Gauge with 8 Segments ─────────────────────────────

const GAUGE_SEGMENTS = [
  { key: "fact",         label: "Facts",         color: "#6B9BF7" },
  { key: "preference",   label: "Preferences",   color: "#A78BFA" },
  { key: "goal",         label: "Goals",          color: "#F59E42" },
  { key: "expertise",    label: "Expertise",      color: "#38BDF8" },
  { key: "pattern",      label: "Patterns",       color: "#F0D68A" },
  { key: "relationship", label: "Relationships",  color: "#5BBA6F" },
  { key: "principle",    label: "Principles",     color: "#D4A547" },
  { key: "context",      label: "Context",        color: "#FF6B9D" },
];

const KnowsMeGauge = ({ score, size = 200, traits = [], onPromptClick }) => {
  const cx = size / 2;
  const cy = size / 2;
  const outerR = (size - 24) / 2;
  const innerR = outerR - 22;
  const segmentCount = GAUGE_SEGMENTS.length;
  const gap = 0.03; // radians gap between segments
  const segmentArc = (2 * Math.PI - gap * segmentCount) / segmentCount;

  // Determine which segments are populated from traits
  const populatedKeys = new Set(traits.map((t) => t.label?.toLowerCase()));

  // Find first missing segment for actionable prompt
  const firstMissing = GAUGE_SEGMENTS.find((s) => !populatedKeys.has(s.key));
  const populatedCount = GAUGE_SEGMENTS.filter((s) => populatedKeys.has(s.key)).length;

  // Build arc path for each segment
  const arcPath = (startAngle, endAngle, r) => {
    const x1 = cx + r * Math.cos(startAngle);
    const y1 = cy + r * Math.sin(startAngle);
    const x2 = cx + r * Math.cos(endAngle);
    const y2 = cy + r * Math.sin(endAngle);
    const large = endAngle - startAngle > Math.PI ? 1 : 0;
    return `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`;
  };

  // Progress arc (inner ring)
  const progressC = 2 * Math.PI * innerR;

  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      gap: 8,
    }}>
      {/* Gauge SVG */}
      <div style={{ position: "relative", width: size, height: size }}>
        <svg width={size} height={size}>
          <defs>
            <linearGradient id="ggrad" x1="0%" y1="0%" x2="100%">
              <stop offset="0%" stopColor="#D4A547" />
              <stop offset="50%" stopColor="#F0D68A" />
              <stop offset="100%" stopColor="#D4A547" />
            </linearGradient>
            {/* Glow filter for active segments */}
            <filter id="segGlow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="2" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Outer segments ring */}
          {GAUGE_SEGMENTS.map((seg, i) => {
            const startAngle = -Math.PI / 2 + i * (segmentArc + gap);
            const endAngle = startAngle + segmentArc;
            const isPopulated = populatedKeys.has(seg.key);
            const midAngle = (startAngle + endAngle) / 2;
            const labelR = outerR + 2;

            return (
              <g key={seg.key}>
                {/* Segment arc */}
                <path
                  d={arcPath(startAngle, endAngle, outerR - 6)}
                  fill="none"
                  stroke={isPopulated ? seg.color : "rgba(255,255,255,0.06)"}
                  strokeWidth="10"
                  strokeLinecap="round"
                  style={{
                    transition: "stroke 0.6s ease, opacity 0.6s ease",
                    opacity: isPopulated ? 1 : 0.4,
                    filter: isPopulated ? "url(#segGlow)" : "none",
                  }}
                />
                {/* Segment label (tiny, around the outside) */}
                <text
                  x={cx + labelR * Math.cos(midAngle)}
                  y={cy + labelR * Math.sin(midAngle)}
                  textAnchor="middle"
                  dominantBaseline="central"
                  style={{
                    fontFamily: "var(--mono)",
                    fontSize: 7,
                    fill: isPopulated ? seg.color : "rgba(255,255,255,0.15)",
                    letterSpacing: 0.3,
                    transition: "fill 0.6s ease",
                  }}
                  transform={`rotate(${(midAngle * 180) / Math.PI + 90}, ${cx + labelR * Math.cos(midAngle)}, ${cy + labelR * Math.sin(midAngle)})`}
                >
                  {seg.label}
                </text>
              </g>
            );
          })}

          {/* Inner progress ring — background */}
          <circle
            cx={cx}
            cy={cy}
            r={innerR}
            fill="none"
            stroke="rgba(212,165,71,0.06)"
            strokeWidth="4"
          />
          {/* Inner progress ring — filled */}
          <circle
            cx={cx}
            cy={cy}
            r={innerR}
            fill="none"
            stroke="url(#ggrad)"
            strokeWidth="4"
            strokeDasharray={progressC}
            strokeDashoffset={progressC - score * progressC}
            strokeLinecap="round"
            style={{
              transform: "rotate(-90deg)",
              transformOrigin: `${cx}px ${cy}px`,
              transition: "stroke-dashoffset 1.2s cubic-bezier(0.4,0,0.2,1)",
            }}
          />
        </svg>

        {/* Center text */}
        <div style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
        }}>
          <span style={{
            fontFamily: "var(--mono)",
            fontSize: 32,
            fontWeight: 700,
            color: "#F0D68A",
            letterSpacing: -1,
          }}>
            {(score * 100).toFixed(1)}
          </span>
          <span style={{
            fontFamily: "var(--mono)",
            fontSize: 9,
            color: "rgba(212,165,71,0.6)",
            letterSpacing: 2,
            textTransform: "uppercase",
            marginTop: 1,
          }}>
            knows me
          </span>
          {/* Segment progress fraction */}
          <span style={{
            fontFamily: "var(--mono)",
            fontSize: 8,
            color: "rgba(255,255,255,0.2)",
            marginTop: 4,
          }}>
            {populatedCount}/{segmentCount} areas
          </span>
        </div>
      </div>

      {/* "Your agent knows X% of you" */}
      <div style={{
        fontFamily: "var(--sans)",
        fontSize: 11,
        color: "rgba(255,255,255,0.4)",
        textAlign: "center",
        lineHeight: 1.4,
      }}>
        Your agent knows{" "}
        <span style={{ color: "#F0D68A", fontWeight: 600 }}>
          {(score * 100).toFixed(0)}%
        </span>{" "}
        of you
      </div>

      {/* Actionable prompt for first missing section */}
      {firstMissing && (
        <button
          onClick={() => onPromptClick && onPromptClick(firstMissing.key)}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            padding: "6px 12px",
            background: `${firstMissing.color}0A`,
            border: `1px solid ${firstMissing.color}25`,
            borderRadius: 8,
            cursor: "pointer",
            transition: "all 0.2s ease",
            maxWidth: "100%",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = `${firstMissing.color}15`;
            e.currentTarget.style.borderColor = `${firstMissing.color}40`;
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = `${firstMissing.color}0A`;
            e.currentTarget.style.borderColor = `${firstMissing.color}25`;
          }}
        >
          <svg width="12" height="12" viewBox="0 0 12 12" style={{ flexShrink: 0 }}>
            <circle cx="6" cy="6" r="5" fill="none" stroke={firstMissing.color} strokeWidth="1" opacity="0.5" />
            <path d="M6 3.5v5M3.5 6h5" stroke={firstMissing.color} strokeWidth="1" strokeLinecap="round" />
          </svg>
          <span style={{
            fontFamily: "var(--sans)",
            fontSize: 10,
            color: firstMissing.color,
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
          }}>
            Tell me about your {firstMissing.label.toLowerCase()}
          </span>
        </button>
      )}
    </div>
  );
};

// ── Ihsan Bar ───────────────────────────────────────────────

const IhsanBar = ({ score }) => {
  const pct = (score / 10000) * 100;
  const color = score >= 9500 ? "#5BBA6F" : score >= 8000 ? "#D4A547" : "#E85D4A";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "rgba(255,255,255,0.4)", width: 50, letterSpacing: 1 }}>إحسان</span>
      <div style={{ flex: 1, height: 4, background: "rgba(255,255,255,0.06)", borderRadius: 2, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 2, transition: "width 0.6s ease" }} />
      </div>
      <span style={{ fontFamily: "var(--mono)", fontSize: 11, color, width: 50, textAlign: "right" }}>{(score / 100).toFixed(1)}%</span>
    </div>
  );
};

// ── Agent Badge ─────────────────────────────────────────────

const AGENT_COLORS = {
  Navigator: "#6B9BF7", Scholar: "#A78BFA", Artisan: "#F59E42",
  Guardian: "#E85D4A", Mentor: "#5BBA6F", Diplomat: "#38BDF8", Oracle: "#F0D68A",
};
const ALL_AGENTS = Object.keys(AGENT_COLORS);

const AgentBadge = ({ name, active }) => {
  const c = AGENT_COLORS[name] || "#888";
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 6, padding: "4px 10px",
      background: active ? `${c}18` : "rgba(255,255,255,0.02)",
      border: `1px solid ${active ? c + "40" : "rgba(255,255,255,0.05)"}`,
      borderRadius: 6, transition: "all 0.3s ease",
    }}>
      <div style={{ width: 6, height: 6, borderRadius: "50%", background: active ? c : "rgba(255,255,255,0.15)", boxShadow: active ? `0 0 8px ${c}60` : "none" }} />
      <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: active ? c : "rgba(255,255,255,0.3)", letterSpacing: 0.5 }}>{name}</span>
    </div>
  );
};

// ── Message Bubble ──────────────────────────────────────────

const Bubble = ({ role, content, meta }) => {
  const isUser = role === "user";
  const [discOpen, setDiscOpen] = useState(false);
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: isUser ? "flex-end" : "flex-start", marginBottom: 12, maxWidth: "85%", alignSelf: isUser ? "flex-end" : "flex-start" }}>
      <div style={{
        padding: "10px 14px",
        background: isUser ? "rgba(212,165,71,0.12)" : "rgba(255,255,255,0.04)",
        border: `1px solid ${isUser ? "rgba(212,165,71,0.2)" : "rgba(255,255,255,0.06)"}`,
        borderRadius: isUser ? "14px 14px 4px 14px" : "14px 14px 14px 4px",
        color: "rgba(255,255,255,0.88)", fontFamily: "var(--sans)", fontSize: 13.5, lineHeight: 1.55,
      }}>{content}</div>
      {meta && (
        <div style={{ display: "flex", flexDirection: "column", gap: 2, marginTop: 4, padding: "0 4px", width: "100%" }}>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
            {meta.agents && <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: "rgba(212,165,71,0.4)" }}>{meta.agents} agents</span>}
            {meta.fragments > 0 && <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: "rgba(91,186,111,0.5)" }}>+{meta.fragments} learned</span>}
            {meta.confidence && <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: "rgba(255,255,255,0.25)" }}>{(parseFloat(meta.confidence) * 100).toFixed(0)}% conf</span>}
            {meta.ihsanScore && <SAPBadge ihsanScore={meta.ihsanScore} sessionActive={true} />}
            {meta.receiptHash && <span style={{ fontFamily: "var(--mono)", fontSize: 8, color: "rgba(107,155,247,0.3)" }} title={`Receipt: ${meta.receiptHash}`}>#{meta.receiptHash.slice(0, 8)}</span>}
          </div>
          {meta.disclosure && (
            <DisclosurePanel disclosure={meta.disclosure} collapsed={!discOpen} onToggle={() => setDiscOpen((o) => !o)} />
          )}
        </div>
      )}
    </div>
  );
};

// ── Trait Pill ───────────────────────────────────────────────

const TraitPill = ({ label, value, confidence }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 10px", background: "rgba(212,165,71,0.06)", border: "1px solid rgba(212,165,71,0.12)", borderRadius: 8 }}>
    <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: "rgba(212,165,71,0.5)", textTransform: "uppercase", letterSpacing: 0.8 }}>{label}</span>
    <span style={{ fontFamily: "var(--sans)", fontSize: 12, color: "rgba(255,255,255,0.75)", flex: 1 }}>{value}</span>
    <div style={{ width: 20, height: 3, background: "rgba(255,255,255,0.06)", borderRadius: 2, overflow: "hidden" }}>
      <div style={{ width: `${(confidence / 10000) * 100}%`, height: "100%", background: "#D4A547", borderRadius: 2 }} />
    </div>
  </div>
);

// ── Metric ──────────────────────────────────────────────────

const Metric = ({ label, value }) => (
  <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
    <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: "rgba(255,255,255,0.3)", letterSpacing: 1.2, textTransform: "uppercase" }}>{label}</span>
    <span style={{ fontFamily: "var(--mono)", fontSize: 20, fontWeight: 600, color: "rgba(255,255,255,0.85)" }}>{value}</span>
  </div>
);

// ── Section Header ──────────────────────────────────────────

const SectionLabel = ({ children, extra }) => (
  <div style={{ fontFamily: "var(--mono)", fontSize: 9, color: "rgba(255,255,255,0.25)", letterSpacing: 1.5, marginBottom: 10, textTransform: "uppercase" }}>
    {children}
    {extra && <span style={{ color: "rgba(212,165,71,0.4)", marginLeft: 6 }}>{extra}</span>}
  </div>
);

// ── SAP v0 Disclosure Badge ─────────────────────────────────

const SAPBadge = ({ ihsanScore, sessionActive }) => {
  const score = parseFloat(ihsanScore || "0");
  const color = score >= 0.95 ? "#5BBA6F" : score >= 0.90 ? "#D4A547" : "#E85D4A";
  const label = score >= 0.95 ? "SAP v0 Conformant" : "SAP v0 Warning";
  return (
    <div style={{
      display: "inline-flex", alignItems: "center", gap: 6, padding: "3px 8px",
      background: `${color}12`, border: `1px solid ${color}30`, borderRadius: 4,
    }}>
      <div style={{ width: 5, height: 5, borderRadius: "50%", background: color, boxShadow: sessionActive ? `0 0 6px ${color}60` : "none" }} />
      <span style={{ fontFamily: "var(--mono)", fontSize: 9, color, letterSpacing: 0.5 }}>{label}</span>
      <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: "rgba(255,255,255,0.3)" }}>{(score * 100).toFixed(1)}%</span>
    </div>
  );
};

const DisclosurePanel = ({ disclosure, collapsed, onToggle }) => {
  let data = null;
  try { data = typeof disclosure === "string" ? JSON.parse(disclosure) : disclosure; } catch { return null; }
  if (!data) return null;

  return (
    <div style={{
      margin: "4px 0 8px", padding: collapsed ? "4px 8px" : "8px 10px",
      background: "rgba(91,186,111,0.04)", border: "1px solid rgba(91,186,111,0.12)", borderRadius: 6,
      cursor: "pointer", transition: "all 0.2s ease",
    }} onClick={onToggle}>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: "rgba(91,186,111,0.6)", letterSpacing: 1, textTransform: "uppercase" }}>
          Disclosure {collapsed ? "+" : "-"}
        </span>
      </div>
      {!collapsed && (
        <div style={{ marginTop: 6 }}>
          {data.claims && data.claims.length > 0 && (
            <div style={{ marginBottom: 4 }}>
              <span style={{ fontFamily: "var(--mono)", fontSize: 8, color: "rgba(255,255,255,0.3)", letterSpacing: 0.8 }}>CLAIMS</span>
              {data.claims.map((c, i) => (
                <div key={i} style={{ fontFamily: "var(--sans)", fontSize: 11, color: "rgba(255,255,255,0.6)", paddingLeft: 8, lineHeight: 1.4 }}>- {c}</div>
              ))}
            </div>
          )}
          {data.uncertainty && data.uncertainty.length > 0 && (
            <div style={{ marginBottom: 4 }}>
              <span style={{ fontFamily: "var(--mono)", fontSize: 8, color: "rgba(232,93,74,0.5)", letterSpacing: 0.8 }}>UNCERTAINTY</span>
              {data.uncertainty.map((u, i) => (
                <div key={i} style={{ fontFamily: "var(--sans)", fontSize: 11, color: "rgba(232,93,74,0.5)", paddingLeft: 8, lineHeight: 1.4 }}>- {u}</div>
              ))}
            </div>
          )}
          {data.source_refs && data.source_refs.length > 0 && (
            <div>
              <span style={{ fontFamily: "var(--mono)", fontSize: 8, color: "rgba(255,255,255,0.2)", letterSpacing: 0.8 }}>SOURCES</span>
              {data.source_refs.map((s, i) => (
                <div key={i} style={{ fontFamily: "var(--mono)", fontSize: 10, color: "rgba(107,155,247,0.5)", paddingLeft: 8 }}>{typeof s === "string" ? s : s.uri || s.ref_id}</div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// ── Quality Tier System ───────────────────────────────────────

const QUALITY_TIERS = [
  { key: "seed", label: "Seed", icon: "\u{1F331}", color: "#8B7355", minAtoms: 0, minMsgs: 0 },
  { key: "sprout", label: "Sprout", icon: "\u{1F33F}", color: "#4CAF50", minAtoms: 1, minMsgs: 0 },
  { key: "growing", label: "Growing", icon: "\u{1F333}", color: "#2196F3", minAtoms: 25, minMsgs: 10 },
  { key: "rooted", label: "Rooted", icon: "\u{1F332}", color: "#9C27B0", minAtoms: 100, minMsgs: 0 },
  { key: "flourishing", label: "Flourishing", icon: "\u{1F31F}", color: "#FFD700", minAtoms: 200, minMsgs: 0 },
];

const TIER_CAPABILITIES = {
  seed: ["Chat", "TEACH"],
  sprout: ["Memory recall", "Bootstrap reflexes"],
  growing: ["Reflex compilation", "Action Bus (ToolCall)"],
  rooted: ["Desktop actions (AHK)", "Token economy"],
  flourishing: ["Full Action Bus", "Agent-as-Service"],
};

const TIER_DESCRIPTIONS = {
  seed: "Begin your journey: chat and teach your node about yourself.",
  sprout: "Your node remembers you across sessions and starts building reflexes.",
  growing: "Reflexes compile into actions. Your node can call tools on your behalf.",
  rooted: "Desktop automation unlocked. Your node participates in the token economy.",
  flourishing: "Full sovereignty. Your node can serve as an agent for others.",
};

function determineTier(nodeData) {
  const atoms = nodeData?.fragments || 0;
  const msgs = nodeData?.messages || 0;
  if (atoms >= 200) return QUALITY_TIERS[4];
  if (atoms >= 100) return QUALITY_TIERS[3];
  if (atoms >= 25 && msgs >= 10) return QUALITY_TIERS[2];
  if (atoms >= 1) return QUALITY_TIERS[1];
  return QUALITY_TIERS[0];
}

function getNextTierHint(currentTier, nodeData) {
  const atoms = nodeData?.fragments || 0;
  const msgs = nodeData?.messages || 0;
  switch (currentTier.key) {
    case "seed":
      return { next: QUALITY_TIERS[1], hint: "Teach 1 fact to begin" };
    case "sprout":
      if (atoms < 25 && msgs < 10) return { next: QUALITY_TIERS[2], hint: `Teach ${25 - atoms} more facts and send ${10 - msgs} more messages` };
      if (atoms < 25) return { next: QUALITY_TIERS[2], hint: `Teach ${25 - atoms} more facts` };
      return { next: QUALITY_TIERS[2], hint: `Send ${10 - msgs} more messages` };
    case "growing":
      return { next: QUALITY_TIERS[3], hint: `Teach ${100 - atoms} more facts to unlock desktop actions` };
    case "rooted":
      return { next: QUALITY_TIERS[4], hint: `Teach ${200 - atoms} more facts across multiple providers` };
    case "flourishing":
      return null;
    default:
      return null;
  }
}

const QualityTierBadge = ({ nodeData }) => {
  const tier = determineTier(nodeData);
  const nextInfo = getNextTierHint(tier, nodeData);
  const isFlourishing = tier.key === "flourishing";
  const description = TIER_DESCRIPTIONS[tier.key] || "";

  return (
    <div style={{
      width: "100%",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      gap: 6,
    }}>
      {/* Badge pill */}
      <div style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 8,
        padding: "6px 14px",
        background: `${tier.color}14`,
        border: `1px solid ${tier.color}30`,
        borderRadius: 20,
        boxShadow: isFlourishing ? `0 0 16px ${tier.color}20` : "none",
      }}>
        <span style={{ fontSize: 16, lineHeight: 1 }}>{tier.icon}</span>
        <span style={{
          fontFamily: "var(--mono)",
          fontSize: 11,
          fontWeight: 600,
          color: tier.color,
          letterSpacing: 0.5,
        }}>
          {tier.label}
        </span>
      </div>

      {/* Tier description */}
      <div style={{
        fontFamily: "var(--sans)",
        fontSize: 10,
        color: "rgba(255,255,255,0.35)",
        textAlign: "center",
        lineHeight: 1.4,
        maxWidth: "90%",
      }}>
        {description}
      </div>

      {/* Unlocked capabilities */}
      <div style={{
        width: "100%",
        display: "flex",
        flexWrap: "wrap",
        justifyContent: "center",
        gap: 4,
      }}>
        {(TIER_CAPABILITIES[tier.key] || []).map((cap) => (
          <span key={cap} style={{
            fontFamily: "var(--mono)",
            fontSize: 8,
            color: tier.color,
            background: `${tier.color}10`,
            border: `1px solid ${tier.color}20`,
            borderRadius: 4,
            padding: "2px 6px",
            letterSpacing: 0.3,
          }}>
            {cap}
          </span>
        ))}
      </div>

      {/* Next tier hint */}
      {nextInfo && (
        <div style={{
          fontFamily: "var(--mono)",
          fontSize: 9,
          color: "rgba(255,255,255,0.25)",
          textAlign: "center",
          lineHeight: 1.4,
        }}>
          Next: <span style={{ color: nextInfo.next.color }}>{nextInfo.next.icon} {nextInfo.next.label}</span>
          {" \u2014 "}
          <span style={{ color: "rgba(255,255,255,0.35)" }}>{nextInfo.hint}</span>
        </div>
      )}
    </div>
  );
};

const GrowthRoadmap = ({ nodeData }) => {
  const [expanded, setExpanded] = useState(false);
  const currentTier = determineTier(nodeData);
  const currentIdx = QUALITY_TIERS.findIndex((t) => t.key === currentTier.key);

  return (
    <div style={{
      width: "100%",
      background: "rgba(255,255,255,0.02)",
      border: "1px solid rgba(255,255,255,0.04)",
      borderRadius: 8,
      overflow: "hidden",
    }}>
      {/* Toggle header */}
      <button
        onClick={() => setExpanded((e) => !e)}
        style={{
          width: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "8px 12px",
          background: "none",
          border: "none",
          cursor: "pointer",
        }}
      >
        <span style={{
          fontFamily: "var(--mono)",
          fontSize: 9,
          color: "rgba(255,255,255,0.25)",
          letterSpacing: 1.5,
          textTransform: "uppercase",
        }}>
          Growth Roadmap
        </span>
        <span style={{
          fontFamily: "var(--mono)",
          fontSize: 10,
          color: "rgba(255,255,255,0.2)",
          transform: expanded ? "rotate(180deg)" : "rotate(0deg)",
          transition: "transform 0.2s ease",
        }}>
          V
        </span>
      </button>

      {/* Timeline */}
      {expanded && (
        <div style={{ padding: "0 12px 12px", display: "flex", flexDirection: "column", gap: 0 }}>
          {QUALITY_TIERS.map((tier, idx) => {
            const isActive = idx === currentIdx;
            const isPast = idx < currentIdx;
            const isFuture = idx > currentIdx;
            const capabilities = TIER_CAPABILITIES[tier.key] || [];

            return (
              <div key={tier.key} style={{ display: "flex", gap: 10 }}>
                {/* Timeline stem */}
                <div style={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  width: 16,
                  flexShrink: 0,
                }}>
                  {/* Dot */}
                  <div style={{
                    width: isActive ? 12 : 8,
                    height: isActive ? 12 : 8,
                    borderRadius: "50%",
                    background: isPast || isActive ? tier.color : "rgba(255,255,255,0.08)",
                    border: isActive ? `2px solid ${tier.color}` : "none",
                    boxShadow: isActive ? `0 0 10px ${tier.color}40` : "none",
                    flexShrink: 0,
                    marginTop: 2,
                  }} />
                  {/* Line */}
                  {idx < QUALITY_TIERS.length - 1 && (
                    <div style={{
                      width: 1,
                      flex: 1,
                      minHeight: 20,
                      background: isPast ? "rgba(212,165,71,0.2)" : "rgba(255,255,255,0.04)",
                    }} />
                  )}
                </div>

                {/* Content */}
                <div style={{
                  paddingBottom: idx < QUALITY_TIERS.length - 1 ? 10 : 0,
                  flex: 1,
                }}>
                  <div style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                  }}>
                    <span style={{ fontSize: 12 }}>{tier.icon}</span>
                    <span style={{
                      fontFamily: "var(--mono)",
                      fontSize: 10,
                      fontWeight: isActive ? 700 : 400,
                      color: isActive ? tier.color : isFuture ? "rgba(255,255,255,0.2)" : "rgba(255,255,255,0.45)",
                      letterSpacing: 0.3,
                    }}>
                      {tier.label}
                    </span>
                    {isActive && (
                      <span style={{
                        fontFamily: "var(--mono)",
                        fontSize: 7,
                        color: tier.color,
                        background: `${tier.color}18`,
                        padding: "1px 5px",
                        borderRadius: 3,
                        letterSpacing: 0.5,
                        textTransform: "uppercase",
                      }}>
                        Current
                      </span>
                    )}
                  </div>
                  <div style={{
                    fontFamily: "var(--sans)",
                    fontSize: 10,
                    color: isFuture ? "rgba(255,255,255,0.12)" : "rgba(255,255,255,0.3)",
                    marginTop: 2,
                    lineHeight: 1.4,
                  }}>
                    {idx === 0
                      ? capabilities.join(" + ")
                      : "+ " + capabilities.join(", + ")}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

// ── Sovereign Agent Card ──────────────────────────────────────

const SovereignAgentCard = ({ agentData }) => {
  if (!agentData) return null;
  const compilation = agentData.compilation || {};
  return (
    <div style={{
      padding: "10px 12px", background: "rgba(212,165,71,0.04)",
      border: "1px solid rgba(212,165,71,0.12)", borderRadius: 8,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
        <div style={{ width: 18, height: 18, borderRadius: 4, background: "linear-gradient(135deg, #D4A547, #8B6914)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9, fontWeight: 700, color: "#0A0B0F", fontFamily: "var(--mono)" }}>S</div>
        <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "rgba(212,165,71,0.8)", letterSpacing: 0.5, fontWeight: 600 }}>SovereignAgentCard</span>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
        {agentData.agent_id && <CardRow label="agent" value={agentData.agent_id} />}
        {agentData.role && <CardRow label="role" value={agentData.role} />}
        {agentData.version && <CardRow label="version" value={agentData.version} />}
        {compilation.compiled_reflex_count != null && <CardRow label="reflexes" value={compilation.compiled_reflex_count} />}
        {compilation.ihsan_threshold != null && <CardRow label="ihsan gate" value={`${(compilation.ihsan_threshold * 100).toFixed(1)}%`} />}
        {compilation.compilation_coverage != null && <CardRow label="coverage" value={`${(compilation.compilation_coverage * 100).toFixed(1)}%`} />}
        {agentData.policy_hash && <CardRow label="policy" value={agentData.policy_hash.slice(0, 16) + "..."} />}
      </div>
    </div>
  );
};

const CardRow = ({ label, value }) => (
  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
    <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: "rgba(255,255,255,0.3)", letterSpacing: 0.6, textTransform: "uppercase" }}>{label}</span>
    <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "rgba(255,255,255,0.6)" }}>{value}</span>
  </div>
);

// ── MissionApprovalCard — Human-in-the-loop for proactive missions ──
// Renders when a scheduled mission wants to execute.
// "Node0 wants to run: Morning Brief" [Approve] [Skip] [Modify]
// Standing on Giants: Boyd (OODA decide phase — human confirms)

const MissionApprovalCard = ({ mission, onApprove, onSkip, onModify }) => {
  if (!mission) return null;

  const agentList = (mission.agents || []).join(", ");

  return (
    <div style={{
      background: "rgba(212,165,71,0.06)",
      border: "1px solid rgba(212,165,71,0.3)",
      borderRadius: 10,
      padding: "12px 14px",
      margin: "8px 0",
      fontFamily: "var(--mono)",
      maxWidth: 420,
      alignSelf: "center",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <div style={{
          width: 8, height: 8, borderRadius: "50%",
          background: "#D4A547",
          animation: "pulse 2s infinite",
        }} />
        <span style={{ fontSize: 11, color: "rgba(212,165,71,0.9)", fontWeight: 600 }}>
          Node0 wants to run: {mission.name || "Mission"}
        </span>
      </div>

      <div style={{ fontSize: 10, color: "rgba(255,255,255,0.5)", marginBottom: 6, lineHeight: 1.5 }}>
        {mission.description}
      </div>

      {agentList && (
        <div style={{ fontSize: 9, color: "rgba(255,255,255,0.3)", marginBottom: 8 }}>
          Agents: {agentList}
        </div>
      )}

      {mission.includes && mission.includes.length > 0 && (
        <div style={{ fontSize: 9, color: "rgba(255,255,255,0.25)", marginBottom: 10 }}>
          Includes: {mission.includes.join(", ")}
        </div>
      )}

      <div style={{ display: "flex", gap: 8 }}>
        <button
          onClick={() => onApprove && onApprove(mission)}
          style={{
            flex: 1,
            background: "rgba(76,175,80,0.15)",
            border: "1px solid rgba(76,175,80,0.4)",
            color: "#4CAF50",
            borderRadius: 6,
            padding: "7px 10px",
            fontFamily: "var(--mono)",
            fontSize: 10,
            cursor: "pointer",
            fontWeight: 600,
          }}
        >
          Approve
        </button>
        <button
          onClick={() => onSkip && onSkip(mission)}
          style={{
            flex: 1,
            background: "rgba(255,255,255,0.04)",
            border: "1px solid rgba(255,255,255,0.12)",
            color: "rgba(255,255,255,0.5)",
            borderRadius: 6,
            padding: "7px 10px",
            fontFamily: "var(--mono)",
            fontSize: 10,
            cursor: "pointer",
          }}
        >
          Skip
        </button>
        <button
          onClick={() => onModify && onModify(mission)}
          style={{
            flex: 1,
            background: "rgba(212,165,71,0.1)",
            border: "1px solid rgba(212,165,71,0.25)",
            color: "#D4A547",
            borderRadius: 6,
            padding: "7px 10px",
            fontFamily: "var(--mono)",
            fontSize: 10,
            cursor: "pointer",
          }}
        >
          Modify
        </button>
      </div>

      <div style={{ marginTop: 8, fontSize: 8, color: "rgba(255,255,255,0.15)", textAlign: "right" }}>
        Scheduled: {mission.schedule || "now"}
      </div>
    </div>
  );
};

// ── ActionProofCard — cryptographic proof of action execution ────────
// Renders in chat pane when an action receipt is present.
// Shows: action name, outcome_hash, pre/post hashes, confidence, status.
// Standing on Giants: General Magic (Telescript permits, 1994)

const ActionProofCard = ({ receipt }) => {
  if (!receipt) return null;

  const confirmed = receipt.outcome_confirmed;
  const statusColor = confirmed ? "rgba(76,175,80,0.9)" : "rgba(255,152,0,0.9)";
  const statusLabel = confirmed ? "VERIFIED" : "UNVERIFIED";
  const hashShort = (h) => h ? `${h.slice(0, 8)}...${h.slice(-4)}` : "\u2014";

  return (
    <div style={{
      background: "rgba(0,0,0,0.3)",
      border: `1px solid ${confirmed ? "rgba(76,175,80,0.3)" : "rgba(255,152,0,0.3)"}`,
      borderRadius: 8,
      padding: "10px 12px",
      margin: "6px 0",
      fontFamily: "var(--mono)",
      fontSize: 10,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <span style={{ color: "rgba(212,165,71,0.9)", fontSize: 11, fontWeight: 600 }}>
          Action Receipt
        </span>
        <span style={{
          color: statusColor,
          fontSize: 9,
          padding: "2px 6px",
          border: `1px solid ${statusColor}`,
          borderRadius: 4,
          letterSpacing: 0.8,
        }}>
          {statusLabel}
        </span>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <span style={{ color: "rgba(255,255,255,0.35)" }}>action_id</span>
          <span style={{ color: "rgba(255,255,255,0.6)" }}>{receipt.action_id || "\u2014"}</span>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <span style={{ color: "rgba(255,255,255,0.35)" }}>outcome_hash</span>
          <span style={{ color: "rgba(212,165,71,0.8)" }}>{hashShort(receipt.outcome_hash)}</span>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <span style={{ color: "rgba(255,255,255,0.35)" }}>pre_hash</span>
          <span style={{ color: "rgba(255,255,255,0.4)" }}>{hashShort(receipt.pre_hash)}</span>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <span style={{ color: "rgba(255,255,255,0.35)" }}>post_hash</span>
          <span style={{ color: "rgba(255,255,255,0.4)" }}>{hashShort(receipt.post_hash)}</span>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <span style={{ color: "rgba(255,255,255,0.35)" }}>state_changed</span>
          <span style={{ color: receipt.state_changed ? "rgba(76,175,80,0.7)" : "rgba(255,255,255,0.4)" }}>
            {receipt.state_changed ? "yes" : "no"}
          </span>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <span style={{ color: "rgba(255,255,255,0.35)" }}>confidence</span>
          <span style={{ color: (receipt.confidence || 0) >= 0.9 ? "rgba(76,175,80,0.8)" : "rgba(255,152,0,0.8)" }}>
            {((receipt.confidence || 0) * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      <div style={{
        marginTop: 8,
        paddingTop: 6,
        borderTop: "1px solid rgba(255,255,255,0.06)",
        fontSize: 8,
        color: "rgba(255,255,255,0.2)",
        textAlign: "right",
      }}>
        {receipt.timestamp ? new Date(receipt.timestamp).toLocaleTimeString() : ""}
      </div>
    </div>
  );
};

// ── ReasoningCard — expandable "Why?" panel for transparent AI decisions ──
// Shows the Graph-of-Thoughts reasoning trace: nodes, scores, verdicts.
// Collapsed by default. Expands on click to reveal the full reasoning graph.
// Standing on Giants: Besta (GoT, 2024) · Shannon (SNR scoring per node)
//   Al-Ghazali (auditable intention) · Boyd (visible orient phase in OODA)

const ReasoningCard = ({ reasoning }) => {
  const [expanded, setExpanded] = React.useState(false);
  if (!reasoning) return null;

  const nodes = reasoning.got_nodes || [];
  const agentScores = reasoning.agent_scores || {};
  const verdicts = reasoning.guardian_verdicts || {};
  const confidence = reasoning.confidence || 0;
  const confColor = confidence >= 0.9 ? "rgba(76,175,80,0.8)"
    : confidence >= 0.7 ? "rgba(212,165,71,0.8)"
    : "rgba(255,82,82,0.8)";

  return (
    <div style={{
      background: "rgba(0,0,0,0.25)",
      border: "1px solid rgba(212,165,71,0.15)",
      borderRadius: 8,
      margin: "4px 0 8px",
      fontFamily: "var(--mono)",
      fontSize: 10,
      overflow: "hidden",
    }}>
      {/* Collapsed header — always visible */}
      <div
        onClick={() => setExpanded(!expanded)}
        style={{
          display: "flex", justifyContent: "space-between", alignItems: "center",
          padding: "8px 12px", cursor: "pointer",
          background: expanded ? "rgba(212,165,71,0.06)" : "transparent",
          transition: "background 0.2s",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ fontSize: 9, color: "rgba(212,165,71,0.7)" }}>
            {expanded ? "\u25BC" : "\u25B6"}
          </span>
          <span style={{ color: "rgba(212,165,71,0.8)", fontSize: 10, fontWeight: 600 }}>
            Why?
          </span>
          <span style={{ color: "rgba(255,255,255,0.3)", fontSize: 9 }}>
            {nodes.length} nodes | {reasoning.alternatives_considered || 0} alternatives
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ color: confColor, fontSize: 9, fontWeight: 600 }}>
            {(confidence * 100).toFixed(0)}%
          </span>
          {reasoning.model_used && (
            <span style={{ color: "rgba(255,255,255,0.2)", fontSize: 8 }}>
              {reasoning.model_used}
            </span>
          )}
        </div>
      </div>

      {/* Expanded reasoning graph */}
      {expanded && (
        <div style={{ padding: "0 12px 10px" }}>
          {/* GoT node graph */}
          {nodes.length > 0 && (
            <div style={{ marginBottom: 8 }}>
              <div style={{ color: "rgba(255,255,255,0.25)", fontSize: 8, marginBottom: 4, letterSpacing: 0.6, textTransform: "uppercase" }}>
                Graph-of-Thoughts
              </div>
              {nodes.map((node, i) => (
                <div key={node.id || i} style={{
                  display: "flex", alignItems: "flex-start", gap: 6,
                  paddingLeft: node.depth * 12,
                  marginBottom: 3,
                }}>
                  <span style={{
                    color: node.is_conclusion ? "rgba(212,165,71,0.9)" : "rgba(255,255,255,0.25)",
                    fontSize: 8, flexShrink: 0,
                  }}>
                    {node.is_conclusion ? "\u2605" : "\u25CB"}
                  </span>
                  <span style={{
                    color: node.is_conclusion ? "rgba(212,165,71,0.8)" : "rgba(255,255,255,0.5)",
                    fontSize: 9, lineHeight: 1.3, flex: 1,
                  }}>
                    {node.content}
                  </span>
                  <span style={{
                    color: node.score >= 0.85 ? "rgba(76,175,80,0.6)" : "rgba(255,255,255,0.2)",
                    fontSize: 8, flexShrink: 0,
                  }}>
                    {(node.score * 100).toFixed(0)}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Agent scores */}
          {Object.keys(agentScores).length > 0 && (
            <div style={{ marginBottom: 8 }}>
              <div style={{ color: "rgba(255,255,255,0.25)", fontSize: 8, marginBottom: 4, letterSpacing: 0.6, textTransform: "uppercase" }}>
                Agent Scores
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {Object.entries(agentScores).map(([agent, score]) => (
                  <span key={agent} style={{
                    padding: "2px 6px", borderRadius: 3,
                    background: score >= 0.9 ? "rgba(76,175,80,0.1)" : "rgba(255,255,255,0.04)",
                    border: `1px solid ${score >= 0.9 ? "rgba(76,175,80,0.2)" : "rgba(255,255,255,0.08)"}`,
                    color: score >= 0.9 ? "rgba(76,175,80,0.7)" : "rgba(255,255,255,0.4)",
                    fontSize: 8,
                  }}>
                    {agent}: {(score * 100).toFixed(0)}%
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Guardian verdicts */}
          {Object.keys(verdicts).length > 0 && (
            <div style={{ marginBottom: 8 }}>
              <div style={{ color: "rgba(255,255,255,0.25)", fontSize: 8, marginBottom: 4, letterSpacing: 0.6, textTransform: "uppercase" }}>
                Guardian Verdicts
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {Object.entries(verdicts).map(([gate, verdict]) => (
                  <span key={gate} style={{
                    padding: "2px 6px", borderRadius: 3,
                    background: verdict === "APPROVED" ? "rgba(76,175,80,0.1)" : "rgba(255,82,82,0.1)",
                    border: `1px solid ${verdict === "APPROVED" ? "rgba(76,175,80,0.2)" : "rgba(255,82,82,0.2)"}`,
                    color: verdict === "APPROVED" ? "rgba(76,175,80,0.7)" : "rgba(255,82,82,0.7)",
                    fontSize: 8,
                  }}>
                    {gate}: {verdict}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Convergence reason */}
          {reasoning.convergence_reason && (
            <div style={{
              borderTop: "1px solid rgba(255,255,255,0.06)",
              paddingTop: 6, marginTop: 4,
              color: "rgba(255,255,255,0.3)", fontSize: 8, lineHeight: 1.4,
            }}>
              {reasoning.convergence_reason}
            </div>
          )}

          {/* Timing */}
          {reasoning.total_reasoning_ms > 0 && (
            <div style={{
              color: "rgba(255,255,255,0.15)", fontSize: 7,
              textAlign: "right", marginTop: 4,
            }}>
              {reasoning.total_reasoning_ms.toFixed(0)}ms
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// ============================================================
// MAIN DASHBOARD
// ============================================================

export default function App() {
  const node = useNode();
  const {
    connected,
    mode,
    send,
    receive,
    teach,
    synthesize,
    refreshHealth,
    nodeReachable,
    queuedActions,
    lastSeenTs,
    sapMeetOpen,
    sapMessage,
    sapDisclosure,
    sapSessionClose,
  } = node;

  // SAP v0 session state
  const [sapSession, setSapSession] = useState(null);
  const [sapDisclosureData, setSapDisclosureData] = useState(null);
  const [sapIhsanScore, setSapIhsanScore] = useState(null);
  const [sapAgentCard, setSapAgentCard] = useState(null);
  const [sapReceiptChain, setSapReceiptChain] = useState([]);
  const [disclosureCollapsed, setDisclosureCollapsed] = useState(true);
  const [onboarded, setOnboarded] = useState(() => {
    try { return localStorage.getItem("bizra_onboarded") === "1"; } catch { return false; }
  });

  // All hooks must be declared before any conditional returns (React rules of hooks)
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [activeAgents, setActiveAgents] = useState([]);
  const [actionPayload, setActionPayload] = useState(
    '{"steps":[{"channel":"DesktopRpc","kind":"Click","payload":{"code":"click notepad"}}]}'
  );
  const [planId, setPlanId] = useState("");
  const [actionId, setActionId] = useState("");
  const [actionRows, setActionRows] = useState([]);
  const [lastReceipt, setLastReceipt] = useState(null);
  const [pendingMissions, setPendingMissions] = useState([]);
  const [nodeData, setNodeData] = useState({
    knowsMe: 0, ihsan: 9900, messages: 0, fragments: 0, insights: 0, traits: [],
  });
  const [opsMetrics, setOpsMetrics] = useState({
    actionsExecuted: 0, receiptsVerified: 0, reasoningDepth: 0, uptimeMinutes: 0,
  });
  const hhmmReportPath = "/reports/hhmm-sparse-tensor-analysis-2026-02-20.html";
  const chatEndRef = useRef(null);

  // Refresh state from node
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
    // Also refresh profile
    const p = await send("PROFILE");
    if (p?.ok && p.fields?.traits) {
      const traits = p.fields.traits
        .split("|")
        .filter(Boolean)
        .map((entry) => {
          const [label, value, conf] = entry.split(":");
          return { label, value, confidence: parseInt(conf || "8000", 10) };
        });
      setNodeData((prev) => ({ ...prev, traits }));
    }
  }, [send]);

  // Derive ops metrics from messages + connection state
  useEffect(() => {
    const actions = messages.filter((m) => m.receipt).length;
    const verified = messages.filter((m) => m.receipt?.outcome_hash).length;
    const maxDepth = messages.reduce((max, m) => {
      const nodes = m.reasoning_summary?.got_nodes?.length || 0;
      return nodes > max ? nodes : max;
    }, 0);
    const uptime = connected && startTimeRef.current
      ? Math.floor((Date.now() - startTimeRef.current) / 60000)
      : opsMetrics.uptimeMinutes;
    setOpsMetrics({ actionsExecuted: actions, receiptsVerified: verified, reasoningDepth: maxDepth, uptimeMinutes: uptime });
  }, [messages, connected]);

  // Track connection start time for uptime
  const startTimeRef = useRef(null);
  useEffect(() => {
    if (connected && !startTimeRef.current) startTimeRef.current = Date.now();
    if (!connected) startTimeRef.current = null;
  }, [connected]);

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Send message (with SAP v0 session support)
  const sendMessage = useCallback(async () => {
    if (!input.trim() || !connected) return;
    const text = input.trim();
    setInput("");

    setMessages((prev) => [...prev, { role: "user", content: text }]);

    // Auto-initiate SAP session on first message if not active
    let currentSession = sapSession;
    if (!currentSession && sapMeetOpen) {
      const meetResult = await sapMeetOpen();
      if (meetResult?.ok && meetResult.fields) {
        currentSession = meetResult.fields.session_id;
        setSapSession(currentSession);
        setSapDisclosureData(meetResult.fields.disclosure);
        if (meetResult.fields.ihsan_score) {
          setSapIhsanScore(meetResult.fields.ihsan_score);
        }
        if (meetResult.fields.agent_card) {
          try {
            const card = typeof meetResult.fields.agent_card === "string" ? JSON.parse(meetResult.fields.agent_card) : meetResult.fields.agent_card;
            setSapAgentCard(card);
          } catch { /* non-critical */ }
        }
      }
    }

    // Use SAP_MESSAGE if session is active, fall back to RECEIVE
    let result;
    if (currentSession && sapMessage) {
      result = await sapMessage(currentSession, text);
    } else {
      result = await receive(text);
    }

    if (result?.ok && result.fields) {
      const f = result.fields;
      // Flash active agents
      const agentCount = parseInt(f.agents_consulted || "0", 10);
      const activeNames = ALL_AGENTS.slice(0, agentCount);
      setActiveAgents(activeNames);
      setTimeout(() => setActiveAgents([]), 1500);

      // Update SAP session state from response
      if (f.disclosure) {
        setSapDisclosureData(f.disclosure);
      }
      if (f.ihsan_score) {
        setSapIhsanScore(f.ihsan_score);
      }
      if (f.receipt_hash) {
        setSapReceiptChain((prev) => [...prev, { hash: f.receipt_hash, ts: Date.now() }]);
      }

      // Parse reasoning_summary from response if present (Task 3.2)
      let reasoningSummary = null;
      if (f.reasoning_summary) {
        try {
          reasoningSummary = typeof f.reasoning_summary === "string"
            ? JSON.parse(f.reasoning_summary)
            : f.reasoning_summary;
        } catch { /* ignore parse errors */ }
      }

      setMessages((prev) => [
        ...prev,
        {
          role: "node",
          content: f.content || "...",
          meta: {
            agents: f.agents_consulted,
            fragments: parseInt(f.fragments_extracted || "0", 10),
            confidence: f.confidence,
            intent: f.intent,
            ihsanScore: f.ihsan_score,
            disclosure: f.disclosure,
            receiptHash: f.receipt_hash,
          },
          reasoning_summary: reasoningSummary,
        },
      ]);
    }

    await syncState();
  }, [input, connected, receive, sapMeetOpen, sapMessage, sapSession, syncState]);

  // Teach shortcut
  const handleTeach = useCallback(
    async (kind, content) => {
      await teach(kind, content);
      setMessages((prev) => [...prev, { role: "system", content: `📚 Taught: "${content}"` }]);
      await syncState();
    },
    [teach, syncState]
  );

  // Synthesize
  const handleSynthesize = useCallback(async () => {
    const result = await synthesize();
    if (result?.ok && result.fields) {
      const n = parseInt(result.fields.insights_generated || "0", 10);
      if (n > 0) {
        setMessages((prev) => [
          ...prev,
          { role: "system", content: `🧬 Synthesis complete — ${n} new insight${n > 1 ? "s" : ""} generated. Knows-me: ${(parseFloat(result.fields.knows_me || "0") * 100).toFixed(1)}%` },
        ]);
      }
    }
    await syncState();
  }, [synthesize, syncState]);

  // Mission approval handlers (Task 2.3)
  const handleApproveMission = useCallback(async (mission) => {
    setPendingMissions((prev) => prev.filter((m) => m.name !== mission.name));
    setMessages((prev) => [
      ...prev,
      { role: "system", content: `Mission approved: ${mission.name}. Executing with ${(mission.agents || []).join(", ")}...` },
    ]);
    // Execute the mission via PLAN_ACTION -> RUN_ACTION
    const planResult = await send("PLAN_ACTION", {
      payload_json: JSON.stringify({ method: "mission", mission_name: mission.name, agents: mission.agents }),
    });
    if (planResult?.ok && planResult.fields?.plan_id) {
      const runResult = await send("RUN_ACTION", {
        plan_id: planResult.fields.plan_id,
        payload_json: JSON.stringify({ mission_name: mission.name }),
      });
      if (runResult?.ok && runResult.fields) {
        const f = runResult.fields;
        const receipt = {
          action_id: f.action_id,
          plan_id: planResult.fields.plan_id,
          status: f.status || "completed",
          pre_hash: f.pre_hash || "",
          post_hash: f.post_hash || "",
          outcome_hash: f.outcome_hash || "",
          state_changed: f.state_changed === "true",
          outcome_confirmed: f.outcome_confirmed === "true",
          confidence: parseFloat(f.confidence || "0"),
          timestamp: parseInt(f.verification_timestamp || Date.now(), 10),
        };
        setLastReceipt(receipt);
        setMessages((prev) => [
          ...prev,
          { role: "system", content: `Mission ${mission.name} completed | hash=${(f.outcome_hash || "").slice(0, 12)}...`, receipt },
        ]);
      }
    }
  }, [send]);

  const handleSkipMission = useCallback((mission) => {
    setPendingMissions((prev) => prev.filter((m) => m.name !== mission.name));
    setMessages((prev) => [
      ...prev,
      { role: "system", content: `Mission skipped: ${mission.name}` },
    ]);
  }, []);

  const handleModifyMission = useCallback((mission) => {
    setPendingMissions((prev) => prev.filter((m) => m.name !== mission.name));
    // Pre-fill the action payload with the mission config for editing
    setActionPayload(JSON.stringify({
      method: "mission",
      mission_name: mission.name,
      agents: mission.agents,
      includes: mission.includes,
    }, null, 2));
    setMessages((prev) => [
      ...prev,
      { role: "system", content: `Mission ${mission.name} loaded into Action Layer for modification` },
    ]);
  }, []);

  const handlePlanAction = useCallback(async () => {
    const result = await send("PLAN_ACTION", { payload_json: actionPayload });
    if (result?.ok && result.fields) {
      setPlanId(result.fields.plan_id || "");
      const method = result.fields.method || "action";
      const permit = result.fields.permit_status || "PENDING";
      setMessages((prev) => [
        ...prev,
        {
          role: "system",
          content: `Action planned: ${result.fields.plan_id} [${method}] permit=${permit}`,
        },
      ]);
    } else {
      setMessages((prev) => [
        ...prev,
        { role: "system", content: `PLAN_ACTION failed${result?.queued ? " (queued)" : ""}` },
      ]);
    }
  }, [send, actionPayload]);

  const handleRunAction = useCallback(async () => {
    const result = await send("RUN_ACTION", {
      plan_id: planId,
      payload_json: actionPayload,
    });
    if (result?.ok && result.fields) {
      const f = result.fields;
      setActionId(f.action_id || "");
      // Store the full receipt for ActionProofCard (Task 1.4)
      const receipt = {
        action_id: f.action_id,
        plan_id: f.plan_id || planId,
        status: f.status || "unknown",
        pre_hash: f.pre_hash || "",
        post_hash: f.post_hash || "",
        outcome_hash: f.outcome_hash || "",
        state_changed: f.state_changed === "true",
        outcome_confirmed: f.outcome_confirmed === "true",
        confidence: parseFloat(f.confidence || "0"),
        timestamp: parseInt(f.verification_timestamp || Date.now(), 10),
      };
      setLastReceipt(receipt);
      const hashShort = (f.outcome_hash || "").slice(0, 12);
      const confirmed = f.outcome_confirmed === "true" ? "CONFIRMED" : "UNVERIFIED";
      setMessages((prev) => [
        ...prev,
        {
          role: "system",
          content: `RUN_ACTION ${f.status} | ${confirmed} | hash=${hashShort}... | confidence=${f.confidence}`,
          receipt,
        },
      ]);
    } else {
      setMessages((prev) => [
        ...prev,
        { role: "system", content: `RUN_ACTION failed${result?.queued ? " (queued)" : ""}` },
      ]);
    }
  }, [send, planId, actionPayload]);

  const handleActionStatus = useCallback(async () => {
    if (!actionId) return;
    const result = await send("ACTION_STATUS", { action_id: actionId });
    if (result?.ok && result.fields) {
      setMessages((prev) => [
        ...prev,
        {
          role: "system",
          content: `📍 ACTION_STATUS ${result.fields.status || "unknown"} for ${result.fields.action_id || actionId}`,
        },
      ]);
    }
  }, [send, actionId]);

  const handleActionHistory = useCallback(async () => {
    const result = await send("ACTION_HISTORY", { limit: 10, cursor: "" });
    if (result?.ok && result.fields) {
      const rows = (result.fields.rows || "")
        .split("||")
        .filter(Boolean)
        .map((line) => {
          try {
            return JSON.parse(line);
          } catch {
            return { raw: line };
          }
        });
      setActionRows(rows);
    }
  }, [send]);

  const openHhmmReport = useCallback(() => {
    if (typeof window !== "undefined") {
      window.open(hhmmReportPath, "_blank", "noopener,noreferrer");
    }
  }, [hhmmReportPath]);

  const quickTeach = [
    { kind: "preference", text: "I prefer dark mode and minimal UI" },
    { kind: "expertise", text: "I specialize in distributed systems" },
    { kind: "goal", text: "My goal is to democratize AI for everyone" },
    { kind: "fact", text: "I live in Dubai and work in GMT+4" },
  ];

  // Demo landing page — accessible via /#/demo
  const isDemo = typeof window !== 'undefined' && window.location.hash === '#/demo';
  if (isDemo) {
    return (
      <LandingDemo
        onEnterApp={() => {
          window.location.hash = '';
        }}
      />
    );
  }

  // Onboarding gate — placed after all hooks to satisfy React rules of hooks
  if (!onboarded) {
    return (
      <OnboardingFlow
        node={node}
        onComplete={() => {
          try { localStorage.setItem("bizra_onboarded", "1"); } catch {}
          setOnboarded(true);
        }}
      />
    );
  }

  return (
    <div style={{
      "--sans": "'DM Sans', sans-serif",
      "--mono": "'JetBrains Mono', monospace",
      minHeight: "100vh", background: "#0A0B0F",
      fontFamily: "var(--sans)", color: "rgba(255,255,255,0.88)",
      display: "flex", flexDirection: "column", position: "relative", overflow: "hidden",
    }}>
      {/* Background */}
      <div style={{ position: "fixed", inset: 0, pointerEvents: "none", zIndex: 0 }}>
        <div style={{ position: "absolute", top: -40, right: -20, opacity: 0.03 }}><SeedOfLife size={400} opacity={1} /></div>
        <div style={{ position: "absolute", bottom: -60, left: -30, opacity: 0.02 }}><SeedOfLife size={500} opacity={1} /></div>
        <div style={{ position: "absolute", inset: 0, background: "radial-gradient(ellipse at 30% 20%, rgba(212,165,71,0.03) 0%, transparent 60%), radial-gradient(ellipse at 70% 80%, rgba(107,155,247,0.02) 0%, transparent 50%)" }} />
      </div>

      {/* Header */}
      <header style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "14px 24px", borderBottom: "1px solid rgba(255,255,255,0.04)",
        background: "rgba(10,11,15,0.8)", backdropFilter: "blur(20px)", position: "relative", zIndex: 10,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ width: 28, height: 28, borderRadius: 8, background: "linear-gradient(135deg, #D4A547, #8B6914)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14, fontWeight: 700, color: "#0A0B0F", fontFamily: "var(--mono)" }}>B</div>
          <div>
            <div style={{ fontWeight: 600, fontSize: 14, letterSpacing: -0.3 }}>
              Node0
              <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "rgba(212,165,71,0.5)", marginLeft: 8, fontWeight: 400 }}>v0.1.0</span>
              <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: "rgba(107,155,247,0.4)", marginLeft: 8 }}>
                {mode === "tauri" ? "NATIVE" : "DEMO"}
              </span>
            </div>
            <div style={{ fontFamily: "var(--mono)", fontSize: 9, color: "rgba(255,255,255,0.2)", letterSpacing: 1.5, marginTop: 1 }}>SOVEREIGN AI NODE</div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <IhsanBar score={nodeData.ihsan} />
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{ width: 7, height: 7, borderRadius: "50%", background: connected ? "#5BBA6F" : "#E85D4A", boxShadow: connected ? "0 0 10px rgba(91,186,111,0.5)" : "none", animation: connected ? "pulse 2s infinite" : "none" }} />
            <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: connected ? "rgba(91,186,111,0.7)" : "rgba(232,93,74,0.7)" }}>
              {connected ? "CONNECTED" : "BOOTING"}
            </span>
          </div>
        </div>
      </header>

      {!nodeReachable && (
        <div
          style={{
            position: "relative",
            zIndex: 9,
            borderBottom: "1px solid rgba(232,93,74,0.25)",
            background: "rgba(232,93,74,0.08)",
            color: "rgba(255,210,205,0.9)",
            fontFamily: "var(--mono)",
            fontSize: 10,
            letterSpacing: 0.6,
            padding: "8px 24px",
          }}
        >
          NODE UNREACHABLE
          <span style={{ marginLeft: 10, color: "rgba(255,255,255,0.5)" }}>
            last seen: {lastSeenTs ? new Date(lastSeenTs).toLocaleTimeString() : "never"}
          </span>
          <span style={{ marginLeft: 10, color: "rgba(212,165,71,0.75)" }}>
            queued actions: {queuedActions}
          </span>
        </div>
      )}

      {/* Main Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "260px 1fr 240px", flex: 1, gap: 0, position: "relative", zIndex: 5, minHeight: 0 }}>

        {/* LEFT — Agents + Metrics */}
        <div style={{ borderRight: "1px solid rgba(255,255,255,0.04)", padding: 16, display: "flex", flexDirection: "column", gap: 20, overflowY: "auto" }}>
          <div>
            <SectionLabel>Personal Agent Team</SectionLabel>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {ALL_AGENTS.map((name) => <AgentBadge key={name} name={name} active={activeAgents.includes(name)} />)}
            </div>
          </div>

          <div>
            <SectionLabel>Runtime Metrics</SectionLabel>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
              <Metric label="Messages" value={nodeData.messages} />
              <Metric label="Fragments" value={nodeData.fragments} />
              <Metric label="Insights" value={nodeData.insights} />
              <Metric label="Traits" value={nodeData.traits.length} />
            </div>
          </div>

          <div>
            <SectionLabel extra={connected ? "LIVE" : "OFF"}>Founder Ops</SectionLabel>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <div style={{ display: "flex", flexDirection: "column", gap: 2, padding: "6px 8px", background: "rgba(212,165,71,0.04)", borderRadius: 6, border: "1px solid rgba(212,165,71,0.08)" }}>
                <span style={{ fontFamily: "var(--mono)", fontSize: 8, color: "rgba(212,165,71,0.5)", letterSpacing: 1, textTransform: "uppercase" }}>Actions</span>
                <span style={{ fontFamily: "var(--mono)", fontSize: 18, fontWeight: 600, color: "#D4A547" }}>{opsMetrics.actionsExecuted}</span>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 2, padding: "6px 8px", background: "rgba(78,205,196,0.04)", borderRadius: 6, border: "1px solid rgba(78,205,196,0.08)" }}>
                <span style={{ fontFamily: "var(--mono)", fontSize: 8, color: "rgba(78,205,196,0.5)", letterSpacing: 1, textTransform: "uppercase" }}>Verified</span>
                <span style={{ fontFamily: "var(--mono)", fontSize: 18, fontWeight: 600, color: "#4ecdc4" }}>{opsMetrics.receiptsVerified}</span>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 2, padding: "6px 8px", background: "rgba(157,78,221,0.04)", borderRadius: 6, border: "1px solid rgba(157,78,221,0.08)" }}>
                <span style={{ fontFamily: "var(--mono)", fontSize: 8, color: "rgba(157,78,221,0.5)", letterSpacing: 1, textTransform: "uppercase" }}>GoT Depth</span>
                <span style={{ fontFamily: "var(--mono)", fontSize: 18, fontWeight: 600, color: "#9d4edd" }}>{opsMetrics.reasoningDepth}</span>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 2, padding: "6px 8px", background: "rgba(255,255,255,0.02)", borderRadius: 6, border: "1px solid rgba(255,255,255,0.06)" }}>
                <span style={{ fontFamily: "var(--mono)", fontSize: 8, color: "rgba(255,255,255,0.3)", letterSpacing: 1, textTransform: "uppercase" }}>Uptime</span>
                <span style={{ fontFamily: "var(--mono)", fontSize: 18, fontWeight: 600, color: "rgba(255,255,255,0.7)" }}>{opsMetrics.uptimeMinutes}<span style={{ fontSize: 9, color: "rgba(255,255,255,0.3)", marginLeft: 2 }}>m</span></span>
              </div>
            </div>
          </div>

          <div>
            <SectionLabel>Quick Teach</SectionLabel>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {quickTeach.map((s, i) => (
                <button key={i} onClick={() => handleTeach(s.kind, s.text)} style={{
                  background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)",
                  borderRadius: 6, padding: "6px 10px", textAlign: "left", cursor: "pointer",
                  fontFamily: "var(--sans)", fontSize: 11, color: "rgba(255,255,255,0.5)", transition: "all 0.2s",
                }}
                  onMouseEnter={(e) => { e.currentTarget.style.background = "rgba(212,165,71,0.08)"; e.currentTarget.style.borderColor = "rgba(212,165,71,0.2)"; }}
                  onMouseLeave={(e) => { e.currentTarget.style.background = "rgba(255,255,255,0.02)"; e.currentTarget.style.borderColor = "rgba(255,255,255,0.06)"; }}
                >
                  <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: "rgba(212,165,71,0.4)", marginRight: 6 }}>{s.kind}</span>
                  {s.text.slice(0, 35)}...
                </button>
              ))}
            </div>
          </div>

          <button onClick={handleSynthesize} style={{
            background: "linear-gradient(135deg, rgba(212,165,71,0.15), rgba(212,165,71,0.05))",
            border: "1px solid rgba(212,165,71,0.25)", borderRadius: 8,
            padding: "10px 16px", cursor: "pointer",
            fontFamily: "var(--mono)", fontSize: 11, fontWeight: 600, color: "#D4A547", letterSpacing: 0.5,
          }}>
            🧬 Synthesize Memory
          </button>

          <div style={{ marginTop: -8 }}>
            <button onClick={openHhmmReport} style={{
              width: "100%",
              background: "rgba(78,205,196,0.08)",
              border: "1px solid rgba(78,205,196,0.25)",
              borderRadius: 8,
              padding: "10px 16px",
              cursor: "pointer",
              fontFamily: "var(--mono)",
              fontSize: 11,
              fontWeight: 600,
              color: "#4ecdc4",
              letterSpacing: 0.5,
            }}>
              📊 Analysis
            </button>
            <div style={{
              marginTop: 6,
              textAlign: "center",
              fontFamily: "var(--mono)",
              fontSize: 9,
              color: "rgba(78,205,196,0.55)",
              letterSpacing: 0.6,
              textTransform: "uppercase",
            }}>
              Internal HHMM snapshot
            </div>
          </div>
        </div>

        {/* CENTER — Chat */}
        <div style={{ display: "flex", flexDirection: "column", minHeight: 0 }}>
          <div style={{ flex: 1, overflowY: "auto", padding: "20px 24px", display: "flex", flexDirection: "column" }}>
            {messages.length === 0 && (
              <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 16, opacity: 0.4 }}>
                <SeedOfLife size={80} opacity={0.3} color="#D4A547" />
                <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "rgba(212,165,71,0.6)", letterSpacing: 2, textTransform: "uppercase" }}>Node0 Ready</div>
                <div style={{ fontSize: 13, color: "rgba(255,255,255,0.3)", textAlign: "center", maxWidth: 300, lineHeight: 1.6 }}>
                  Start a conversation. Every message teaches me who you are.
                </div>
              </div>
            )}
            {messages.map((msg, i) =>
              msg.role === "system" ? (
                <div key={i} style={{ alignSelf: "center", width: "100%", maxWidth: 420 }}>
                  <div style={{ textAlign: "center", padding: "8px 16px", marginBottom: msg.receipt ? 0 : 12, fontFamily: "var(--mono)", fontSize: 10, color: "rgba(212,165,71,0.5)", background: "rgba(212,165,71,0.04)", borderRadius: msg.receipt ? "20px 20px 0 0" : 20 }}>{msg.content}</div>
                  {msg.receipt && <ActionProofCard receipt={msg.receipt} />}
                  {msg.reasoning_summary && <ReasoningCard reasoning={msg.reasoning_summary} />}
                </div>
              ) : (
                <div key={i}>
                  <Bubble role={msg.role} content={msg.content} meta={msg.meta} />
                  {msg.reasoning_summary && <ReasoningCard reasoning={msg.reasoning_summary} />}
                </div>
              )
            )}
            {/* Pending mission approval cards */}
            {pendingMissions.map((mission, i) => (
              <MissionApprovalCard
                key={`mission-${mission.name}-${i}`}
                mission={mission}
                onApprove={handleApproveMission}
                onSkip={handleSkipMission}
                onModify={handleModifyMission}
              />
            ))}
            <div ref={chatEndRef} />
          </div>

          {/* Input */}
          <div style={{ padding: "12px 20px 16px", borderTop: "1px solid rgba(255,255,255,0.04)", background: "rgba(10,11,15,0.6)", backdropFilter: "blur(10px)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 12, padding: "6px 6px 6px 16px" }}>
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && sendMessage()}
                placeholder={connected ? "Talk to your node..." : "Connecting..."}
                disabled={!connected}
                style={{ flex: 1, background: "none", border: "none", outline: "none", fontFamily: "var(--sans)", fontSize: 14, color: "rgba(255,255,255,0.88)", padding: "6px 0" }}
              />
              <button
                onClick={sendMessage}
                disabled={!input.trim() || !connected}
                style={{
                  width: 36, height: 36, borderRadius: 8, border: "none",
                  background: input.trim() ? "linear-gradient(135deg, #D4A547, #8B6914)" : "rgba(255,255,255,0.04)",
                  color: input.trim() ? "#0A0B0F" : "rgba(255,255,255,0.15)",
                  cursor: input.trim() ? "pointer" : "default",
                  display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, fontWeight: 700,
                }}>↑</button>
            </div>
            <div style={{ display: "flex", justifyContent: "center", gap: 16, marginTop: 8, fontFamily: "var(--mono)", fontSize: 9, color: "rgba(255,255,255,0.15)" }}>
              <span>Try: "I'm a software architect who loves Rust"</span>
              <span>•</span>
              <span>"My goal is to build sovereign AI"</span>
            </div>
          </div>
        </div>

        {/* RIGHT — Knowledge */}
        <div style={{ borderLeft: "1px solid rgba(255,255,255,0.04)", padding: 16, display: "flex", flexDirection: "column", gap: 20, alignItems: "center", overflowY: "auto" }}>
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
            <KnowsMeGauge
              score={nodeData.knowsMe}
              traits={nodeData.traits}
              onPromptClick={(kind) => {
                const prompts = {
                  fact: "Tell me a fact about yourself",
                  preference: "What do you prefer?",
                  goal: "What are you working toward?",
                  expertise: "What do you specialize in?",
                  pattern: "Describe a pattern in how you work",
                  relationship: "Who matters in your life?",
                  principle: "What principle guides you?",
                  context: "What context should I know about?",
                };
                setInput(prompts[kind] || "");
              }}
            />
            <div style={{ fontFamily: "var(--mono)", fontSize: 9, color: "rgba(255,255,255,0.2)", letterSpacing: 1.5, textTransform: "uppercase", marginTop: 4 }}>Understanding Depth</div>
          </div>

          {/* Quality Tier Badge */}
          <QualityTierBadge nodeData={nodeData} />

          {/* Growth Roadmap */}
          <GrowthRoadmap nodeData={nodeData} />

          {/* SAP v0 Transparency */}
          {sapSession && (
            <div style={{ width: "100%", display: "flex", flexDirection: "column", gap: 8 }}>
              <div style={{ padding: "10px 12px", background: "rgba(91,186,111,0.03)", border: "1px solid rgba(91,186,111,0.08)", borderRadius: 8 }}>
                <SectionLabel extra="SAP v0">Transparency</SectionLabel>
                <SAPBadge ihsanScore={sapIhsanScore || "0.95"} sessionActive={true} />
                <DisclosurePanel disclosure={sapDisclosureData} collapsed={disclosureCollapsed} onToggle={() => setDisclosureCollapsed((c) => !c)} />
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 6 }}>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 9, color: "rgba(255,255,255,0.2)" }}>
                    Session: {sapSession.slice(0, 12)}...
                  </div>
                  <div style={{ display: "flex", gap: 4 }}>
                    <button onClick={async () => {
                      const d = await sapDisclosure(sapSession);
                      if (d?.ok && d.fields?.disclosure) setSapDisclosureData(d.fields.disclosure);
                    }} style={{ background: "none", border: "1px solid rgba(91,186,111,0.2)", borderRadius: 4, padding: "2px 6px", fontFamily: "var(--mono)", fontSize: 8, color: "rgba(91,186,111,0.5)", cursor: "pointer" }} title="Refresh disclosure">
                      Refresh
                    </button>
                    <button onClick={async () => {
                      await sapSessionClose(sapSession);
                      setSapSession(null);
                      setSapDisclosureData(null);
                      setSapIhsanScore(null);
                      setSapAgentCard(null);
                      setSapReceiptChain([]);
                    }} style={{ background: "none", border: "1px solid rgba(232,93,74,0.2)", borderRadius: 4, padding: "2px 6px", fontFamily: "var(--mono)", fontSize: 8, color: "rgba(232,93,74,0.5)", cursor: "pointer" }} title="Close SAP session">
                      Close
                    </button>
                  </div>
                </div>
                {sapReceiptChain.length > 0 && (
                  <div style={{ marginTop: 6, paddingTop: 6, borderTop: "1px solid rgba(255,255,255,0.04)" }}>
                    <span style={{ fontFamily: "var(--mono)", fontSize: 8, color: "rgba(107,155,247,0.4)", letterSpacing: 0.8, textTransform: "uppercase" }}>Receipt Chain ({sapReceiptChain.length})</span>
                    <div style={{ display: "flex", flexDirection: "column", gap: 1, marginTop: 3, maxHeight: 60, overflowY: "auto" }}>
                      {sapReceiptChain.slice(-5).map((r, i) => (
                        <span key={i} style={{ fontFamily: "var(--mono)", fontSize: 8, color: "rgba(107,155,247,0.3)" }}>#{r.hash.slice(0, 12)}</span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
              <SovereignAgentCard agentData={sapAgentCard || {
                agent_id: "node0-user-zero",
                role: "sovereign_personal",
                version: "0.1.0",
                compilation: { genesis_version: "GENESIS", ihsan_threshold: 0.95, compiled_reflex_count: 81, compilation_coverage: 0.92 },
                policy_hash: "504145f781412a4103249f78f46d61609eb1d02f",
              }} />
            </div>
          )}

          <div style={{ width: "100%" }}>
            <SectionLabel extra={nodeData.traits.length}>Learned Traits</SectionLabel>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {nodeData.traits.length === 0 ? (
                <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "rgba(255,255,255,0.12)", textAlign: "center", padding: "20px 0" }}>No traits yet.<br />Talk to teach me.</div>
              ) : (
                nodeData.traits.map((t, i) => <TraitPill key={`${t.label}-${i}`} label={t.label} value={t.value} confidence={t.confidence} />)
              )}
            </div>
          </div>

          <div style={{ width: "100%", padding: "10px 12px", background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.04)", borderRadius: 8 }}>
            <SectionLabel>Architecture</SectionLabel>
            {["bizra-hooks → nerves", "bizra-memory → brain", "bizra-agent → being", "bizra-node → process"].map((l, i) => (
              <div key={i} style={{ fontFamily: "var(--mono)", fontSize: 9, color: "rgba(255,255,255,0.18)", marginBottom: 2 }}>{l}</div>
            ))}
            <div style={{ marginTop: 8, paddingTop: 8, borderTop: "1px solid rgba(255,255,255,0.04)", fontFamily: "var(--mono)", fontSize: 9, color: "rgba(212,165,71,0.3)", textAlign: "center", lineHeight: 1.6 }}>
              10,000 lines • 205 tests<br />Zero dependencies<br />ربي لا يعرف المستحيل
            </div>
          </div>

          <div style={{ width: "100%", padding: "10px 12px", background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.04)", borderRadius: 8 }}>
            <SectionLabel>Action Layer</SectionLabel>
            <textarea
              value={actionPayload}
              onChange={(e) => setActionPayload(e.target.value)}
              style={{
                width: "100%",
                minHeight: 72,
                background: "rgba(0,0,0,0.2)",
                border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: 6,
                color: "rgba(255,255,255,0.75)",
                fontFamily: "var(--mono)",
                fontSize: 10,
                padding: 8,
                resize: "vertical",
              }}
            />
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, marginTop: 8 }}>
              <button onClick={handlePlanAction} style={actionBtnStyle}>Plan</button>
              <button onClick={handleRunAction} style={actionBtnStyle}>Run</button>
              <button onClick={handleActionStatus} style={actionBtnStyle} disabled={!actionId}>Status</button>
              <button onClick={handleActionHistory} style={actionBtnStyle}>History</button>
            </div>
            <div style={{ marginTop: 8, fontFamily: "var(--mono)", fontSize: 9, color: "rgba(255,255,255,0.25)" }}>
              plan: {planId || "\u2014"}<br />
              action: {actionId || "\u2014"}
              {lastReceipt && (
                <>
                  <br />
                  <span style={{ color: lastReceipt.outcome_confirmed ? "rgba(76,175,80,0.8)" : "rgba(255,152,0,0.8)" }}>
                    {lastReceipt.outcome_confirmed ? "VERIFIED" : "UNVERIFIED"}
                  </span>
                  {" "}hash: {(lastReceipt.outcome_hash || "").slice(0, 16)}...
                  <br />confidence: {(lastReceipt.confidence * 100).toFixed(0)}%
                </>
              )}
            </div>
            {actionRows.length > 0 && (
              <div style={{ marginTop: 8, maxHeight: 120, overflowY: "auto", display: "flex", flexDirection: "column", gap: 4 }}>
                {actionRows.map((row, idx) => (
                  <div key={idx} style={{ fontFamily: "var(--mono)", fontSize: 9, color: "rgba(212,165,71,0.6)", border: "1px solid rgba(212,165,71,0.15)", borderRadius: 6, padding: "4px 6px" }}>
                    {(row.id || row.raw || "row").toString().slice(0, 32)}
                    <span style={{ color: "rgba(255,255,255,0.35)", marginLeft: 6 }}>{row.result || ""}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      <style>{`
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
      `}</style>
    </div>
  );
}

const actionBtnStyle = {
  background: "rgba(212,165,71,0.1)",
  border: "1px solid rgba(212,165,71,0.25)",
  color: "#D4A547",
  borderRadius: 6,
  padding: "6px 8px",
  fontFamily: "var(--mono)",
  fontSize: 10,
  cursor: "pointer",
};
