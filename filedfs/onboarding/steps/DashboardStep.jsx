// ============================================================
// DashboardStep — Onboarding summary and promotion info
// ============================================================
// Shows: KnowsMe gauge, reflex stats, health summary,
// manual promotion instructions, and "Enter Dashboard" button.
// ============================================================

import { useState, useEffect, useCallback, useRef } from 'react';

// ── KnowsMe Gauge (full size, matching App.jsx) ──────────────

const KnowsMeGauge = ({ score, size = 160 }) => {
  const r = (size - 16) / 2;
  const c = 2 * Math.PI * r;
  return (
    <div style={{ position: 'relative', width: size, height: size }}>
      <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
        <defs>
          <linearGradient id="onb-dash-ggrad" x1="0%" y1="0%" x2="100%">
            <stop offset="0%" stopColor="#C9A962" />
            <stop offset="50%" stopColor="#E8D5A3" />
            <stop offset="100%" stopColor="#C9A962" />
          </linearGradient>
        </defs>
        <circle
          cx={size / 2} cy={size / 2} r={r}
          fill="none"
          stroke="rgba(212,165,71,0.08)"
          strokeWidth="5"
        />
        <circle
          cx={size / 2} cy={size / 2} r={r}
          fill="none"
          stroke="url(#onb-dash-ggrad)"
          strokeWidth="5"
          strokeDasharray={c}
          strokeDashoffset={c - score * c}
          strokeLinecap="round"
          style={{ transition: 'stroke-dashoffset 1.2s cubic-bezier(0.4,0,0.2,1)' }}
        />
      </svg>
      <div style={{
        position: 'absolute',
        inset: 0,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
      }}>
        <span style={{
          fontFamily: 'var(--mono)',
          fontSize: 32,
          fontWeight: 700,
          color: '#E8D5A3',
          letterSpacing: -1,
        }}>
          {(score * 100).toFixed(1)}
        </span>
        <span style={{
          fontFamily: 'var(--mono)',
          fontSize: 9,
          color: 'rgba(212,165,71,0.5)',
          letterSpacing: 2,
          textTransform: 'uppercase',
          marginTop: 2,
        }}>
          knows me
        </span>
      </div>
    </div>
  );
};

// ── Stat Card ─────────────────────────────────────────────────

const StatCard = ({ label, value, color = 'rgba(255,255,255,0.85)' }) => (
  <div style={{
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: 4,
    padding: '10px 12px',
    background: 'rgba(255,255,255,0.02)',
    border: '1px solid rgba(255,255,255,0.04)',
    borderRadius: 8,
    minWidth: 0,
  }}>
    <span style={{
      fontFamily: 'var(--mono)',
      fontSize: 18,
      fontWeight: 600,
      color,
    }}>
      {value}
    </span>
    <span style={{
      fontFamily: 'var(--mono)',
      fontSize: 8,
      color: 'rgba(255,255,255,0.25)',
      letterSpacing: 1,
      textTransform: 'uppercase',
      textAlign: 'center',
    }}>
      {label}
    </span>
  </div>
);

// ── Quality Tier (inline, matches App.jsx tiers) ────────────

const QUALITY_TIERS = [
  { key: "seed", label: "Seed", icon: "\u{1F331}", color: "#8B7355", minAtoms: 0, minMsgs: 0 },
  { key: "sprout", label: "Sprout", icon: "\u{1F33F}", color: "#4CAF50", minAtoms: 1, minMsgs: 0 },
  { key: "growing", label: "Growing", icon: "\u{1F333}", color: "#2196F3", minAtoms: 25, minMsgs: 10 },
  { key: "rooted", label: "Rooted", icon: "\u{1F332}", color: "#9C27B0", minAtoms: 100, minMsgs: 0 },
  { key: "flourishing", label: "Flourishing", icon: "\u{1F31F}", color: "#FFD700", minAtoms: 200, minMsgs: 0 },
];

const TIER_UNLOCKS = {
  seed: ["Chat with your node", "TEACH commands to build knowledge"],
  sprout: ["Memory recall across sessions", "Bootstrap reflexes (shadow mode)"],
  growing: ["Reflex compilation", "Action Bus (ToolCall)"],
  rooted: ["Desktop actions via AHK bridge", "Token economy participation"],
  flourishing: ["Full Action Bus orchestration", "Agent-as-Service publishing"],
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

const TIER_CONGRATS = {
  seed: "Your sovereign node is planted. Every conversation is a seed of understanding.",
  sprout: "You have already begun teaching your node. It remembers you now.",
  growing: "Your node is growing strong with knowledge. Reflexes are forming.",
  rooted: "Deeply rooted. Your node has real depth of understanding.",
  flourishing: "Full sovereignty achieved. Your node is a living agent.",
};

const TierBadgeOnboarding = ({ health }) => {
  const nodeData = {
    fragments: parseInt(health?.fragments || '0', 10),
    messages: parseInt(health?.messages || '0', 10),
  };
  const tier = determineTier(nodeData);
  const unlocks = TIER_UNLOCKS[tier.key] || [];
  const isFlourishing = tier.key === 'flourishing';
  const congrats = TIER_CONGRATS[tier.key] || '';

  return (
    <div style={{
      padding: '14px 16px',
      background: `${tier.color}08`,
      border: `1px solid ${tier.color}18`,
      borderRadius: 10,
      boxShadow: isFlourishing ? `0 0 20px ${tier.color}10` : 'none',
    }}>
      {/* Badge row */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        marginBottom: 10,
      }}>
        <div style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 8,
          padding: '5px 12px',
          background: `${tier.color}14`,
          border: `1px solid ${tier.color}30`,
          borderRadius: 20,
        }}>
          <span style={{ fontSize: 16, lineHeight: 1 }}>{tier.icon}</span>
          <span style={{
            fontFamily: 'var(--mono)',
            fontSize: 12,
            fontWeight: 600,
            color: tier.color,
            letterSpacing: 0.5,
          }}>
            {tier.label}
          </span>
        </div>
        <span style={{
          fontFamily: 'var(--sans)',
          fontSize: 12,
          color: 'rgba(255,255,255,0.4)',
        }}>
          You are starting at <span style={{ color: tier.color, fontWeight: 600 }}>{tier.label}</span> level
        </span>
      </div>

      {/* Congratulatory message */}
      <div style={{
        fontFamily: 'var(--sans)',
        fontSize: 12,
        color: 'rgba(255,255,255,0.5)',
        lineHeight: 1.5,
        marginBottom: 12,
        padding: '8px 10px',
        background: `${tier.color}06`,
        borderRadius: 6,
        borderLeft: `3px solid ${tier.color}40`,
      }}>
        {congrats}
      </div>

      {/* Available capabilities */}
      <div style={{
        fontFamily: 'var(--mono)',
        fontSize: 9,
        color: 'rgba(255,255,255,0.2)',
        letterSpacing: 1,
        textTransform: 'uppercase',
        marginBottom: 6,
      }}>
        Available at this tier
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        {unlocks.map((cap, i) => (
          <div key={i} style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '4px 0',
          }}>
            <div style={{
              width: 5,
              height: 5,
              borderRadius: '50%',
              background: tier.color,
              flexShrink: 0,
              opacity: 0.7,
            }} />
            <span style={{
              fontFamily: 'var(--sans)',
              fontSize: 11,
              color: 'rgba(255,255,255,0.5)',
            }}>
              {cap}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

// ── Section Label ─────────────────────────────────────────────

const SectionLabel = ({ children }) => (
  <div style={{
    fontFamily: 'var(--mono)',
    fontSize: 9,
    color: 'rgba(255,255,255,0.25)',
    letterSpacing: 1.5,
    marginBottom: 8,
    textTransform: 'uppercase',
  }}>
    {children}
  </div>
);

// ── Criteria Checklist Item ───────────────────────────────────

const CriteriaItem = ({ label, description }) => (
  <div style={{
    display: 'flex',
    alignItems: 'flex-start',
    gap: 10,
    padding: '6px 0',
  }}>
    <div style={{
      width: 14,
      height: 14,
      borderRadius: 3,
      border: '1.5px solid rgba(255,255,255,0.12)',
      flexShrink: 0,
      marginTop: 1,
    }} />
    <div>
      <span style={{
        fontFamily: 'var(--sans)',
        fontSize: 12,
        color: 'rgba(255,255,255,0.5)',
      }}>
        {label}
      </span>
      {description && (
        <div style={{
          fontFamily: 'var(--sans)',
          fontSize: 10,
          color: 'rgba(255,255,255,0.2)',
          marginTop: 2,
        }}>
          {description}
        </div>
      )}
    </div>
  </div>
);

// ============================================================
// MAIN COMPONENT
// ============================================================

export default function DashboardStep({ node, state, setState, onNext }) {
  const [knowsMe, setKnowsMe] = useState(0);
  const [health, setHealth] = useState(null);
  const [reflexStats, setReflexStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const fetchedRef = useRef(false);

  // Fetch all data on mount
  useEffect(() => {
    if (fetchedRef.current) return;
    fetchedRef.current = true;

    const fetchData = async () => {
      setLoading(true);

      try {
        // Fetch knows_me
        const kmResult = await node.send('KNOWS_ME');
        if (kmResult?.ok && kmResult.fields?.score) {
          setKnowsMe(parseFloat(kmResult.fields.score));
        }

        // Fetch health
        const hResult = await node.send('HEALTH');
        if (hResult?.ok && hResult.fields) {
          setHealth(hResult.fields);
        }

        // Fetch reflex stats (may not be supported by all bridges)
        try {
          const rResult = await node.send('REFLEX_STATS');
          if (rResult?.ok && rResult.fields) {
            setReflexStats(rResult.fields);
          }
        } catch (err) {
          // REFLEX_STATS may not be available; use fallback
          setReflexStats({
            mode: 'shadow',
            hits: '0',
            misses: '0',
            compiled: '0',
          });
        }
      } catch (err) {
        // Graceful degradation — show what we have
      }

      setLoading(false);
    };

    fetchData();
  }, [node]);

  const reflexMode = reflexStats?.mode || 'shadow';
  const isShadow = reflexMode === 'shadow';
  const policyHash = state.policyHash || 'sha256:<pending>';

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      gap: 20,
      animation: 'onb-fadeUp 0.4s ease',
    }}>
      {/* Title */}
      <div style={{ textAlign: 'center' }}>
        <h2 style={{
          fontFamily: 'var(--sans)',
          fontSize: 20,
          fontWeight: 600,
          color: 'rgba(255,255,255,0.88)',
          margin: '0 0 4px 0',
        }}>
          Your Node is Ready
        </h2>
        <p style={{
          fontFamily: 'var(--sans)',
          fontSize: 13,
          color: 'rgba(255,255,255,0.35)',
          margin: 0,
        }}>
          Here is a summary of your sovereign node's state.
        </p>
      </div>

      {/* Gauge + Stats row */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 24,
        flexWrap: 'wrap',
      }}>
        <KnowsMeGauge score={knowsMe} />

        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: 8,
          minWidth: 180,
        }}>
          <StatCard
            label="Agents"
            value={health?.agents_registered || '0'}
            color="#6B9BF7"
          />
          <StatCard
            label="Fragments"
            value={health?.fragments || '0'}
            color="#5BBA6F"
          />
          <StatCard
            label="Insights"
            value={health?.insights || '0'}
            color="#A78BFA"
          />
          <StatCard
            label="Ihsan"
            value={health?.ihsan ? (parseInt(health.ihsan, 10) / 100).toFixed(1) + '%' : '?'}
            color="#C9A962"
          />
        </div>
      </div>

      {/* Quality Tier Badge */}
      <TierBadgeOnboarding health={health} />

      {/* Reflex Stats Panel */}
      <div style={{
        padding: '14px 16px',
        background: 'rgba(255,255,255,0.02)',
        border: '1px solid rgba(255,255,255,0.06)',
        borderRadius: 10,
      }}>
        <SectionLabel>Reflex Engine</SectionLabel>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 16,
          flexWrap: 'wrap',
        }}>
          {/* Mode badge */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            padding: '4px 10px',
            background: isShadow
              ? 'rgba(107,155,247,0.08)'
              : 'rgba(91,186,111,0.08)',
            border: `1px solid ${
              isShadow
                ? 'rgba(107,155,247,0.15)'
                : 'rgba(91,186,111,0.15)'
            }`,
            borderRadius: 6,
          }}>
            <div style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: isShadow ? '#6B9BF7' : '#5BBA6F',
            }} />
            <span style={{
              fontFamily: 'var(--mono)',
              fontSize: 10,
              color: isShadow ? '#6B9BF7' : '#5BBA6F',
              textTransform: 'uppercase',
              letterSpacing: 0.5,
            }}>
              {reflexMode}
            </span>
          </div>

          {/* Stats */}
          <div style={{
            display: 'flex',
            gap: 16,
          }}>
            {[
              { label: 'Hits', value: reflexStats?.hits || '0' },
              { label: 'Misses', value: reflexStats?.misses || '0' },
              { label: 'Compiled', value: reflexStats?.compiled || '0' },
            ].map((stat) => (
              <div key={stat.label} style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}>
                <span style={{
                  fontFamily: 'var(--mono)',
                  fontSize: 14,
                  fontWeight: 600,
                  color: 'rgba(255,255,255,0.7)',
                }}>
                  {stat.value}
                </span>
                <span style={{
                  fontFamily: 'var(--mono)',
                  fontSize: 8,
                  color: 'rgba(255,255,255,0.2)',
                  letterSpacing: 0.5,
                  textTransform: 'uppercase',
                }}>
                  {stat.label}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Manual Promotion Section */}
      {isShadow && (
        <div style={{
          padding: '14px 16px',
          background: 'rgba(212,165,71,0.03)',
          border: '1px solid rgba(212,165,71,0.1)',
          borderRadius: 10,
        }}>
          <SectionLabel>Promotion to Active Mode</SectionLabel>
          <p style={{
            fontFamily: 'var(--sans)',
            fontSize: 12,
            color: 'rgba(255,255,255,0.4)',
            margin: '0 0 12px 0',
            lineHeight: 1.5,
          }}>
            Your reflex engine is in shadow mode (observing only). When ready, promote manually:
          </p>

          {/* Command */}
          <div style={{
            padding: '10px 14px',
            background: 'rgba(0,0,0,0.3)',
            border: '1px solid rgba(255,255,255,0.06)',
            borderRadius: 8,
            fontFamily: 'var(--mono)',
            fontSize: 11,
            color: '#E8D5A3',
            lineHeight: 1.6,
            overflowX: 'auto',
            whiteSpace: 'nowrap',
            userSelect: 'all',
            cursor: 'text',
          }}>
            bizra-node --reflex-mode active --policy-hash {policyHash}
          </div>

          {/* Criteria checklist */}
          <div style={{ marginTop: 14 }}>
            <div style={{
              fontFamily: 'var(--mono)',
              fontSize: 9,
              color: 'rgba(255,255,255,0.2)',
              letterSpacing: 1,
              textTransform: 'uppercase',
              marginBottom: 6,
            }}>
              Readiness Criteria (informational)
            </div>
            <CriteriaItem
              label="Stable knows_me trend"
              description="Score should be consistently rising, not volatile."
            />
            <CriteriaItem
              label="No quarantine spikes"
              description="No sudden increases in quarantined fragments."
            />
            <CriteriaItem
              label="No guardian-veto regressions"
              description="Guardian council has not blocked recent actions."
            />
          </div>
        </div>
      )}

      {/* Already active */}
      {!isShadow && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '10px 14px',
          background: 'rgba(91,186,111,0.04)',
          border: '1px solid rgba(91,186,111,0.1)',
          borderRadius: 8,
        }}>
          <div style={{
            width: 6,
            height: 6,
            borderRadius: '50%',
            background: '#5BBA6F',
          }} />
          <span style={{
            fontFamily: 'var(--sans)',
            fontSize: 11,
            color: 'rgba(91,186,111,0.7)',
          }}>
            Reflex engine is active. Your node is fully operational.
          </span>
        </div>
      )}

      {/* Enter Dashboard button */}
      <button
        onClick={onNext}
        style={{
          alignSelf: 'center',
          marginTop: 4,
          padding: '14px 44px',
          background: 'linear-gradient(135deg, #C9A962, #8B7340)',
          border: 'none',
          borderRadius: 10,
          fontFamily: 'var(--sans)',
          fontSize: 15,
          fontWeight: 600,
          color: '#030810',
          cursor: 'pointer',
          boxShadow: '0 4px 24px rgba(212,165,71,0.3)',
          transition: 'all 0.3s ease',
          letterSpacing: 0.3,
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.boxShadow = '0 8px 32px rgba(212,165,71,0.4)';
          e.currentTarget.style.transform = 'translateY(-2px)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.boxShadow = '0 4px 24px rgba(212,165,71,0.3)';
          e.currentTarget.style.transform = 'translateY(0)';
        }}
      >
        Enter Dashboard
      </button>

      <style>{`
        @keyframes onb-fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
      `}</style>
    </div>
  );
}
