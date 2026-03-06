// ============================================================
// LandingDemo — Founder Ops Agent Demo Landing Page
// ============================================================
// Accessible at /#/demo — showcases the 4-step agent loop:
// Observe → Reason → Act → Verify
// Includes feature highlights and CTA to enter the main app.
//
// Standing on Giants:
// - General Magic (1994): Agent = persona + capabilities + permits
// - Boyd (1976): OODA loop → Observe-Orient-Decide-Act
// - Shannon (1948): Signal quality as hard constraint
// ============================================================

import { useState, useEffect, useRef } from 'react';

// ── Animated Loop Step ──────────────────────────────────────

const LoopStep = ({ step, index, active }) => {
  const colors = ['#C9A962', '#6B9BF7', '#5BBA6F', '#A78BFA'];
  const color = colors[index % 4];

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: 12,
      flex: 1,
      minWidth: 140,
      padding: '24px 16px',
      background: active ? `${color}08` : 'rgba(255,255,255,0.02)',
      border: `1px solid ${active ? `${color}25` : 'rgba(255,255,255,0.04)'}`,
      borderRadius: 16,
      transition: 'all 0.6s ease',
      transform: active ? 'translateY(-4px)' : 'translateY(0)',
    }}>
      {/* Step number circle */}
      <div style={{
        width: 40,
        height: 40,
        borderRadius: '50%',
        background: active
          ? `linear-gradient(135deg, ${color}, ${color}80)`
          : 'rgba(255,255,255,0.04)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        transition: 'all 0.6s ease',
        boxShadow: active ? `0 4px 20px ${color}30` : 'none',
      }}>
        <span style={{
          fontFamily: 'var(--mono)',
          fontSize: 16,
          fontWeight: 700,
          color: active ? '#030810' : 'rgba(255,255,255,0.2)',
          transition: 'color 0.6s ease',
        }}>
          {index + 1}
        </span>
      </div>

      {/* Icon */}
      <div style={{
        fontSize: 28,
        lineHeight: 1,
        filter: active ? 'none' : 'grayscale(1) opacity(0.3)',
        transition: 'filter 0.6s ease',
      }}>
        {step.icon}
      </div>

      {/* Title */}
      <div style={{
        fontFamily: 'var(--sans)',
        fontSize: 15,
        fontWeight: 600,
        color: active ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.3)',
        transition: 'color 0.6s ease',
      }}>
        {step.title}
      </div>

      {/* Description */}
      <div style={{
        fontFamily: 'var(--sans)',
        fontSize: 12,
        color: active ? 'rgba(255,255,255,0.5)' : 'rgba(255,255,255,0.15)',
        textAlign: 'center',
        lineHeight: 1.5,
        transition: 'color 0.6s ease',
      }}>
        {step.description}
      </div>

      {/* Tech label */}
      <span style={{
        fontFamily: 'var(--mono)',
        fontSize: 9,
        color: active ? `${color}90` : 'rgba(255,255,255,0.1)',
        background: active ? `${color}10` : 'transparent',
        border: `1px solid ${active ? `${color}20` : 'transparent'}`,
        borderRadius: 4,
        padding: '2px 8px',
        letterSpacing: 0.5,
        transition: 'all 0.6s ease',
      }}>
        {step.tech}
      </span>
    </div>
  );
};

// ── Feature Card ────────────────────────────────────────────

const FeatureCard = ({ title, description, color, icon }) => (
  <div style={{
    padding: '20px 24px',
    background: 'rgba(255,255,255,0.02)',
    border: '1px solid rgba(255,255,255,0.04)',
    borderRadius: 14,
    display: 'flex',
    gap: 14,
    alignItems: 'flex-start',
  }}>
    <div style={{
      width: 36,
      height: 36,
      borderRadius: 10,
      background: `${color}12`,
      border: `1px solid ${color}20`,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      flexShrink: 0,
      fontSize: 18,
    }}>
      {icon}
    </div>
    <div>
      <div style={{
        fontFamily: 'var(--sans)',
        fontSize: 14,
        fontWeight: 600,
        color: 'rgba(255,255,255,0.85)',
        marginBottom: 4,
      }}>
        {title}
      </div>
      <div style={{
        fontFamily: 'var(--sans)',
        fontSize: 12,
        color: 'rgba(255,255,255,0.4)',
        lineHeight: 1.5,
      }}>
        {description}
      </div>
    </div>
  </div>
);

// ── Stat Counter ────────────────────────────────────────────

const StatCounter = ({ value, label, color }) => (
  <div style={{ textAlign: 'center' }}>
    <div style={{
      fontFamily: 'var(--mono)',
      fontSize: 28,
      fontWeight: 700,
      color,
      letterSpacing: -1,
    }}>
      {value}
    </div>
    <div style={{
      fontFamily: 'var(--mono)',
      fontSize: 9,
      color: 'rgba(255,255,255,0.3)',
      letterSpacing: 1,
      textTransform: 'uppercase',
      marginTop: 4,
    }}>
      {label}
    </div>
  </div>
);

// ============================================================
// MAIN COMPONENT
// ============================================================

const LOOP_STEPS = [
  {
    title: 'Observe',
    description: 'Desktop context capture — window list, foreground app, clipboard state',
    tech: 'AHK Bridge + HDA',
    icon: '\u{1F441}', // eye
  },
  {
    title: 'Reason',
    description: 'Graph-of-Thoughts with PAT agents — strategist, analyst, guardian',
    tech: 'GoT + Entropy Router',
    icon: '\u{1F9E0}', // brain
  },
  {
    title: 'Act',
    description: 'Desktop automation with Telescript Permits — capability-scoped, time-limited',
    tech: 'Telescript + Permits',
    icon: '\u{26A1}', // lightning
  },
  {
    title: 'Verify',
    description: 'Cryptographic proof of every action — SHA-256 outcome hash, audit trail',
    tech: 'ActionReceipt + BLAKE3',
    icon: '\u{1F512}', // lock
  },
];

const FEATURES = [
  {
    title: 'Morning Briefs',
    description: 'Automated daily briefings at 08:00 — overnight alerts, priority tasks, calendar context.',
    color: '#C9A962',
    icon: '\u{2600}', // sun
  },
  {
    title: '10 Desktop Skills',
    description: 'Open apps, switch windows, type text, click elements, screenshot, read clipboard, navigate browser.',
    color: '#6B9BF7',
    icon: '\u{1F5A5}', // desktop
  },
  {
    title: 'Reasoning Transparency',
    description: 'Every answer shows its reasoning graph — GoT nodes, SNR scores, guardian verdicts.',
    color: '#5BBA6F',
    icon: '\u{1F50D}', // magnifier
  },
  {
    title: 'Constitutional Safety',
    description: 'Telescript Permits enforce capability scope, budget limits, and TTL. FATE gates prevent unsafe actions.',
    color: '#A78BFA',
    icon: '\u{1F6E1}', // shield
  },
  {
    title: 'Proactive Kernel',
    description: 'Scheduled missions with confidence gates — auto-execute low-risk, human-in-the-loop for high-risk.',
    color: '#F59E42',
    icon: '\u{23F0}', // alarm
  },
  {
    title: 'Exportable Agent',
    description: 'Package as .bizra-agent archive — manifest, permits, and README. Load on any node.',
    color: '#4ecdc4',
    icon: '\u{1F4E6}', // package
  },
];

export default function LandingDemo({ onEnterApp }) {
  const [activeStep, setActiveStep] = useState(0);
  const intervalRef = useRef(null);

  // Auto-cycle through loop steps
  useEffect(() => {
    intervalRef.current = setInterval(() => {
      setActiveStep((prev) => (prev + 1) % LOOP_STEPS.length);
    }, 3000);
    return () => clearInterval(intervalRef.current);
  }, []);

  return (
    <div style={{
      minHeight: '100vh',
      background: '#030810',
      color: 'rgba(255,255,255,0.85)',
      fontFamily: 'var(--sans)',
      overflowY: 'auto',
    }}>
      {/* Background radials */}
      <div style={{
        position: 'fixed',
        inset: 0,
        pointerEvents: 'none',
        background: 'radial-gradient(ellipse at 30% 10%, rgba(212,165,71,0.06) 0%, transparent 50%), radial-gradient(ellipse at 70% 90%, rgba(107,155,247,0.03) 0%, transparent 50%)',
        zIndex: 0,
      }} />

      <div style={{
        position: 'relative',
        zIndex: 1,
        maxWidth: 960,
        margin: '0 auto',
        padding: '60px 32px 80px',
      }}>
        {/* ── Hero Section ──────────────────────────────────── */}
        <div style={{ textAlign: 'center', marginBottom: 64 }}>
          {/* Logo */}
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: 56,
            height: 56,
            borderRadius: 16,
            background: 'linear-gradient(135deg, #C9A962, #8B7340)',
            marginBottom: 24,
            boxShadow: '0 8px 32px rgba(212,165,71,0.25)',
          }}>
            <span style={{
              fontFamily: 'var(--mono)',
              fontSize: 24,
              fontWeight: 700,
              color: '#030810',
            }}>
              B
            </span>
          </div>

          <h1 style={{
            fontFamily: 'var(--sans)',
            fontSize: 36,
            fontWeight: 700,
            color: 'rgba(255,255,255,0.95)',
            margin: '0 0 12px 0',
            letterSpacing: -0.5,
            lineHeight: 1.2,
          }}>
            Founder Ops Agent
          </h1>

          <p style={{
            fontFamily: 'var(--sans)',
            fontSize: 16,
            color: 'rgba(255,255,255,0.4)',
            margin: '0 0 8px 0',
            lineHeight: 1.6,
            maxWidth: 560,
            marginLeft: 'auto',
            marginRight: 'auto',
          }}>
            Your AI operations agent that acts on your desktop,
            proves what it did, and explains why.
          </p>

          <p style={{
            fontFamily: 'var(--mono)',
            fontSize: 11,
            color: 'rgba(212,165,71,0.5)',
            letterSpacing: 1,
            margin: 0,
          }}>
            OBSERVE &rarr; REASON &rarr; ACT &rarr; VERIFY
          </p>
        </div>

        {/* ── 4-Step Loop Visualizer ────────────────────────── */}
        <div style={{
          display: 'flex',
          gap: 12,
          marginBottom: 64,
          flexWrap: 'wrap',
          justifyContent: 'center',
        }}>
          {LOOP_STEPS.map((step, i) => (
            <LoopStep
              key={step.title}
              step={step}
              index={i}
              active={activeStep === i}
            />
          ))}
        </div>

        {/* ── Connector arrows between steps ─────────────────── */}
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          gap: 32,
          marginTop: -48,
          marginBottom: 48,
        }}>
          {[0, 1, 2].map((i) => (
            <span key={i} style={{
              fontFamily: 'var(--mono)',
              fontSize: 18,
              color: activeStep === i || activeStep === i + 1
                ? 'rgba(212,165,71,0.5)'
                : 'rgba(255,255,255,0.08)',
              transition: 'color 0.6s ease',
            }}>
              &rarr;
            </span>
          ))}
        </div>

        {/* ── Stats Bar ─────────────────────────────────────── */}
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          gap: 48,
          padding: '28px 0',
          marginBottom: 48,
          borderTop: '1px solid rgba(255,255,255,0.04)',
          borderBottom: '1px solid rgba(255,255,255,0.04)',
        }}>
          <StatCounter value="10" label="HDA Skills" color="#C9A962" />
          <StatCounter value="3" label="PAT Agents" color="#6B9BF7" />
          <StatCounter value="4" label="Daily Missions" color="#5BBA6F" />
          <StatCounter value="100%" label="Proof Coverage" color="#A78BFA" />
        </div>

        {/* ── Feature Grid ──────────────────────────────────── */}
        <div style={{ marginBottom: 64 }}>
          <h2 style={{
            fontFamily: 'var(--sans)',
            fontSize: 22,
            fontWeight: 600,
            color: 'rgba(255,255,255,0.85)',
            textAlign: 'center',
            marginBottom: 28,
          }}>
            What It Does
          </h2>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
            gap: 12,
          }}>
            {FEATURES.map((f) => (
              <FeatureCard key={f.title} {...f} />
            ))}
          </div>
        </div>

        {/* ── Giants Attribution ─────────────────────────────── */}
        <div style={{
          textAlign: 'center',
          padding: '24px 0',
          borderTop: '1px solid rgba(255,255,255,0.04)',
          marginBottom: 48,
        }}>
          <div style={{
            fontFamily: 'var(--mono)',
            fontSize: 9,
            color: 'rgba(255,255,255,0.15)',
            letterSpacing: 1,
            textTransform: 'uppercase',
            marginBottom: 8,
          }}>
            Standing on Giants
          </div>
          <div style={{
            fontFamily: 'var(--sans)',
            fontSize: 12,
            color: 'rgba(255,255,255,0.3)',
            lineHeight: 1.6,
          }}>
            General Magic (1994) &middot; Shannon (1948) &middot; Boyd (1976) &middot; Al-Ghazali (1095) &middot; Besta (2024) &middot; Lamport (1978) &middot; Anthropic (2023)
          </div>
        </div>

        {/* ── CTA ────────────────────────────────────────────── */}
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 12,
        }}>
          <button
            onClick={onEnterApp}
            style={{
              padding: '14px 48px',
              background: 'linear-gradient(135deg, #C9A962, #8B7340)',
              border: 'none',
              borderRadius: 12,
              fontFamily: 'var(--sans)',
              fontSize: 16,
              fontWeight: 600,
              color: '#030810',
              cursor: 'pointer',
              boxShadow: '0 8px 32px rgba(212,165,71,0.25)',
              transition: 'all 0.3s ease',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.boxShadow = '0 12px 40px rgba(212,165,71,0.35)';
              e.currentTarget.style.transform = 'translateY(-2px)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.boxShadow = '0 8px 32px rgba(212,165,71,0.25)';
              e.currentTarget.style.transform = 'translateY(0)';
            }}
          >
            Enter Node0
          </button>

          <span style={{
            fontFamily: 'var(--mono)',
            fontSize: 10,
            color: 'rgba(255,255,255,0.2)',
            letterSpacing: 0.5,
          }}>
            BIZRA Node0 &middot; Alpha-100 &middot; v1.0.0
          </span>
        </div>
      </div>
    </div>
  );
}
