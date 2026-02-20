// ============================================================
// TeachStep — "Tell me about yourself"
// ============================================================
// Three text areas that send TEACH commands to the node:
//   1. "What do you do?" -> TEACH fact
//   2. "What matters to you?" -> TEACH preference
//   3. "What are you working toward?" -> TEACH goal
// Then triggers SYNTHESIZE. Animated feedback per field.
// ============================================================

import { useState, useCallback } from 'react';

const MAX_CHARS = 500;

const FIELDS = [
  {
    key: 'role',
    kind: 'fact',
    label: 'What do you do?',
    placeholder: 'I am a software architect building distributed systems...',
    hint: 'Your role, profession, or primary activity.',
  },
  {
    key: 'values',
    kind: 'preference',
    label: 'What matters to you?',
    placeholder: 'I care deeply about privacy, open source, and craftsmanship...',
    hint: 'Values, preferences, things you care about.',
  },
  {
    key: 'goal',
    kind: 'goal',
    label: 'What are you working toward?',
    placeholder: 'My goal is to build sovereign AI that serves everyone...',
    hint: 'Your current goal or aspiration.',
  },
];

// ── Character Counter ─────────────────────────────────────────

const CharCounter = ({ current, max }) => {
  const ratio = current / max;
  const color = ratio > 0.9
    ? '#E85D4A'
    : ratio > 0.7
      ? '#D4A547'
      : 'rgba(255,255,255,0.2)';
  return (
    <span style={{
      fontFamily: 'var(--mono)',
      fontSize: 9,
      color,
      transition: 'color 0.2s ease',
    }}>
      {current}/{max}
    </span>
  );
};

// ── Status Badge ──────────────────────────────────────────────

const StatusBadge = ({ status }) => {
  if (status === 'idle') return null;

  const configs = {
    sending: {
      text: 'Learning...',
      color: '#D4A547',
      bg: 'rgba(212,165,71,0.08)',
      border: 'rgba(212,165,71,0.15)',
    },
    done: {
      text: 'Remembered',
      color: '#5BBA6F',
      bg: 'rgba(91,186,111,0.08)',
      border: 'rgba(91,186,111,0.15)',
    },
    error: {
      text: 'Failed',
      color: '#E85D4A',
      bg: 'rgba(232,93,74,0.08)',
      border: 'rgba(232,93,74,0.15)',
    },
  };

  const cfg = configs[status] || configs.idle;
  if (!cfg) return null;

  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 5,
      fontFamily: 'var(--mono)',
      fontSize: 9,
      color: cfg.color,
      background: cfg.bg,
      border: `1px solid ${cfg.border}`,
      borderRadius: 4,
      padding: '2px 8px',
      letterSpacing: 0.5,
      animation: status === 'sending' ? 'onb-pulse 1.5s ease infinite' : 'onb-fadeUp 0.3s ease',
    }}>
      {status === 'done' && (
        <svg width="10" height="10" viewBox="0 0 10 10">
          <path d="M2 5.5 L4 7.5 L8 3" fill="none" stroke="#5BBA6F" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      )}
      {cfg.text}
    </span>
  );
};

// ── Teach Field ───────────────────────────────────────────────

const TeachField = ({ field, value, onChange, status, disabled }) => (
  <div style={{
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
  }}>
    {/* Label row */}
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
    }}>
      <label style={{
        fontFamily: 'var(--sans)',
        fontSize: 13,
        fontWeight: 500,
        color: 'rgba(255,255,255,0.7)',
      }}>
        {field.label}
      </label>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <StatusBadge status={status} />
        <CharCounter current={value.length} max={MAX_CHARS} />
      </div>
    </div>

    {/* Hint */}
    <span style={{
      fontFamily: 'var(--sans)',
      fontSize: 10,
      color: 'rgba(255,255,255,0.2)',
      marginTop: -2,
    }}>
      {field.hint}
    </span>

    {/* Textarea */}
    <textarea
      value={value}
      onChange={(e) => {
        if (e.target.value.length <= MAX_CHARS) {
          onChange(field.key, e.target.value);
        }
      }}
      placeholder={field.placeholder}
      disabled={disabled}
      rows={3}
      style={{
        width: '100%',
        background: status === 'done'
          ? 'rgba(91,186,111,0.03)'
          : 'rgba(255,255,255,0.03)',
        border: `1px solid ${
          status === 'done'
            ? 'rgba(91,186,111,0.12)'
            : 'rgba(255,255,255,0.06)'
        }`,
        borderRadius: 10,
        padding: '10px 14px',
        fontFamily: 'var(--sans)',
        fontSize: 13,
        color: 'rgba(255,255,255,0.8)',
        lineHeight: 1.5,
        resize: 'none',
        outline: 'none',
        transition: 'all 0.25s ease',
        opacity: disabled ? 0.6 : 1,
        boxSizing: 'border-box',
      }}
      onFocus={(e) => {
        if (status !== 'done') {
          e.currentTarget.style.borderColor = 'rgba(212,165,71,0.3)';
        }
      }}
      onBlur={(e) => {
        if (status !== 'done') {
          e.currentTarget.style.borderColor = 'rgba(255,255,255,0.06)';
        }
      }}
    />
  </div>
);

// ============================================================
// MAIN COMPONENT
// ============================================================

export default function TeachStep({ node, state, setState, onNext }) {
  const [values, setValues] = useState({
    role: state.teachData?.role || '',
    values: state.teachData?.values || '',
    goal: state.teachData?.goal || '',
  });
  const [statuses, setStatuses] = useState({
    role: 'idle',
    values: 'idle',
    goal: 'idle',
  });
  const [synthesizeStatus, setSynthesizeStatus] = useState('idle');
  const [submitting, setSubmitting] = useState(false);

  const allDone = Object.values(statuses).every((s) => s === 'done') && synthesizeStatus === 'done';
  const hasContent = Object.values(values).some((v) => v.trim().length > 0);

  const handleChange = useCallback((key, value) => {
    setValues((prev) => ({ ...prev, [key]: value }));
  }, []);

  const handleSubmit = useCallback(async () => {
    if (submitting || !hasContent) return;
    setSubmitting(true);

    // Save to onboarding state
    setState({ teachData: { ...values } });

    // Send TEACH commands sequentially with staggered feedback
    for (const field of FIELDS) {
      const text = values[field.key]?.trim();
      if (!text) {
        setStatuses((prev) => ({ ...prev, [field.key]: 'done' }));
        continue;
      }

      setStatuses((prev) => ({ ...prev, [field.key]: 'sending' }));

      try {
        const result = await node.send('TEACH', {
          kind: field.kind,
          content: text,
          confidence: 9000,
          timestamp: Date.now(),
        });

        // Brief pause for animation
        await new Promise((r) => setTimeout(r, 400));

        if (result?.ok) {
          setStatuses((prev) => ({ ...prev, [field.key]: 'done' }));
        } else {
          setStatuses((prev) => ({ ...prev, [field.key]: 'error' }));
        }
      } catch (err) {
        setStatuses((prev) => ({ ...prev, [field.key]: 'error' }));
      }
    }

    // SYNTHESIZE
    await new Promise((r) => setTimeout(r, 300));
    setSynthesizeStatus('sending');

    try {
      const synResult = await node.send('SYNTHESIZE', { timestamp: Date.now() });
      await new Promise((r) => setTimeout(r, 500));

      if (synResult?.ok) {
        setSynthesizeStatus('done');
      } else {
        setSynthesizeStatus('error');
      }
    } catch (err) {
      setSynthesizeStatus('error');
    }

    setSubmitting(false);
  }, [submitting, hasContent, values, node, setState]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Title */}
      <div style={{ textAlign: 'center', marginBottom: 4 }}>
        <h2 style={{
          fontFamily: 'var(--sans)',
          fontSize: 20,
          fontWeight: 600,
          color: 'rgba(255,255,255,0.88)',
          margin: '0 0 6px 0',
        }}>
          Tell Me About Yourself
        </h2>
        <p style={{
          fontFamily: 'var(--sans)',
          fontSize: 13,
          color: 'rgba(255,255,255,0.35)',
          margin: 0,
          lineHeight: 1.5,
        }}>
          These become your node's foundational memory. You can teach more later.
        </p>
      </div>

      {/* Fields */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        {FIELDS.map((field) => (
          <TeachField
            key={field.key}
            field={field}
            value={values[field.key]}
            onChange={handleChange}
            status={statuses[field.key]}
            disabled={submitting}
          />
        ))}
      </div>

      {/* Synthesize status */}
      {synthesizeStatus !== 'idle' && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 8,
          padding: '10px 16px',
          background: synthesizeStatus === 'done'
            ? 'rgba(91,186,111,0.04)'
            : synthesizeStatus === 'error'
              ? 'rgba(232,93,74,0.04)'
              : 'rgba(212,165,71,0.04)',
          border: `1px solid ${
            synthesizeStatus === 'done'
              ? 'rgba(91,186,111,0.12)'
              : synthesizeStatus === 'error'
                ? 'rgba(232,93,74,0.12)'
                : 'rgba(212,165,71,0.12)'
          }`,
          borderRadius: 8,
          animation: 'onb-fadeUp 0.3s ease',
        }}>
          {synthesizeStatus === 'sending' && (
            <svg width="14" height="14" viewBox="0 0 14 14" style={{ animation: 'onb-spin 1s linear infinite' }}>
              <circle cx="7" cy="7" r="5.5" fill="none" stroke="rgba(212,165,71,0.2)" strokeWidth="1.5" />
              <path d="M7 1.5 A5.5 5.5 0 0 1 12.5 7" fill="none" stroke="#D4A547" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          )}
          {synthesizeStatus === 'done' && (
            <svg width="14" height="14" viewBox="0 0 14 14">
              <path d="M3 7.5 L5.5 10 L11 4" fill="none" stroke="#5BBA6F" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          )}
          <span style={{
            fontFamily: 'var(--mono)',
            fontSize: 10,
            color: synthesizeStatus === 'done'
              ? '#5BBA6F'
              : synthesizeStatus === 'error'
                ? '#E85D4A'
                : '#D4A547',
            letterSpacing: 0.5,
          }}>
            {synthesizeStatus === 'sending'
              ? 'Synthesizing memories...'
              : synthesizeStatus === 'done'
                ? 'Memory synthesis complete'
                : 'Synthesis failed'}
          </span>
        </div>
      )}

      {/* Action buttons */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 12,
        marginTop: 8,
      }}>
        {!allDone && (
          <button
            onClick={handleSubmit}
            disabled={!hasContent || submitting}
            style={{
              padding: '12px 36px',
              background: hasContent && !submitting
                ? 'linear-gradient(135deg, #D4A547, #8B6914)'
                : 'rgba(255,255,255,0.04)',
              border: 'none',
              borderRadius: 10,
              fontFamily: 'var(--sans)',
              fontSize: 14,
              fontWeight: 600,
              color: hasContent && !submitting ? '#0A0B0F' : 'rgba(255,255,255,0.15)',
              cursor: hasContent && !submitting ? 'pointer' : 'default',
              boxShadow: hasContent && !submitting ? '0 4px 20px rgba(212,165,71,0.25)' : 'none',
              transition: 'all 0.3s ease',
            }}
            onMouseEnter={(e) => {
              if (hasContent && !submitting) {
                e.currentTarget.style.boxShadow = '0 6px 28px rgba(212,165,71,0.35)';
                e.currentTarget.style.transform = 'translateY(-1px)';
              }
            }}
            onMouseLeave={(e) => {
              if (hasContent && !submitting) {
                e.currentTarget.style.boxShadow = '0 4px 20px rgba(212,165,71,0.25)';
                e.currentTarget.style.transform = 'translateY(0)';
              }
            }}
          >
            {submitting ? 'Teaching...' : 'Teach My Node'}
          </button>
        )}

        {/* Skip button (only before submitting) */}
        {!submitting && !allDone && (
          <button
            onClick={onNext}
            style={{
              padding: '8px 16px',
              background: 'none',
              border: 'none',
              fontFamily: 'var(--mono)',
              fontSize: 11,
              color: 'rgba(255,255,255,0.2)',
              cursor: 'pointer',
              transition: 'color 0.2s ease',
            }}
            onMouseEnter={(e) => { e.currentTarget.style.color = 'rgba(255,255,255,0.4)'; }}
            onMouseLeave={(e) => { e.currentTarget.style.color = 'rgba(255,255,255,0.2)'; }}
          >
            Skip for now
          </button>
        )}

        {/* Continue after done */}
        {allDone && (
          <button
            onClick={onNext}
            style={{
              padding: '12px 36px',
              background: 'linear-gradient(135deg, #D4A547, #8B6914)',
              border: 'none',
              borderRadius: 10,
              fontFamily: 'var(--sans)',
              fontSize: 14,
              fontWeight: 600,
              color: '#0A0B0F',
              cursor: 'pointer',
              boxShadow: '0 4px 20px rgba(212,165,71,0.25)',
              transition: 'all 0.3s ease',
              animation: 'onb-fadeUp 0.4s ease',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.boxShadow = '0 6px 28px rgba(212,165,71,0.35)';
              e.currentTarget.style.transform = 'translateY(-1px)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.boxShadow = '0 4px 20px rgba(212,165,71,0.25)';
              e.currentTarget.style.transform = 'translateY(0)';
            }}
          >
            Continue
          </button>
        )}
      </div>

      <style>{`
        @keyframes onb-spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        @keyframes onb-fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes onb-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }
      `}</style>
    </div>
  );
}
