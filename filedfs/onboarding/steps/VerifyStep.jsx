// ============================================================
// VerifyStep — Installation verification checklist
// ============================================================
// Sends PING, VERSION, HEALTH to the node.
// Displays animated checkmarks as each succeeds.
// Continue button appears only when all three pass.
// ============================================================

import { useState, useEffect, useCallback, useRef } from 'react';

const CHECK_ITEMS = [
  { key: 'ping', label: 'Node connection', description: 'Sending PING to bizra-node...' },
  { key: 'version', label: 'Protocol version', description: 'Checking VERSION compatibility...' },
  { key: 'health', label: 'System health', description: 'Fetching HEALTH diagnostics...' },
];

// ── Animated Checkmark SVG ────────────────────────────────────

const Checkmark = ({ visible, size = 20 }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 20 20"
    style={{
      opacity: visible ? 1 : 0,
      transform: visible ? 'scale(1)' : 'scale(0.5)',
      transition: 'all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)',
    }}
  >
    <circle cx="10" cy="10" r="9" fill="rgba(91,186,111,0.12)" stroke="#5BBA6F" strokeWidth="1.5" />
    <path
      d="M6 10.5 L9 13.5 L14.5 7"
      fill="none"
      stroke="#5BBA6F"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{
        strokeDasharray: 20,
        strokeDashoffset: visible ? 0 : 20,
        transition: 'stroke-dashoffset 0.5s ease 0.15s',
      }}
    />
  </svg>
);

// ── Spinner SVG ───────────────────────────────────────────────

const Spinner = ({ size = 20 }) => (
  <svg width={size} height={size} viewBox="0 0 20 20" style={{ animation: 'onb-spin 1s linear infinite' }}>
    <circle cx="10" cy="10" r="8" fill="none" stroke="rgba(212,165,71,0.15)" strokeWidth="2" />
    <path d="M10 2 A8 8 0 0 1 18 10" fill="none" stroke="#C9A962" strokeWidth="2" strokeLinecap="round" />
  </svg>
);

// ── Error Icon ────────────────────────────────────────────────

const ErrorIcon = ({ size = 20 }) => (
  <svg width={size} height={size} viewBox="0 0 20 20">
    <circle cx="10" cy="10" r="9" fill="rgba(232,93,74,0.12)" stroke="#E85D4A" strokeWidth="1.5" />
    <path d="M7 7 L13 13 M13 7 L7 13" stroke="#E85D4A" strokeWidth="1.8" strokeLinecap="round" />
  </svg>
);

// ── Check Item Row ────────────────────────────────────────────

const CheckItem = ({ label, description, status, detail, delay }) => (
  <div style={{
    display: 'flex',
    alignItems: 'center',
    gap: 14,
    padding: '14px 16px',
    background: status === 'pass'
      ? 'rgba(91,186,111,0.04)'
      : status === 'fail'
        ? 'rgba(232,93,74,0.04)'
        : 'rgba(255,255,255,0.02)',
    border: `1px solid ${
      status === 'pass'
        ? 'rgba(91,186,111,0.12)'
        : status === 'fail'
          ? 'rgba(232,93,74,0.12)'
          : 'rgba(255,255,255,0.04)'
    }`,
    borderRadius: 10,
    transition: 'all 0.4s ease',
    opacity: status === 'pending' ? 0.5 : 1,
    transform: status !== 'pending' ? 'translateX(0)' : 'translateX(-4px)',
  }}>
    <div style={{ flexShrink: 0, width: 20, height: 20 }}>
      {status === 'checking' && <Spinner />}
      {status === 'pass' && <Checkmark visible={true} />}
      {status === 'fail' && <ErrorIcon />}
      {status === 'pending' && (
        <div style={{
          width: 20,
          height: 20,
          borderRadius: '50%',
          border: '1.5px solid rgba(255,255,255,0.08)',
        }} />
      )}
    </div>
    <div style={{ flex: 1 }}>
      <div style={{
        fontFamily: 'var(--sans)',
        fontSize: 13,
        fontWeight: 500,
        color: status === 'pass'
          ? 'rgba(91,186,111,0.9)'
          : status === 'fail'
            ? 'rgba(232,93,74,0.9)'
            : 'rgba(255,255,255,0.6)',
        marginBottom: 2,
      }}>
        {label}
      </div>
      <div style={{
        fontFamily: 'var(--mono)',
        fontSize: 10,
        color: 'rgba(255,255,255,0.25)',
      }}>
        {status === 'checking' ? description : detail || description}
      </div>
    </div>
  </div>
);

// ============================================================
// MAIN COMPONENT
// ============================================================

export default function VerifyStep({ node, state, setState, onNext }) {
  const [checks, setChecks] = useState({
    ping: { status: 'pending', detail: '' },
    version: { status: 'pending', detail: '' },
    health: { status: 'pending', detail: '' },
  });
  const [running, setRunning] = useState(false);
  const ranRef = useRef(false);

  const allPassed = Object.values(checks).every((c) => c.status === 'pass');

  const runChecks = useCallback(async () => {
    if (running) return;
    setRunning(true);

    // Reset
    setChecks({
      ping: { status: 'pending', detail: '' },
      version: { status: 'pending', detail: '' },
      health: { status: 'pending', detail: '' },
    });

    // 1. PING
    await new Promise((r) => setTimeout(r, 300));
    setChecks((prev) => ({ ...prev, ping: { status: 'checking', detail: '' } }));

    try {
      const pingResult = await node.send('PING');
      if (pingResult?.ok) {
        setChecks((prev) => ({
          ...prev,
          ping: { status: 'pass', detail: 'Node responded — connection active' },
        }));
      } else {
        throw new Error('PING returned not-ok');
      }
    } catch (err) {
      setChecks((prev) => ({
        ...prev,
        ping: { status: 'fail', detail: 'Could not reach node' },
      }));
      setRunning(false);
      return;
    }

    // 2. VERSION
    await new Promise((r) => setTimeout(r, 500));
    setChecks((prev) => ({ ...prev, version: { status: 'checking', detail: '' } }));

    try {
      const verResult = await node.send('VERSION');
      if (verResult?.ok && verResult.fields) {
        const f = verResult.fields;
        const versionStr = `${f.node || 'bizra-node'} v${f.version || '?'} (protocol ${f.protocol || '?'})`;
        setChecks((prev) => ({
          ...prev,
          version: { status: 'pass', detail: versionStr },
        }));
      } else {
        throw new Error('VERSION returned not-ok');
      }
    } catch (err) {
      setChecks((prev) => ({
        ...prev,
        version: { status: 'fail', detail: 'Version check failed' },
      }));
      setRunning(false);
      return;
    }

    // 3. HEALTH
    await new Promise((r) => setTimeout(r, 500));
    setChecks((prev) => ({ ...prev, health: { status: 'checking', detail: '' } }));

    try {
      const healthResult = await node.send('HEALTH');
      if (healthResult?.ok && healthResult.fields) {
        const f = healthResult.fields;
        const agents = f.agents_registered || '0';
        const ihsan = f.ihsan ? (parseInt(f.ihsan, 10) / 100).toFixed(1) + '%' : '?';
        const healthStr = `${agents} agents registered | Ihsan ${ihsan} | State: ${f.state || 'Unknown'}`;
        setChecks((prev) => ({
          ...prev,
          health: { status: 'pass', detail: healthStr },
        }));
      } else {
        throw new Error('HEALTH returned not-ok');
      }
    } catch (err) {
      setChecks((prev) => ({
        ...prev,
        health: { status: 'fail', detail: 'Health check failed' },
      }));
      setRunning(false);
      return;
    }

    // All passed
    setState({ installVerified: true });
    setRunning(false);
  }, [node, running, setState]);

  // Auto-run on mount
  useEffect(() => {
    if (!ranRef.current && node.connected) {
      ranRef.current = true;
      runChecks();
    }
  }, [node.connected, runChecks]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Title */}
      <div style={{ textAlign: 'center', marginBottom: 8 }}>
        <h2 style={{
          fontFamily: 'var(--sans)',
          fontSize: 20,
          fontWeight: 600,
          color: 'rgba(255,255,255,0.88)',
          margin: '0 0 6px 0',
        }}>
          Verifying Installation
        </h2>
        <p style={{
          fontFamily: 'var(--sans)',
          fontSize: 13,
          color: 'rgba(255,255,255,0.35)',
          margin: 0,
          lineHeight: 1.5,
        }}>
          Checking that your node is running and healthy.
        </p>
        {state?.contactEmail && (
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 6,
            marginTop: 10,
            padding: '5px 10px',
            background: 'rgba(212,165,71,0.08)',
            border: '1px solid rgba(212,165,71,0.15)',
            borderRadius: 999,
            fontFamily: 'var(--mono)',
            fontSize: 10,
            color: 'rgba(212,165,71,0.7)',
            letterSpacing: 0.4,
          }}>
            <span>Genesis contact:</span>
            <span style={{ color: 'rgba(255,255,255,0.65)' }}>{state.contactEmail}</span>
          </div>
        )}
      </div>

      {/* Checklist */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {CHECK_ITEMS.map((item) => (
          <CheckItem
            key={item.key}
            label={item.label}
            description={item.description}
            status={checks[item.key].status}
            detail={checks[item.key].detail}
          />
        ))}
      </div>

      {/* Retry button (if any check failed) */}
      {Object.values(checks).some((c) => c.status === 'fail') && !running && (
        <button
          onClick={() => {
            ranRef.current = false;
            runChecks();
          }}
          style={{
            alignSelf: 'center',
            marginTop: 8,
            padding: '8px 20px',
            background: 'rgba(232,93,74,0.1)',
            border: '1px solid rgba(232,93,74,0.2)',
            borderRadius: 8,
            fontFamily: 'var(--mono)',
            fontSize: 11,
            color: '#E85D4A',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = 'rgba(232,93,74,0.15)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'rgba(232,93,74,0.1)';
          }}
        >
          Retry Checks
        </button>
      )}

      {/* Continue button */}
      {allPassed && (
        <button
          onClick={onNext}
          style={{
            alignSelf: 'center',
            marginTop: 12,
            padding: '12px 36px',
            background: 'linear-gradient(135deg, #C9A962, #8B7340)',
            border: 'none',
            borderRadius: 10,
            fontFamily: 'var(--sans)',
            fontSize: 14,
            fontWeight: 600,
            color: '#030810',
            cursor: 'pointer',
            boxShadow: '0 4px 20px rgba(212,165,71,0.25)',
            transition: 'all 0.3s ease',
            opacity: 1,
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

      {/* Keyframe styles */}
      <style>{`
        @keyframes onb-spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        @keyframes onb-fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
      `}</style>
    </div>
  );
}
