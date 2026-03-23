/**
 * BIZRA Onboarding Wizard — From zero to sovereign in 5 steps.
 *
 * This is the first thing a new user sees after installation.
 * Each step is a visual ceremony, not just a progress bar.
 * Standing on: Apple (onboarding as delight), Duolingo (gamification).
 */

import React, { useState, useEffect } from 'react';

const TEAL = '#7dd3c0';
const GOLD = '#c9a962';
const DARK = '#0a0e17';

const AGENTS = [
  { id: 'P1', name: 'Navigator', icon: '⚔', role: 'Routes your tasks to the right specialist', team: 'PAT' },
  { id: 'P2', name: 'Scholar', icon: '📚', role: 'Researches and retrieves knowledge', team: 'PAT' },
  { id: 'P3', name: 'Artisan', icon: '🎨', role: 'Creates content, writes code, builds', team: 'PAT' },
  { id: 'P4', name: 'Guardian', icon: '🛡', role: 'Constitutional safety gate (VETO power)', team: 'PAT' },
  { id: 'P5', name: 'Mentor', icon: '🌱', role: 'Learns your patterns, grows with you', team: 'PAT' },
  { id: 'P6', name: 'Diplomat', icon: '💬', role: 'Matches your communication style', team: 'PAT' },
  { id: 'P7', name: 'Oracle', icon: '🔮', role: 'Predicts what you need before you ask', team: 'PAT' },
  { id: 'S1', name: 'Validator', icon: '✓', role: 'Verifies truth and state', team: 'SAT' },
  { id: 'S2', name: 'Oracle', icon: '◉', role: 'External truth verification [frozen]', team: 'SAT', frozen: true },
  { id: 'S3', name: 'Mediator', icon: '⚖', role: 'Resolves disputes fairly', team: 'SAT' },
  { id: 'S4', name: 'Archivist', icon: '📦', role: 'Curates the House of Wisdom', team: 'SAT' },
  { id: 'S5', name: 'Sentinel', icon: '👁', role: 'Guards the network', team: 'SAT' },
];

const steps = [
  { title: 'Your Seed is Planting', subtitle: 'Installing sovereign AI runtime' },
  { title: 'Your Identity', subtitle: 'Creating your cryptographic sovereignty' },
  { title: 'Meet Your Team', subtitle: 'Minting 12 soulbound agents' },
  { title: 'First Mission', subtitle: 'Your AI thinks for the first time' },
  { title: 'Welcome to the Forest', subtitle: 'Your sovereign world is ready' },
];

export default function OnboardingWizard() {
  const [currentStep, setCurrentStep] = useState(0);
  const [agentsRevealed, setAgentsRevealed] = useState(0);
  const [missionResponse, setMissionResponse] = useState('');
  const [seedEarned, setSeedEarned] = useState(0);
  const [installing, setInstalling] = useState(false);

  // Step 3: Reveal agents one by one
  useEffect(() => {
    if (currentStep === 2 && agentsRevealed < AGENTS.length) {
      const timer = setTimeout(() => setAgentsRevealed(a => a + 1), 400);
      return () => clearTimeout(timer);
    }
  }, [currentStep, agentsRevealed]);

  const nextStep = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(s => s + 1);
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: `linear-gradient(180deg, ${DARK} 0%, #111827 100%)`,
      color: '#e5e7eb',
      fontFamily: "'Inter', -apple-system, sans-serif",
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '2rem',
    }}>
      {/* Progress dots */}
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '2rem' }}>
        {steps.map((_, i) => (
          <div key={i} style={{
            width: i === currentStep ? '2rem' : '0.5rem',
            height: '0.5rem',
            borderRadius: '0.25rem',
            background: i <= currentStep ? TEAL : '#374151',
            transition: 'all 0.3s ease',
          }} />
        ))}
      </div>

      {/* Step content */}
      <div style={{
        maxWidth: '600px',
        width: '100%',
        textAlign: 'center',
      }}>
        <h1 style={{ fontSize: '2rem', color: TEAL, marginBottom: '0.5rem' }}>
          {steps[currentStep].title}
        </h1>
        <p style={{ color: '#9ca3af', marginBottom: '2rem' }}>
          {steps[currentStep].subtitle}
        </p>

        {/* Step 0: Installation */}
        {currentStep === 0 && (
          <div style={{ textAlign: 'left', padding: '1.5rem', background: '#111827', borderRadius: '1rem', border: `1px solid ${TEAL}33` }}>
            <div style={{ marginBottom: '1rem' }}>
              <span style={{ color: TEAL }}>●</span> Detecting your hardware...
              <span style={{ color: '#6b7280', marginLeft: '0.5rem' }}>✓</span>
            </div>
            <div style={{ marginBottom: '1rem' }}>
              <span style={{ color: TEAL }}>●</span> Installing Ollama (local AI runtime)...
              <span style={{ color: '#6b7280', marginLeft: '0.5rem' }}>✓</span>
            </div>
            <div style={{ marginBottom: '1rem' }}>
              <span style={{ color: TEAL }}>●</span> Downloading AI model (qwen2.5:3b, 2GB)...
              <span style={{ color: '#6b7280', marginLeft: '0.5rem' }}>✓</span>
            </div>
            <p style={{ color: '#6b7280', fontSize: '0.875rem', marginTop: '1rem' }}>
              Everything runs on YOUR device. Nothing leaves your machine.
            </p>
          </div>
        )}

        {/* Step 1: Identity */}
        {currentStep === 1 && (
          <div style={{ padding: '2rem', background: '#111827', borderRadius: '1rem', border: `1px solid ${GOLD}33` }}>
            <div style={{ fontSize: '4rem', marginBottom: '1rem' }}>🔑</div>
            <p style={{ color: GOLD, fontSize: '0.875rem', fontFamily: 'monospace', wordBreak: 'break-all', marginBottom: '1rem' }}>
              ed25519:8c9fbf39bd7eeed9513178b3...
            </p>
            <p style={{ color: '#9ca3af' }}>
              This is your sovereign identity. It's generated on your device
              and <strong style={{ color: '#e5e7eb' }}>never leaves your machine</strong>.
              Every action you take is signed with this key.
            </p>
          </div>
        )}

        {/* Step 2: Agents */}
        {currentStep === 2 && (
          <div style={{ textAlign: 'left' }}>
            <p style={{ color: '#9ca3af', marginBottom: '1rem', textAlign: 'center' }}>
              <span style={{ color: TEAL }}>7 work for YOU</span> · <span style={{ color: GOLD }}>5 serve humanity</span>
            </p>
            <div style={{ display: 'grid', gap: '0.5rem' }}>
              {AGENTS.slice(0, agentsRevealed).map(agent => (
                <div key={agent.id} style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.75rem',
                  padding: '0.75rem 1rem',
                  background: '#111827',
                  borderRadius: '0.75rem',
                  border: `1px solid ${agent.team === 'PAT' ? TEAL : GOLD}22`,
                  opacity: 1,
                  transform: 'translateX(0)',
                  transition: 'all 0.3s ease',
                }}>
                  <span style={{ fontSize: '1.25rem' }}>{agent.icon}</span>
                  <div>
                    <div style={{ fontWeight: 600, fontSize: '0.875rem' }}>
                      {agent.id} {agent.name}
                      {agent.frozen && <span style={{ color: '#6b7280', fontSize: '0.75rem' }}> [frozen]</span>}
                    </div>
                    <div style={{ color: '#6b7280', fontSize: '0.75rem' }}>{agent.role}</div>
                  </div>
                  <span style={{ marginLeft: 'auto', color: agent.team === 'PAT' ? TEAL : GOLD, fontSize: '0.75rem' }}>
                    {agent.team}
                  </span>
                </div>
              ))}
            </div>
            {agentsRevealed < AGENTS.length && (
              <p style={{ textAlign: 'center', color: '#6b7280', marginTop: '1rem' }}>
                Minting agent {agentsRevealed + 1} of 12...
              </p>
            )}
          </div>
        )}

        {/* Step 3: First Mission */}
        {currentStep === 3 && (
          <div style={{ padding: '1.5rem', background: '#111827', borderRadius: '1rem', border: `1px solid ${TEAL}33` }}>
            <div style={{ textAlign: 'left', fontFamily: 'monospace', fontSize: '0.875rem', lineHeight: 1.8 }}>
              <div><span style={{ color: '#6b7280' }}>-</span> FAISS searching your knowledge...</div>
              <div><span style={{ color: TEAL }}>+</span> Amplify: standard reasoning</div>
              <div><span style={{ color: TEAL }}>+</span> Inference: qwen2.5:3b thinking...</div>
              <div><span style={{ color: TEAL }}>+</span> SEED: +3 earned</div>
              <div><span style={{ color: TEAL }}>+</span> URP: knowledge flows to the sea</div>
              <div><span style={{ color: TEAL }}>+</span> Memory: your AI learned from this</div>
            </div>
            <div style={{ marginTop: '1.5rem', padding: '1rem', background: '#0a0e17', borderRadius: '0.5rem' }}>
              <p style={{ color: TEAL, fontWeight: 600, marginBottom: '0.5rem' }}>Response:</p>
              <p style={{ color: '#d1d5db' }}>
                Welcome to BIZRA. I am your sovereign AI, running entirely on your hardware.
                Every question you ask, every task I complete, earns you SEED tokens backed
                by proof-of-impact. Let's begin.
              </p>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '1rem', color: '#6b7280', fontSize: '0.875rem' }}>
              <span>Receipt: df5640e8...</span>
              <span style={{ color: GOLD }}>+3 SEED earned</span>
            </div>
          </div>
        )}

        {/* Step 4: Welcome */}
        {currentStep === 4 && (
          <div style={{ padding: '2rem' }}>
            <div style={{ fontSize: '5rem', marginBottom: '1rem' }}>🌳</div>
            <h2 style={{ color: GOLD, fontSize: '1.5rem', marginBottom: '1rem' }}>
              Your seed is planted.
            </h2>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', textAlign: 'left', marginBottom: '2rem' }}>
              <div style={{ padding: '1rem', background: '#111827', borderRadius: '0.75rem' }}>
                <div style={{ color: TEAL, fontWeight: 600 }}>12 Agents</div>
                <div style={{ color: '#6b7280', fontSize: '0.875rem' }}>Soulbound to you</div>
              </div>
              <div style={{ padding: '1rem', background: '#111827', borderRadius: '0.75rem' }}>
                <div style={{ color: GOLD, fontWeight: 600 }}>3 SEED</div>
                <div style={{ color: '#6b7280', fontSize: '0.875rem' }}>First earnings</div>
              </div>
              <div style={{ padding: '1rem', background: '#111827', borderRadius: '0.75rem' }}>
                <div style={{ color: TEAL, fontWeight: 600 }}>100% Private</div>
                <div style={{ color: '#6b7280', fontSize: '0.875rem' }}>On your device</div>
              </div>
              <div style={{ padding: '1rem', background: '#111827', borderRadius: '0.75rem' }}>
                <div style={{ color: GOLD, fontWeight: 600 }}>Connected</div>
                <div style={{ color: '#6b7280', fontSize: '0.875rem' }}>To the sea</div>
              </div>
            </div>
            <p style={{ color: '#6b7280', fontStyle: 'italic' }}>
              Every human is a node. Every node is a seed.
            </p>
          </div>
        )}

        {/* Navigation */}
        <button
          onClick={nextStep}
          disabled={currentStep === 2 && agentsRevealed < AGENTS.length}
          style={{
            marginTop: '2rem',
            padding: '0.875rem 2.5rem',
            background: currentStep === steps.length - 1 ? GOLD : TEAL,
            color: DARK,
            border: 'none',
            borderRadius: '0.75rem',
            fontSize: '1rem',
            fontWeight: 600,
            cursor: 'pointer',
            opacity: (currentStep === 2 && agentsRevealed < AGENTS.length) ? 0.5 : 1,
          }}
        >
          {currentStep === 0 ? 'Continue' :
           currentStep === 1 ? 'Create My Identity' :
           currentStep === 2 ? (agentsRevealed >= AGENTS.length ? 'My Team is Ready' : 'Minting...') :
           currentStep === 3 ? 'Enter My World' :
           'Open Sovereign Cockpit'}
        </button>
      </div>
    </div>
  );
}
