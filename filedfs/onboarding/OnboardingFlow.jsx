// ============================================================
// OnboardingFlow — Alpha-100 Onboarding Wizard
// ============================================================
// 5-step guided setup for BIZRA Node0.
// Manages step progression and shared onboarding state.
// Each step receives { node, state, setState, onNext }.
// ============================================================

import { useState, useCallback, useEffect } from 'react';
import VerifyStep from './steps/VerifyStep';
import ProviderStep from './steps/ProviderStep';
import TeachStep from './steps/TeachStep';
import FirstChatStep from './steps/FirstChatStep';
import DashboardStep from './steps/DashboardStep';

const STEPS = [
  { label: 'Verify', component: VerifyStep },
  { label: 'Provider', component: ProviderStep },
  { label: 'Teach', component: TeachStep },
  { label: 'Chat', component: FirstChatStep },
  { label: 'Dashboard', component: DashboardStep },
];

const TOTAL_STEPS = STEPS.length;

// ── Step Indicator Dots ───────────────────────────────────────

const StepIndicator = ({ currentStep, totalSteps }) => (
  <div style={{
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    marginBottom: 24,
  }}>
    {Array.from({ length: totalSteps }, (_, i) => (
      <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <div style={{
          width: i === currentStep ? 10 : 8,
          height: i === currentStep ? 10 : 8,
          borderRadius: '50%',
          background: i < currentStep
            ? '#C9A962'
            : i === currentStep
              ? 'linear-gradient(135deg, #C9A962, #E8D5A3)'
              : 'rgba(255,255,255,0.08)',
          border: i === currentStep
            ? '2px solid rgba(212,165,71,0.4)'
            : i < currentStep
              ? '2px solid rgba(212,165,71,0.3)'
              : '2px solid rgba(255,255,255,0.04)',
          boxShadow: i === currentStep ? '0 0 12px rgba(212,165,71,0.3)' : 'none',
          transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
        }} />
        {i < totalSteps - 1 && (
          <div style={{
            width: 32,
            height: 1,
            background: i < currentStep
              ? 'rgba(212,165,71,0.3)'
              : 'rgba(255,255,255,0.06)',
            transition: 'background 0.4s ease',
          }} />
        )}
      </div>
    ))}
  </div>
);

// ── Step Label ────────────────────────────────────────────────

const StepLabel = ({ currentStep, totalSteps, labels }) => (
  <div style={{
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 8,
  }}>
    <span style={{
      fontFamily: 'var(--mono)',
      fontSize: 10,
      color: 'rgba(212,165,71,0.5)',
      letterSpacing: 1.5,
      textTransform: 'uppercase',
    }}>
      Step {currentStep + 1} of {totalSteps}
    </span>
    <span style={{
      fontFamily: 'var(--mono)',
      fontSize: 10,
      color: 'rgba(255,255,255,0.25)',
      letterSpacing: 1,
      textTransform: 'uppercase',
    }}>
      {labels[currentStep]}
    </span>
  </div>
);

// ── Progress Bar ──────────────────────────────────────────────

const ProgressBar = ({ currentStep, totalSteps }) => {
  const progress = ((currentStep + 1) / totalSteps) * 100;
  return (
    <div style={{
      width: '100%',
      height: 2,
      background: 'rgba(255,255,255,0.04)',
      borderRadius: 1,
      overflow: 'hidden',
      marginBottom: 28,
    }}>
      <div style={{
        width: `${progress}%`,
        height: '100%',
        background: 'linear-gradient(90deg, #C9A962, #E8D5A3)',
        borderRadius: 1,
        transition: 'width 0.6s cubic-bezier(0.4, 0, 0.2, 1)',
      }} />
    </div>
  );
};

// ============================================================
// MAIN COMPONENT
// ============================================================

export default function OnboardingFlow({ node, onComplete }) {
  const [step, setStep] = useState(0);
  const [direction, setDirection] = useState(1); // 1 = forward, -1 = back
  const [transitioning, setTransitioning] = useState(false);
  const [onboardingState, setOnboardingState] = useState({
    provider: 'local',
    model: '',
    apiKey: '',
    policyHash: '',
    installVerified: false,
    teachData: {
      role: '',
      values: '',
      goal: '',
    },
    agentOps: {
      work_schedule: '',
      primary_tools: '',
      communication_pref: '',
      priority_domains: '',
      automation_comfort: '',
    },
    firstChatComplete: false,
  });

  const goNext = useCallback(() => {
    if (step >= TOTAL_STEPS - 1) return;
    setDirection(1);
    setTransitioning(true);
    setTimeout(() => {
      setStep((s) => s + 1);
      setTransitioning(false);
    }, 200);
  }, [step]);

  const goBack = useCallback(() => {
    if (step <= 0) return;
    setDirection(-1);
    setTransitioning(true);
    setTimeout(() => {
      setStep((s) => s - 1);
      setTransitioning(false);
    }, 200);
  }, [step]);

  const updateState = useCallback((updates) => {
    setOnboardingState((prev) => ({
      ...prev,
      ...(typeof updates === 'function' ? updates(prev) : updates),
    }));
  }, []);

  const CurrentStepComponent = STEPS[step].component;

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      zIndex: 1000,
      background: '#030810',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: 'var(--sans)',
    }}>
      {/* Background radials */}
      <div style={{
        position: 'absolute',
        inset: 0,
        pointerEvents: 'none',
        background: 'radial-gradient(ellipse at 30% 20%, rgba(212,165,71,0.04) 0%, transparent 60%), radial-gradient(ellipse at 70% 80%, rgba(107,155,247,0.02) 0%, transparent 50%)',
      }} />

      {/* Main card */}
      <div style={{
        position: 'relative',
        width: '100%',
        maxWidth: 640,
        maxHeight: 'calc(100vh - 48px)',
        margin: '24px',
        background: 'rgba(255,255,255,0.03)',
        border: '1px solid rgba(255,255,255,0.06)',
        borderRadius: 16,
        backdropFilter: 'blur(16px)',
        WebkitBackdropFilter: 'blur(16px)',
        padding: '32px 36px',
        display: 'flex',
        flexDirection: 'column',
        overflowY: 'auto',
      }}>
        {/* Header */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 10,
          marginBottom: 20,
        }}>
          <div style={{
            width: 28,
            height: 28,
            borderRadius: 8,
            background: 'linear-gradient(135deg, #C9A962, #8B7340)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 14,
            fontWeight: 700,
            color: '#030810',
            fontFamily: 'var(--mono)',
          }}>
            B
          </div>
          <span style={{
            fontFamily: 'var(--sans)',
            fontSize: 16,
            fontWeight: 600,
            color: 'rgba(255,255,255,0.88)',
            letterSpacing: -0.3,
          }}>
            Node0 Setup
          </span>
        </div>

        {/* Step label + indicator */}
        <StepLabel
          currentStep={step}
          totalSteps={TOTAL_STEPS}
          labels={STEPS.map((s) => s.label)}
        />
        <ProgressBar currentStep={step} totalSteps={TOTAL_STEPS} />
        <StepIndicator currentStep={step} totalSteps={TOTAL_STEPS} />

        {/* Step content with transition */}
        <div style={{
          flex: 1,
          opacity: transitioning ? 0 : 1,
          transform: transitioning
            ? `translateX(${direction * 20}px)`
            : 'translateX(0)',
          transition: 'opacity 0.2s ease, transform 0.2s ease',
          minHeight: 0,
        }}>
          <CurrentStepComponent
            node={node}
            state={onboardingState}
            setState={updateState}
            onNext={step === TOTAL_STEPS - 1 ? onComplete : goNext}
          />
        </div>

        {/* Back button (visible on steps 1-3) */}
        {step > 0 && step < TOTAL_STEPS - 1 && (
          <button
            onClick={goBack}
            style={{
              position: 'absolute',
              top: 32,
              left: 36,
              background: 'none',
              border: 'none',
              fontFamily: 'var(--mono)',
              fontSize: 11,
              color: 'rgba(255,255,255,0.25)',
              cursor: 'pointer',
              padding: '4px 8px',
              borderRadius: 4,
              transition: 'color 0.2s ease',
            }}
            onMouseEnter={(e) => { e.currentTarget.style.color = 'rgba(255,255,255,0.5)'; }}
            onMouseLeave={(e) => { e.currentTarget.style.color = 'rgba(255,255,255,0.25)'; }}
          >
            &larr; Back
          </button>
        )}
      </div>
    </div>
  );
}
