// ============================================================
// TeachStep — Conversational Interview (7 Questions, 10 Atom Kinds)
// ============================================================
// A stepped conversational interview that generates TEACH atoms
// across all 10 kinds. Each question maps to specific atom kinds.
// Shows one question at a time with smooth transitions.
// After all 7 questions, auto-triggers SYNTHESIZE.
// ============================================================

import { useState, useCallback, useRef, useEffect } from 'react';

const MAX_CHARS = 500;

// ── Interview Questions ─────────────────────────────────────
// Each question maps to one or more atom kinds.
// The `parse` function extracts atoms from the answer text.

const QUESTIONS = [
  {
    id: 'identity',
    question: 'Who are you? Tell me about yourself.',
    placeholder: 'I am a software architect from Dubai. I build distributed systems. I have two kids and love hiking on weekends...',
    hint: 'Write naturally — each sentence becomes a memory atom.',
    kinds: ['fact'],
    parse: (text) => {
      // Split on sentence boundaries, generate 1 atom per sentence
      const sentences = text
        .split(/(?<=[.!?])\s+/)
        .map((s) => s.trim())
        .filter((s) => s.length > 5);
      return sentences.slice(0, 5).map((s) => ({
        kind: 'fact',
        content: s,
        confidence: 9000,
      }));
    },
  },
  {
    id: 'values',
    question: 'What matters most to you?',
    placeholder: 'I believe in open source, privacy as a human right, and craftsmanship over speed...',
    hint: 'Your principles and preferences become guiding constraints.',
    kinds: ['principle', 'preference'],
    parse: (text) => {
      const atoms = [];
      const sentences = text
        .split(/(?<=[.!?])\s+|,\s*/)
        .map((s) => s.trim())
        .filter((s) => s.length > 5);
      // First sentence -> principle, rest -> preference
      if (sentences.length > 0) {
        atoms.push({ kind: 'principle', content: sentences[0], confidence: 9200 });
      }
      for (let i = 1; i < Math.min(sentences.length, 4); i++) {
        atoms.push({ kind: 'preference', content: sentences[i], confidence: 8800 });
      }
      // If only one sentence, also add as preference
      if (sentences.length === 1) {
        atoms.push({ kind: 'preference', content: sentences[0], confidence: 8800 });
      }
      return atoms;
    },
  },
  {
    id: 'expertise',
    question: 'What do you know well? What\'s your expertise?',
    placeholder: 'I specialize in Rust, distributed consensus, and zero-knowledge proofs. I have 12 years in backend systems...',
    hint: 'Skills, domains, and deep knowledge areas.',
    kinds: ['expertise'],
    parse: (text) => {
      const sentences = text
        .split(/(?<=[.!?])\s+|,\s*/)
        .map((s) => s.trim())
        .filter((s) => s.length > 5);
      return sentences.slice(0, 4).map((s) => ({
        kind: 'expertise',
        content: s,
        confidence: 9000,
      }));
    },
  },
  {
    id: 'workstyle',
    question: 'How do you prefer to work?',
    placeholder: 'I work best in deep focus blocks of 3 hours. I prefer async communication. I always start with architecture before code...',
    hint: 'Work habits, communication style, routines.',
    kinds: ['preference', 'pattern'],
    parse: (text) => {
      const atoms = [];
      const sentences = text
        .split(/(?<=[.!?])\s+/)
        .map((s) => s.trim())
        .filter((s) => s.length > 5);
      sentences.slice(0, 4).forEach((s, i) => {
        atoms.push({
          kind: i % 2 === 0 ? 'preference' : 'pattern',
          content: s,
          confidence: 8800,
        });
      });
      return atoms;
    },
  },
  {
    id: 'boundaries',
    question: 'What should your AI never do?',
    placeholder: 'Never share my data with third parties. Never make assumptions about my intentions. Never use a patronizing tone...',
    hint: 'Hard boundaries and constraints for your sovereign agent.',
    kinds: ['negation'],
    parse: (text) => {
      const sentences = text
        .split(/(?<=[.!?])\s+/)
        .map((s) => s.trim())
        .filter((s) => s.length > 5);
      return sentences.slice(0, 4).map((s) => ({
        kind: 'negation',
        content: s,
        confidence: 9500,
      }));
    },
  },
  {
    id: 'relationships',
    question: 'Who matters in your life?',
    placeholder: 'My co-founder Sarah leads our design team. My mentor Dr. Ahmed taught me systems thinking. My team of 5 engineers...',
    hint: 'People, teams, and relationships your agent should know about.',
    kinds: ['relationship'],
    parse: (text) => {
      const sentences = text
        .split(/(?<=[.!?])\s+/)
        .map((s) => s.trim())
        .filter((s) => s.length > 5);
      return sentences.slice(0, 4).map((s) => ({
        kind: 'relationship',
        content: s,
        confidence: 8500,
      }));
    },
  },
  {
    id: 'current',
    question: 'What are you working on right now?',
    placeholder: 'I am building BIZRA, a decentralized AI platform. This quarter we are shipping the genesis node. The deadline is March 15...',
    hint: 'Current projects, goals, timelines, and context.',
    kinds: ['goal', 'temporal', 'context'],
    parse: (text) => {
      const atoms = [];
      const sentences = text
        .split(/(?<=[.!?])\s+/)
        .map((s) => s.trim())
        .filter((s) => s.length > 5);
      // First sentence -> goal, second -> context, any with dates -> temporal
      const datePattern = /\b(january|february|march|april|may|june|july|august|september|october|november|december|q[1-4]|202[0-9]|deadline|by|before|until|this week|this month|this quarter|this year)\b/i;
      sentences.slice(0, 5).forEach((s, i) => {
        if (datePattern.test(s)) {
          atoms.push({ kind: 'temporal', content: s, confidence: 8500 });
        } else if (i === 0) {
          atoms.push({ kind: 'goal', content: s, confidence: 9000 });
        } else {
          atoms.push({ kind: 'context', content: s, confidence: 8500 });
        }
      });
      return atoms;
    },
  },

  // ── Agent Ops Questions (from Founder Ops manifest) ──────────
  // These 5 questions configure the Founder Ops Agent's operational
  // behavior: schedule, tools, communication style, priorities, autonomy.

  {
    id: 'work_schedule',
    question: "What's your typical work schedule?",
    placeholder: '8am to 6pm weekdays. Sometimes I work Saturday mornings on deep focus tasks...',
    hint: 'Your agent will schedule morning briefs and standups around this.',
    kinds: ['pattern', 'temporal'],
    parse: (text) => {
      const atoms = [];
      const sentences = text
        .split(/(?<=[.!?])\s+|,\s*/)
        .map((s) => s.trim())
        .filter((s) => s.length > 3);
      sentences.slice(0, 3).forEach((s, i) => {
        atoms.push({
          kind: i === 0 ? 'pattern' : 'temporal',
          content: `Work schedule: ${s}`,
          confidence: 9000,
        });
      });
      return atoms;
    },
  },
  {
    id: 'primary_tools',
    question: 'Which apps and tools do you use most?',
    placeholder: 'VS Code for coding, Chrome for research, Terminal for git and builds, Slack for team comms...',
    hint: 'Your agent can automate actions in these apps via desktop skills.',
    kinds: ['preference', 'context'],
    parse: (text) => {
      const tools = text
        .split(/(?<=[.!?])\s+|,\s*/)
        .map((s) => s.trim())
        .filter((s) => s.length > 2);
      return tools.slice(0, 6).map((t, i) => ({
        kind: i < 2 ? 'preference' : 'context',
        content: `Primary tool: ${t}`,
        confidence: 8800,
      }));
    },
  },
  {
    id: 'communication_pref',
    question: 'How should your agent communicate with you?',
    placeholder: 'Concise bullet points. Only interrupt me for critical issues. No fluff or filler...',
    hint: 'Sets the tone for briefs, alerts, and status updates.',
    kinds: ['preference'],
    parse: (text) => {
      const sentences = text
        .split(/(?<=[.!?])\s+/)
        .map((s) => s.trim())
        .filter((s) => s.length > 5);
      return sentences.slice(0, 3).map((s) => ({
        kind: 'preference',
        content: `Communication: ${s}`,
        confidence: 9200,
      }));
    },
  },
  {
    id: 'priority_domains',
    question: 'What are your top priority domains right now?',
    placeholder: 'Engineering is #1 — shipping the product. Then business strategy and operations...',
    hint: 'Helps your agent prioritize what to surface in morning briefs.',
    kinds: ['goal', 'preference'],
    parse: (text) => {
      const atoms = [];
      const sentences = text
        .split(/(?<=[.!?])\s+|,\s*/)
        .map((s) => s.trim())
        .filter((s) => s.length > 5);
      sentences.slice(0, 4).forEach((s, i) => {
        atoms.push({
          kind: i === 0 ? 'goal' : 'preference',
          content: `Priority domain: ${s}`,
          confidence: 9000,
        });
      });
      return atoms;
    },
  },
  {
    id: 'automation_comfort',
    question: 'How much autonomy should your agent have?',
    placeholder: 'Auto-execute low-risk things like health checks. Ask me before opening apps or sending messages...',
    hint: 'Controls the approval gate for proactive actions.',
    kinds: ['preference', 'negation'],
    parse: (text) => {
      const atoms = [];
      const sentences = text
        .split(/(?<=[.!?])\s+/)
        .map((s) => s.trim())
        .filter((s) => s.length > 5);
      const negPattern = /\b(never|don't|do not|no |ask me|ask first|confirm|approval)\b/i;
      sentences.slice(0, 4).forEach((s) => {
        atoms.push({
          kind: negPattern.test(s) ? 'negation' : 'preference',
          content: `Autonomy: ${s}`,
          confidence: negPattern.test(s) ? 9500 : 8800,
        });
      });
      return atoms;
    },
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

// ── Atom Kind Pill ────────────────────────────────────────────

const KindPill = ({ kind }) => {
  const colors = {
    fact: '#6B9BF7',
    preference: '#A78BFA',
    goal: '#F59E42',
    expertise: '#38BDF8',
    pattern: '#F0D68A',
    negation: '#E85D4A',
    relationship: '#5BBA6F',
    principle: '#D4A547',
    temporal: '#4ecdc4',
    context: '#FF6B9D',
  };
  const c = colors[kind] || '#888';
  return (
    <span style={{
      display: 'inline-block',
      fontFamily: 'var(--mono)',
      fontSize: 8,
      color: c,
      background: `${c}15`,
      border: `1px solid ${c}30`,
      borderRadius: 3,
      padding: '1px 6px',
      letterSpacing: 0.5,
      textTransform: 'uppercase',
    }}>
      {kind}
    </span>
  );
};

// ── Previous Answer Summary ───────────────────────────────────

const AnswerSummary = ({ questionIndex, answer, atomCount }) => {
  const q = QUESTIONS[questionIndex];
  if (!q || !answer) return null;

  return (
    <div style={{
      display: 'flex',
      alignItems: 'flex-start',
      gap: 10,
      padding: '8px 12px',
      background: 'rgba(91,186,111,0.03)',
      border: '1px solid rgba(91,186,111,0.08)',
      borderRadius: 8,
      animation: 'onb-fadeUp 0.3s ease',
    }}>
      {/* Check mark */}
      <div style={{
        width: 18,
        height: 18,
        borderRadius: '50%',
        background: 'rgba(91,186,111,0.15)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
        marginTop: 1,
      }}>
        <svg width="10" height="10" viewBox="0 0 10 10">
          <path d="M2 5.5 L4 7.5 L8 3" fill="none" stroke="#5BBA6F" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>

      <div style={{ flex: 1, minWidth: 0 }}>
        {/* Question (compact) */}
        <div style={{
          fontFamily: 'var(--sans)',
          fontSize: 10,
          color: 'rgba(255,255,255,0.35)',
          marginBottom: 3,
        }}>
          {q.question}
        </div>
        {/* Truncated answer */}
        <div style={{
          fontFamily: 'var(--sans)',
          fontSize: 11,
          color: 'rgba(255,255,255,0.5)',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}>
          {answer.length > 80 ? answer.slice(0, 80) + '...' : answer}
        </div>
        {/* Atom kinds */}
        <div style={{ display: 'flex', gap: 4, marginTop: 4, flexWrap: 'wrap', alignItems: 'center' }}>
          {q.kinds.map((k) => <KindPill key={k} kind={k} />)}
          <span style={{
            fontFamily: 'var(--mono)',
            fontSize: 8,
            color: 'rgba(91,186,111,0.5)',
            marginLeft: 4,
          }}>
            +{atomCount} atoms
          </span>
        </div>
      </div>
    </div>
  );
};

// ── Progress Bar ──────────────────────────────────────────────

const ProgressIndicator = ({ current, total, atomCount }) => (
  <div style={{
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    padding: '0 4px',
  }}>
    {/* Step dots */}
    <div style={{ display: 'flex', gap: 4 }}>
      {Array.from({ length: total }).map((_, i) => (
        <div
          key={i}
          style={{
            width: i === current ? 20 : 6,
            height: 6,
            borderRadius: 3,
            background: i < current
              ? '#5BBA6F'
              : i === current
                ? 'linear-gradient(90deg, #D4A547, #F0D68A)'
                : 'rgba(255,255,255,0.08)',
            transition: 'all 0.3s ease',
          }}
        />
      ))}
    </div>

    {/* Text */}
    <span style={{
      fontFamily: 'var(--mono)',
      fontSize: 9,
      color: 'rgba(255,255,255,0.3)',
      letterSpacing: 0.5,
    }}>
      Question {Math.min(current + 1, total)} of {total}
    </span>

    {/* Atom count */}
    {atomCount > 0 && (
      <span style={{
        fontFamily: 'var(--mono)',
        fontSize: 9,
        color: 'rgba(212,165,71,0.6)',
        letterSpacing: 0.5,
        marginLeft: 'auto',
      }}>
        {atomCount} atoms generated
      </span>
    )}
  </div>
);

// ============================================================
// MAIN COMPONENT
// ============================================================

export default function TeachStep({ node, state, setState, onNext }) {
  const [currentStep, setCurrentStep] = useState(0);
  const [currentAnswer, setCurrentAnswer] = useState('');
  const [answers, setAnswers] = useState(state.teachData?.answers || {});
  const [atomCounts, setAtomCounts] = useState(state.teachData?.atomCounts || {});
  const [totalAtoms, setTotalAtoms] = useState(state.teachData?.totalAtoms || 0);
  const [stepStatus, setStepStatus] = useState('idle'); // idle | sending | done | error
  const [synthesizeStatus, setSynthesizeStatus] = useState('idle');
  const [submitting, setSubmitting] = useState(false);
  const textareaRef = useRef(null);
  const containerRef = useRef(null);

  const isLastQuestion = currentStep >= QUESTIONS.length;
  const allComplete = isLastQuestion && synthesizeStatus === 'done';
  const currentQuestion = QUESTIONS[currentStep];

  // Auto-focus textarea when step changes
  useEffect(() => {
    if (textareaRef.current && !isLastQuestion) {
      // Small delay for transition animation
      const t = setTimeout(() => textareaRef.current?.focus(), 300);
      return () => clearTimeout(t);
    }
  }, [currentStep, isLastQuestion]);

  // Agent ops question IDs (questions 8-12, indices 7-11)
  const AGENT_OPS_IDS = ['work_schedule', 'primary_tools', 'communication_pref', 'priority_domains', 'automation_comfort'];

  // Persist state — split personal teachData from agent ops answers
  useEffect(() => {
    const agentOps = {};
    AGENT_OPS_IDS.forEach((id) => {
      if (answers[id]) agentOps[id] = answers[id];
    });
    setState({
      teachData: { answers, atomCounts, totalAtoms },
      agentOps,
    });
  }, [answers, atomCounts, totalAtoms, setState]);

  // Submit current answer
  const handleSubmitAnswer = useCallback(async () => {
    if (submitting || !currentAnswer.trim() || !currentQuestion) return;
    setSubmitting(true);
    setStepStatus('sending');

    const text = currentAnswer.trim();
    const atoms = currentQuestion.parse(text);

    // Send TEACH commands for each atom
    let successCount = 0;
    for (const atom of atoms) {
      try {
        const result = await node.send('TEACH', {
          kind: atom.kind,
          content: atom.content,
          confidence: atom.confidence,
          timestamp: Date.now(),
        });
        if (result?.ok) {
          successCount++;
        }
        // Brief pause between atoms for visual feedback
        await new Promise((r) => setTimeout(r, 150));
      } catch (err) {
        // Continue with other atoms even if one fails
      }
    }

    if (successCount > 0) {
      setStepStatus('done');

      // Store the answer and atom count
      const newAnswers = { ...answers, [currentQuestion.id]: text };
      const newAtomCounts = { ...atomCounts, [currentQuestion.id]: successCount };
      const newTotal = totalAtoms + successCount;
      setAnswers(newAnswers);
      setAtomCounts(newAtomCounts);
      setTotalAtoms(newTotal);

      // Auto-advance after a brief pause
      await new Promise((r) => setTimeout(r, 600));
      setStepStatus('idle');
      setCurrentAnswer('');
      const nextStep = currentStep + 1;
      setCurrentStep(nextStep);

      // If that was the last question, auto-synthesize
      if (nextStep >= QUESTIONS.length) {
        await handleSynthesize();
      }
    } else {
      setStepStatus('error');
      await new Promise((r) => setTimeout(r, 1500));
      setStepStatus('idle');
    }

    setSubmitting(false);
  }, [submitting, currentAnswer, currentQuestion, currentStep, answers, atomCounts, totalAtoms, node]);

  // Skip current question
  const handleSkip = useCallback(() => {
    if (submitting) return;
    setCurrentAnswer('');
    setStepStatus('idle');
    const nextStep = currentStep + 1;
    setCurrentStep(nextStep);

    // If that was the last question, auto-synthesize
    if (nextStep >= QUESTIONS.length) {
      handleSynthesize();
    }
  }, [currentStep, submitting]);

  // Synthesize memories
  const handleSynthesize = useCallback(async () => {
    setSynthesizeStatus('sending');
    await new Promise((r) => setTimeout(r, 300));

    try {
      const result = await node.send('SYNTHESIZE', { timestamp: Date.now() });
      await new Promise((r) => setTimeout(r, 500));

      if (result?.ok) {
        setSynthesizeStatus('done');
      } else {
        setSynthesizeStatus('error');
      }
    } catch (err) {
      setSynthesizeStatus('error');
    }
  }, [node]);

  // Handle Enter key (Shift+Enter for newline)
  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmitAnswer();
    }
  }, [handleSubmitAnswer]);

  return (
    <div ref={containerRef} style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      {/* Title — switches between personal and ops phases */}
      <div style={{ textAlign: 'center', marginBottom: 2 }}>
        <h2 style={{
          fontFamily: 'var(--sans)',
          fontSize: 20,
          fontWeight: 600,
          color: 'rgba(255,255,255,0.88)',
          margin: '0 0 6px 0',
        }}>
          {currentStep < 7 ? 'Tell Me About Yourself' : 'Configure Your Agent'}
        </h2>
        <p style={{
          fontFamily: 'var(--sans)',
          fontSize: 13,
          color: 'rgba(255,255,255,0.35)',
          margin: 0,
          lineHeight: 1.5,
        }}>
          {currentStep < 7
            ? 'A short conversation to build your node\'s foundational memory.'
            : 'Operational preferences for your Founder Ops Agent.'}
        </p>
      </div>

      {/* Progress Indicator */}
      <ProgressIndicator
        current={currentStep}
        total={QUESTIONS.length}
        atomCount={totalAtoms}
      />

      {/* Previous answers (compact) */}
      {currentStep > 0 && (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: 4,
          maxHeight: currentStep > 3 ? 160 : 'none',
          overflowY: currentStep > 3 ? 'auto' : 'visible',
          paddingRight: currentStep > 3 ? 4 : 0,
        }}>
          {QUESTIONS.slice(0, currentStep).map((q, i) => {
            // Phase divider between personal (0-6) and agent ops (7-11)
            const showDivider = i === 7 && currentStep >= 7;
            return (
              <div key={q.id}>
                {showDivider && (
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 10,
                    padding: '6px 0',
                    margin: '2px 0',
                  }}>
                    <div style={{ flex: 1, height: 1, background: 'rgba(212,165,71,0.15)' }} />
                    <span style={{
                      fontFamily: 'var(--mono)',
                      fontSize: 8,
                      color: 'rgba(212,165,71,0.4)',
                      letterSpacing: 1,
                      textTransform: 'uppercase',
                      whiteSpace: 'nowrap',
                    }}>
                      Agent Ops
                    </span>
                    <div style={{ flex: 1, height: 1, background: 'rgba(212,165,71,0.15)' }} />
                  </div>
                )}
                {answers[q.id] ? (
                  <AnswerSummary
                    questionIndex={i}
                    answer={answers[q.id]}
                    atomCount={atomCounts[q.id] || 0}
                  />
                ) : null}
              </div>
            );
          })}
        </div>
      )}

      {/* Phase transition banner (personal -> agent ops) */}
      {currentStep === 7 && !isLastQuestion && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          padding: '10px 16px',
          background: 'rgba(212,165,71,0.04)',
          border: '1px solid rgba(212,165,71,0.12)',
          borderRadius: 10,
          animation: 'onb-fadeUp 0.4s ease',
          marginBottom: 4,
        }}>
          <div style={{
            width: 22,
            height: 22,
            borderRadius: 6,
            background: 'linear-gradient(135deg, #D4A547, #8B6914)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
          }}>
            <span style={{ fontFamily: 'var(--mono)', fontSize: 10, fontWeight: 700, color: '#0A0B0F' }}>
              FO
            </span>
          </div>
          <div>
            <div style={{
              fontFamily: 'var(--sans)',
              fontSize: 12,
              fontWeight: 600,
              color: 'rgba(212,165,71,0.8)',
            }}>
              Activate Founder Ops
            </div>
            <div style={{
              fontFamily: 'var(--mono)',
              fontSize: 9,
              color: 'rgba(255,255,255,0.25)',
              letterSpacing: 0.5,
            }}>
              5 operational questions to configure your agent
            </div>
          </div>
        </div>
      )}

      {/* Current Question */}
      {!isLastQuestion && currentQuestion && (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: 10,
          animation: 'onb-fadeUp 0.4s ease',
        }}>
          {/* Question text */}
          <div style={{
            display: 'flex',
            alignItems: 'flex-start',
            gap: 12,
            padding: '14px 16px',
            background: 'rgba(212,165,71,0.04)',
            border: '1px solid rgba(212,165,71,0.12)',
            borderRadius: 12,
          }}>
            {/* Step number */}
            <div style={{
              width: 28,
              height: 28,
              borderRadius: 8,
              background: 'linear-gradient(135deg, #D4A547, #8B6914)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
            }}>
              <span style={{
                fontFamily: 'var(--mono)',
                fontSize: 13,
                fontWeight: 700,
                color: '#0A0B0F',
              }}>
                {currentStep + 1}
              </span>
            </div>

            <div style={{ flex: 1 }}>
              <div style={{
                fontFamily: 'var(--sans)',
                fontSize: 16,
                fontWeight: 600,
                color: 'rgba(255,255,255,0.88)',
                lineHeight: 1.4,
                marginBottom: 4,
              }}>
                {currentQuestion.question}
              </div>
              <div style={{
                fontFamily: 'var(--sans)',
                fontSize: 11,
                color: 'rgba(255,255,255,0.25)',
              }}>
                {currentQuestion.hint}
              </div>
              {/* Atom kind tags */}
              <div style={{ display: 'flex', gap: 4, marginTop: 6 }}>
                {currentQuestion.kinds.map((k) => <KindPill key={k} kind={k} />)}
              </div>
            </div>
          </div>

          {/* Text area */}
          <div style={{ position: 'relative' }}>
            <textarea
              ref={textareaRef}
              value={currentAnswer}
              onChange={(e) => {
                if (e.target.value.length <= MAX_CHARS) {
                  setCurrentAnswer(e.target.value);
                }
              }}
              onKeyDown={handleKeyDown}
              placeholder={currentQuestion.placeholder}
              disabled={submitting}
              rows={4}
              style={{
                width: '100%',
                background: stepStatus === 'done'
                  ? 'rgba(91,186,111,0.03)'
                  : 'rgba(255,255,255,0.03)',
                border: `1px solid ${
                  stepStatus === 'done'
                    ? 'rgba(91,186,111,0.12)'
                    : stepStatus === 'sending'
                      ? 'rgba(212,165,71,0.25)'
                      : 'rgba(255,255,255,0.06)'
                }`,
                borderRadius: 10,
                padding: '12px 14px',
                fontFamily: 'var(--sans)',
                fontSize: 13,
                color: 'rgba(255,255,255,0.8)',
                lineHeight: 1.55,
                resize: 'none',
                outline: 'none',
                transition: 'all 0.25s ease',
                opacity: submitting ? 0.6 : 1,
                boxSizing: 'border-box',
              }}
              onFocus={(e) => {
                if (stepStatus !== 'done') {
                  e.currentTarget.style.borderColor = 'rgba(212,165,71,0.3)';
                }
              }}
              onBlur={(e) => {
                if (stepStatus !== 'done' && stepStatus !== 'sending') {
                  e.currentTarget.style.borderColor = 'rgba(255,255,255,0.06)';
                }
              }}
            />

            {/* Bottom bar: char counter + status */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginTop: 4,
              padding: '0 4px',
            }}>
              <span style={{
                fontFamily: 'var(--mono)',
                fontSize: 9,
                color: 'rgba(255,255,255,0.15)',
              }}>
                Enter to submit, Shift+Enter for new line
              </span>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <StatusBadge status={stepStatus} />
                <CharCounter current={currentAnswer.length} max={MAX_CHARS} />
              </div>
            </div>
          </div>

          {/* Action buttons */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 12,
          }}>
            <button
              onClick={handleSubmitAnswer}
              disabled={!currentAnswer.trim() || submitting}
              style={{
                padding: '10px 28px',
                background: currentAnswer.trim() && !submitting
                  ? 'linear-gradient(135deg, #D4A547, #8B6914)'
                  : 'rgba(255,255,255,0.04)',
                border: 'none',
                borderRadius: 10,
                fontFamily: 'var(--sans)',
                fontSize: 13,
                fontWeight: 600,
                color: currentAnswer.trim() && !submitting ? '#0A0B0F' : 'rgba(255,255,255,0.15)',
                cursor: currentAnswer.trim() && !submitting ? 'pointer' : 'default',
                boxShadow: currentAnswer.trim() && !submitting ? '0 4px 20px rgba(212,165,71,0.25)' : 'none',
                transition: 'all 0.3s ease',
              }}
              onMouseEnter={(e) => {
                if (currentAnswer.trim() && !submitting) {
                  e.currentTarget.style.boxShadow = '0 6px 28px rgba(212,165,71,0.35)';
                  e.currentTarget.style.transform = 'translateY(-1px)';
                }
              }}
              onMouseLeave={(e) => {
                if (currentAnswer.trim() && !submitting) {
                  e.currentTarget.style.boxShadow = '0 4px 20px rgba(212,165,71,0.25)';
                  e.currentTarget.style.transform = 'translateY(0)';
                }
              }}
            >
              {submitting ? 'Teaching...' : 'Next'}
            </button>

            <button
              onClick={handleSkip}
              disabled={submitting}
              style={{
                padding: '8px 16px',
                background: 'none',
                border: 'none',
                fontFamily: 'var(--mono)',
                fontSize: 11,
                color: 'rgba(255,255,255,0.2)',
                cursor: submitting ? 'default' : 'pointer',
                transition: 'color 0.2s ease',
              }}
              onMouseEnter={(e) => { if (!submitting) e.currentTarget.style.color = 'rgba(255,255,255,0.4)'; }}
              onMouseLeave={(e) => { if (!submitting) e.currentTarget.style.color = 'rgba(255,255,255,0.2)'; }}
            >
              Skip
            </button>
          </div>
        </div>
      )}

      {/* Completion state */}
      {isLastQuestion && (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: 12,
          animation: 'onb-fadeUp 0.4s ease',
        }}>
          {/* Synthesize status */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 8,
            padding: '12px 16px',
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
            borderRadius: 10,
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
              {synthesizeStatus === 'idle'
                ? 'Preparing synthesis...'
                : synthesizeStatus === 'sending'
                  ? 'Synthesizing memories...'
                  : synthesizeStatus === 'done'
                    ? `Memory synthesis complete — ${totalAtoms} atoms across ${Object.keys(answers).length} areas`
                    : 'Synthesis failed'}
            </span>
          </div>

          {/* Summary of all kinds covered */}
          {synthesizeStatus === 'done' && (
            <div style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: 4,
              justifyContent: 'center',
              padding: '8px 0',
              animation: 'onb-fadeUp 0.5s ease',
            }}>
              {Array.from(new Set(
                Object.keys(answers).flatMap((id) => {
                  const q = QUESTIONS.find((q) => q.id === id);
                  return q ? q.kinds : [];
                })
              )).map((kind) => (
                <KindPill key={kind} kind={kind} />
              ))}
            </div>
          )}

          {/* Continue button */}
          <div style={{
            display: 'flex',
            justifyContent: 'center',
            marginTop: 4,
          }}>
            <button
              onClick={onNext}
              disabled={synthesizeStatus === 'sending'}
              style={{
                padding: '12px 36px',
                background: synthesizeStatus !== 'sending'
                  ? 'linear-gradient(135deg, #D4A547, #8B6914)'
                  : 'rgba(255,255,255,0.04)',
                border: 'none',
                borderRadius: 10,
                fontFamily: 'var(--sans)',
                fontSize: 14,
                fontWeight: 600,
                color: synthesizeStatus !== 'sending' ? '#0A0B0F' : 'rgba(255,255,255,0.15)',
                cursor: synthesizeStatus !== 'sending' ? 'pointer' : 'default',
                boxShadow: synthesizeStatus !== 'sending' ? '0 4px 20px rgba(212,165,71,0.25)' : 'none',
                transition: 'all 0.3s ease',
                animation: allComplete ? 'onb-fadeUp 0.4s ease' : 'none',
              }}
              onMouseEnter={(e) => {
                if (synthesizeStatus !== 'sending') {
                  e.currentTarget.style.boxShadow = '0 6px 28px rgba(212,165,71,0.35)';
                  e.currentTarget.style.transform = 'translateY(-1px)';
                }
              }}
              onMouseLeave={(e) => {
                if (synthesizeStatus !== 'sending') {
                  e.currentTarget.style.boxShadow = '0 4px 20px rgba(212,165,71,0.25)';
                  e.currentTarget.style.transform = 'translateY(0)';
                }
              }}
            >
              Continue
            </button>
          </div>
        </div>
      )}

      {/* Skip all (only on first question and not submitting) */}
      {currentStep === 0 && !submitting && (
        <div style={{ textAlign: 'center' }}>
          <button
            onClick={onNext}
            style={{
              padding: '8px 16px',
              background: 'none',
              border: 'none',
              fontFamily: 'var(--mono)',
              fontSize: 11,
              color: 'rgba(255,255,255,0.15)',
              cursor: 'pointer',
              transition: 'color 0.2s ease',
            }}
            onMouseEnter={(e) => { e.currentTarget.style.color = 'rgba(255,255,255,0.3)'; }}
            onMouseLeave={(e) => { e.currentTarget.style.color = 'rgba(255,255,255,0.15)'; }}
          >
            Skip interview for now
          </button>
        </div>
      )}

      <style>{`
        @keyframes onb-spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        @keyframes onb-fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes onb-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }
      `}</style>
    </div>
  );
}
