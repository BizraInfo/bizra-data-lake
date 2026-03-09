import { useEffect, useState } from 'react';
import { Reveal } from '../components/Reveal';
import { TEACH_QUESTIONS } from '../lib/agents';
import { color } from '../tokens';
import type { TeachDraftState, UserConfig } from '../types';

interface TeachStepsProps {
  initialDraft?: TeachDraftState;
  onDraftChange?: (draft: TeachDraftState) => void;
  onDone: (config: UserConfig) => void;
}

const EMPTY_DRAFT: TeachDraftState = {
  step: 0,
  answers: {},
  textValue: '',
  selected: [],
};

export default function TeachSteps({ initialDraft = EMPTY_DRAFT, onDraftChange, onDone }: TeachStepsProps) {
  const [step, setStep] = useState(initialDraft.step);
  const [answers, setAnswers] = useState<Record<string, string | string[]>>(initialDraft.answers);
  const [textVal, setTextVal] = useState(initialDraft.textValue);
  const [selected, setSelected] = useState<string[]>(initialDraft.selected);

  useEffect(() => {
    setStep(initialDraft.step);
    setAnswers(initialDraft.answers);
    setTextVal(initialDraft.textValue);
    setSelected(initialDraft.selected);
  }, [initialDraft]);

  useEffect(() => {
    onDraftChange?.({
      step,
      answers,
      textValue: textVal,
      selected,
    });
  }, [answers, onDraftChange, selected, step, textVal]);

  const q = TEACH_QUESTIONS[step];
  const total = TEACH_QUESTIONS.length;

  const next = () => {
    const ans = { ...answers };
    if (q.type === 'text') ans[q.id] = textVal || q.default || '';
    else if (q.type === 'single') ans[q.id] = selected[0] || q.default || '';
    else ans[q.id] = selected.length ? selected : [];
    setAnswers(ans);
    setSelected([]);
    setTextVal('');
    if (step < total - 1) setStep(step + 1);
    else onDone(ans as unknown as UserConfig);
  };

  const toggleOpt = (o: string) => {
    if (q.type === 'single') setSelected([o]);
    else setSelected(p => (p.includes(o) ? p.filter(x => x !== o) : [...p, o]));
  };

  const canNext = q.type === 'text' || selected.length > 0;

  return (
    <div style={{
      minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      fontFamily: 'var(--font-mono)',
    }}>
      {/* Progress pips */}
      <Reveal delay={100}>
        <div style={{ display: 'flex', gap: 6, marginBottom: 36 }}>
          {TEACH_QUESTIONS.map((_, i) => (
            <div key={i} style={{
              width: i === step ? 36 : 16, height: 2, borderRadius: 99,
              background: i < step ? color.emerald : i === step ? color.gold : 'rgba(255,255,255,.08)',
              transition: 'all .5s cubic-bezier(.16,1,.3,1)',
            }} />
          ))}
        </div>
      </Reveal>

      <Reveal delay={200}>
        <div style={{ fontFamily: 'var(--font-label)', color: color.gold, fontSize: 8, letterSpacing: 4, marginBottom: 6 }}>
          TEACH {'\u00B7'} STEP {step + 1} OF {total}
        </div>
      </Reveal>

      <Reveal delay={300} key={step}>
        <div style={{ textAlign: 'center', maxWidth: 440, padding: '0 28px' }}>
          <div style={{ fontSize: 36, marginBottom: 18 }}>{q.icon}</div>
          <div style={{ fontSize: 17, fontFamily: 'var(--font-display)', fontWeight: 400, marginBottom: 6 }}>{q.prompt}</div>
          <div style={{ fontSize: 9, color: color.ghost, marginBottom: 28, letterSpacing: 1 }}>This configures your PAT-7 agent team</div>

          {q.type === 'text' && (
            <input value={textVal} onChange={e => setTextVal(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && next()} placeholder={q.default} autoFocus
              style={{
                background: 'transparent', border: 'none', borderBottom: '1px solid rgba(201,169,98,.2)',
                color: color.text, fontSize: 15, fontFamily: 'var(--font-mono)', padding: '10px 0',
                width: '100%', outline: 'none', textAlign: 'center', letterSpacing: 1,
              }}
            />
          )}

          {(q.type === 'single' || q.type === 'multi') && q.options && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, textAlign: 'left' }}>
              {q.options.map((o, i) => {
                const sel = selected.includes(o);
                return (
                  <button key={i} onClick={() => toggleOpt(o)} style={{
                    padding: '12px 16px', borderRadius: 6,
                    background: sel ? 'rgba(201,169,98,.06)' : 'transparent',
                    border: `1px solid ${sel ? 'rgba(201,169,98,.25)' : 'var(--line)'}`,
                    color: sel ? color.gold : color.muted, fontSize: 12,
                    fontFamily: 'var(--font-mono)', cursor: 'pointer', transition: 'all .25s',
                    display: 'flex', alignItems: 'center', gap: 12,
                  }}>
                    <div style={{
                      width: 16, height: 16, borderRadius: q.type === 'single' ? '50%' : 3,
                      border: `1.5px solid ${sel ? color.gold : 'rgba(255,255,255,.12)'}`,
                      display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
                      transition: 'border-color .2s',
                    }}>
                      {sel && <div style={{ width: 8, height: 8, borderRadius: q.type === 'single' ? '50%' : 2, background: color.gold }} />}
                    </div>
                    {o}
                  </button>
                );
              })}
            </div>
          )}

          <button onClick={next} disabled={q.type !== 'text' && !canNext} style={{
            marginTop: 28, padding: '12px 40px', borderRadius: 2,
            background: (canNext || q.type === 'text') ? 'rgba(201,169,98,.08)' : 'transparent',
            border: `1px solid ${(canNext || q.type === 'text') ? 'rgba(201,169,98,.25)' : 'var(--line)'}`,
            color: (canNext || q.type === 'text') ? color.gold : color.ghost,
            fontSize: 9, letterSpacing: 4, fontFamily: 'var(--font-mono)',
            cursor: (canNext || q.type === 'text') ? 'pointer' : 'default', transition: 'all .3s',
          }}>
            {step === total - 1 ? 'CONFIGURE AGENTS' : 'NEXT \u2192'}
          </button>
        </div>
      </Reveal>
    </div>
  );
}
