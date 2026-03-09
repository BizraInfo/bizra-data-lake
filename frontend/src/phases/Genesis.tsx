import { useEffect, useRef, useState } from 'react';
import { Reveal } from '../components/Reveal';
import { AstrolabeSVG } from '../components/AstrolabeSVG';
import { PAT } from '../lib/agents';
import { color } from '../tokens';

interface GenesisProps {
  initialName?: string;
  onNameChange?: (name: string) => void;
  onDone: (name: string) => void;
}

const delay = (ms: number) => new Promise(r => setTimeout(r, ms));

export default function Genesis({ initialName = '', onNameChange, onDone }: GenesisProps) {
  const [name, setName] = useState(initialName);
  const [phase, setPhase] = useState<'input' | 'generating'>('input');
  const [lines, setLines] = useState<string[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { setTimeout(() => inputRef.current?.focus(), 600); }, []);
  useEffect(() => { setName(initialName); }, [initialName]);

  const handleNameChange = (value: string) => {
    setName(value);
    onNameChange?.(value);
  };

  const generate = async () => {
    if (!name.trim()) return;
    setPhase('generating');
    const id = Array.from(crypto.getRandomValues(new Uint8Array(16)))
      .map(b => b.toString(16).padStart(2, '0'))
      .join('');
    const steps: [string, number][] = [
      ['Generating Ed25519 sovereign keypair...', 400],
      [`Node ID: ${id.slice(0, 24)}...`, 300],
      ['Deriving 12 agent child keys (HD-Ed25519)...', 500],
      ['Loading constitution v5.0.0-GENESIS...', 300],
      ['Covenant: 859649ea...verified', 400],
      ['7 constitutional rights bound.', 300],
      [`Genesis complete. Welcome, ${name.trim()}.`, 600],
    ];
    for (const [text, d] of steps) {
      await delay(d);
      setLines(p => [...p, text]);
    }
    await delay(800);
    onDone(name.trim());
  };

  const agentNodes = Object.values(PAT).map(a => ({
    color: a.color,
    booted: phase !== 'input',
  }));

  return (
    <div style={{
      minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      fontFamily: 'var(--font-mono)',
    }}>
      <Reveal delay={200}>
        <div style={{ fontFamily: 'var(--font-label)', color: color.gold, fontSize: 10, letterSpacing: 5, marginBottom: 28 }}>
          IDENTITY GENESIS
        </div>
      </Reveal>

      <Reveal delay={300}>
        <div style={{ marginBottom: 28, opacity: .6 }}>
          <AstrolabeSVG size={120} agents={agentNodes} active={phase !== 'input'} />
        </div>
      </Reveal>

      {phase === 'input' && (
        <Reveal delay={500}>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 22 }}>
            <div style={{ color: color.muted, fontSize: 15, fontFamily: 'var(--font-display)', fontStyle: 'italic', fontWeight: 300 }}>
              What shall the network know you as?
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span style={{ color: color.gold, fontSize: 12 }}>{'\u25B8'}</span>
              <input ref={inputRef} value={name} onChange={e => handleNameChange(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && generate()}
                placeholder="Your sovereign name"
                style={{
                  background: 'transparent', border: 'none',
                  borderBottom: '1px solid rgba(201,169,98,.2)',
                  color: color.text, fontSize: 16, fontFamily: 'var(--font-mono)',
                  padding: '10px 2px', width: 260, outline: 'none', letterSpacing: 1,
                  transition: 'border-color .3s',
                }}
                onFocus={e => e.target.style.borderBottomColor = 'rgba(201,169,98,.5)'}
                onBlur={e => e.target.style.borderBottomColor = 'rgba(201,169,98,.2)'}
              />
            </div>
            <button onClick={generate} disabled={!name.trim()} style={{
              marginTop: 6, background: 'transparent',
              border: `1px solid ${name.trim() ? 'rgba(201,169,98,.25)' : 'var(--line)'}`,
              color: name.trim() ? color.gold : color.ghost,
              padding: '11px 36px', borderRadius: 2, fontSize: 9, letterSpacing: 4,
              fontFamily: 'var(--font-mono)', cursor: name.trim() ? 'pointer' : 'default',
              transition: 'all .3s',
            }}>GENERATE IDENTITY</button>
          </div>
        </Reveal>
      )}

      {phase !== 'input' && (
        <div style={{ maxWidth: 460, width: '100%', padding: '0 28px' }}>
          {lines.map((l, i) => (
            <Reveal key={i} delay={i * 50}>
              <div style={{
                padding: '4px 0', fontSize: 11,
                color: l.includes('Welcome') ? color.gold : l.includes('verified') ? color.emerald : 'rgba(156,163,175,.9)',
                display: 'flex', alignItems: 'center', gap: 10,
              }}>
                {l.includes('Welcome')
                  ? <span style={{ color: color.gold, fontFamily: 'var(--font-display)', fontWeight: 500, fontSize: 13 }}>{l}</span>
                  : <><span style={{ color: color.emerald, fontSize: 10 }}>{'\u2713'}</span><span>{l}</span></>
                }
              </div>
            </Reveal>
          ))}
        </div>
      )}
    </div>
  );
}
