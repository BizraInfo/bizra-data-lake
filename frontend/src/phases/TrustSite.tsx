import { useEffect, useRef, useState } from 'react';
import { Reveal } from '../components/Reveal';
import { color } from '../tokens';

interface TrustSiteProps {
  onEnter: () => void;
}

const LAYERS = [
  { name: 'Human Seed', code: '\u0627\u0644\u0631\u0633\u0627\u0644\u0629 + \u0627\u0644\u0628\u0630\u0631\u0629', tests: '\u2014', color: color.gold },
  { name: 'Sovereign Node', code: 'identity_genesis.py', tests: '332', color: color.sapphire },
  { name: 'Agentic Dev', code: 'mission_pipeline.py', tests: '151', color: color.emerald },
  { name: 'Verification', code: 'evidence_receipt.py', tests: '50+', color: color.amber },
  { name: 'Learning', code: 'seed_engine.py', tests: '46', color: color.cyan },
  { name: 'Economic', code: 'algorithms.py', tests: '100+', color: color.amethyst },
  { name: 'Civilizational', code: 'federation/', tests: '60+', color: color.rose },
];

const INVARIANTS = [
  { id: 'I-1', name: 'Excellence', value: 'Ihsan \u2265 0.95', color: color.gold },
  { id: 'I-2', name: 'Signal', value: 'SNR \u2265 0.85', color: color.sapphire },
  { id: 'I-3', name: 'Justice', value: 'Gini \u2264 0.35', color: color.emerald },
  { id: 'I-4', name: 'Sovereignty', value: 'Keys LOCAL', color: color.amethyst },
  { id: 'I-5', name: 'Proof', value: 'Hash-chained', color: color.cyan },
];

const STATS = [
  { value: '8,237', label: 'Tests Passing' },
  { value: '22', label: 'Rust Crates' },
  { value: '31+', label: 'Days Live' },
  { value: '0.95+', label: 'Ihsan Floor' },
];

export default function TrustSite({ onEnter }: TrustSiteProps) {
  const [scrollY, setScrollY] = useState(0);
  const [enterHov, setEnterHov] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    const h = () => setScrollY(el?.scrollTop || 0);
    el?.addEventListener('scroll', h);
    return () => el?.removeEventListener('scroll', h);
  }, []);

  const navVisible = scrollY > 60;

  return (
    <div ref={ref} style={{ height: '100vh', overflow: 'auto', position: 'relative' }}>
      {/* Sticky Nav */}
      <nav style={{
        position: 'sticky', top: 0, zIndex: 50,
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '14px 40px',
        background: navVisible ? 'rgba(4,11,20,.88)' : 'transparent',
        backdropFilter: navVisible ? 'blur(24px)' : 'none',
        borderBottom: navVisible ? '1px solid rgba(201,169,98,.08)' : 'none',
        transition: 'all .5s ease',
      }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 14 }}>
          <span style={{ fontFamily: 'var(--font-label)', color: color.gold, fontSize: 13, fontWeight: 600, letterSpacing: 5 }}>BIZRA</span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 8, color: color.ghost, letterSpacing: 3 }}>DDAGI OS</span>
        </div>
        <button onClick={onEnter} style={{
          background: 'rgba(201,169,98,.06)', border: '1px solid rgba(201,169,98,.2)',
          color: color.gold, padding: '8px 22px', borderRadius: 2, fontSize: 10,
          fontFamily: 'var(--font-mono)', letterSpacing: 3, cursor: 'pointer',
          transition: 'all .3s',
        }} onMouseEnter={e => { (e.target as HTMLElement).style.background = 'rgba(201,169,98,.12)'; }}
           onMouseLeave={e => { (e.target as HTMLElement).style.background = 'rgba(201,169,98,.06)'; }}>
          INITIALIZE
        </button>
      </nav>

      {/* Hero */}
      <section style={{
        position: 'relative', padding: '100px 60px 90px',
        background: `radial-gradient(ellipse 60% 50% at 20% 20%, rgba(201,169,98,.07), transparent),
                     radial-gradient(ellipse 40% 40% at 80% 30%, rgba(96,165,250,.04), transparent),
                     linear-gradient(175deg, #071119, #040B14 60%)`,
        minHeight: '85vh', display: 'flex', alignItems: 'center',
      }}>
        <div style={{
          position: 'absolute', inset: 0, opacity: .4,
          backgroundImage: 'linear-gradient(rgba(201,169,98,.02) 1px, transparent 1px), linear-gradient(90deg, rgba(201,169,98,.02) 1px, transparent 1px)',
          backgroundSize: '56px 56px',
          maskImage: 'linear-gradient(180deg, rgba(0,0,0,.5), transparent 70%)',
          WebkitMaskImage: 'linear-gradient(180deg, rgba(0,0,0,.5), transparent 70%)',
        }} />

        <div style={{ position: 'relative', maxWidth: 1100, margin: '0 auto', width: '100%' }}>
          <Reveal delay={200}>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: color.gold, letterSpacing: 4, fontWeight: 500, marginBottom: 6 }}>
              DISTRIBUTED DECENTRALIZED AGI OPERATING SYSTEM
            </div>
          </Reveal>

          <Reveal delay={500}>
            <h1 style={{
              fontFamily: 'var(--font-display)', fontSize: 'clamp(38px, 5vw, 62px)',
              lineHeight: .94, margin: '16px 0 0', maxWidth: 780, fontWeight: 300,
              letterSpacing: '-0.02em', color: color.text,
            }}>
              From human need<br />
              <span style={{ color: color.gold, fontWeight: 500 }}>to sovereign</span> intelligence.
            </h1>
          </Reveal>

          <Reveal delay={800}>
            <p style={{
              color: color.muted, fontSize: 16, maxWidth: 640, lineHeight: 1.75, marginTop: 20,
              fontFamily: 'var(--font-display)', fontWeight: 400, fontStyle: 'italic',
            }}>
              BIZRA turns every human into a sovereign node, every node into a living seed,
              and every verified act of growth into shared intelligence, capability, and value.
            </p>
          </Reveal>

          <Reveal delay={1100}>
            <button
              onClick={onEnter}
              onMouseEnter={() => setEnterHov(true)}
              onMouseLeave={() => setEnterHov(false)}
              style={{
                marginTop: 36, padding: '16px 44px', borderRadius: 2,
                background: enterHov ? color.gold : 'transparent',
                color: enterHov ? color.bg : color.gold,
                border: `1.5px solid ${enterHov ? color.gold : 'rgba(201,169,98,.35)'}`,
                fontSize: 11, fontFamily: 'var(--font-mono)', letterSpacing: 4,
                cursor: 'pointer', transition: 'all .35s cubic-bezier(.16,1,.3,1)',
                fontWeight: 500,
              }}
            >
              BEGIN YOUR JOURNEY
            </button>
          </Reveal>

          <Reveal delay={1400}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 14, marginTop: 52 }}>
              {STATS.map((k, i) => (
                <div key={i} style={{
                  padding: '18px 20px', borderRadius: 8,
                  background: 'rgba(201,169,98,.02)', border: '1px solid rgba(201,169,98,.06)',
                }}>
                  <div style={{ fontSize: 28, fontWeight: 300, letterSpacing: -1, color: color.goldBright, fontFamily: 'var(--font-display)' }}>{k.value}</div>
                  <div style={{ fontSize: 9, color: color.dim, letterSpacing: 2, textTransform: 'uppercase', marginTop: 4, fontFamily: 'var(--font-mono)' }}>{k.label}</div>
                </div>
              ))}
            </div>
          </Reveal>
        </div>
      </section>

      {/* Bismillah + Invariants + Layers */}
      <section style={{ maxWidth: 1100, margin: '0 auto', padding: '64px 60px' }}>
        <Reveal delay={100}>
          <div style={{
            borderLeft: '2px solid rgba(201,169,98,.25)', padding: '24px 28px',
            background: 'rgba(201,169,98,.03)', borderRadius: '0 12px 12px 0',
            marginBottom: 56,
          }}>
            <div style={{ fontFamily: 'var(--font-arabic)', fontSize: 18, color: 'rgba(201,169,98,.4)', direction: 'rtl', marginBottom: 10 }}>
              {'\u0628\u0633\u0645 \u0627\u0644\u0644\u0647 \u0627\u0644\u0631\u062D\u0645\u0646 \u0627\u0644\u0631\u062D\u064A\u0645'}
            </div>
            <div style={{ fontSize: 18, lineHeight: 1.65, fontFamily: 'var(--font-display)', fontStyle: 'italic', fontWeight: 400 }}>
              "Every human is a node, and every node is a seed, and every seed has infinite potential."
            </div>
            <div style={{ fontSize: 10, color: color.dim, marginTop: 10, fontFamily: 'var(--font-mono)', letterSpacing: 1 }}>
              {'\u2014 \u0627\u0644\u0628\u0630\u0631\u0629, Ramadan 2023'}
            </div>
          </div>
        </Reveal>

        {/* Invariants */}
        <Reveal delay={200}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: color.gold, letterSpacing: 4, marginBottom: 8 }}>
            FIVE NON-NEGOTIABLE INVARIANTS
          </div>
          <h2 style={{ fontFamily: 'var(--font-display)', fontSize: 30, margin: '0 0 24px', fontWeight: 400, letterSpacing: '-0.01em' }}>
            Machine-enforced. No exceptions.
          </h2>
        </Reveal>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 10, marginBottom: 56 }}>
          {INVARIANTS.map((v, i) => (
            <Reveal key={i} delay={300 + i * 80}>
              <div style={{
                padding: '18px 16px', borderRadius: 8,
                background: 'rgba(255,255,255,.015)', border: '1px solid var(--line)',
                transition: 'border-color .3s, background .3s',
              }} onMouseEnter={e => { e.currentTarget.style.borderColor = `${v.color}25`; e.currentTarget.style.background = `${v.color}05`; }}
                 onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--line)'; e.currentTarget.style.background = 'rgba(255,255,255,.015)'; }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 8, color: v.color, letterSpacing: 3, marginBottom: 10, fontWeight: 500 }}>{v.id}</div>
                <div style={{ fontSize: 15, fontWeight: 500, fontFamily: 'var(--font-display)', marginBottom: 3 }}>{v.name}</div>
                <div style={{ fontSize: 10, color: color.muted, fontFamily: 'var(--font-mono)' }}>{v.value}</div>
              </div>
            </Reveal>
          ))}
        </div>

        {/* Seven Layers */}
        <Reveal delay={200}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: color.gold, letterSpacing: 4, marginBottom: 8 }}>SEVEN-LAYER DDAGI STACK</div>
          <h2 style={{ fontFamily: 'var(--font-display)', fontSize: 30, margin: '0 0 24px', fontWeight: 400 }}>
            Every layer has code. Every layer has tests.
          </h2>
        </Reveal>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginBottom: 56 }}>
          {LAYERS.map((l, i) => (
            <Reveal key={i} delay={300 + i * 60}>
              <div style={{
                display: 'grid', gridTemplateColumns: '36px 1fr 200px 80px', gap: 16, alignItems: 'center',
                padding: '14px 18px', borderRadius: 6,
                background: 'rgba(255,255,255,.012)', border: '1px solid var(--line)',
                transition: 'background .3s',
              }} onMouseEnter={e => e.currentTarget.style.background = `${l.color}06`}
                 onMouseLeave={e => e.currentTarget.style.background = 'rgba(255,255,255,.012)'}>
                <div style={{ fontFamily: 'var(--font-label)', fontSize: 9, color: l.color, fontWeight: 600, letterSpacing: 2 }}>L{i}</div>
                <div style={{ fontSize: 14, fontWeight: 400, fontFamily: 'var(--font-display)', letterSpacing: '.02em' }}>{l.name}</div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: color.dim }}>{l.code}</div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: color.emerald, textAlign: 'right' }}>{l.tests}</div>
              </div>
            </Reveal>
          ))}
        </div>

        {/* CTA */}
        <Reveal delay={400}>
          <div style={{ textAlign: 'center', padding: '56px 0 32px' }}>
            <div style={{ fontFamily: 'var(--font-arabic)', fontSize: 20, color: 'rgba(201,169,98,.3)', direction: 'rtl', marginBottom: 20 }}>
              {'\u0643\u0644 \u0628\u0630\u0631\u0629 \u062A\u062D\u0645\u0644 \u0641\u064A \u062F\u0627\u062E\u0644\u0647\u0627 \u0645\u062E\u0637\u0637 \u063A\u0627\u0628\u0629 \u0628\u0623\u0643\u0645\u0644\u0647\u0627'}
            </div>
            <button onClick={onEnter} style={{
              background: color.gold, color: color.bg, border: 'none', padding: '16px 52px', borderRadius: 2,
              fontSize: 11, fontFamily: 'var(--font-mono)', letterSpacing: 4, cursor: 'pointer', fontWeight: 600,
              transition: 'transform .2s, box-shadow .3s',
              boxShadow: '0 4px 30px rgba(201,169,98,.2)',
            }} onMouseEnter={e => { (e.target as HTMLElement).style.transform = 'translateY(-1px)'; }}
               onMouseLeave={e => { (e.target as HTMLElement).style.transform = 'translateY(0)'; }}>
              BECOME A NODE
            </button>
            <div style={{ marginTop: 14, fontSize: 11, color: color.dim, fontFamily: 'var(--font-mono)', letterSpacing: 1 }}>
              Zero cloud. Zero cost. Your keys. Your sovereignty.
            </div>
          </div>
        </Reveal>
      </section>

      {/* Footer */}
      <footer style={{
        borderTop: '1px solid var(--line)', padding: '20px 60px',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 10,
      }}>
        <span style={{ fontFamily: 'var(--font-label)', letterSpacing: 4, color: color.goldDim, fontSize: 11 }}>BIZRA</span>
        <span style={{ fontFamily: 'var(--font-arabic)', color: color.dim }}>
          {'\u0628\u0633\u0645 \u0627\u0644\u0644\u0647 \u0627\u0644\u0631\u062D\u0645\u0646 \u0627\u0644\u0631\u062D\u064A\u0645 \u00B7 Dubai'}
        </span>
        <span style={{ fontFamily: 'var(--font-mono)', color: color.dim, letterSpacing: 1 }}>v{__VERSION__}</span>
      </footer>
    </div>
  );
}
