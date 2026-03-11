import { useState } from 'react';
import { PrimordialBloom } from '../components/PrimordialBloom';
import { Reveal } from '../components/Reveal';
import { color } from '../tokens';

interface SplashProps {
  onStart: () => void;
}

export default function Splash({ onStart }: SplashProps) {
  const [hov, setHov] = useState(false);

  return (
    <div style={{
      minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      position: 'relative', overflow: 'hidden',
    }}>
      {/* Radial glow */}
      <div style={{
        position: 'absolute', width: 500, height: 500, borderRadius: '50%', opacity: .04,
        background: `radial-gradient(circle, ${color.gold}, transparent 70%)`,
        top: '20%', left: '50%', transform: 'translateX(-50%)', filter: 'blur(80px)',
      }} />

      <div style={{
        position: 'absolute',
        inset: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        pointerEvents: 'none',
      }}>
        <PrimordialBloom
          size={420}
          style={{
            opacity: 0.8,
            transform: 'translateY(-26px)',
            maskImage: 'radial-gradient(circle, rgba(0,0,0,1) 56%, rgba(0,0,0,0.15) 82%, transparent 100%)',
            WebkitMaskImage: 'radial-gradient(circle, rgba(0,0,0,1) 56%, rgba(0,0,0,0.15) 82%, transparent 100%)',
          }}
        />
      </div>

      <Reveal delay={400}>
        <div style={{ position: 'relative', width: 170, height: 170, marginBottom: 36 }}>
          <svg width="170" height="170" viewBox="0 0 140 140" style={{ position: 'absolute', inset: 0 }}>
            <circle cx="70" cy="70" r="60" fill="none" stroke="rgba(201,169,98,.08)" strokeWidth=".5" />
            <circle cx="70" cy="70" r="48" fill="none" stroke="rgba(201,169,98,.12)" strokeWidth="1"
              strokeDasharray="2 5" style={{ animation: 'spinSlow 90s linear infinite', transformOrigin: '70px 70px' }} />
            <circle cx="70" cy="70" r="36" fill="none" stroke="rgba(201,169,98,.06)" strokeWidth=".5"
              style={{ animation: 'spinSlow 60s linear infinite reverse', transformOrigin: '70px 70px' }} />
            <line x1="70" y1="22" x2="70" y2="118" stroke="rgba(201,169,98,.04)" strokeWidth=".5" />
            <line x1="22" y1="70" x2="118" y2="70" stroke="rgba(201,169,98,.04)" strokeWidth=".5" />
          </svg>
          <div style={{
            position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%,-50%)',
            width: 44, height: 44, borderRadius: '50%',
            background: 'radial-gradient(circle, rgba(201,169,98,.2), transparent)',
            animation: 'breathe 4s ease-in-out infinite',
          }} />
        </div>
      </Reveal>

      <Reveal delay={800}>
        <div style={{ fontFamily: 'var(--font-label)', color: color.gold, fontSize: 15, letterSpacing: 7, fontWeight: 500 }}>BIZRA</div>
      </Reveal>
      <Reveal delay={1000}>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 8, color: color.ghost, letterSpacing: 4, marginTop: 3 }}>SOVEREIGN AI OPERATING SYSTEM</div>
      </Reveal>

      <Reveal delay={1400}>
        <div style={{ marginTop: 32, textAlign: 'center' }}>
          <div style={{ fontFamily: 'var(--font-arabic)', fontSize: 17, color: 'rgba(201,169,98,.25)', direction: 'rtl', marginBottom: 10 }}>
            {'\u0628\u0633\u0645 \u0627\u0644\u0644\u0647 \u0627\u0644\u0631\u062D\u0645\u0646 \u0627\u0644\u0631\u062D\u064A\u0645'}
          </div>
          <div style={{
            color: 'rgba(232,228,219,.4)', fontSize: 15, lineHeight: 2,
            fontFamily: 'var(--font-display)', fontStyle: 'italic', fontWeight: 300, maxWidth: 340,
          }}>
            Every human is a node. Every node is a seed.<br />Every seed has infinite potential.
          </div>
          <div style={{
            marginTop: 18,
            color: 'rgba(96,165,250,.48)',
            fontSize: 9,
            letterSpacing: 3,
            fontFamily: 'var(--font-mono)',
          }}>
            PRIMORDIAL BLOOM / GOLDEN-ANGLE EMERGENCE
          </div>
        </div>
      </Reveal>

      <Reveal delay={2000}>
        <button onClick={onStart} onMouseEnter={() => setHov(true)} onMouseLeave={() => setHov(false)}
          style={{
            marginTop: 40, background: hov ? 'rgba(201,169,98,.06)' : 'transparent',
            border: `1px solid ${hov ? 'rgba(201,169,98,.35)' : 'rgba(201,169,98,.12)'}`,
            color: color.gold, padding: '14px 48px', borderRadius: 2, fontSize: 10,
            letterSpacing: 5, cursor: 'pointer', fontFamily: 'var(--font-mono)',
            transition: 'all .4s cubic-bezier(.16,1,.3,1)', fontWeight: 400,
            boxShadow: hov ? '0 0 50px rgba(201,169,98,.06)' : 'none',
          }}>
          INITIALIZE NODE
        </button>
      </Reveal>
    </div>
  );
}
