import { useEffect, useState } from 'react';
import { Reveal } from '../components/Reveal';
import { AstrolabeSVG } from '../components/AstrolabeSVG';
import { PAT, SAT_AGENTS, AGENT_IDS } from '../lib/agents';
import { color } from '../tokens';
import type { UserConfig } from '../types';

interface AssemblyProps {
  userName: string;
  config: UserConfig;
  onDone: () => void;
}

const delay = (ms: number) => new Promise(r => setTimeout(r, ms));

export default function Assembly({ userName, config, onDone }: AssemblyProps) {
  const [booted, setBooted] = useState<string[]>([]);
  const [satReady, setSatReady] = useState(false);
  const [done, setDone] = useState(false);
  const [configLines, setConfigLines] = useState<string[]>([]);

  useEffect(() => {
    (async () => {
      if (config.work_schedule) { await delay(300); setConfigLines(p => [...p, `Schedule: ${config.work_schedule}`]); }
      if (config.primary_tools?.length) { await delay(200); setConfigLines(p => [...p, `Tools: ${(config.primary_tools ?? []).join(', ')}`]); }
      if (config.communication_pref) { await delay(200); setConfigLines(p => [...p, `Comms: ${config.communication_pref}`]); }
      if (config.priority_domains?.length) { await delay(200); setConfigLines(p => [...p, `Domains: ${(config.priority_domains ?? []).join(', ')}`]); }
      if (config.autonomy) { await delay(200); setConfigLines(p => [...p, `Autonomy: ${config.autonomy}`]); }
      await delay(500);
      for (const id of AGENT_IDS) { await delay(280 + Math.random() * 150); setBooted(p => [...p, id]); }
      await delay(500); setSatReady(true); await delay(700); setDone(true); await delay(600); onDone();
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const agentNodes = AGENT_IDS.map(id => ({
    color: PAT[id].color,
    booted: booted.includes(id),
  }));

  return (
    <div style={{
      minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      gap: 24, fontFamily: 'var(--font-mono)',
    }}>
      <Reveal delay={100}>
        <div style={{ fontFamily: 'var(--font-label)', color: color.gold, fontSize: 10, letterSpacing: 5, textAlign: 'center' }}>
          ASSEMBLING YOUR TEAM
        </div>
      </Reveal>

      <Reveal delay={200}>
        <div style={{ margin: '8px 0' }}>
          <AstrolabeSVG size={180} agents={agentNodes} active={booted.length > 0} />
        </div>
      </Reveal>

      {/* Config lines */}
      {configLines.length > 0 && (
        <div style={{ minWidth: 320 }}>
          {configLines.map((l, i) => (
            <Reveal key={i} delay={i * 40}>
              <div style={{ fontSize: 9, color: color.cyan, padding: '2px 0', display: 'flex', gap: 8, alignItems: 'center' }}>
                <span style={{ color: color.emerald }}>{'\u2699'}</span>{l}
              </div>
            </Reveal>
          ))}
        </div>
      )}

      {/* Agent boot list */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 3, minWidth: 380, maxWidth: 420 }}>
        {AGENT_IDS.map((id, i) => {
          const ag = PAT[id];
          const on = booted.includes(id);
          return (
            <Reveal key={id} delay={100 + i * 50}>
              <div style={{
                display: 'flex', alignItems: 'center', gap: 12, padding: '9px 16px', borderRadius: 6,
                background: on ? `${ag.color}06` : 'transparent',
                border: `1px solid ${on ? `${ag.color}15` : 'var(--line)'}`,
                transition: 'all .6s cubic-bezier(.16,1,.3,1)',
              }}>
                <div style={{
                  width: 26, height: 26, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: 11, border: `1px solid ${on ? `${ag.color}30` : 'rgba(255,255,255,.06)'}`,
                  color: on ? ag.color : color.ghost, transition: 'all .5s',
                }}>{ag.icon}</div>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', gap: 10, alignItems: 'baseline' }}>
                    <span style={{ fontSize: 9, fontWeight: 600, letterSpacing: 2, color: on ? ag.color : color.ghost, transition: 'color .5s' }}>{ag.callsign}</span>
                    <span style={{ fontSize: 9, color: on ? color.muted : color.ghost, transition: 'color .5s' }}>{ag.name}</span>
                  </div>
                  <div style={{
                    fontSize: 8, marginTop: 1, color: on ? color.dim : color.ghost,
                    fontFamily: 'var(--font-display)', fontStyle: 'italic', transition: 'color .5s',
                  }}>{on ? ag.bootMsg : '...'}</div>
                </div>
                <div style={{
                  width: 6, height: 6, borderRadius: '50%',
                  background: on ? ag.color : 'rgba(255,255,255,.06)',
                  boxShadow: on ? `0 0 8px ${ag.color}40` : 'none',
                  transition: 'all .5s',
                }} />
              </div>
            </Reveal>
          );
        })}
      </div>

      {/* SAT-5 */}
      {satReady && (
        <Reveal>
          <div style={{
            padding: '8px 18px', borderRadius: 6,
            border: `1px solid ${color.amethyst}10`, background: `${color.amethyst}04`,
          }}>
            <div style={{ fontSize: 7, letterSpacing: 2, color: color.amethyst, marginBottom: 5, fontWeight: 500 }}>SAT-5 {'\u2014'} ZERO USER CONTROL</div>
            <div style={{ display: 'flex', gap: 14 }}>
              {SAT_AGENTS.map((s, i) => (
                <div key={i} style={{ textAlign: 'center' }}>
                  <div style={{ width: 4, height: 4, borderRadius: '50%', background: s.color, margin: '0 auto 3px', boxShadow: `0 0 4px ${s.color}35` }} />
                  <div style={{ fontSize: 7, color: color.dim }}>{s.name}</div>
                </div>
              ))}
            </div>
          </div>
        </Reveal>
      )}

      {done && (
        <Reveal>
          <div style={{ color: color.gold, fontSize: 13, fontFamily: 'var(--font-display)', fontStyle: 'italic', fontWeight: 400 }}>
            Your sovereign AI team is configured, {userName}.
          </div>
        </Reveal>
      )}
    </div>
  );
}
