import { useCallback, useEffect, useRef, useState } from 'react';
import { PAT, SAT_AGENTS, AGENT_IDS, QUICK_MISSIONS, SCHEDULED_MISSIONS } from '../lib/agents';
import { SKILLS } from '../lib/agents';
import { color, THRESHOLDS, TIERS, TIER_COLORS, STAGES, getStage } from '../tokens';
import { useWebSocket } from '../hooks/useWebSocket';
import { useMission } from '../hooks/useMission';
import { useWallet } from '../hooks/useWallet';
import type { UserConfig, WSEvent } from '../types';

interface DashboardProps {
  userName: string;
  config: UserConfig;
}

type TabId = 'cmd' | 'char' | 'skill' | 'quest' | 'comm' | 'prog';

const TABS: { id: TabId; label: string; icon: string }[] = [
  { id: 'cmd', label: 'COMMAND', icon: '\u25B8' },
  { id: 'char', label: 'CHARACTER', icon: '\u25C8' },
  { id: 'skill', label: 'SKILLS', icon: '\u2B21' },
  { id: 'quest', label: 'QUESTS', icon: '\u2657' },
  { id: 'comm', label: 'COMMUNITY', icon: '\u2318' },
  { id: 'prog', label: 'PROGRESS', icon: '\u2197' },
];

export default function Dashboard({ userName, config }: DashboardProps) {
  const [tab, setTab] = useState<TabId>('cmd');
  const [input, setInput] = useState('');
  const [time, setTime] = useState(new Date());
  const feedEnd = useRef<HTMLDivElement>(null);

  const commStyle = config?.communication_pref || 'Concise bullet points';
  const greeting = commStyle.includes('critical')
    ? `${userName}. Systems nominal.`
    : commStyle.includes('Detailed')
      ? `Good evening, ${userName}. All seven agents online. Schedule loaded, domains configured. Awaiting your first mission.`
      : `Good evening, ${userName}. All agents reporting. What shall we work on?`;

  const { msgs, running, nodeState: st, exec, add, inputRef } = useMission(userName);
  const w = useWallet(st);

  // Greeting on mount
  useEffect(() => { add('NEXUS', greeting, 'greet'); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Clock
  useEffect(() => { const t = setInterval(() => setTime(new Date()), 1000); return () => clearInterval(t); }, []);

  // Auto-scroll feed
  useEffect(() => { feedEnd.current?.scrollIntoView({ behavior: 'smooth' }); }, [msgs]);

  // WebSocket for live events — receipt_minted triggers wallet refresh
  const { connected } = useWebSocket({
    url: `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/ghost`,
    onMessage: useCallback((evt: WSEvent) => {
      if (evt.type === 'proactive_message') {
        const p = evt.payload as { agent?: string; text?: string };
        add(p.agent || 'SYS', p.text || '', 'pro');
      } else if (evt.type === 'receipt_minted') {
        w.refresh();
      }
    }, [add, w]),
  });

  // Proactive morning brief
  useEffect(() => {
    const t = setTimeout(() => {
      add('ATLAS', "I've prepared your morning brief based on overnight activity.", 'pro');
      setTimeout(() => add('ORACLE', `Priority domains loaded: ${(config?.priority_domains || ['Engineering']).join(', ')}. Your knowledge graph has grown 12% this week.`, 'pro'), 2500);
      setTimeout(() => add('FORGE', 'Build pipeline is green. All 219 tests passing.', 'pro'), 4500);
      setTimeout(() => add('NEXUS', `${SCHEDULED_MISSIONS.filter(m => !m.auto).length} scheduled missions pending approval. Cross-agent coordination score: 94%.`, 'pro'), 6500);
    }, 5000);
    return () => clearTimeout(t);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Idle chatter
  useEffect(() => {
    if (running) return;
    const interval = setInterval(() => {
      const ids = AGENT_IDS;
      const id = ids[Math.floor(Math.random() * ids.length)];
      const ag = PAT[id];
      const msg = ag.idle[Math.floor(Math.random() * ag.idle.length)];
      add(ag.callsign, msg, 'pro');
    }, 18000 + Math.random() * 12000);
    return () => clearInterval(interval);
  }, [running, add]);

  const stage = getStage(st.sovereignty);
  const nv = +(st.sovereignty * Math.max(st.rac, 0.01) * (st.ihsan || 0.01) * (1 + Math.log(1 + st.streak) / Math.log(10))).toFixed(2);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', fontFamily: 'var(--font-mono)', fontSize: 11 }}>
      {/* Top Bar */}
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '6px 20px', borderBottom: '1px solid var(--line)',
        background: 'rgba(4,11,20,.6)', backdropFilter: 'blur(12px)',
      }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 12 }}>
          <span style={{ fontFamily: 'var(--font-label)', color: color.gold, fontSize: 11, letterSpacing: 4, fontWeight: 600 }}>BIZRA</span>
          <span style={{ fontSize: 7, color: color.ghost, letterSpacing: 2 }}>NODE0</span>
          <span style={{ fontSize: 7, letterSpacing: 1, color: running ? color.flame : color.emerald }}>
            {running ? '\u25CF EXECUTING' : '\u25CF READY'}
          </span>
          {connected && <span style={{ fontSize: 7, color: color.cyan, letterSpacing: 1 }}>{'\u25CF'} WS</span>}
        </div>
        <div style={{ display: 'flex', gap: 16, fontSize: 9, alignItems: 'center' }}>
          <span style={{ color: color.emerald }}>{st.seed.toFixed(1)} SEED</span>
          <span style={{ color: color.amethyst }}>{st.bloom.toFixed(3)} BLOOM</span>
          <span style={{ color: TIER_COLORS[st.tier], fontWeight: 500 }}>{TIERS[st.tier]}</span>
          <span style={{ color: color.gold, fontVariantNumeric: 'tabular-nums' }}>
            {time.toLocaleTimeString('en', { hour12: false })}
          </span>
        </div>
      </div>

      {/* Agent Status Strip */}
      <div style={{
        display: 'flex', gap: 1, padding: '3px 20px', borderBottom: '1px solid rgba(255,255,255,.03)',
        background: 'rgba(8,18,32,.4)',
      }}>
        {AGENT_IDS.map(id => (
          <div key={id} style={{ flex: 1, textAlign: 'center', padding: '3px 0', borderRadius: 2, border: '1px solid rgba(255,255,255,.03)' }}>
            <div style={{ fontSize: 7, letterSpacing: 1, fontWeight: 500, color: PAT[id].color }}>{PAT[id].callsign}</div>
          </div>
        ))}
        <div style={{ width: 1, background: 'var(--line)', margin: '0 4px' }} />
        {SAT_AGENTS.map((s, i) => (
          <div key={i} style={{ padding: '3px 3px', display: 'flex', alignItems: 'center' }}>
            <div style={{ width: 4, height: 4, borderRadius: '50%', background: `${s.color}50` }} />
          </div>
        ))}
      </div>

      {/* Tab Bar */}
      <div style={{ display: 'flex', padding: '0 20px', borderBottom: '1px solid var(--line)', background: 'rgba(4,11,20,.3)' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            background: 'transparent', border: 'none',
            borderBottom: tab === t.id ? '2px solid var(--gold)' : '2px solid transparent',
            color: tab === t.id ? color.gold : color.dim,
            padding: '8px 14px', fontSize: 8, letterSpacing: 2, cursor: 'pointer',
            fontFamily: 'var(--font-mono)', transition: 'all .2s',
          }}>
            <span style={{ marginRight: 5, opacity: .7 }}>{t.icon}</span>{t.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>

        {/* COMMAND TAB */}
        {tab === 'cmd' && (<>
          <div style={{ flex: 1, overflowY: 'auto', padding: '8px 20px' }}>
            {msgs.map((m, i) => {
              const isU = m.type === 'user', isM = m.type === 'mint', isP = m.type === 'pro', isD = m.type === 'done';
              const col = isU ? color.gold : isM ? color.emerald : isD ? color.gold : Object.values(PAT).find(a => a.callsign === m.agent)?.color || '#6B7280';
              return (
                <div key={i} style={{
                  display: 'flex', gap: 10, alignItems: 'flex-start', marginBottom: 1, padding: '2px 0',
                  opacity: m.type === 'route' ? .4 : isP ? .6 : 1,
                }}>
                  <span style={{ fontWeight: 600, minWidth: 54, textAlign: 'right', fontSize: 9, color: col }}>{isU ? 'YOU' : m.agent}</span>
                  <span style={{
                    color: isU ? color.text : isM ? color.emerald : isD ? color.gold : isP ? col : '#9CA3AF',
                    fontSize: isU ? 11 : 10, lineHeight: 1.65,
                    fontStyle: isP ? 'italic' : 'normal',
                  }}>
                    {isM ? '\u25B6 ' + m.text : isD ? '\u2713 ' + m.text : m.text}
                  </span>
                </div>
              );
            })}
            <div ref={feedEnd} />
          </div>

          {/* Quick actions */}
          {!running && (
            <div style={{ padding: '4px 20px', display: 'flex', gap: 4, flexWrap: 'wrap', borderTop: '1px solid rgba(255,255,255,.03)' }}>
              {QUICK_MISSIONS.slice(0, 4).map((m, i) => (
                <button key={i} onClick={() => exec(m)} style={{
                  background: 'rgba(255,255,255,.02)', border: '1px solid var(--line)',
                  color: color.dim, padding: '4px 8px', borderRadius: 3, fontSize: 8,
                  cursor: 'pointer', fontFamily: 'var(--font-mono)', transition: 'all .2s', letterSpacing: .5,
                }} onMouseEnter={e => { (e.target as HTMLElement).style.borderColor = 'rgba(201,169,98,.15)'; (e.target as HTMLElement).style.color = color.muted; }}
                   onMouseLeave={e => { (e.target as HTMLElement).style.borderColor = 'var(--line)'; (e.target as HTMLElement).style.color = color.dim; }}>
                  {m}
                </button>
              ))}
            </div>
          )}

          {/* Input */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: 10, padding: '8px 20px',
            borderTop: '1px solid rgba(201,169,98,.06)', background: 'rgba(201,169,98,.01)',
          }}>
            <span style={{ color: color.gold, fontSize: 12 }}>{'\u25B8'}</span>
            <input ref={inputRef} value={input} onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') { const t = input; setInput(''); exec(t); } }}
              placeholder={running ? 'Executing...' : 'Speak your mission...'}
              disabled={running}
              style={{
                flex: 1, background: 'transparent', border: 'none', color: color.text,
                fontSize: 12, fontFamily: 'var(--font-mono)', outline: 'none', letterSpacing: .5,
              }}
            />
            <div style={{ display: 'flex', gap: 10, fontSize: 8, color: color.ghost }}>
              <span>RAC:{st.rac}</span>
              <span>{st.reflexes}{'\u26A1'}</span>
            </div>
          </div>
        </>)}

        {/* CHARACTER TAB */}
        {tab === 'char' && (
          <div style={{ flex: 1, overflowY: 'auto', padding: 20 }}>
            <div style={{
              padding: 20, borderRadius: 10, border: '1px solid rgba(201,169,98,.1)',
              background: 'rgba(201,169,98,.02)', marginBottom: 14, textAlign: 'center',
            }}>
              <div style={{ fontSize: 8, letterSpacing: 3, color: color.goldDim, fontFamily: 'var(--font-label)' }}>NODE VALUE</div>
              <div style={{ fontSize: 40, fontWeight: 300, color: color.gold, fontFamily: 'var(--font-display)', marginTop: 4 }}>{nv}</div>
            </div>

            <div style={{ padding: 16, borderRadius: 10, border: '1px solid var(--line)', background: 'var(--bg-raised)', marginBottom: 14 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
                <div>
                  <div style={{ fontSize: 8, letterSpacing: 2, color: color.dim }}>LIFECYCLE</div>
                  <div style={{ fontSize: 16, color: color.gold, fontWeight: 400, fontFamily: 'var(--font-display)' }}>{stage.name}</div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: 8, color: color.dim }}>Sovereignty</div>
                  <div style={{ fontSize: 16, color: color.gold, fontFamily: 'var(--font-display)' }}>{(st.sovereignty * 100).toFixed(1)}%</div>
                </div>
              </div>
              <div style={{ width: '100%', height: 4, borderRadius: 99, background: 'rgba(255,255,255,.05)' }}>
                <div style={{
                  height: '100%', borderRadius: 99, background: `linear-gradient(90deg, ${color.goldDim}, ${color.gold})`,
                  transition: 'width .7s cubic-bezier(.16,1,.3,1)',
                  width: `${Math.min(100, stage.high > stage.low ? ((st.sovereignty - stage.low) / (stage.high - stage.low)) * 100 : 100)}%`,
                }} />
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              {[
                { l: 'SEED', v: st.seed.toFixed(2), c: color.emerald },
                { l: 'BLOOM', v: st.bloom.toFixed(3), c: color.amethyst },
                { l: 'IHSAN', v: st.ihsan.toFixed(4), c: color.gold },
                { l: 'TIER', v: TIERS[st.tier], c: TIER_COLORS[st.tier] },
                { l: 'MYELINATION', v: (st.mye * 100).toFixed(0) + '%', c: color.sapphire },
                { l: 'STREAK', v: '' + st.streak, c: color.amber },
              ].map((s, i) => (
                <div key={i} style={{ padding: 12, borderRadius: 8, border: '1px solid var(--line)', background: 'var(--bg-raised)' }}>
                  <div style={{ fontSize: 7, letterSpacing: 2, color: color.dim }}>{s.l}</div>
                  <div style={{ fontSize: 22, fontWeight: 300, color: s.c, fontFamily: 'var(--font-display)', marginTop: 2 }}>{s.v}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* SKILLS TAB */}
        {tab === 'skill' && (
          <div style={{ flex: 1, overflowY: 'auto', padding: 20 }}>
            <div style={{ fontSize: 8, letterSpacing: 2, color: color.dim, marginBottom: 4 }}>
              HDA SKILLS {'\u2014'} {SKILLS.filter(s => s.unlocked).length}/{SKILLS.length}
            </div>
            <div style={{ fontSize: 8, color: color.ghost, marginBottom: 12 }}>
              8 productized desktop actions from founder-ops-agent manifest
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 6 }}>
              {SKILLS.map(sk => {
                const tc = TIER_COLORS[sk.tier];
                return (
                  <div key={sk.id} style={{
                    padding: 10, borderRadius: 6,
                    border: `1px solid ${sk.unlocked ? tc + '18' : 'var(--line)'}`,
                    background: sk.unlocked ? `${tc}04` : 'var(--bg-raised)',
                    opacity: sk.unlocked ? 1 : .35, transition: 'all .2s',
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: 14 }}>{sk.icon}</span>
                      <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
                        {sk.hda && <span style={{ fontSize: 6, color: color.cyan, letterSpacing: 1 }}>HDA</span>}
                        <span style={{ fontSize: 6, color: tc, letterSpacing: 1 }}>{TIERS[sk.tier]}</span>
                      </div>
                    </div>
                    <div style={{ fontSize: 9, marginTop: 4, fontWeight: sk.unlocked ? 500 : 400, color: sk.unlocked ? tc : color.dim }}>{sk.name}</div>
                    <div style={{ fontSize: 7, marginTop: 3, color: sk.unlocked ? color.emerald : color.ghost }}>{sk.unlocked ? '\u2713 Unlocked' : '\u{1F512} Locked'}</div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* QUESTS TAB */}
        {tab === 'quest' && (
          <div style={{ flex: 1, overflowY: 'auto', padding: 20 }}>
            <div style={{ fontSize: 8, letterSpacing: 2, color: color.dim, marginBottom: 4 }}>SCHEDULED MISSIONS</div>
            <div style={{ fontSize: 8, color: color.ghost, marginBottom: 14 }}>
              From founder-ops-agent manifest {'\u00B7'} {config?.work_schedule || '8:00-18:00'}
            </div>
            {SCHEDULED_MISSIONS.map((q, i) => (
              <div key={i} style={{
                display: 'flex', alignItems: 'center', gap: 12, padding: '12px 14px', marginBottom: 5,
                borderRadius: 8, border: '1px solid var(--line)', background: 'var(--bg-raised)',
                transition: 'border-color .2s',
              }} onMouseEnter={e => e.currentTarget.style.borderColor = 'rgba(201,169,98,.12)'}
                 onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--line)'}>
                <span style={{ fontSize: 20 }}>{q.icon}</span>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 12, fontWeight: 500, fontFamily: 'var(--font-display)' }}>{q.name}</div>
                  <div style={{ fontSize: 9, color: color.dim, fontFamily: 'var(--font-display)', fontStyle: 'italic', marginTop: 1 }}>{q.description}</div>
                  <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
                    {q.agents.map((a, j) => (
                      <span key={j} style={{
                        fontSize: 7, letterSpacing: 1,
                        color: Object.values(PAT).find(p => p.callsign === a)?.color || color.dim,
                      }}>{a}</span>
                    ))}
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ color: color.emerald, fontSize: 10, fontFamily: 'var(--font-mono)' }}>+{q.seedReward} SEED</div>
                  <div style={{ fontSize: 7, color: color.dim, marginTop: 2 }}>{q.cron}</div>
                  <div style={{ fontSize: 7, color: q.auto ? color.cyan : color.amber, marginTop: 3, letterSpacing: 1 }}>
                    {q.auto ? 'AUTO' : 'APPROVAL'}
                  </div>
                </div>
              </div>
            ))}

            <div style={{ fontSize: 8, letterSpacing: 2, color: color.dim, marginTop: 24, marginBottom: 10 }}>AD-HOC MISSIONS</div>
            {[
              { name: 'File Janitor', seed: '0.50', icon: '\u{1F9F9}', desc: 'Organize a folder' },
              { name: 'Report Generator', seed: '1.00', icon: '\u{1F4CA}', desc: 'Create report from data' },
              { name: 'Build Pipeline', seed: '2.00', icon: '\u{1F3D7}\uFE0F', desc: 'Full CI/CD execution' },
              { name: 'Knowledge Crawl', seed: '5.00', icon: '\u{1F9E0}', desc: 'Index your digital life' },
            ].map((q, i) => (
              <div key={i} style={{
                display: 'flex', alignItems: 'center', gap: 12, padding: '10px 14px', marginBottom: 4,
                borderRadius: 8, border: '1px solid var(--line)', background: 'transparent', opacity: .6,
              }}>
                <span style={{ fontSize: 18 }}>{q.icon}</span>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 11, fontWeight: 500, fontFamily: 'var(--font-display)' }}>{q.name}</div>
                  <div style={{ fontSize: 8, color: color.dim, fontStyle: 'italic' }}>{q.desc}</div>
                </div>
                <div style={{ color: color.emerald, fontSize: 9, fontFamily: 'var(--font-mono)' }}>+{q.seed}</div>
              </div>
            ))}
          </div>
        )}

        {/* COMMUNITY TAB */}
        {tab === 'comm' && (
          <div style={{ flex: 1, overflowY: 'auto', padding: 20 }}>
            {/* Network Asabiyyah */}
            <div style={{
              padding: 20, borderRadius: 10, border: '1px solid rgba(96,165,250,.1)',
              background: 'rgba(96,165,250,.02)', marginBottom: 14, textAlign: 'center',
            }}>
              <div style={{ fontSize: 8, letterSpacing: 3, color: color.sapphire, fontFamily: 'var(--font-label)' }}>
                NETWORK ASABIYYAH
              </div>
              <div style={{ fontSize: 40, fontWeight: 300, color: color.sapphire, fontFamily: 'var(--font-display)', marginTop: 4 }}>
                {(0.72).toFixed(3)}
              </div>
              <div style={{ fontSize: 8, color: color.dim, marginTop: 4 }}>
                Ibn Khaldun social cohesion index {'\u00B7'} Modulates network minting rate
              </div>
              <div style={{ width: '100%', height: 4, borderRadius: 99, background: 'rgba(255,255,255,.05)', marginTop: 8 }}>
                <div style={{
                  height: '100%', borderRadius: 99,
                  background: `linear-gradient(90deg, ${color.sapphire}80, ${color.sapphire})`,
                  width: '72%', transition: 'width .7s cubic-bezier(.16,1,.3,1)',
                  boxShadow: `0 0 10px ${color.sapphire}20`,
                }} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4, fontSize: 7, color: color.ghost }}>
                <span>0.00 Fragmented</span>
                <span>0.50 Neutral</span>
                <span>1.00 Cohesive</span>
              </div>
            </div>

            {/* Gini + Network Stats */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 14 }}>
              <div style={{ padding: 14, borderRadius: 8, border: '1px solid var(--line)', background: 'var(--bg-raised)' }}>
                <div style={{ fontSize: 7, letterSpacing: 2, color: color.dim }}>GINI COEFFICIENT</div>
                <div style={{ fontSize: 26, fontWeight: 300, color: color.emerald, fontFamily: 'var(--font-display)', marginTop: 2 }}>0.218</div>
                <div style={{ fontSize: 7, color: color.emerald, marginTop: 2 }}>{'\u2713'} Below 0.35 ADL gate</div>
              </div>
              <div style={{ padding: 14, borderRadius: 8, border: '1px solid var(--line)', background: 'var(--bg-raised)' }}>
                <div style={{ fontSize: 7, letterSpacing: 2, color: color.dim }}>MINT MULTIPLIER</div>
                <div style={{ fontSize: 26, fontWeight: 300, color: color.gold, fontFamily: 'var(--font-display)', marginTop: 2 }}>1.088x</div>
                <div style={{ fontSize: 7, color: color.dim, marginTop: 2 }}>Range: 0.80x {'\u2014'} 1.20x</div>
              </div>
              <div style={{ padding: 14, borderRadius: 8, border: '1px solid var(--line)', background: 'var(--bg-raised)' }}>
                <div style={{ fontSize: 7, letterSpacing: 2, color: color.dim }}>NETWORK SIZE</div>
                <div style={{ fontSize: 26, fontWeight: 300, color: color.cyan, fontFamily: 'var(--font-display)', marginTop: 2 }}>1</div>
                <div style={{ fontSize: 7, color: color.dim, marginTop: 2 }}>Active sovereign nodes</div>
              </div>
              <div style={{ padding: 14, borderRadius: 8, border: '1px solid var(--line)', background: 'var(--bg-raised)' }}>
                <div style={{ fontSize: 7, letterSpacing: 2, color: color.dim }}>ATTESTATIONS</div>
                <div style={{ fontSize: 26, fontWeight: 300, color: color.amethyst, fontFamily: 'var(--font-display)', marginTop: 2 }}>0</div>
                <div style={{ fontSize: 7, color: color.dim, marginTop: 2 }}>Peer trust links</div>
              </div>
            </div>

            {/* Peer Attestation Status */}
            <div style={{ fontSize: 8, letterSpacing: 2, color: color.dim, marginBottom: 8 }}>PEER ATTESTATION GRAPH</div>
            <div style={{
              padding: 20, borderRadius: 10, border: '1px solid var(--line)',
              background: 'var(--bg-raised)', textAlign: 'center', marginBottom: 14,
            }}>
              <div style={{
                width: 64, height: 64, borderRadius: '50%', margin: '0 auto 10px',
                border: `2px solid ${color.gold}30`,
                background: `radial-gradient(circle, ${color.gold}08 0%, transparent 70%)`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 9, color: color.gold, fontFamily: 'var(--font-label)', letterSpacing: 2,
              }}>
                YOU
              </div>
              <div style={{ fontSize: 9, color: color.muted }}>
                Genesis node {'\u2014'} no peers connected yet
              </div>
              <div style={{ fontSize: 8, color: color.dim, marginTop: 6 }}>
                Reach <span style={{ color: color.emerald }}>Verifier</span> stage (55% sovereignty) to attest others
              </div>
            </div>

            {/* Constitutional Health */}
            <div style={{ fontSize: 8, letterSpacing: 2, color: color.dim, marginBottom: 8 }}>CONSTITUTIONAL HEALTH</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              {[
                { gate: 'ADL Gini', status: true, value: '0.218 \u2264 0.35', c: color.emerald },
                { gate: 'Ihsan Floor', status: true, value: '\u2265 0.95', c: color.emerald },
                { gate: 'BFT Quorum', status: true, value: 'f < n/3', c: color.emerald },
                { gate: 'Zakat Collection', status: true, value: '2.5% applied', c: color.emerald },
                { gate: 'Demurrage', status: true, value: 'Active', c: color.emerald },
                { gate: 'Declaration Hash', status: true, value: 'BLAKE2b verified', c: color.emerald },
              ].map((g, i) => (
                <div key={i} style={{
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  padding: '6px 12px', borderRadius: 6, border: '1px solid var(--line)', background: 'var(--bg-raised)',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ fontSize: 9, color: g.c }}>{g.status ? '\u2713' : '\u2717'}</span>
                    <span style={{ fontSize: 9, color: color.text }}>{g.gate}</span>
                  </div>
                  <span style={{ fontSize: 8, color: g.c, fontFamily: 'var(--font-mono)' }}>{g.value}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* PROGRESS / WALLET TAB */}
        {tab === 'prog' && (
          <div style={{ flex: 1, overflowY: 'auto', padding: 20 }}>
            {/* Connection status */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
              <div style={{ fontSize: 8, letterSpacing: 2, color: color.dim }}>SOVEREIGN WALLET</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ width: 5, height: 5, borderRadius: '50%', background: w.live ? color.emerald : color.flame }} />
                <span style={{ fontSize: 7, color: w.live ? color.emerald : color.flame, letterSpacing: 1 }}>
                  {w.live ? 'LIVE' : 'OFFLINE'}
                </span>
                {w.lastSync && (
                  <span style={{ fontSize: 7, color: color.ghost }}>
                    {Math.round((Date.now() - w.lastSync) / 1000)}s ago
                  </span>
                )}
              </div>
            </div>

            {/* Token Balances */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginBottom: 14 }}>
              <div style={{ padding: 16, borderRadius: 10, border: `1px solid ${color.emerald}18`, background: `${color.emerald}04`, textAlign: 'center' }}>
                <div style={{ fontSize: 7, letterSpacing: 2, color: color.dim }}>SEED</div>
                <div style={{ fontSize: 28, fontWeight: 300, color: color.emerald, fontFamily: 'var(--font-display)', marginTop: 4 }}>
                  {w.seed.toFixed(2)}
                </div>
                {w.lockedSeed > 0 && (
                  <div style={{ fontSize: 7, color: color.ghost, marginTop: 4 }}>
                    {w.lockedSeed.toFixed(2)} staked
                  </div>
                )}
              </div>
              <div style={{ padding: 16, borderRadius: 10, border: `1px solid ${color.amethyst}18`, background: `${color.amethyst}04`, textAlign: 'center' }}>
                <div style={{ fontSize: 7, letterSpacing: 2, color: color.dim }}>BLOOM</div>
                <div style={{ fontSize: 28, fontWeight: 300, color: color.amethyst, fontFamily: 'var(--font-display)', marginTop: 4 }}>
                  {w.bloom.toFixed(3)}
                </div>
                <div style={{ fontSize: 7, color: color.ghost, marginTop: 4 }}>Soulbound</div>
              </div>
              <div style={{ padding: 16, borderRadius: 10, border: `1px solid ${color.gold}18`, background: `${color.gold}04`, textAlign: 'center' }}>
                <div style={{ fontSize: 7, letterSpacing: 2, color: color.dim }}>ZAKAT</div>
                <div style={{ fontSize: 28, fontWeight: 300, color: color.gold, fontFamily: 'var(--font-display)', marginTop: 4 }}>
                  {w.zakatContributed.toFixed(4)}
                </div>
                <div style={{ fontSize: 7, color: color.ghost, marginTop: 4 }}>2.5% contributed</div>
              </div>
            </div>

            {/* Supply Cap Gauge */}
            <div style={{
              padding: 14, borderRadius: 10, border: '1px solid var(--line)',
              background: 'var(--bg-raised)', marginBottom: 14,
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                <span style={{ fontSize: 8, letterSpacing: 2, color: color.dim }}>ANNUAL SUPPLY CAP</span>
                <span style={{ fontSize: 8, color: color.muted, fontVariantNumeric: 'tabular-nums' }}>
                  {(w.supplyCapUtilization * 100).toFixed(3)}% of {(THRESHOLDS.SEED_SUPPLY_CAP_PER_YEAR / 1000).toFixed(0)}K
                </span>
              </div>
              <div style={{ width: '100%', height: 6, borderRadius: 99, background: 'rgba(255,255,255,.05)' }}>
                <div style={{
                  height: '100%', borderRadius: 99,
                  background: w.supplyCapUtilization > 0.9
                    ? `linear-gradient(90deg, ${color.flame}, ${color.ruby})`
                    : `linear-gradient(90deg, ${color.emerald}80, ${color.emerald})`,
                  width: `${Math.min(100, w.supplyCapUtilization * 100)}%`,
                  transition: 'width .7s cubic-bezier(.16,1,.3,1)',
                  boxShadow: `0 0 8px ${w.supplyCapUtilization > 0.9 ? color.ruby : color.emerald}20`,
                }} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4, fontSize: 7, color: color.ghost }}>
                <span>Minted: {w.totalSeed.toFixed(0)}</span>
                <span>Circulating: {w.circulating.toFixed(0)}</span>
              </div>
            </div>

            {/* Seed Potential (PoI Factors) */}
            <div style={{ fontSize: 8, letterSpacing: 2, color: color.dim, marginBottom: 12 }}>SEED POTENTIAL FACTORS</div>
            {[
              { l: 'Sovereignty', v: w.factors.sovereignty, c: color.gold },
              { l: 'Activation', v: w.factors.activation, c: color.emerald },
              { l: 'Quality (Ihsan)', v: w.factors.quality, c: color.amber },
              { l: 'Compounding', v: w.factors.compounding, c: color.sapphire },
              { l: 'Synergy', v: w.factors.synergy, c: color.amethyst },
            ].map((f, i) => (
              <div key={i} style={{ marginBottom: 14 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{ fontSize: 10, color: f.c }}>{f.l}</span>
                  <span style={{ fontSize: 10, color: f.c, fontVariantNumeric: 'tabular-nums' }}>{f.v.toFixed(3)}</span>
                </div>
                <div style={{ width: '100%', height: 3, borderRadius: 99, background: 'rgba(255,255,255,.05)' }}>
                  <div style={{
                    height: '100%', borderRadius: 99, background: f.c,
                    transition: 'width .6s cubic-bezier(.16,1,.3,1)',
                    width: Math.min(100, f.v * 100) + '%',
                    boxShadow: `0 0 8px ${f.c}25`,
                  }} />
                </div>
              </div>
            ))}

            {/* Node Value Composite */}
            <div style={{
              padding: 18, borderRadius: 10, textAlign: 'center',
              border: '1px solid rgba(201,169,98,.1)', background: 'rgba(201,169,98,.02)', marginTop: 12,
            }}>
              <div style={{ fontSize: 8, letterSpacing: 3, color: color.goldDim, fontFamily: 'var(--font-label)' }}>NODE VALUE</div>
              <div style={{ fontSize: 36, fontWeight: 300, color: color.gold, marginTop: 6, fontFamily: 'var(--font-display)' }}>{nv}</div>
            </div>

            {/* Lifecycle Tracker */}
            <div style={{ marginTop: 20 }}>
              <div style={{ fontSize: 8, letterSpacing: 2, color: color.dim, marginBottom: 10 }}>SEED {'\u2192'} CATALYST</div>
              {STAGES.map((s, i) => {
                const active = st.sovereignty >= s.low;
                const cur = stage.name === s.name;
                return (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 5 }}>
                    <div style={{
                      width: 20, height: 20, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center',
                      fontSize: 8, background: cur ? 'rgba(201,169,98,.08)' : active ? 'rgba(52,211,153,.05)' : 'transparent',
                      border: `1px solid ${cur ? color.gold : active ? 'rgba(52,211,153,.2)' : 'var(--line)'}`,
                      color: cur ? color.gold : active ? color.emerald : color.ghost,
                    }}>
                      {cur ? '\u25C9' : active ? '\u2713' : '\u25CB'}
                    </div>
                    <span style={{ fontSize: 10, color: cur ? color.gold : active ? color.emerald : color.dim, fontWeight: cur ? 500 : 400 }}>{s.name}</span>
                    <span style={{ fontSize: 8, color: color.dim }}>{(s.low * 100).toFixed(0)}%</span>
                    {cur && <span style={{ fontSize: 8, color: color.gold }}>{'\u25C4'}</span>}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Status Bar */}
      <div style={{
        display: 'flex', justifyContent: 'space-between', padding: '4px 20px',
        fontSize: 7, letterSpacing: 1, color: color.ghost, borderTop: '1px solid var(--line)',
        background: 'rgba(4,11,20,.5)',
      }}>
        <span>{userName.toUpperCase()} {'\u00B7'} {TIERS[st.tier].toUpperCase()} {'\u00B7'} {stage.name.toUpperCase()}</span>
        <span>PAT-7 {'\u00B7'} SAT-5 {'\u00B7'} 15 ALG {'\u00B7'} 7 INV {'\u00B7'} {
          config?.autonomy?.includes('Full') ? 'FULL AUTO' : config?.autonomy?.includes('Ask') ? 'MANUAL' : 'SEMI-AUTO'
        }</span>
      </div>
    </div>
  );
}
