"use client";

import { useSovereignHealth, useSeedPotential, useTokenBalance } from "@/hooks/use-sovereign-api";
import { PAT_AGENT_MANIFEST, SAT_AGENT_MANIFEST, TERMINAL_THEME } from "./terminal/terminal-manifest";

const PIPELINE = [
  { key: "intent", label: "Intent", c: TERMINAL_THEME.gold },
  { key: "guardian", label: "Guardian", c: TERMINAL_THEME.success },
  { key: "execution", label: "Execute", c: TERMINAL_THEME.info },
  { key: "receipt", label: "Receipt", c: "#AFA9EC" },
  { key: "chain", label: "Chain", c: "#1D9E75" },
  { key: "evidence", label: "Evidence", c: TERMINAL_THEME.alert },
] as const;

function IhsanGauge({ score }: { score: number }) {
  const r = 44, circ = 2 * Math.PI * r;
  const offset = circ * (1 - Math.min(score, 1));
  const ok = score >= 0.95;
  const color = ok ? TERMINAL_THEME.success : score > 0.7 ? TERMINAL_THEME.alert : TERMINAL_THEME.danger;
  return (
    <div style={{ position: "relative", width: 100, height: 100 }}>
      <svg width={100} height={100} style={{ transform: "rotate(-90deg)" }}>
        <circle cx={50} cy={50} r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={5} />
        <circle cx={50} cy={50} r={r} fill="none" stroke={color} strokeWidth={5}
          strokeDasharray={circ} strokeDashoffset={offset} strokeLinecap="round"
          style={{ transition: "stroke-dashoffset 1s ease, stroke 0.4s" }} />
      </svg>
      <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
        <span style={{ fontSize: 20, fontWeight: 600, color, fontFamily: "var(--font-mono)" }}>{(score * 100).toFixed(1)}</span>
        <span style={{ fontSize: 9, color: "rgba(255,255,255,0.35)", letterSpacing: 1, textTransform: "uppercase" }}>{ok ? "passed" : "held"}</span>
      </div>
    </div>
  );
}

function PipelineBar({ stage }: { stage: number }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 2, padding: "12px 0" }}>
      {PIPELINE.map((s, i) => {
        const on = i <= stage;
        return (
          <div key={s.key} style={{ display: "flex", alignItems: "center", flex: 1 }}>
            <div style={{ flex: 1, textAlign: "center" }}>
              <div style={{ width: 26, height: 26, borderRadius: "50%", margin: "0 auto", background: on ? s.c : "rgba(255,255,255,0.04)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, color: on ? "#020408" : "rgba(255,255,255,0.15)", fontWeight: 600, transition: "all 0.3s" }}>{i + 1}</div>
              <div style={{ fontSize: 9, marginTop: 3, color: on ? "rgba(255,255,255,0.6)" : "rgba(255,255,255,0.15)", fontFamily: "var(--font-mono)" }}>{s.label}</div>
            </div>
            {i < PIPELINE.length - 1 && <div style={{ height: 1.5, width: 14, background: i < stage ? PIPELINE[i + 1].c : "rgba(255,255,255,0.05)", borderRadius: 1 }} />}
          </div>
        );
      })}
    </div>
  );
}

function AgentGrid() {
  const all = [...PAT_AGENT_MANIFEST, ...SAT_AGENT_MANIFEST];
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 5 }}>
      {all.map((a) => {
        const pat = a.team === "PAT";
        const on = a.status === "active";
        return (
          <div key={a.id} style={{ padding: "5px 7px", borderRadius: 5, background: on ? (pat ? "rgba(201,169,98,0.07)" : "rgba(93,202,165,0.07)") : "rgba(255,255,255,0.01)", border: `1px solid ${on ? (pat ? "rgba(201,169,98,0.18)" : "rgba(93,202,165,0.18)") : "rgba(255,255,255,0.03)"}` }}>
            <div style={{ fontSize: 10, fontWeight: 600, color: on ? "#fff" : "rgba(255,255,255,0.2)", fontFamily: "var(--font-mono)" }}>{a.emoji} {a.call}</div>
            <div style={{ fontSize: 8, color: pat ? TERMINAL_THEME.gold : TERMINAL_THEME.success, fontFamily: "var(--font-mono)", marginTop: 1 }}>{a.team} · {a.role}</div>
          </div>
        );
      })}
    </div>
  );
}

export default function OperatorCockpit() {
  const { data: health } = useSovereignHealth();
  const { data: potential } = useSeedPotential();
  const { data: balance } = useTokenBalance();

  const isLive = health.live_status === "LIVE" || health.running === true;
  const ihsan = health.ihsan_score ?? potential.reward_ema ?? 0;
  const snr = health.snr_score ?? 0;
  const gini = health.gini ?? 0;
  const w = health.wallet_snapshot ?? { seed: balance?.seed ?? 0, bloom: balance?.bloom ?? 0 };

  return (
    <div className="p-4 max-w-4xl mx-auto">
      <div className="flex items-start justify-between mb-3">
        <div>
          <h2 className="text-lg font-bold text-slate-100">Operator Cockpit</h2>
          <p className="text-xs text-slate-500">Constitutional pipeline · Intent → Evidence</p>
        </div>
        <div className="flex items-center gap-2">
          <span className={`w-1.5 h-1.5 rounded-full ${isLive ? "bg-emerald-500 animate-pulse" : "bg-red-500"}`} />
          <span className={`text-[10px] font-mono ${isLive ? "text-emerald-400" : "text-red-400"}`}>{isLive ? "LIVE" : "OFFLINE"}</span>
        </div>
      </div>
      <PipelineBar stage={isLive ? 5 : 0} />
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 8 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 14 }}>
            <IhsanGauge score={ihsan} />
            <div>
              <div className="text-[10px] text-slate-500 font-mono mb-1">GUARDIAN VERDICT</div>
              <div className={`text-xs font-bold font-mono px-2 py-1 rounded inline-block ${ihsan >= 0.95 ? "bg-emerald-900/30 text-emerald-400 border border-emerald-800/30" : "bg-amber-900/30 text-amber-400 border border-amber-800/30"}`}>
                {ihsan >= 0.95 ? "APPROVED" : "HELD — Below floor"}
              </div>
              <div className="text-[10px] text-slate-600 font-mono mt-2">SNR {snr.toFixed(4)} | Gini {gini.toFixed(4)}</div>
            </div>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 5, marginBottom: 10 }}>
            {[{ l: "IHSAN", v: ihsan.toFixed(2), ok: ihsan >= 0.95, c: "#AFA9EC" }, { l: "AMANAH", v: "Signed", ok: true, c: "#5DCAA5" }, { l: "ADL", v: "Exact", ok: true, c: TERMINAL_THEME.gold }].map((t) => (
              <div key={t.l} className="text-center p-2 rounded" style={{ background: t.ok ? `${t.c}08` : "rgba(239,68,68,0.06)", border: `1px solid ${t.ok ? `${t.c}20` : "rgba(239,68,68,0.2)"}` }}>
                <div style={{ fontSize: 9, color: t.c, fontFamily: "var(--font-mono)", fontWeight: 600, letterSpacing: 1 }}>{t.l}</div>
                <div style={{ fontSize: 12, fontWeight: 700, color: t.ok ? t.c : TERMINAL_THEME.danger, marginTop: 2, fontFamily: "var(--font-mono)" }}>{t.v}</div>
              </div>
            ))}
          </div>
          <div className="text-[10px] font-mono font-semibold mb-1.5" style={{ color: TERMINAL_THEME.gold, letterSpacing: 1 }}>12 AGENTS (7 PAT + 5 SAT)</div>
          <AgentGrid />
        </div>
        <div>
          <div className="bg-slate-900/40 rounded-lg p-3 border border-slate-800/50 mb-3">
            <div className="text-[10px] font-mono font-semibold mb-2" style={{ color: TERMINAL_THEME.gold, letterSpacing: 1 }}>RECEIPT</div>
            {[["Algorithm", "BLAKE3"], ["SNR", snr.toFixed(4)], ["Ihsan", ihsan.toFixed(4)], ["SEED", w.seed.toFixed(2)], ["BLOOM", w.bloom.toFixed(2)], ["Heartbeat", `${health.tick_interval_s ?? 60}s`]].map(([k, v]) => (
              <div key={k} className="flex justify-between py-0.5 border-b border-slate-800/30"><span className="text-[10px] text-slate-500">{k}</span><span className="text-[10px] text-slate-300 font-mono">{v}</span></div>
            ))}
          </div>
          <div className="bg-slate-900/40 rounded-lg p-3 border border-slate-800/50">
            <div className="text-[10px] font-mono font-semibold mb-2" style={{ color: "#5DCAA5", letterSpacing: 1 }}>ECONOMIC STATE</div>
            <div className="grid grid-cols-2 gap-2">
              <div className="text-center p-2 bg-slate-800/30 rounded"><div className="text-xl font-bold text-amber-400">{w.seed.toFixed(1)}</div><div className="text-[9px] text-slate-500">SEED</div></div>
              <div className="text-center p-2 bg-slate-800/30 rounded"><div className="text-xl font-bold text-sky-400">{w.bloom.toFixed(1)}</div><div className="text-[9px] text-slate-500">BLOOM</div></div>
            </div>
            <div className="mt-2 text-[10px] text-slate-600 font-mono">Gini: {gini.toFixed(4)} {gini <= 0.35 ? "✓ Adl" : "⚠ above"}</div>
          </div>
        </div>
      </div>
    </div>
  );
}
