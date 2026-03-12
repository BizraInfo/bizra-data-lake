"use client";

import { useSeedPotential, useSovereignHealth } from "@/hooks/use-sovereign-api";

function metricColor(
  value: number,
  greenWhen: (value: number) => boolean,
  amberWhen?: (value: number) => boolean,
): string {
  if (greenWhen(value)) {
    return "text-emerald-400";
  }
  if (amberWhen?.(value)) {
    return "text-amber-400";
  }
  return "text-red-400";
}

function formatTick(timestamp: string): string {
  if (!timestamp) {
    return "Awaiting first tick";
  }

  const parsed = new Date(timestamp);
  if (Number.isNaN(parsed.getTime())) {
    return timestamp;
  }

  return parsed.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function TerminalDashboard() {
  const { data: health } = useSovereignHealth();
  const { data: potential } = useSeedPotential();

  const isLive = health.live_status === "LIVE" || health.running === true;
  const ihsan = health.ihsan_score ?? potential.reward_ema;
  const snr = health.snr_score ?? 0;
  const gini = health.gini ?? 0;
  const wallet = health.wallet_snapshot ?? { seed: 0, bloom: 0 };
  const heartbeat = health.tick_interval_s ?? 60;

  return (
    <div className="p-4 max-w-4xl mx-auto">
      <div className="flex items-start justify-between mb-4 gap-4">
        <div>
          <h2 className="text-lg font-bold text-slate-100">Dashboard</h2>
          <p className="text-xs text-slate-500">Immediate node readiness and sovereignty overview</p>
        </div>
        <div className="text-right">
          <div className="flex items-center justify-end gap-1.5">
            <span className={`w-2 h-2 rounded-full ${isLive ? "bg-emerald-500 animate-pulse" : "bg-red-500"}`} />
            <span className={`text-[10px] font-bold ${isLive ? "text-emerald-400" : "text-red-400"}`}>
              {isLive ? "LIVE" : "OFFLINE"}
            </span>
          </div>
          <p className="text-[10px] text-slate-600 mt-1">Tier {potential.tier}</p>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
        <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-amber-400">{wallet.seed.toFixed(1)}</div>
          <div className="text-[10px] text-slate-500">SEED</div>
        </div>
        <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-sky-400">{wallet.bloom.toFixed(1)}</div>
          <div className="text-[10px] text-slate-500">BLOOM</div>
        </div>
        <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-3 text-center">
          <div className={`text-2xl font-bold ${metricColor(ihsan, (value) => value >= 0.95, (value) => value >= 0.85)}`}>
            {ihsan.toFixed(2)}
          </div>
          <div className="text-[10px] text-slate-500">Ihsan</div>
        </div>
        <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-3 text-center">
          <div className={`text-2xl font-bold ${metricColor(snr, (value) => value >= 0.85)}`}>
            {snr.toFixed(2)}
          </div>
          <div className="text-[10px] text-slate-500">SNR</div>
        </div>
      </div>

      <div className="grid md:grid-cols-[1.1fr_0.9fr] gap-4">
        <section className="border border-slate-700/50 rounded-lg p-4">
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
            Constitutional Status
          </h3>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-300">Ihsan</span>
              <span className={`text-xs font-mono ${metricColor(ihsan, (value) => value >= 0.95, (value) => value >= 0.85)}`}>
                {ihsan.toFixed(2)}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-300">SNR</span>
              <span className={`text-xs font-mono ${metricColor(snr, (value) => value >= 0.85)}`}>
                {snr.toFixed(2)}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-300">Gini</span>
              <span className={`text-xs font-mono ${metricColor(gini, (value) => value <= 0.35, (value) => value <= 0.5)}`}>
                {gini.toFixed(2)}
              </span>
            </div>
            <div className="flex items-center justify-between pt-2 border-t border-slate-800">
              <span className="text-xs text-slate-400">Heartbeat</span>
              <span className="text-xs text-slate-300">
                {formatTick(health.last_tick_timestamp ?? "")} | every {heartbeat}s
              </span>
            </div>
          </div>
        </section>

        <section className="border border-slate-700/50 rounded-lg p-4">
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
            Readiness Snapshot
          </h3>
          <div className="space-y-3 text-xs">
            <div>
              <div className="text-slate-500 mb-1">Last mission</div>
              <p className="text-slate-200">{health.last_mission_summary || "No mission recorded yet."}</p>
            </div>
            <div>
              <div className="text-slate-500 mb-1">Next action</div>
              <p className="text-slate-300">Review the envelope and launch a mission.</p>
            </div>
            <div className="grid grid-cols-2 gap-2 pt-2 border-t border-slate-800">
              <div className="bg-slate-900/60 rounded p-2">
                <div className="text-slate-200 font-mono">{potential.streak}</div>
                <div className="text-slate-600">Qualified streak</div>
              </div>
              <div className="bg-slate-900/60 rounded p-2">
                <div className="text-slate-200 font-mono">{potential.episodes_total}</div>
                <div className="text-slate-600">Episodes</div>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
