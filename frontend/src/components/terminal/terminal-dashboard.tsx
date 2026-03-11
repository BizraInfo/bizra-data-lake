"use client";

import {
  useSovereignHealth,
  useSeedPotential,
  useTokenBalance,
  useConstitutionalStatus,
} from "@/hooks/use-sovereign-api";

export default function TerminalDashboard() {
  const { data: health } = useSovereignHealth();
  const { data: potential } = useSeedPotential();
  const { data: balance } = useTokenBalance();
  const { data: constitutional } = useConstitutionalStatus();

  const isLive = health.status !== "unknown";

  return (
    <div className="p-4 max-w-3xl mx-auto">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-bold text-slate-100">Dashboard</h2>
          <p className="text-xs text-slate-500">Sovereign node overview</p>
        </div>
        <div className="flex items-center gap-1.5">
          <span className={`w-2 h-2 rounded-full ${isLive ? "bg-emerald-500 animate-pulse" : "bg-red-500"}`} />
          <span className={`text-[10px] ${isLive ? "text-emerald-400" : "text-red-400"}`}>
            {isLive ? "LIVE" : "OFFLINE"}
          </span>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
        <div className="bg-slate-800/50 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-amber-400">{balance.seed.toFixed(1)}</div>
          <div className="text-[10px] text-slate-500">SEED</div>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-purple-400">{balance.bloom.toFixed(1)}</div>
          <div className="text-[10px] text-slate-500">BLOOM</div>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-3 text-center">
          <div className={`text-2xl font-bold ${constitutional.ihsan >= 0.95 ? "text-emerald-400" : "text-amber-400"}`}>
            {constitutional.ihsan.toFixed(2)}
          </div>
          <div className="text-[10px] text-slate-500">Ihsan</div>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-teal-400">{potential.tier}</div>
          <div className="text-[10px] text-slate-500">Tier</div>
        </div>
      </div>

      {/* Constitutional Gates */}
      <div className="border border-slate-700/50 rounded-lg p-4 mb-3">
        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
          Constitutional Gates
        </h3>
        <div className="space-y-2">
          {[
            { label: "Ihsan", value: constitutional.ihsan, threshold: 0.95 },
            { label: "SNR", value: constitutional.snr, threshold: 0.85 },
            { label: "Gini", value: constitutional.gini, threshold: 0.35, inverted: true },
          ].map((gate) => {
            const passed = gate.inverted ? gate.value <= gate.threshold : gate.value >= gate.threshold;
            return (
              <div key={gate.label} className="flex items-center justify-between">
                <span className="text-xs text-slate-300">{gate.label}</span>
                <div className="flex items-center gap-2">
                  <span className={`text-xs font-mono ${passed ? "text-emerald-400" : "text-red-400"}`}>
                    {gate.value.toFixed(2)}
                  </span>
                  <span className={`text-[10px] ${passed ? "text-emerald-600" : "text-red-600"}`}>
                    {passed ? "PASS" : "FAIL"}
                  </span>
                </div>
              </div>
            );
          })}
          <div className="flex items-center justify-between pt-1 border-t border-slate-800">
            <span className="text-xs text-slate-400">Gates</span>
            <span className="text-xs text-emerald-400 font-bold">
              {constitutional.gates_passed}/{constitutional.gates_total}
            </span>
          </div>
        </div>
      </div>

      {/* Health */}
      <div className="border border-slate-700/50 rounded-lg p-3">
        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
          Node Health
        </h3>
        <div className="grid grid-cols-3 gap-2 text-xs text-center">
          <div className="bg-slate-800/50 rounded p-2">
            <div className="text-slate-300 font-mono">{health.version}</div>
            <div className="text-slate-600">Version</div>
          </div>
          <div className="bg-slate-800/50 rounded p-2">
            <div className="text-slate-300 font-mono">{Math.floor(health.uptime_s / 3600)}h</div>
            <div className="text-slate-600">Uptime</div>
          </div>
          <div className="bg-slate-800/50 rounded p-2">
            <div className={`font-mono ${health.snr_score >= 0.85 ? "text-emerald-400" : "text-red-400"}`}>
              {health.snr_score.toFixed(2)}
            </div>
            <div className="text-slate-600">SNR</div>
          </div>
        </div>
      </div>
    </div>
  );
}
