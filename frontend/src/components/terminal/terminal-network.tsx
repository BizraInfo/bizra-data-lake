"use client";

import {
  useNetworkEffect,
  useNetworkMilestones,
  useNodeLifecycle,
  useNodeValue,
} from "@/hooks/use-sovereign-api";

function formatNodes(value: number): string {
  if (value >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(0)}B`;
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(0)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(0)}K`;
  return value.toString();
}

function scoreColor(value: number): string {
  if (value >= 0.8) return "text-emerald-400";
  if (value >= 0.6) return "text-amber-400";
  return "text-red-400";
}

export default function TerminalNetwork() {
  const { data: nodeValue } = useNodeValue();
  const { data: lifecycle } = useNodeLifecycle();
  const { data: networkEffect } = useNetworkEffect();
  const { data: milestones } = useNetworkMilestones();

  const factorRows = [
    ["Potential", Number(nodeValue.potential ?? 0)],
    ["Activation", Number(nodeValue.activation ?? 0)],
    ["Quality", Number(nodeValue.quality ?? 0)],
    ["Compounding", Number(nodeValue.compounding ?? 0)],
    ["Synergy", Number(nodeValue.synergy ?? 0)],
  ] as const;

  const composite = Number(nodeValue.composite ?? 0);
  const stageRank = Number(lifecycle.rank ?? 0);
  const sovereigntyScore = Number(lifecycle.sovereignty_score ?? 0);
  const progress = Number(lifecycle.progress ?? 0);
  const nodes = Number(networkEffect.nodes ?? 0);
  const skillsAvailable = Number(networkEffect.skills_available ?? 0);
  const computeTflops = Number(networkEffect.compute_tflops ?? 0);
  const latencyFactor = Number(networkEffect.latency_factor ?? 0);

  return (
    <div className="p-4 max-w-3xl mx-auto">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-bold text-slate-100">Network & Forest</h2>
          <p className="text-xs text-slate-500">Your place in the sovereign lattice</p>
        </div>
        <div className="text-xs text-slate-600">node value {composite.toFixed(2)}</div>
      </div>

      <section className="border border-slate-700/50 rounded-lg p-4 mb-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider">
            5-Factor Node Value
          </h3>
          <span className={`text-2xl font-bold ${scoreColor(composite)}`}>{composite.toFixed(2)}</span>
        </div>
        <div className="space-y-2">
          {factorRows.map(([label, value]) => (
            <div key={label}>
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="text-slate-300">{label}</span>
                <span className={scoreColor(value)}>{value.toFixed(2)}</span>
              </div>
              <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-teal-500 to-amber-400 rounded-full"
                  style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="border border-slate-700/50 rounded-lg p-4 mb-4">
        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">Lifecycle</h3>
        <div className="flex items-center justify-between text-sm">
          <span className="text-slate-200">{lifecycle.current_stage}</span>
          <span className="text-slate-500">rank {stageRank}</span>
        </div>
        <p className="text-xs text-slate-500 mt-1">
          Sovereignty {(sovereigntyScore * 100).toFixed(0)}%
          {lifecycle.next_stage ? ` -> ${lifecycle.next_stage}` : ""}
        </p>
        <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden mt-3">
          <div
            className="h-full bg-gradient-to-r from-emerald-500 to-teal-400 rounded-full"
            style={{ width: `${Math.max(0, Math.min(100, progress * 100))}%` }}
          />
        </div>
      </section>

      <section className="border border-slate-700/50 rounded-lg p-4 mb-4">
        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">Projection</h3>
        <div className="grid grid-cols-2 gap-3 text-xs">
          <div className="bg-slate-900/40 rounded-lg p-3">
            <div className="text-slate-500">Nodes</div>
            <div className="text-slate-200 font-mono">{formatNodes(nodes)}</div>
          </div>
          <div className="bg-slate-900/40 rounded-lg p-3">
            <div className="text-slate-500">Skills</div>
            <div className="text-slate-200 font-mono">{formatNodes(skillsAvailable)}</div>
          </div>
          <div className="bg-slate-900/40 rounded-lg p-3">
            <div className="text-slate-500">Compute</div>
            <div className="text-slate-200 font-mono">{computeTflops.toFixed(2)} TFLOPS</div>
          </div>
          <div className="bg-slate-900/40 rounded-lg p-3">
            <div className="text-slate-500">Latency Factor</div>
            <div className="text-slate-200 font-mono">{latencyFactor.toFixed(2)}</div>
          </div>
        </div>
      </section>

      <section className="border border-slate-700/50 rounded-lg p-4 mb-4">
        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
          Milestone Projections
        </h3>
        <div className="space-y-2 text-xs">
          {milestones.map((milestone) => {
            const milestoneNodes = Number(milestone.nodes ?? 0);
            const milestoneSkills = Number(milestone.skills ?? 0);
            const milestoneLatency = Number(milestone.latency_factor ?? 0);
            return (
              <div
                key={milestoneNodes}
                className="flex items-center justify-between border-b border-slate-800/40 pb-2 last:border-0 last:pb-0"
              >
                <span className="text-slate-300">{formatNodes(milestoneNodes)} nodes</span>
                <span className="text-slate-500">
                  skills {formatNodes(milestoneSkills)} | latency {milestoneLatency.toFixed(2)}
                </span>
              </div>
            );
          })}
        </div>
      </section>

      <section className="border border-slate-700/50 rounded-lg p-4">
        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
          Diffusion Eligibility
        </h3>
        <p className="text-xs text-slate-500">
          Diffusion eligibility is not yet exposed by the current read model.
        </p>
      </section>
    </div>
  );
}
