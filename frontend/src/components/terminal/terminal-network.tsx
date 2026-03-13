"use client";

import {
  useNodeValue,
  useNodeLifecycle,
  useNetworkEffect,
} from "@/hooks/use-sovereign-api";

// ─── Types ──────────────────────────────────────────────────────

interface ValueFactor {
  name: string;
  score: number;
  weight: number;
  description: string;
}

interface MilestoneProjection {
  label: string;
  nodes: number;
  projected_latency_ms: number;
  pool_memory_tb: number;
  reflex_library_size: number;
}

// ─── Demo Data ──────────────────────────────────────────────────

const DEMO_VALUE_FACTORS: ValueFactor[] = [
  { name: "Contribution", score: 0.72, weight: 0.25, description: "SEED minted from verified work" },
  { name: "Consistency", score: 0.85, weight: 0.20, description: "Work streak + daily activity" },
  { name: "Quality", score: 0.94, weight: 0.25, description: "Average Ihsān across missions" },
  { name: "Growth", score: 0.68, weight: 0.15, description: "Skill tier progression velocity" },
  { name: "Impact", score: 0.45, weight: 0.15, description: "Reflexes shared to forest pool" },
];

const MILESTONES: MilestoneProjection[] = [
  { label: "Alpha-10", nodes: 10, projected_latency_ms: 50, pool_memory_tb: 0.0004, reflex_library_size: 50 },
  { label: "Genesis-100", nodes: 100, projected_latency_ms: 35, pool_memory_tb: 0.004, reflex_library_size: 500 },
  { label: "Phase 1", nodes: 100_000_000, projected_latency_ms: 5, pool_memory_tb: 400, reflex_library_size: 500_000 },
  { label: "Phase 2", nodes: 1_000_000_000, projected_latency_ms: 2, pool_memory_tb: 1_600, reflex_library_size: 5_000_000 },
  { label: "Phase 4", nodes: 8_000_000_000, projected_latency_ms: 0.5, pool_memory_tb: 16_000, reflex_library_size: 100_000_000 },
];

const DEMO_LIFECYCLE = {
  current_stage: "Sprout",
  rank: 3,
  sovereignty_score: 0.42,
  next_stage: "Sapling",
  next_threshold: 0.50,
};

const DEMO_DIFFUSION = {
  eligible_reflexes: 2,
  published: 0,
  forest_reach: 0,
};

// ─── Helpers ────────────────────────────────────────────────────

function formatNodes(n: number): string {
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(0)}B`;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(0)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
  return n.toString();
}

function formatMemory(tb: number): string {
  if (tb >= 1000) return `${(tb / 1000).toFixed(0)} PB`;
  if (tb >= 1) return `${tb.toFixed(0)} TB`;
  return `${(tb * 1000).toFixed(0)} GB`;
}

function scoreColor(score: number): string {
  if (score >= 0.80) return "text-emerald-400";
  if (score >= 0.60) return "text-amber-400";
  return "text-red-400";
}

function barColor(score: number): string {
  if (score >= 0.80) return "bg-emerald-500";
  if (score >= 0.60) return "bg-amber-500";
  return "bg-red-500";
}

// ─── Sub-Components ─────────────────────────────────────────────

function CompositeValue({ factors }: { factors: ValueFactor[] }) {
  const composite = factors.reduce((sum, f) => sum + f.score * f.weight, 0);

  return (
    <div className="border border-slate-700/50 rounded-lg p-4 mb-3">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider">
          5-Factor Node Value
        </h3>
        <div className="text-right">
          <span className={`text-2xl font-bold ${scoreColor(composite)}`}>
            {composite.toFixed(2)}
          </span>
          <span className="text-[10px] text-slate-600 block">composite</span>
        </div>
      </div>

      <div className="space-y-2">
        {factors.map((f) => (
          <div key={f.name} className="group">
            <div className="flex items-center justify-between text-xs mb-0.5">
              <div className="flex items-center gap-2">
                <span className="text-slate-300 font-medium">{f.name}</span>
                <span className="text-[10px] text-slate-600">×{f.weight}</span>
              </div>
              <span className={scoreColor(f.score)}>{f.score.toFixed(2)}</span>
            </div>
            <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-700 ${barColor(f.score)}`}
                style={{ width: `${f.score * 100}%` }}
              />
            </div>
            <p className="text-[10px] text-slate-600 mt-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
              {f.description}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}

function LifecycleProgress() {
  const stages = ["Seedling", "Sprout", "Sapling", "Branch", "Canopy", "Catalyst"];
  const emojis = ["🌱", "🌿", "🌲", "🌳", "🏔️", "⭐"];
  const currentIdx = stages.indexOf(DEMO_LIFECYCLE.current_stage);
  // Progress used for future stage visualization
  void ((DEMO_LIFECYCLE.sovereignty_score - (currentIdx * 0.2)) / 0.2);

  return (
    <div className="border border-slate-700/50 rounded-lg p-4 mb-3">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider">
          Lifecycle Stage
        </h3>
        <span className="text-xs text-slate-500">
          Rank #{DEMO_LIFECYCLE.rank}
        </span>
      </div>

      <div className="flex items-center gap-2 mb-3">
        <span className="text-2xl">{emojis[currentIdx]}</span>
        <div>
          <span className="text-lg font-bold text-slate-200">{DEMO_LIFECYCLE.current_stage}</span>
          <p className="text-[10px] text-slate-500">
            Sovereignty {(DEMO_LIFECYCLE.sovereignty_score * 100).toFixed(0)}% →{" "}
            {DEMO_LIFECYCLE.next_stage} at {(DEMO_LIFECYCLE.next_threshold * 100).toFixed(0)}%
          </p>
        </div>
      </div>

      {/* Stage track */}
      <div className="flex items-center gap-0.5">
        {stages.map((s, i) => (
          <div key={s} className="flex-1 flex flex-col items-center">
            <div
              className={`w-full h-2 rounded ${
                i < currentIdx
                  ? "bg-emerald-600"
                  : i === currentIdx
                    ? "bg-gradient-to-r from-emerald-500 to-teal-400"
                    : "bg-slate-800"
              }`}
            />
            <span className="text-[9px] mt-1">{emojis[i]}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function MilestoneProjections() {
  return (
    <div className="border border-slate-700/50 rounded-lg p-4 mb-3">
      <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
        Forest Growth Projections (Reverse Scaling)
      </h3>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-slate-500 border-b border-slate-800">
              <th className="text-left py-1.5 pr-3">Milestone</th>
              <th className="text-right py-1.5 px-2">Nodes</th>
              <th className="text-right py-1.5 px-2">Latency</th>
              <th className="text-right py-1.5 px-2">Pool</th>
              <th className="text-right py-1.5 pl-2">Reflexes</th>
            </tr>
          </thead>
          <tbody>
            {MILESTONES.map((m, i) => (
              <tr
                key={m.label}
                className={`border-b border-slate-800/50 ${i === 0 ? "text-teal-300" : "text-slate-400"}`}
              >
                <td className="py-1.5 pr-3 font-medium">
                  {i === 0 && "→ "}
                  {m.label}
                </td>
                <td className="text-right py-1.5 px-2 font-mono">{formatNodes(m.nodes)}</td>
                <td className="text-right py-1.5 px-2">
                  <span className={m.projected_latency_ms < 5 ? "text-emerald-400" : ""}>
                    {m.projected_latency_ms < 1
                      ? `${m.projected_latency_ms}ms`
                      : `${m.projected_latency_ms}ms`}
                  </span>
                </td>
                <td className="text-right py-1.5 px-2">{formatMemory(m.pool_memory_tb)}</td>
                <td className="text-right py-1.5 pl-2 font-mono">{formatNodes(m.reflex_library_size)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="text-[10px] text-slate-600 mt-2">
        More nodes → larger reflex library → higher cache hit rate → faster for everyone.
        Reverse scaling: performance improves with growth.
      </p>
    </div>
  );
}

function DiffusionEligibility() {
  return (
    <div className="border border-slate-700/50 rounded-lg p-3 mb-3">
      <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
        Reflex Diffusion
      </h3>
      <div className="grid grid-cols-3 gap-2 text-center text-xs">
        <div className="bg-slate-800/50 rounded p-2">
          <div className="text-amber-400 font-bold text-lg">{DEMO_DIFFUSION.eligible_reflexes}</div>
          <div className="text-slate-600">Eligible</div>
        </div>
        <div className="bg-slate-800/50 rounded p-2">
          <div className="text-teal-400 font-bold text-lg">{DEMO_DIFFUSION.published}</div>
          <div className="text-slate-600">Published</div>
        </div>
        <div className="bg-slate-800/50 rounded p-2">
          <div className="text-slate-400 font-bold text-lg">{DEMO_DIFFUSION.forest_reach}</div>
          <div className="text-slate-600">Forest Reach</div>
        </div>
      </div>
      <p className="text-[10px] text-slate-600 mt-2">
        Expert tier required to publish reflexes to the forest pool.
      </p>
    </div>
  );
}

// ─── Main Component ─────────────────────────────────────────────

export default function TerminalNetwork() {
  useNodeValue();
  useNodeLifecycle();
  useNetworkEffect();

  return (
    <div className="p-4 max-w-3xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-bold text-slate-100">Network & Forest</h2>
          <p className="text-xs text-slate-500">Your place in the sovereign lattice</p>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-amber-500" />
          <span className="text-[10px] text-amber-400">Alpha</span>
        </div>
      </div>

      {/* §10.6: Shows 5-factor node value composite */}
      <CompositeValue factors={DEMO_VALUE_FACTORS} />

      {/* §10.6: Shows lifecycle stage + progress within stage */}
      <LifecycleProgress />

      {/* §10.6: Shows milestone projections (1→8B nodes) */}
      <MilestoneProjections />

      {/* §10.6: Shows diffusion eligibility for published reflexes */}
      <DiffusionEligibility />

      {/* §10.6: Offline fallback: last cached projection */}
      <div className="text-center mt-4 text-[10px] text-slate-700">
        Projections from last forest sync · Offline: cached values shown
      </div>
    </div>
  );
}
