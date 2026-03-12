"use client";

import { useState } from "react";
import {
  useSeedPotential,
  useMemoryProfile,
  type MemoryProfileCompiledReflex,
  type MemoryProfilePattern,
} from "@/hooks/use-sovereign-api";
import {
  PAT_AGENT_MANIFEST,
  SAT_AGENT_MANIFEST,
  TERMINAL_LIFECYCLE_STAGES,
  TERMINAL_TIER_DEFS,
  type LifecycleStageDef,
  type TerminalAgentDef,
} from "./terminal-manifest";

// ─── Constants ──────────────────────────────────────────────────

const PAT_AGENTS = PAT_AGENT_MANIFEST;
const SAT_AGENTS = SAT_AGENT_MANIFEST;
const TIER_DEFS = TERMINAL_TIER_DEFS;
const LIFECYCLE_STAGES = TERMINAL_LIFECYCLE_STAGES;

// ─── Helpers ────────────────────────────────────────────────────

function getCurrentTier(actions: number, ihsan: number): number {
  let tier = 0;
  for (let i = TIER_DEFS.length - 1; i >= 0; i--) {
    if (actions >= TIER_DEFS[i].min_actions && ihsan >= TIER_DEFS[i].min_ihsan) {
      tier = i;
      break;
    }
  }
  return tier;
}

function getLifecycleStage(score: number): LifecycleStageDef {
  let stage = LIFECYCLE_STAGES[0];
  for (const s of LIFECYCLE_STAGES) {
    if (score >= s.threshold) stage = s;
  }
  return stage;
}

function statusDot(status: string): string {
  switch (status) {
    case "active": return "bg-emerald-500";
    case "idle": return "bg-amber-500";
    case "standby": return "bg-slate-600";
    default: return "bg-slate-700";
  }
}

function tempBar(t: number): string {
  if (t >= 0.6) return "bg-orange-500";
  if (t >= 0.3) return "bg-amber-500";
  return "bg-teal-500";
}

function formatRecordedAt(ts: string): string {
  if (!ts) return "Recorded in procedural memory";
  try {
    return new Date(ts).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return ts;
  }
}

function formatLatency(ms: number): string {
  return ms > 0 ? `${ms.toFixed(0)}ms` : "timing pending";
}

// ─── Sub-Components ─────────────────────────────────────────────

function AgentCard({ agent }: { agent: TerminalAgentDef }) {
  return (
    <div className="flex items-center gap-3 py-2 px-3 rounded-lg hover:bg-white/5 transition-colors">
      <span className="text-lg">{agent.emoji}</span>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-bold text-slate-200">{agent.name}</span>
          <span className="text-[10px] text-slate-500">{agent.id}</span>
          <span className="text-[10px] font-mono text-amber-300/80">{agent.call}</span>
          <span className={`w-1.5 h-1.5 rounded-full ${statusDot(agent.status)}`} />
        </div>
        <div className="flex items-center gap-2 text-[10px] text-slate-500">
          <span>{agent.role}</span>
          <span className="text-slate-700">·</span>
          <span className="text-slate-600">{agent.domain}</span>
        </div>
      </div>
      <div className="flex items-center gap-1.5 flex-shrink-0">
        <span className="text-[10px] text-slate-600">T={agent.temperature}</span>
        <div className="w-8 h-1 bg-slate-800 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full ${tempBar(agent.temperature)}`}
            style={{ width: `${agent.temperature * 100}%` }}
          />
        </div>
      </div>
    </div>
  );
}

function TierProgress({
  currentActions,
  currentIhsan,
}: {
  currentActions: number;
  currentIhsan: number;
}) {
  const currentTier = getCurrentTier(currentActions, currentIhsan);
  const nextTier = currentTier < TIER_DEFS.length - 1 ? TIER_DEFS[currentTier + 1] : null;
  const def = TIER_DEFS[currentTier];

  const actionsProgress = nextTier
    ? Math.min(1, (currentActions - def.min_actions) / (nextTier.min_actions - def.min_actions))
    : 1;

  return (
    <div className="border border-slate-700/50 rounded-lg p-3 mb-3">
      <div className="flex items-center justify-between mb-2">
        <div>
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Current Tier</h3>
          <span className={`text-lg font-bold ${def.color}`}>{def.name}</span>
        </div>
        {nextTier && (
          <div className="text-right text-[10px] text-slate-500">
            Next: <span className={nextTier.color}>{nextTier.name}</span>
            <br />{nextTier.min_actions - currentActions} actions + Ihsān ≥ {nextTier.min_ihsan}
          </div>
        )}
      </div>

      {/* Progress bar */}
      {nextTier && (
        <div className="mb-3">
          <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-teal-500 to-amber-400 rounded-full transition-all duration-500"
              style={{ width: `${actionsProgress * 100}%` }}
            />
          </div>
          <div className="flex justify-between text-[10px] text-slate-600 mt-0.5">
            <span>{currentActions} actions</span>
            <span>{nextTier.min_actions} needed</span>
          </div>
        </div>
      )}

      {/* Unlocked/Locked skills */}
      <div className="grid grid-cols-2 gap-2">
        {TIER_DEFS.map((t, i) => (
          <div key={t.name} className={`text-[10px] p-2 rounded ${i <= currentTier ? "bg-slate-800/50" : "bg-slate-900/30 opacity-50"}`}>
            <div className={`font-bold ${i <= currentTier ? t.color : "text-slate-600"}`}>
              {i <= currentTier ? "✓" : "🔒"} {t.name}
            </div>
            {t.unlocks.map((u) => (
              <div key={u} className={i <= currentTier ? "text-slate-400" : "text-slate-700"}>
                {u}
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

function LifecycleDisplay({ sovereigntyScore }: { sovereigntyScore: number }) {
  const stage = getLifecycleStage(sovereigntyScore);
  const nextStage = LIFECYCLE_STAGES.find((s) => s.threshold > sovereigntyScore);
  const progress = nextStage
    ? (sovereigntyScore - stage.threshold) / (nextStage.threshold - stage.threshold)
    : 1;

  return (
    <div className="border border-slate-700/50 rounded-lg p-3 mb-3">
      <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
        Human Lifecycle Stage
      </h3>
      <div className="flex items-center gap-3">
        <span className="text-3xl">{stage.emoji}</span>
        <div className="flex-1">
          <div className="text-base font-bold text-slate-200">{stage.name}</div>
          <div className="text-[10px] text-slate-500">
            Sovereignty: {(sovereigntyScore * 100).toFixed(0)}%
            {nextStage && ` → ${nextStage.name} at ${(nextStage.threshold * 100).toFixed(0)}%`}
          </div>
          {nextStage && (
            <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden mt-1.5">
              <div
                className="h-full bg-gradient-to-r from-emerald-500 to-teal-400 rounded-full transition-all duration-500"
                style={{ width: `${Math.max(0, Math.min(100, progress * 100))}%` }}
              />
            </div>
          )}
        </div>
      </div>
      {/* Stage progression */}
      <div className="flex items-center gap-0.5 mt-3">
        {LIFECYCLE_STAGES.map((s, i) => (
          <div key={s.name} className="flex items-center flex-1">
            <div className={`w-full h-1 rounded ${sovereigntyScore >= s.threshold ? "bg-emerald-600" : "bg-slate-800"}`} />
            {i < LIFECYCLE_STAGES.length - 1 && <div className="w-0.5" />}
          </div>
        ))}
      </div>
      <div className="flex justify-between text-[9px] text-slate-600 mt-0.5 px-0.5">
        {LIFECYCLE_STAGES.map((s) => (
          <span key={s.name}>{s.emoji}</span>
        ))}
      </div>
    </div>
  );
}

function CompiledReflexes({
  reflexes,
}: {
  reflexes: MemoryProfileCompiledReflex[];
}) {
  return (
    <div className="border border-emerald-800/30 rounded-lg p-3 mb-3 bg-emerald-950/10">
      <h3 className="text-xs font-bold text-emerald-400 uppercase tracking-wider mb-2">
        ⚡ Compiled Reflexes (System-1)
      </h3>
      {reflexes.length === 0 ? (
        <div className="text-xs text-emerald-200/80">
          No compiled reflexes yet. Three excellent repetitions will precipitate a System-1 path.
        </div>
      ) : (
        reflexes.map((r) => (
          <div key={r.name} className="flex items-center justify-between py-1.5 border-b border-slate-800/30 last:border-0">
            <div>
              <span className="text-sm text-slate-200 font-medium">"{r.name}"</span>
              <div className="text-[10px] text-slate-500 mt-0.5">
                {r.execution_count} executions · Ihsān {r.avg_ihsan.toFixed(2)} · {formatRecordedAt(r.compiled_at)}
              </div>
            </div>
            <div className="text-right flex-shrink-0">
              <div className="text-sm font-bold text-emerald-400">{formatLatency(r.avg_latency_ms)}</div>
              <div className="text-[10px] text-emerald-600">System-1</div>
            </div>
          </div>
        ))
      )}
    </div>
  );
}

function NearCompileList({
  patterns,
}: {
  patterns: MemoryProfilePattern[];
}) {
  return (
    <div className="border border-amber-800/30 rounded-lg p-3 mb-3 bg-amber-950/10">
      <h3 className="text-xs font-bold text-amber-400 uppercase tracking-wider mb-2">
        🔥 Near-Compile Candidates
      </h3>
      {patterns.length === 0 ? (
        <div className="text-xs text-amber-200/80">
          No near-compile patterns yet. High-Ihsan repetition will appear here as the learning loop matures.
        </div>
      ) : (
        patterns.map((p) => (
          <div key={p.name} className="flex items-center justify-between py-1.5">
            <div>
              <span className="text-sm text-slate-200">"{p.name}"</span>
              <span className="text-[10px] text-slate-500 ml-2">Ihsān {p.avg_ihsan.toFixed(2)}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-14 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-amber-400 rounded-full"
                  style={{ width: `${(p.count / Math.max(1, p.threshold)) * 100}%` }}
                />
              </div>
              <span className="text-xs text-amber-300 font-mono font-bold">{p.count}/{p.threshold}</span>
            </div>
          </div>
        ))
      )}
    </div>
  );
}

// ─── Main Component ─────────────────────────────────────────────

export default function TerminalSkills() {
  const { data: potential } = useSeedPotential();
  const { data: memoryProfile } = useMemoryProfile();
  const [showSAT, setShowSAT] = useState(false);

  const currentActions = Math.max(
    potential?.episodes_total ?? 0,
    memoryProfile.missions.length,
  );
  const currentIhsan =
    memoryProfile.missions[0]?.ihsan_score ?? potential?.reward_ema ?? 0.42;
  const sovereigntyScore = potential?.sovereignty_score ?? 0.42;
  const compiledReflexes = memoryProfile.compiled_reflex_summary;
  const nearCompilePatterns = memoryProfile.near_compile_patterns;
  const reflexCount = compiledReflexes.length;

  return (
    <div className="p-4 max-w-3xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-bold text-slate-100">Agents & Skills</h2>
          <p className="text-xs text-slate-500">
            Your sovereign cognitive topology, promoted from the design canon into the live terminal.
          </p>
        </div>
        <div className="text-xs text-slate-600">
          {PAT_AGENTS.length + SAT_AGENTS.length} agents · {reflexCount} reflexes · {memoryProfile.work_streak} streak
        </div>
      </div>

      {/* §10.5: Shows current tier with progress bar */}
      <TierProgress currentActions={currentActions} currentIhsan={currentIhsan} />

      {/* §10.5: Shows human lifecycle stage with progress */}
      <LifecycleDisplay sovereigntyScore={sovereigntyScore} />

      {/* §10.5: Shows PAT-7 agent list with status */}
      <div className="border border-slate-700/50 rounded-lg overflow-hidden mb-3">
        <div className="bg-slate-800/60 px-3 py-1.5 flex items-center justify-between">
          <span className="text-xs font-bold text-teal-400">
            💜 PAT-7 — Personal Agent Team
          </span>
          <span className="text-[10px] text-slate-500">Human → DEMA / NEXUS → PAT → Pool → SAT</span>
        </div>
        <div className="divide-y divide-slate-800/50">
          {PAT_AGENTS.map((a) => (
            <AgentCard key={a.id} agent={a} />
          ))}
        </div>
      </div>

      {/* §10.5: Shows SAT-5 agent list */}
      <div className="border border-slate-700/50 rounded-lg overflow-hidden mb-3">
        <button
          onClick={() => setShowSAT(!showSAT)}
          className="w-full bg-slate-800/60 px-3 py-1.5 flex items-center justify-between hover:bg-slate-800/80 transition-colors"
        >
          <span className="text-xs font-bold text-slate-400">
            🛡️ SAT-5 — System Agent Team
          </span>
          <span className="text-xs text-slate-600">{showSAT ? "▼" : "▶"}</span>
        </button>
        {showSAT && (
          <div className="divide-y divide-slate-800/50">
            {SAT_AGENTS.map((a) => (
              <AgentCard key={a.id} agent={a} />
            ))}
          </div>
        )}
      </div>

      {/* §10.5: Shows compiled reflexes with avg Ihsān, count, latency */}
      <CompiledReflexes reflexes={compiledReflexes} />

      {/* §10.5: Shows near-compile candidates with N/3 threshold */}
      <NearCompileList patterns={nearCompilePatterns} />

      {/* Boundary model */}
      <div className="text-center mt-4 text-[10px] text-slate-700">
        Human → DEMA → PAT-7 → Pool → SAT-5 (Boundary Model) · Reflex visibility derives from procedural memory and constitutional state.
      </div>
    </div>
  );
}
