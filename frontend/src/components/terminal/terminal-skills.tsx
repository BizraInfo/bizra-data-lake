"use client";

import { useState } from "react";
import {
  useSeedPotential,
  useSeedEpisodes,
} from "@/hooks/use-sovereign-api";

// ─── Types ──────────────────────────────────────────────────────

interface AgentInfo {
  id: string;
  name: string;
  role: string;
  emoji: string;
  temperature: number;
  status: "active" | "idle" | "standby";
  team: "PAT" | "SAT";
  raidRole: string;
}

interface CompiledReflex {
  name: string;
  compiled_at: string;
  avg_ihsan: number;
  execution_count: number;
  avg_latency_ms: number;
}

interface NearCompile {
  name: string;
  count: number;
  threshold: number;
  avg_ihsan: number;
}

type TierName = "Novice" | "Adept" | "Expert" | "Master";

interface TierDef {
  name: TierName;
  min_actions: number;
  min_ihsan: number;
  unlocks: string[];
  color: string;
}

type LifecycleStage = "Seedling" | "Sprout" | "Sapling" | "Branch" | "Canopy" | "Catalyst";

// ─── Constants ──────────────────────────────────────────────────

const PAT_AGENTS: AgentInfo[] = [
  { id: "P1", name: "Atlas", role: "Planner", emoji: "🗺️", temperature: 0.2, status: "active", team: "PAT", raidRole: "Strategist" },
  { id: "P2", name: "Oracle", role: "Researcher", emoji: "🔭", temperature: 0.7, status: "active", team: "PAT", raidRole: "Scout" },
  { id: "P3", name: "Forge", role: "Coder", emoji: "⚒️", temperature: 0.3, status: "active", team: "PAT", raidRole: "DPS" },
  { id: "P4", name: "Judge", role: "Evaluator", emoji: "⚖️", temperature: 0.1, status: "active", team: "PAT", raidRole: "Healer" },
  { id: "P5", name: "Crown", role: "Ethicist", emoji: "👑", temperature: 0.1, status: "active", team: "PAT", raidRole: "Tank" },
  { id: "P6", name: "Herald", role: "Publisher", emoji: "📢", temperature: 0.4, status: "idle", team: "PAT", raidRole: "Bard" },
  { id: "P7", name: "DEMA", role: "Nexus", emoji: "💜", temperature: 0.2, status: "active", team: "PAT", raidRole: "Raid Leader" },
];

const SAT_AGENTS: AgentInfo[] = [
  { id: "S1", name: "Sentinel", role: "Security", emoji: "🛡️", temperature: 0.05, status: "active", team: "SAT", raidRole: "Anti-Cheat" },
  { id: "S2", name: "Oracle-S", role: "Forest Health", emoji: "🌳", temperature: 0.3, status: "active", team: "SAT", raidRole: "World State" },
  { id: "S3", name: "Ledger", role: "Economy", emoji: "📊", temperature: 0.05, status: "active", team: "SAT", raidRole: "Economy Mgr" },
  { id: "S4", name: "Conductor", role: "Capacity", emoji: "🎵", temperature: 0.2, status: "active", team: "SAT", raidRole: "Load Balancer" },
  { id: "S5", name: "Ambassador", role: "Federation", emoji: "🤝", temperature: 0.4, status: "standby", team: "SAT", raidRole: "Diplomacy" },
];

const TIER_DEFS: TierDef[] = [
  { name: "Novice", min_actions: 0, min_ihsan: 0, unlocks: ["Read files", "Clipboard", "Basic queries"], color: "text-slate-400" },
  { name: "Adept", min_actions: 10, min_ihsan: 0.85, unlocks: ["Write files", "Local scripts", "Browser automation"], color: "text-teal-400" },
  { name: "Expert", min_actions: 100, min_ihsan: 0.90, unlocks: ["Network access", "API calls", "Multi-app orchestration"], color: "text-amber-400" },
  { name: "Master", min_actions: 1000, min_ihsan: 0.95, unlocks: ["Unsandboxed processes", "Marketplace publish", "Mentor others"], color: "text-purple-400" },
];

const LIFECYCLE_STAGES: { name: LifecycleStage; threshold: number; emoji: string }[] = [
  { name: "Seedling", threshold: 0, emoji: "🌱" },
  { name: "Sprout", threshold: 0.15, emoji: "🌿" },
  { name: "Sapling", threshold: 0.30, emoji: "🌲" },
  { name: "Branch", threshold: 0.50, emoji: "🌳" },
  { name: "Canopy", threshold: 0.75, emoji: "🏔️" },
  { name: "Catalyst", threshold: 0.95, emoji: "⭐" },
];

const DEMO_REFLEXES: CompiledReflex[] = [
  { name: "file_organization", compiled_at: new Date(Date.now() - 86400000).toISOString(), avg_ihsan: 0.97, execution_count: 8, avg_latency_ms: 48 },
  { name: "git_commit_flow", compiled_at: new Date(Date.now() - 172800000).toISOString(), avg_ihsan: 0.96, execution_count: 14, avg_latency_ms: 52 },
];

const DEMO_NEAR: NearCompile[] = [
  { name: "test_generation", count: 2, threshold: 3, avg_ihsan: 0.97 },
  { name: "code_review", count: 1, threshold: 3, avg_ihsan: 0.94 },
  { name: "deploy_staging", count: 2, threshold: 3, avg_ihsan: 0.96 },
];

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

function getLifecycleStage(score: number): typeof LIFECYCLE_STAGES[number] {
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

// ─── Sub-Components ─────────────────────────────────────────────

function AgentCard({ agent }: { agent: AgentInfo }) {
  return (
    <div className="flex items-center gap-3 py-2 px-3 rounded-lg hover:bg-white/5 transition-colors">
      <span className="text-lg">{agent.emoji}</span>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-bold text-slate-200">{agent.name}</span>
          <span className="text-[10px] text-slate-500">{agent.id}</span>
          <span className={`w-1.5 h-1.5 rounded-full ${statusDot(agent.status)}`} />
        </div>
        <div className="flex items-center gap-2 text-[10px] text-slate-500">
          <span>{agent.role}</span>
          <span className="text-slate-700">·</span>
          <span className="text-slate-600">{agent.raidRole}</span>
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

function TierProgress({ currentActions, currentIhsan }: { currentActions: number; currentIhsan: number }) {
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

function CompiledReflexes() {
  if (DEMO_REFLEXES.length === 0) return null;

  return (
    <div className="border border-emerald-800/30 rounded-lg p-3 mb-3 bg-emerald-950/10">
      <h3 className="text-xs font-bold text-emerald-400 uppercase tracking-wider mb-2">
        ⚡ Compiled Reflexes (System-1)
      </h3>
      {DEMO_REFLEXES.map((r) => (
        <div key={r.name} className="flex items-center justify-between py-1.5 border-b border-slate-800/30 last:border-0">
          <div>
            <span className="text-sm text-slate-200 font-medium">"{r.name}"</span>
            <div className="text-[10px] text-slate-500 mt-0.5">
              {r.execution_count} executions · Ihsān {r.avg_ihsan.toFixed(2)}
            </div>
          </div>
          <div className="text-right flex-shrink-0">
            <div className="text-sm font-bold text-emerald-400">{r.avg_latency_ms}ms</div>
            <div className="text-[10px] text-emerald-600">System-1</div>
          </div>
        </div>
      ))}
    </div>
  );
}

function NearCompileList() {
  if (DEMO_NEAR.length === 0) return null;

  return (
    <div className="border border-amber-800/30 rounded-lg p-3 mb-3 bg-amber-950/10">
      <h3 className="text-xs font-bold text-amber-400 uppercase tracking-wider mb-2">
        🔥 Near-Compile Candidates
      </h3>
      {DEMO_NEAR.map((p) => (
        <div key={p.name} className="flex items-center justify-between py-1.5">
          <div>
            <span className="text-sm text-slate-200">"{p.name}"</span>
            <span className="text-[10px] text-slate-500 ml-2">Ihsān {p.avg_ihsan.toFixed(2)}</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-14 h-1.5 bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-amber-400 rounded-full"
                style={{ width: `${(p.count / p.threshold) * 100}%` }}
              />
            </div>
            <span className="text-xs text-amber-300 font-mono font-bold">{p.count}/{p.threshold}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

// ─── Main Component ─────────────────────────────────────────────

export default function TerminalSkills() {
  const { data: potential } = useSeedPotential();
  const [showSAT, setShowSAT] = useState(false);

  const currentActions = potential?.episodes_total ?? 25;
  const currentIhsan = potential?.sovereignty_score ?? 0.42;
  const sovereigntyScore = potential?.sovereignty_score ?? 0.42;

  return (
    <div className="p-4 max-w-3xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-bold text-slate-100">Agents & Skills</h2>
          <p className="text-xs text-slate-500">Your sovereign cognitive team</p>
        </div>
        <div className="text-xs text-slate-600">
          {PAT_AGENTS.length + SAT_AGENTS.length} agents · {DEMO_REFLEXES.length} reflexes
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
          <span className="text-[10px] text-slate-500">Human → DEMA → PAT → Pool → SAT</span>
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
      <CompiledReflexes />

      {/* §10.5: Shows near-compile candidates with N/3 threshold */}
      <NearCompileList />

      {/* Boundary model */}
      <div className="text-center mt-4 text-[10px] text-slate-700">
        Human → DEMA → PAT-7 → Pool → SAT-5 (Boundary Model)
      </div>
    </div>
  );
}
