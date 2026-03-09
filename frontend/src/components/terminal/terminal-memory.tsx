"use client";

import { useState } from "react";
import {
  useMemoryStats,
  useTerminalBriefing,
  useSeedEpisodes,
  useSeedPotential,
} from "@/hooks/use-sovereign-api";

// ─── Types ──────────────────────────────────────────────────────

interface MissionSummary {
  mission_id: string;
  description: string;
  status: "COMPLETE" | "PARTIAL" | "FAILED" | "BLOCKED";
  ihsan_score: number;
  seed_earned: number;
  timestamp: string;
}

interface NearCompilePattern {
  name: string;
  count: number;
  threshold: number;
  avg_ihsan: number;
}

interface ActiveProject {
  name: string;
  last_activity: string;
  mission_count: number;
}

// ─── Demo Data ──────────────────────────────────────────────────

const DEMO_MISSIONS: MissionSummary[] = [
  { mission_id: "m_010", description: "Organize Downloads folder", status: "COMPLETE", ihsan_score: 0.97, seed_earned: 2.41, timestamp: new Date(Date.now() - 3600000).toISOString() },
  { mission_id: "m_009", description: "Generate invoice report", status: "COMPLETE", ihsan_score: 0.95, seed_earned: 1.89, timestamp: new Date(Date.now() - 7200000).toISOString() },
  { mission_id: "m_008", description: "Refactor auth middleware", status: "COMPLETE", ihsan_score: 0.96, seed_earned: 3.12, timestamp: new Date(Date.now() - 14400000).toISOString() },
  { mission_id: "m_007", description: "Deploy staging environment", status: "PARTIAL", ihsan_score: 0.88, seed_earned: 0.95, timestamp: new Date(Date.now() - 28800000).toISOString() },
  { mission_id: "m_006", description: "Write unit tests for bloom.py", status: "COMPLETE", ihsan_score: 0.98, seed_earned: 4.20, timestamp: new Date(Date.now() - 43200000).toISOString() },
  { mission_id: "m_005", description: "Fix atomic write in vault.py", status: "COMPLETE", ihsan_score: 0.97, seed_earned: 2.67, timestamp: new Date(Date.now() - 86400000).toISOString() },
  { mission_id: "m_004", description: "Design terminal Timeline view", status: "COMPLETE", ihsan_score: 0.94, seed_earned: 1.75, timestamp: new Date(Date.now() - 172800000).toISOString() },
  { mission_id: "m_003", description: "Reconcile bank statements", status: "FAILED", ihsan_score: 0.72, seed_earned: 0, timestamp: new Date(Date.now() - 259200000).toISOString() },
  { mission_id: "m_002", description: "Set up CI pipeline gate", status: "COMPLETE", ihsan_score: 0.96, seed_earned: 2.90, timestamp: new Date(Date.now() - 345600000).toISOString() },
  { mission_id: "m_001", description: "Initialize BIZRA node", status: "COMPLETE", ihsan_score: 0.99, seed_earned: 5.00, timestamp: new Date(Date.now() - 432000000).toISOString() },
];

const DEMO_PATTERNS: NearCompilePattern[] = [
  { name: "file_organization", count: 2, threshold: 3, avg_ihsan: 0.97 },
  { name: "code_refactoring", count: 1, threshold: 3, avg_ihsan: 0.96 },
  { name: "test_generation", count: 2, threshold: 3, avg_ihsan: 0.97 },
];

const DEMO_PROJECTS: ActiveProject[] = [
  { name: "BIZRA Terminal v1", last_activity: "2 hours ago", mission_count: 12 },
  { name: "Invoice Automation", last_activity: "1 day ago", mission_count: 5 },
  { name: "CI/CD Pipeline", last_activity: "3 days ago", mission_count: 8 },
];

const DEMO_PROFILE = {
  preferred_domains: ["software engineering", "DevOps", "documentation"],
  active_hours: "08:00-18:00 GST",
  vocabulary_signature: "technical, constitutional, Arabic-English",
  work_window: "weekdays, morning-focused",
};

// ─── Helpers ────────────────────────────────────────────────────

function statusColor(status: string): string {
  switch (status) {
    case "COMPLETE": return "text-emerald-400";
    case "PARTIAL": return "text-amber-400";
    case "FAILED": return "text-red-400";
    case "BLOCKED": return "text-red-500";
    default: return "text-slate-400";
  }
}

function ihsanColor(score: number): string {
  if (score >= 0.95) return "text-emerald-400";
  if (score >= 0.85) return "text-amber-400";
  return "text-red-400";
}

function timeAgo(ts: string): string {
  const diff = Date.now() - new Date(ts).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

function calculateStreak(missions: MissionSummary[]): number {
  let streak = 0;
  const today = new Date();
  today.setHours(0, 0, 0, 0);

  for (let d = 0; d < 30; d++) {
    const day = new Date(today.getTime() - d * 86400000);
    const nextDay = new Date(day.getTime() + 86400000);
    const hasWork = missions.some((m) => {
      const mt = new Date(m.timestamp).getTime();
      return mt >= day.getTime() && mt < nextDay.getTime() && m.status === "COMPLETE";
    });
    if (hasWork) streak++;
    else if (d > 0) break;
  }
  return streak;
}

// ─── Sub-Components ─────────────────────────────────────────────

function BriefingCard() {
  const { data: briefing } = useTerminalBriefing();
  const { data: potential } = useSeedPotential();
  const streak = calculateStreak(DEMO_MISSIONS);
  const qualityTrend = DEMO_MISSIONS.slice(0, 5)
    .reduce((sum, m) => sum + m.ihsan_score, 0) / 5;

  return (
    <div className="bg-gradient-to-br from-teal-950/50 to-slate-900/50 border border-teal-800/30 rounded-xl p-4 mb-4">
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className="text-sm font-bold text-teal-300">🌅 Morning Briefing</h3>
          <p className="text-xs text-slate-500 mt-0.5">from DEMA</p>
        </div>
        <span className="text-xs text-slate-600">
          {new Date().toLocaleDateString(undefined, { weekday: "long", month: "short", day: "numeric" })}
        </span>
      </div>

      <div className="space-y-2 text-sm text-slate-300">
        <p>
          Your last mission was{" "}
          <span className="text-teal-300 font-medium">
            "{DEMO_MISSIONS[0].description}"
          </span>{" "}
          — Ihsān{" "}
          <span className={ihsanColor(DEMO_MISSIONS[0].ihsan_score)}>
            {DEMO_MISSIONS[0].ihsan_score.toFixed(2)}
          </span>
          , +{DEMO_MISSIONS[0].seed_earned.toFixed(1)} SEED.
        </p>

        <p>
          🔥 Work streak:{" "}
          <span className="text-amber-400 font-bold">{streak} days</span>.
          Quality trend:{" "}
          <span className={ihsanColor(qualityTrend)}>
            {qualityTrend.toFixed(2)} avg
          </span>{" "}
          (last 5 missions).
        </p>

        {DEMO_PATTERNS.filter((p) => p.count >= 2).length > 0 && (
          <p>
            ⚡ Near-compile:{" "}
            {DEMO_PATTERNS.filter((p) => p.count >= 2).map((p) => (
              <span key={p.name} className="text-amber-300">
                "{p.name}" ({p.count}/{p.threshold})
              </span>
            )).reduce((prev, curr, i) => (
              <>{prev}{i > 0 ? ", " : ""}{curr}</>
            ), <></>)}
            . One more excellent execution compiles a reflex.
          </p>
        )}

        <p className="text-teal-400 font-medium">
          💡 Suggested: complete "{DEMO_PATTERNS[0]?.name}" to trigger your first System-1 reflex.
        </p>
      </div>
    </div>
  );
}

function SemanticProfile() {
  return (
    <div className="border border-slate-700/50 rounded-lg p-3 mb-3">
      <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
        Semantic Profile
      </h3>
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div>
          <span className="text-slate-500">Domains:</span>
          <div className="flex flex-wrap gap-1 mt-1">
            {DEMO_PROFILE.preferred_domains.map((d) => (
              <span key={d} className="bg-teal-900/30 text-teal-300 px-1.5 py-0.5 rounded text-[10px]">
                {d}
              </span>
            ))}
          </div>
        </div>
        <div>
          <span className="text-slate-500">Active hours:</span>
          <p className="text-slate-300 mt-0.5">{DEMO_PROFILE.active_hours}</p>
        </div>
        <div>
          <span className="text-slate-500">Language:</span>
          <p className="text-slate-300 mt-0.5">{DEMO_PROFILE.vocabulary_signature}</p>
        </div>
        <div>
          <span className="text-slate-500">Work window:</span>
          <p className="text-slate-300 mt-0.5">{DEMO_PROFILE.work_window}</p>
        </div>
      </div>
    </div>
  );
}

function MissionHistory() {
  return (
    <div className="border border-slate-700/50 rounded-lg p-3 mb-3">
      <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
        Last 10 Missions
      </h3>
      <div className="space-y-1">
        {DEMO_MISSIONS.map((m) => (
          <div key={m.mission_id} className="flex items-center justify-between py-1 border-b border-slate-800/50 last:border-0">
            <div className="flex items-center gap-2 min-w-0 flex-1">
              <span className={`text-xs font-bold ${statusColor(m.status)}`}>
                {m.status === "COMPLETE" ? "✓" : m.status === "FAILED" ? "✗" : "◐"}
              </span>
              <span className="text-xs text-slate-300 truncate">{m.description}</span>
            </div>
            <div className="flex items-center gap-3 flex-shrink-0 text-[10px]">
              <span className={ihsanColor(m.ihsan_score)}>{m.ihsan_score.toFixed(2)}</span>
              {m.seed_earned > 0 && <span className="text-amber-400">+{m.seed_earned.toFixed(1)}</span>}
              <span className="text-slate-600">{timeAgo(m.timestamp)}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function NearCompilePatterns() {
  return (
    <div className="border border-amber-800/30 rounded-lg p-3 mb-3 bg-amber-950/10">
      <h3 className="text-xs font-bold text-amber-400 uppercase tracking-wider mb-2">
        ⚡ Near-Compilation Patterns
      </h3>
      {DEMO_PATTERNS.map((p) => (
        <div key={p.name} className="flex items-center justify-between py-1.5">
          <div>
            <span className="text-sm text-slate-200 font-medium">"{p.name}"</span>
            <span className="text-[10px] text-slate-500 ml-2">avg Ihsān: {p.avg_ihsan.toFixed(2)}</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-amber-400 rounded-full transition-all"
                style={{ width: `${(p.count / p.threshold) * 100}%` }}
              />
            </div>
            <span className="text-xs text-amber-300 font-mono font-bold">
              {p.count}/{p.threshold}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

function ActiveProjects() {
  return (
    <div className="border border-slate-700/50 rounded-lg p-3 mb-3">
      <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
        Active Projects
      </h3>
      {DEMO_PROJECTS.map((p) => (
        <div key={p.name} className="flex items-center justify-between py-1.5 border-b border-slate-800/50 last:border-0">
          <div>
            <span className="text-sm text-slate-200">{p.name}</span>
            <span className="text-[10px] text-slate-600 ml-2">{p.mission_count} missions</span>
          </div>
          <span className="text-[10px] text-slate-500">{p.last_activity}</span>
        </div>
      ))}
    </div>
  );
}

// ─── Main Component ─────────────────────────────────────────────

export default function TerminalMemory() {
  const { data: memStats } = useMemoryStats();

  return (
    <div className="p-4 max-w-3xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-bold text-slate-100">Memory</h2>
          <p className="text-xs text-slate-500">Persistent personal continuity</p>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <span className="text-[10px] text-emerald-400">Local</span>
        </div>
      </div>

      {/* §10.4: Morning briefing renders on startup */}
      <BriefingCard />

      {/* §10.4: Shows semantic profile */}
      <SemanticProfile />

      {/* §10.4: Shows last 10 missions with outcomes */}
      <MissionHistory />

      {/* §10.4: Shows near-compilation patterns */}
      <NearCompilePatterns />

      {/* §10.4: Shows active projects with last activity */}
      <ActiveProjects />

      {/* Memory Stats */}
      {memStats && (
        <div className="border border-slate-700/50 rounded-lg p-3 mb-3">
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
            Memory Store
          </h3>
          <div className="grid grid-cols-3 gap-2 text-xs text-center">
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-slate-300 font-bold">{memStats.episodic ?? 0}</div>
              <div className="text-slate-600">Episodic</div>
            </div>
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-slate-300 font-bold">{memStats.semantic ?? 0}</div>
              <div className="text-slate-600">Semantic</div>
            </div>
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-slate-300 font-bold">{memStats.procedural ?? 0}</div>
              <div className="text-slate-600">Procedural</div>
            </div>
          </div>
        </div>
      )}

      {/* §10.4: Privacy note visible */}
      <div className="text-center mt-4 py-2 border border-emerald-900/30 rounded-lg bg-emerald-950/10">
        <span className="text-[10px] text-emerald-500">
          🔒 All data is local. Nothing leaves your device. Sovereign by design.
        </span>
      </div>
    </div>
  );
}
