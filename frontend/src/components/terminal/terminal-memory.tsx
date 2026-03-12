"use client";

import { useMemoryProfile } from "@/hooks/use-sovereign-api";

function timeAgo(timestamp: string): string {
  if (!timestamp) {
    return "No activity";
  }

  const diffMs = Date.now() - new Date(timestamp).getTime();
  if (!Number.isFinite(diffMs) || diffMs < 0) {
    return timestamp;
  }

  const minutes = Math.floor(diffMs / 60000);
  if (minutes < 60) {
    return `${minutes}m ago`;
  }
  const hours = Math.floor(minutes / 60);
  if (hours < 24) {
    return `${hours}h ago`;
  }
  return `${Math.floor(hours / 24)}d ago`;
}

export default function TerminalMemory() {
  const { data: profile } = useMemoryProfile();

  return (
    <div className="p-4 max-w-4xl mx-auto">
      <div className="flex items-start justify-between mb-4">
        <div>
          <h2 className="text-lg font-bold text-slate-100">Memory</h2>
          <p className="text-xs text-slate-500">Persistent personal continuity</p>
        </div>
        <span className="text-[10px] px-2 py-1 rounded bg-emerald-950/30 text-emerald-400 border border-emerald-900/40">
          Local continuity
        </span>
      </div>

      <div className="bg-gradient-to-br from-teal-950/50 to-slate-900/60 border border-teal-800/30 rounded-xl p-4 mb-4">
        <div className="flex items-start justify-between gap-4 mb-3">
          <div>
            <h3 className="text-sm font-bold text-teal-300">Morning Briefing</h3>
            <p className="text-xs text-slate-500">Continuity substrate for the next mission.</p>
          </div>
          <span className="text-xs text-slate-600">
            {new Date().toLocaleDateString(undefined, {
              weekday: "long",
              month: "short",
              day: "numeric",
            })}
          </span>
        </div>
        <div className="space-y-2 text-sm text-slate-300">
          <p>{profile.briefing.last_mission_summary || "No prior mission summary recorded yet."}</p>
          <p>
            Active project <span className="text-teal-300">{profile.briefing.active_project || "local node"}</span> ·
            Quality trend <span className="text-emerald-400">{profile.briefing.quality_trend}</span>
          </p>
          <p>{profile.briefing.next_action_suggestion || "Submit a mission to establish continuity."}</p>
        </div>
      </div>

      <div className="grid lg:grid-cols-[1fr_1fr] gap-4">
        <section className="border border-slate-700/50 rounded-lg p-4">
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">Semantic Profile</h3>
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div>
              <div className="text-slate-500 mb-1">Domains</div>
              <div className="flex flex-wrap gap-1">
                {profile.semantic_profile.preferred_domains.length > 0 ? (
                  profile.semantic_profile.preferred_domains.map((domain) => (
                    <span key={domain} className="bg-teal-900/30 text-teal-300 px-1.5 py-0.5 rounded text-[10px]">
                      {domain}
                    </span>
                  ))
                ) : (
                  <span className="text-slate-600">No learned domains yet</span>
                )}
              </div>
            </div>
            <div>
              <div className="text-slate-500 mb-1">Active hours</div>
              <div className="text-slate-300">{profile.semantic_profile.active_hours || "Not learned yet"}</div>
            </div>
            <div>
              <div className="text-slate-500 mb-1">Vocabulary</div>
              <div className="text-slate-300">{profile.semantic_profile.vocabulary_signature || "Local-first constitutional terminal"}</div>
            </div>
            <div>
              <div className="text-slate-500 mb-1">Work window</div>
              <div className="text-slate-300">{profile.semantic_profile.work_window || "Adaptive"}</div>
            </div>
          </div>
        </section>

        <section className="border border-slate-700/50 rounded-lg p-4">
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">Memory Store</h3>
          <div className="grid grid-cols-3 gap-2 text-xs text-center">
            <div className="bg-slate-900/60 rounded p-3">
              <div className="text-slate-200 font-bold">{profile.stats.episodic_count}</div>
              <div className="text-slate-600">Episodic</div>
            </div>
            <div className="bg-slate-900/60 rounded p-3">
              <div className="text-slate-200 font-bold">{profile.stats.semantic_count}</div>
              <div className="text-slate-600">Semantic</div>
            </div>
            <div className="bg-slate-900/60 rounded p-3">
              <div className="text-slate-200 font-bold">{profile.stats.procedural_count}</div>
              <div className="text-slate-600">Procedural</div>
            </div>
          </div>
        </section>
      </div>

      <div className="grid lg:grid-cols-[1fr_1fr] gap-4 mt-4">
        <section className="border border-slate-700/50 rounded-lg p-4">
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">Last 10 Missions</h3>
          <div className="space-y-2">
            {profile.missions.length > 0 ? (
              profile.missions.map((mission) => (
                <div key={mission.receipt_hash || mission.mission_id} className="flex items-center justify-between gap-3 border-b border-slate-800/50 pb-2 last:border-0 last:pb-0">
                  <div className="min-w-0">
                    <div className="text-sm text-slate-200 truncate">{mission.description}</div>
                    <div className="text-[10px] text-slate-500">{timeAgo(mission.timestamp)}</div>
                  </div>
                  <div className="text-right text-xs flex-shrink-0">
                    <div className="text-emerald-400">{mission.ihsan_score.toFixed(2)}</div>
                    <div className="text-amber-400">+{mission.seed_earned.toFixed(2)}</div>
                  </div>
                </div>
              ))
            ) : (
              <p className="text-xs text-slate-600">No missions recorded yet.</p>
            )}
          </div>
        </section>

        <section className="border border-slate-700/50 rounded-lg p-4">
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">Near-Compilation Patterns</h3>
          <div className="space-y-3">
            {profile.near_compile_patterns.length > 0 ? (
              profile.near_compile_patterns.map((pattern) => (
                <div key={pattern.name}>
                  <div className="flex items-center justify-between text-xs mb-1">
                    <span className="text-slate-200">{pattern.name}</span>
                    <span className="text-amber-300 font-mono">
                      {pattern.count}/{pattern.threshold}
                    </span>
                  </div>
                  <div className="w-full h-1.5 bg-slate-900 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-amber-400 rounded-full"
                      style={{ width: `${Math.min(100, (pattern.count / pattern.threshold) * 100)}%` }}
                    />
                  </div>
                  <div className="text-[10px] text-slate-500 mt-1">avg Ihsan {pattern.avg_ihsan.toFixed(2)}</div>
                </div>
              ))
            ) : (
              <p className="text-xs text-slate-600">No near-compile patterns yet.</p>
            )}
          </div>
        </section>
      </div>

      <section className="border border-slate-700/50 rounded-lg p-4 mt-4">
        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">Active Projects</h3>
        <div className="space-y-2">
          {profile.active_projects.length > 0 ? (
            profile.active_projects.map((project) => (
              <div key={project.name} className="flex items-center justify-between text-sm">
                <div className="text-slate-200">{project.name}</div>
                <div className="text-right text-xs text-slate-500">
                  {project.mission_count} missions · {timeAgo(project.last_activity)}
                </div>
              </div>
            ))
          ) : (
            <p className="text-xs text-slate-600">No active projects yet.</p>
          )}
        </div>
      </section>

      <div className="text-center mt-4 py-2 border border-emerald-900/30 rounded-lg bg-emerald-950/10">
        <span className="text-[10px] text-emerald-500">{profile.privacy_note}</span>
      </div>
    </div>
  );
}
