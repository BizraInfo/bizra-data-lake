"use client";

/**
 * Terminal Dashboard — Overview panel for the terminal shell.
 * Placeholder component; main dashboard is in phases/Dashboard.tsx.
 */

export default function TerminalDashboard() {
  return (
    <div className="p-4 max-w-3xl mx-auto">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-bold text-slate-100">Dashboard</h2>
          <p className="text-xs text-slate-500">
            Sovereign node overview
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="bg-slate-900/60 border border-slate-800/50 rounded-lg p-3">
          <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">
            Status
          </div>
          <div className="text-sm text-emerald-400 font-medium">
            Online
          </div>
        </div>
        <div className="bg-slate-900/60 border border-slate-800/50 rounded-lg p-3">
          <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">
            Agents
          </div>
          <div className="text-sm text-slate-200 font-medium">
            7 PAT · 5 SAT
          </div>
        </div>
      </div>

      <div className="text-xs text-slate-600 text-center py-8">
        Use keyboard shortcuts 1-7 to navigate views
      </div>
    </div>
  );
}
