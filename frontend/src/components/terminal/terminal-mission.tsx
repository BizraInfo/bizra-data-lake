"use client";

/**
 * Terminal Mission — Mission execution panel for the terminal shell.
 * Placeholder component; main mission logic is in hooks/useMission.ts.
 */

import { useState } from "react";

export default function TerminalMission() {
  const [input, setInput] = useState("");

  return (
    <div className="p-4 max-w-3xl mx-auto flex flex-col h-full">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-bold text-slate-100">Mission</h2>
          <p className="text-xs text-slate-500">
            Execute tasks through the agent mesh
          </p>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
          <span className="text-[10px] text-emerald-400">READY</span>
        </div>
      </div>

      <div className="flex-1 bg-slate-900/40 border border-slate-800/50 rounded-lg p-4 mb-3 min-h-[200px]">
        <p className="text-xs text-slate-600 text-center py-12">
          Enter a mission below to begin
        </p>
      </div>

      <div className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Describe your mission..."
          className="flex-1 bg-slate-900/60 border border-slate-800/50 rounded-md px-3 py-2 text-sm text-slate-200 placeholder-slate-600 outline-none focus:border-teal-700"
        />
        <button
          disabled={!input.trim()}
          className="px-4 py-2 bg-teal-900/50 border border-teal-700/50 rounded-md text-xs text-teal-300 disabled:opacity-30 disabled:cursor-not-allowed hover:bg-teal-800/50 transition-colors"
        >
          Execute
        </button>
      </div>
    </div>
  );
}
