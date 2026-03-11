"use client";

import { useState } from "react";
import { useSeedPotential } from "@/hooks/use-sovereign-api";

interface Mission {
  id: string;
  description: string;
  status: "pending" | "executing" | "complete" | "failed";
  ihsan_score?: number;
  seed_earned?: number;
}

const DEMO_MISSIONS: Mission[] = [
  { id: "m_alpha_003", description: "Organize project files", status: "complete", ihsan_score: 0.97, seed_earned: 2.41 },
  { id: "m_alpha_002", description: "Generate weekly report", status: "complete", ihsan_score: 0.95, seed_earned: 1.89 },
  { id: "m_alpha_001", description: "Initialize sovereign node", status: "complete", ihsan_score: 0.99, seed_earned: 5.00 },
];

export default function TerminalMission() {
  const { data: potential } = useSeedPotential();
  const [input, setInput] = useState("");

  return (
    <div className="p-4 max-w-3xl mx-auto">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-bold text-slate-100">Mission</h2>
          <p className="text-xs text-slate-500">Submit work, earn sovereignty</p>
        </div>
        <span className="text-xs text-slate-600">{potential.tier}</span>
      </div>

      {/* Mission input */}
      <div className="border border-slate-700/50 rounded-lg p-4 mb-4">
        <label className="text-xs text-slate-400 block mb-2">What would you like to accomplish?</label>
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Describe your mission..."
            className="flex-1 bg-slate-800/50 border border-slate-700 rounded px-3 py-2 text-sm text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-teal-500"
          />
          <button
            className="px-4 py-2 bg-teal-600 hover:bg-teal-500 text-white text-xs rounded font-bold transition-colors"
            onClick={() => setInput("")}
          >
            Submit
          </button>
        </div>
        <p className="text-[10px] text-slate-600 mt-2">
          OBSERVE → DECOMPOSE → EXECUTE → SYNTHESIZE → GATE → EVIDENCE
        </p>
      </div>

      {/* Recent missions */}
      <div className="border border-slate-700/50 rounded-lg p-3">
        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
          Recent Missions
        </h3>
        <div className="space-y-1">
          {DEMO_MISSIONS.map((m) => (
            <div key={m.id} className="flex items-center justify-between py-1.5 border-b border-slate-800/50 last:border-0">
              <div className="flex items-center gap-2 min-w-0 flex-1">
                <span className={`text-xs font-bold ${m.status === "complete" ? "text-emerald-400" : "text-red-400"}`}>
                  {m.status === "complete" ? "✓" : "✗"}
                </span>
                <span className="text-xs text-slate-300 truncate">{m.description}</span>
              </div>
              <div className="flex items-center gap-3 flex-shrink-0 text-[10px]">
                {m.ihsan_score && (
                  <span className={m.ihsan_score >= 0.95 ? "text-emerald-400" : "text-amber-400"}>
                    {m.ihsan_score.toFixed(2)}
                  </span>
                )}
                {m.seed_earned && m.seed_earned > 0 && (
                  <span className="text-amber-400">+{m.seed_earned.toFixed(1)}</span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
