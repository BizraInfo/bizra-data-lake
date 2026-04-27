"use client";

import { useDemaGoalState } from "@/hooks/use-dema-goal-state";

import GoalStateCard from "./goal-state-card";
import GoalTruthBadge from "./goal-truth-badge";

const ACCENT_CURRENT = "#C9A962";
const ACCENT_IDEAL = "#3B82F6";
const ACCENT_GAP = "#F59E0B";
const ACCENT_NEXT = "#22C55E";

const IHSAN_BAND_COLOR: Record<string, string> = {
  ideal: "text-emerald-300",
  warn: "text-amber-300",
  halt: "text-red-400",
  unknown: "text-slate-500",
};

export default function TerminalGoal() {
  const goal = useDemaGoalState();

  return (
    <div data-testid="terminal-goal" className="flex flex-col gap-4 p-4">
      <header className="flex flex-col gap-1">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-mono uppercase tracking-wider text-slate-200">
            §9 — Current → Ideal → Gap → Next
          </h2>
          <GoalTruthBadge
            label={goal.trust.truthLabel}
            hint="trust signals are measured from /v1/health when available"
          />
        </div>
        <p className="text-xs text-slate-500">
          The Dema goal loop: every meaningful action is admissibility-gated and
          receipted. No fake metrics — unavailable data is labelled UNKNOWN or
          PLANNED.
        </p>
      </header>

      {/* Trust strip */}
      <section
        data-testid="goal-trust-strip"
        className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] font-mono"
      >
        <div className="flex flex-col gap-0.5 p-2 rounded border border-slate-800 bg-slate-950/60">
          <span className="text-slate-500 uppercase tracking-wider">CHAIN</span>
          <span
            data-testid="goal-trust-chain"
            className="text-slate-200"
          >
            {goal.trust.chainHeadShort
              ? `#${goal.trust.chainLength ?? 0} ${goal.trust.chainHeadShort}`
              : "—"}
          </span>
        </div>
        <div className="flex flex-col gap-0.5 p-2 rounded border border-slate-800 bg-slate-950/60">
          <span className="text-slate-500 uppercase tracking-wider">RECEIPT</span>
          <span data-testid="goal-trust-receipt" className="text-slate-200">
            {goal.trust.receiptKind ?? "—"}
          </span>
        </div>
        <div className="flex flex-col gap-0.5 p-2 rounded border border-slate-800 bg-slate-950/60">
          <span className="text-slate-500 uppercase tracking-wider">IHSĀN</span>
          <span
            data-testid="goal-trust-ihsan"
            className={IHSAN_BAND_COLOR[goal.trust.ihsanBand] ?? "text-slate-300"}
          >
            {goal.trust.ihsanScore === null
              ? "—"
              : `${goal.trust.ihsanScore.toFixed(2)} ${goal.trust.ihsanBand}`}
          </span>
        </div>
        <div className="flex flex-col gap-0.5 p-2 rounded border border-slate-800 bg-slate-950/60">
          <span className="text-slate-500 uppercase tracking-wider">GINI</span>
          <span data-testid="goal-trust-gini" className="text-slate-200">
            {goal.trust.gini === null ? "—" : goal.trust.gini.toFixed(2)}
          </span>
        </div>
      </section>

      {/* Four state cards */}
      <section className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <GoalStateCard
          testId="goal-card-current"
          title="Current State"
          body={goal.current.body}
          truthLabel={goal.current.truthLabel}
          hint={goal.current.hint}
          accentHex={ACCENT_CURRENT}
        />
        <GoalStateCard
          testId="goal-card-ideal"
          title="Ideal State"
          body={goal.ideal.body}
          truthLabel={goal.ideal.truthLabel}
          hint={goal.ideal.hint}
          accentHex={ACCENT_IDEAL}
        />
        <GoalStateCard
          testId="goal-card-gap"
          title="Gap"
          body={goal.gap.body}
          truthLabel={goal.gap.truthLabel}
          hint={goal.gap.hint}
          accentHex={ACCENT_GAP}
        />
        <GoalStateCard
          testId="goal-card-next"
          title="Next Admissible Action"
          body={goal.nextAdmissibleAction.body}
          truthLabel={goal.nextAdmissibleAction.truthLabel}
          hint={goal.nextAdmissibleAction.hint}
          accentHex={ACCENT_NEXT}
        />
      </section>

      {goal.error && (
        <p
          data-testid="goal-error"
          className="text-xs text-red-400 font-mono"
        >
          backend error: {goal.error}
        </p>
      )}
    </div>
  );
}
