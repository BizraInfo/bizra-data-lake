"use client";

import GoalTruthBadge, { type GoalTruthLabel } from "./goal-truth-badge";

interface GoalStateCardProps {
  testId: string;
  title: string;
  body: string;
  truthLabel: GoalTruthLabel;
  hint?: string;
  accentHex?: string;
}

export default function GoalStateCard({
  testId,
  title,
  body,
  truthLabel,
  hint,
  accentHex = "#C9A962",
}: GoalStateCardProps) {
  return (
    <div
      data-testid={testId}
      className="flex flex-col gap-2 p-4 rounded-lg border border-slate-800 bg-slate-950/60"
    >
      <div className="flex items-center justify-between">
        <h3
          className="text-xs font-mono uppercase tracking-wider"
          style={{ color: accentHex }}
        >
          {title}
        </h3>
        <GoalTruthBadge label={truthLabel} hint={hint} />
      </div>
      <p
        data-testid={`${testId}-body`}
        className="text-sm text-slate-300 whitespace-pre-line"
      >
        {body || "—"}
      </p>
    </div>
  );
}
