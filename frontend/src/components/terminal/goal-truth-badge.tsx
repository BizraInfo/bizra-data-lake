"use client";

export type GoalTruthLabel = "MEASURED" | "DERIVED" | "PLANNED" | "UNKNOWN";

interface GoalTruthBadgeProps {
  label: GoalTruthLabel;
  hint?: string;
}

const STYLES: Record<GoalTruthLabel, { bg: string; text: string; ring: string }> = {
  MEASURED: {
    bg: "bg-emerald-950/40",
    text: "text-emerald-300",
    ring: "ring-emerald-700/40",
  },
  DERIVED: {
    bg: "bg-sky-950/40",
    text: "text-sky-300",
    ring: "ring-sky-700/40",
  },
  PLANNED: {
    bg: "bg-amber-950/40",
    text: "text-amber-300",
    ring: "ring-amber-700/40",
  },
  UNKNOWN: {
    bg: "bg-slate-900/60",
    text: "text-slate-400",
    ring: "ring-slate-700/40",
  },
};

export default function GoalTruthBadge({ label, hint }: GoalTruthBadgeProps) {
  const s = STYLES[label];
  return (
    <span
      data-testid={`goal-truth-badge-${label.toLowerCase()}`}
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono uppercase tracking-wider ring-1 ${s.bg} ${s.text} ${s.ring}`}
      title={hint ?? `truth label: ${label}`}
    >
      {label}
    </span>
  );
}
