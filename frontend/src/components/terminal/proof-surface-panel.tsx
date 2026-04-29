"use client";

import type { ProofSurfacePayload } from "@/lib/dema-proof-surface";

interface ProofSurfacePanelProps {
  surface: ProofSurfacePayload;
}

function verdictColor(verdict: ProofSurfacePayload["decision"]): string {
  if (verdict === "forbid") {
    return "text-red-300 border-red-800/50 bg-red-950/20";
  }
  if (verdict === "require_approval") {
    return "text-amber-300 border-amber-800/50 bg-amber-950/20";
  }
  if (verdict === "notify") {
    return "text-sky-300 border-sky-800/50 bg-sky-950/20";
  }
  return "text-emerald-300 border-emerald-800/50 bg-emerald-950/20";
}

function listValue(values: string[]): string {
  return values.length > 0 ? values.join(", ") : "none";
}

function EvidenceList({
  title,
  values,
  tone = "slate",
}: {
  title: string;
  values: string[];
  tone?: "slate" | "amber" | "red";
}) {
  const color =
    tone === "red"
      ? "text-red-300"
      : tone === "amber"
        ? "text-amber-300"
        : "text-slate-300";

  return (
    <div className="border border-slate-800 rounded-lg p-3">
      <div className="text-slate-500 text-[10px] uppercase tracking-wider mb-1">
        {title}
      </div>
      <div className={`text-xs font-mono break-all ${color}`}>
        {listValue(values)}
      </div>
    </div>
  );
}

export default function ProofSurfacePanel({ surface }: ProofSurfacePanelProps) {
  return (
    <section
      data-testid="proof-surface-panel"
      className="border border-violet-800/40 rounded-lg p-4 bg-violet-950/10"
    >
      <div className="flex items-start justify-between gap-3 mb-3">
        <div>
          <h3 className="text-sm font-bold text-slate-100">Proof Surface</h3>
          <p className="text-xs text-slate-500">
            Claim, source, evidence-auditor verdict, and export readiness.
          </p>
        </div>
        <div
          data-testid="proof-surface-decision"
          className={`text-[10px] px-2 py-1 rounded border uppercase tracking-wider ${verdictColor(
            surface.decision,
          )}`}
        >
          {surface.decision}
        </div>
      </div>

      <div className="grid md:grid-cols-[1.15fr_0.85fr] gap-3 mb-3">
        <div className="border border-slate-800 rounded-lg p-3">
          <div className="flex items-center justify-between gap-3 mb-2">
            <div className="text-slate-500 text-[10px] uppercase tracking-wider">
              Claim
            </div>
            <span
              data-testid="proof-surface-truth-label"
              className="text-[10px] text-violet-300 font-mono"
            >
              {surface.truth_label}
            </span>
          </div>
          <p
            data-testid="proof-surface-claim"
            className="text-sm text-slate-200"
          >
            {surface.claim}
          </p>
          <p className="text-xs text-slate-500 mt-2">
            Source:{" "}
            <span
              data-testid="proof-surface-source"
              className="font-mono text-slate-300"
            >
              {surface.source}
            </span>
          </p>
        </div>

        <div className="border border-slate-800 rounded-lg p-3">
          <div className="text-slate-500 text-[10px] uppercase tracking-wider mb-2">
            Auditor + Export
          </div>
          <div className="space-y-2 text-xs">
            <div className="flex items-center justify-between gap-3">
              <span className="text-slate-400">Evidence auditor</span>
              <span
                data-testid="proof-surface-auditor"
                className={`font-mono ${verdictColor(
                  surface.evidence_auditor_verdict,
                ).split(" ")[0]}`}
              >
                {surface.evidence_auditor_verdict}
              </span>
            </div>
            <div className="flex items-center justify-between gap-3">
              <span className="text-slate-400">Converged</span>
              <span
                data-testid="proof-surface-converged"
                className={surface.converged ? "text-emerald-300" : "text-amber-300"}
              >
                {surface.converged ? "yes" : "no"}
              </span>
            </div>
            <div className="flex items-center justify-between gap-3">
              <span className="text-slate-400">Signed export</span>
              <span
                data-testid="proof-surface-export"
                className={
                  surface.receipt_export_ready
                    ? "text-emerald-300"
                    : "text-slate-500"
                }
              >
                {surface.receipt_export_ready ? "ready" : "locked"}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="grid md:grid-cols-3 gap-3 mb-3">
        <EvidenceList title="Evidence refs" values={surface.evidence_refs} />
        <EvidenceList
          title="Missing sources"
          values={surface.missing_sources}
          tone={surface.missing_sources.length > 0 ? "amber" : "slate"}
        />
        <EvidenceList
          title="Blocking sources"
          values={surface.blocking_sources}
          tone={surface.blocking_sources.length > 0 ? "red" : "slate"}
        />
      </div>

      <div className="grid md:grid-cols-2 gap-3 text-xs">
        <div className="border border-slate-800 rounded-lg p-3">
          <div className="text-slate-500 text-[10px] uppercase tracking-wider mb-1">
            Decision reason
          </div>
          <p data-testid="proof-surface-reason" className="text-slate-300">
            {surface.decision_reason}
          </p>
        </div>
        <div className="border border-slate-800 rounded-lg p-3">
          <div className="text-slate-500 text-[10px] uppercase tracking-wider mb-1">
            Surface ID
          </div>
          <p
            data-testid="proof-surface-id"
            className="font-mono text-slate-300 break-all"
          >
            {surface.surface_id}
          </p>
          {surface.receipt_id && (
            <p className="text-slate-500 mt-1">
              Receipt <span className="font-mono">{surface.receipt_id}</span>
            </p>
          )}
        </div>
      </div>
    </section>
  );
}
