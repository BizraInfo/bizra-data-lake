"use client";

import {
  useChainLatest,
  useMemoryStats,
  useNode0Readiness,
  useSovereignHealth,
  useTerminalState,
} from "@/hooks/use-sovereign-api";

type ReadinessState = "ready" | "warn" | "planned";

interface ReadinessCardProps {
  title: string;
  state: ReadinessState;
  body: string;
  testId: string;
}

const STATE_STYLES: Record<ReadinessState, string> = {
  ready: "border-emerald-800/50 bg-emerald-950/15 text-emerald-300",
  warn: "border-amber-800/50 bg-amber-950/15 text-amber-300",
  planned: "border-slate-800 bg-slate-950/50 text-slate-400",
};

function ReadinessCard({ title, state, body, testId }: ReadinessCardProps) {
  return (
    <div
      data-testid={testId}
      className={`rounded-lg border p-3 ${STATE_STYLES[state]}`}
    >
      <div className="flex items-center justify-between gap-3 mb-2">
        <h3 className="text-xs font-bold uppercase tracking-wider">{title}</h3>
        <span className="text-[10px] font-mono uppercase">{state}</span>
      </div>
      <p className="text-xs text-slate-300">{body}</p>
    </div>
  );
}

function formatLastReceipt(kind: string, timestamp: number | null): string {
  if (!kind) {
    return "no receipt yet";
  }
  if (timestamp === null) {
    return kind;
  }
  const rendered = new Date(timestamp * 1000).toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  });
  return `${kind} at ${rendered}`;
}

function readinessState(ready: boolean, failed = false): ReadinessState {
  if (ready) {
    return "ready";
  }
  return failed ? "warn" : "planned";
}

export default function TerminalNode0() {
  const { data: health, error: healthError, loading: healthLoading } = useSovereignHealth();
  const { data: terminalState } = useTerminalState();
  const { data: chainLatest, error: chainError, loading: chainLoading } = useChainLatest();
  const { data: memory } = useMemoryStats();
  const { data: readiness } = useNode0Readiness();

  const live = !healthLoading && !healthError && (health.running || health.live_status === "LIVE");
  const ihsan = typeof health.ihsan_score === "number" ? health.ihsan_score : null;
  const snr = typeof health.snr_score === "number" ? health.snr_score : null;
  const chainReady =
    !chainLoading &&
    !chainError &&
    chainLatest.length > 0 &&
    Boolean(chainLatest.head);
  const latestReceipt = chainLatest.latestReceipt;
  const receiptLabel = formatLastReceipt(
    latestReceipt?.kind ?? "",
    latestReceipt?.timestamp ?? null,
  );
  const memoryEntries = memory.total_entries;
  const missionReady = live && terminalState.state !== "error";
  const proofReady = chainReady || latestReceipt !== null;
  const bootReady = readiness.boot_service.booted;
  const spearpointReady = readiness.spearpoint.status === "pass";
  const spearpointFailed = readiness.spearpoint.status === "fail";
  const readinessLabel = readiness.status.toUpperCase();

  return (
    <div data-testid="node0-shell" className="p-4 max-w-5xl mx-auto space-y-4">
      <header className="rounded-2xl border border-violet-800/40 bg-violet-950/10 p-5">
        <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
          <div>
            <p className="text-[10px] uppercase tracking-[0.35em] text-violet-300 mb-2">
              Dema Node0 Product Shell v0.1
            </p>
            <h1 className="text-2xl font-bold text-slate-100">
              Open Dema. Submit a mission. Inspect the proof.
            </h1>
            <p className="text-sm text-slate-400 mt-2 max-w-2xl">
              This is the visible Node0 entry point: local readiness, mission
              path, receipt/proof visibility, and the next action without
              claiming voice, desktop action, or daemon packaging as complete.
            </p>
          </div>
          <div
            data-testid="node0-live-state"
            className={`rounded-lg border px-3 py-2 text-xs font-mono ${
              readiness.status === "green"
                ? "border-emerald-800/60 bg-emerald-950/20 text-emerald-300"
                : readiness.status === "yellow"
                ? "border-amber-800/60 bg-amber-950/20 text-amber-300"
                : "border-red-800/60 bg-red-950/20 text-red-300"
            }`}
          >
            NODE0 {readinessLabel}
          </div>
        </div>

        <div className="grid sm:grid-cols-4 gap-3 mt-5 text-xs">
          <div className="rounded-lg bg-slate-950/50 border border-slate-800 p-3">
            <div className="text-slate-500 uppercase tracking-wider mb-1">
              Runtime / Boot
            </div>
            <div className="text-slate-200 font-mono">
              {terminalState.state} / {readiness.boot_service.status}
            </div>
          </div>
          <div className="rounded-lg bg-slate-950/50 border border-slate-800 p-3">
            <div className="text-slate-500 uppercase tracking-wider mb-1">
              Ihsan / SNR
            </div>
            <div className="text-slate-200 font-mono">
              {ihsan === null ? "--" : ihsan.toFixed(2)} /{" "}
              {snr === null ? "--" : snr.toFixed(2)}
            </div>
          </div>
          <div className="rounded-lg bg-slate-950/50 border border-slate-800 p-3">
            <div className="text-slate-500 uppercase tracking-wider mb-1">
              Receipt
            </div>
            <div className="text-slate-200">{receiptLabel}</div>
          </div>
          <div className="rounded-lg bg-slate-950/50 border border-slate-800 p-3">
            <div className="text-slate-500 uppercase tracking-wider mb-1">
              Spearpoint
            </div>
            <div data-testid="node0-spearpoint-status" className="text-slate-200 font-mono">
              {readiness.spearpoint.status}
              {readiness.spearpoint.run_id ? ` #${readiness.spearpoint.run_id}` : ""}
            </div>
          </div>
        </div>
      </header>

      <section className="grid md:grid-cols-4 gap-3">
        <ReadinessCard
          testId="node0-step-dema"
          title="1. See Dema"
          state={live ? "ready" : "warn"}
          body={
            live
              ? "Local health is reporting live; Dema can accept a guided mission."
              : "Start or repair the local Dema service before trusting runtime metrics."
          }
        />
        <ReadinessCard
          testId="node0-step-mission"
          title="2. Submit task"
          state={missionReady ? "ready" : "warn"}
          body="Press key 2 or open Mission, review the permission envelope, then execute one task."
        />
        <ReadinessCard
          testId="node0-step-proof"
          title="3. Inspect proof"
          state={proofReady ? "ready" : "warn"}
          body="Mission receipts now render a Proof Surface panel with verdict, source, and export readiness."
        />
        <ReadinessCard
          testId="node0-step-next"
          title="4. Evaluate"
          state={readinessState(spearpointReady, spearpointFailed)}
          body={
            spearpointReady
              ? "Last internal Spearpoint strict run passed; treat it as readiness evidence, not public leaderboard proof."
              : "Run the internal Spearpoint strict harness before making benchmark-readiness claims."
          }
        />
      </section>

      <section className="grid md:grid-cols-[1fr_1fr] gap-4">
        <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-4">
          <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
            Minimum usable loop
          </h2>
          <ol className="space-y-2 text-sm text-slate-300">
            <li>1. Dema opens to this Node0 shell.</li>
             <li>2. Operator presses Mission and submits one task.</li>
             <li>3. Runtime returns a mission receipt.</li>
             <li>4. Proof Surface shows claim, source, verdict, and export lock.</li>
             <li>5. Node0 readiness shows boot state and last Spearpoint status.</li>
             <li>6. Next action is explicit instead of hidden in backend logs.</li>
           </ol>
         </div>

        <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-4">
          <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
            Honest readiness
          </h2>
          <div className="space-y-2 text-xs">
            <div className="flex items-center justify-between gap-3">
              <span className="text-slate-400">Proof panel</span>
              <span className={readiness.proof_surface.available ? "text-emerald-300" : "text-red-300"}>
                {readiness.proof_surface.available ? "visible" : "unavailable"}
              </span>
            </div>
            <div className="flex items-center justify-between gap-3">
              <span className="text-slate-400">Memory entries</span>
              <span className="font-mono text-slate-200">{memoryEntries}</span>
            </div>
            <div className="flex items-center justify-between gap-3">
              <span className="text-slate-400">Boot/service packaging</span>
              <span
                data-testid="node0-boot-service-status"
                className={bootReady ? "text-emerald-300" : "text-amber-300"}
              >
                {readiness.boot_service.status}
              </span>
            </div>
            <div className="flex items-center justify-between gap-3">
              <span className="text-slate-400">Last Spearpoint strict run</span>
              <span className={spearpointReady ? "text-emerald-300" : "text-amber-300"}>
                {readiness.spearpoint.status}
              </span>
            </div>
            <div className="flex items-center justify-between gap-3">
              <span className="text-slate-400">Voice + desktop action</span>
              <span className="text-slate-500">not integrated</span>
            </div>
            <div className="border-t border-slate-800 pt-2 flex items-center justify-between gap-3">
              <span className="text-slate-400">Next action</span>
              <span data-testid="node0-next-action" className="text-violet-300">
                {readiness.next_action}
              </span>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
