"use client";

import { useState } from "react";
import {
  type MissionReceipt,
  type PermissionEnvelope,
  useMissionPlanner,
  useSeedPotential,
  useTerminalState,
  useTerminalStream,
} from "@/hooks/use-sovereign-api";

type MissionStage = "draft" | "review" | "executing" | "completed";

const LIVE_MISSION_ALERT_TOPICS = [
  "mission.failed",
  "auth.boundary.crossed",
  "ihsan.breach",
  "invariant.violation",
];

function executionLabel(path: string, durationMs?: number): string {
  const latency = typeof durationMs === "number" ? ` (${Math.round(durationMs)}ms)` : "";
  if (path === "SYSTEM_1_CACHE_HIT") {
    return `System-1${latency}`;
  }
  if (path === "MIXED") {
    return `Mixed${latency}`;
  }
  return `System-2${latency}`;
}

function statusColor(status: MissionReceipt["status"]): string {
  if (status === "COMPLETE") {
    return "text-emerald-400";
  }
  if (status === "PARTIAL") {
    return "text-amber-400";
  }
  return "text-red-400";
}

function routeBadge(path: string): string {
  if (path === "SYSTEM_1_CACHE_HIT") {
    return "bg-emerald-950/40 text-emerald-300 border-emerald-700/40";
  }
  if (path === "MIXED") {
    return "bg-sky-950/40 text-sky-300 border-sky-700/40";
  }
  return "bg-indigo-950/40 text-indigo-300 border-indigo-700/40";
}

function prettyEnvelope(envelope: PermissionEnvelope): string {
  return [
    `filesystem: ${envelope.filesystem.join(", ") || "none"}`,
    `applications: ${envelope.applications.join(", ") || "none"}`,
    `network: ${envelope.network.join(", ") || "none"}`,
    `budget: $${envelope.spend_budget_usd.toFixed(2)} / ${envelope.time_budget_seconds}s`,
  ].join("\n");
}

function summarizeAlert(topic: string, payload: Record<string, unknown>): string {
  if (topic === "mission.failed") {
    return "Mission failed and requires review.";
  }
  if (topic === "auth.boundary.crossed") {
    const reason = payload.reason;
    return typeof reason === "string"
      ? `Auth boundary crossed: ${reason}.`
      : "Auth boundary crossed and needs review.";
  }
  if (topic === "ihsan.breach") {
    const rejected = payload.rejected_count;
    return typeof rejected === "number"
      ? `Ihsan breach detected: ${rejected} receipts rejected.`
      : "Ihsan breach detected in the constitutional lane.";
  }
  if (topic === "invariant.violation") {
    const metric = payload.metric;
    return typeof metric === "string"
      ? `Invariant violation detected: ${metric}.`
      : "Invariant violation detected.";
  }
  return topic;
}

function ReceiptView({ receipt }: { receipt: MissionReceipt }) {
  const hasCacheProof =
    receipt.execution_path === "SYSTEM_1_CACHE_HIT" &&
    Boolean(receipt.reflex_pattern);
  const reasoningProof = receipt.reasoning_proof;

  return (
    <div className="border border-slate-700/50 rounded-lg p-4 bg-slate-900/40">
      <div className="flex items-start justify-between gap-3 mb-3">
        <div>
          <h3 className="text-sm font-bold text-slate-100">Final Receipt</h3>
          <p className="text-xs text-slate-500">One mission, one proof.</p>
        </div>
        <div className={`text-xs px-2 py-1 rounded border ${routeBadge(receipt.execution_path)}`}>
          {executionLabel(receipt.execution_path, receipt.duration_ms)}
        </div>
      </div>

      <div className="grid sm:grid-cols-2 gap-3 text-xs mb-3">
        <div className="border border-slate-800 rounded-lg p-3">
          <div className="text-slate-500 mb-1">Status</div>
          <div className={`font-bold ${statusColor(receipt.status)}`}>{receipt.status}</div>
        </div>
        <div className="border border-slate-800 rounded-lg p-3">
          <div className="text-slate-500 mb-1">Proof</div>
          <div className="text-slate-200 font-mono break-all">
            {receipt.hash_chain_ref || receipt.receipt_id}
          </div>
        </div>
      </div>

      <p className="text-sm text-slate-200 mb-3">{receipt.synthesis}</p>

      <div className="grid sm:grid-cols-3 gap-3 mb-3">
        <div className="bg-slate-950/60 rounded-lg p-3">
          <div className="text-slate-500 text-[10px] uppercase tracking-wider">Ihsan</div>
          <div className="text-lg font-bold text-emerald-400">{receipt.ihsan_score.toFixed(2)}</div>
        </div>
        <div className="bg-slate-950/60 rounded-lg p-3">
          <div className="text-slate-500 text-[10px] uppercase tracking-wider">SNR</div>
          <div className="text-lg font-bold text-sky-400">{receipt.snr_score.toFixed(2)}</div>
        </div>
        <div className="bg-slate-950/60 rounded-lg p-3">
          <div className="text-slate-500 text-[10px] uppercase tracking-wider">SEED Earned</div>
          <div className="text-lg font-bold text-amber-400">{receipt.wallet_delta.seed.toFixed(2)}</div>
        </div>
      </div>

      <div className="border border-slate-800 rounded-lg p-3 mb-3">
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Channels Executed</h4>
          <span className="text-[10px] text-slate-600">{receipt.channels_executed.length} channels</span>
        </div>
        <div className="space-y-2">
          {receipt.channels_executed.map((channel) => (
            <div key={`${receipt.receipt_id}-${channel.channel}`} className="flex items-center justify-between text-xs">
              <span className="text-slate-300">{channel.channel}</span>
              <span className={channel.success ? "text-emerald-400" : "text-red-400"}>
                {channel.success ? "success" : "failed"} · {Math.round(channel.duration_ms)}ms
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="grid sm:grid-cols-3 gap-3 text-xs">
        <div className="border border-slate-800 rounded-lg p-3">
          <div className="text-slate-500 mb-1">Reflex Progress</div>
          <div className="text-slate-200">
            {receipt.reflex_delta.compiled
              ? "Compiled"
              : `${receipt.reflex_delta.compile_count}/${receipt.reflex_delta.threshold} toward compile`}
          </div>
        </div>
        <div className="border border-slate-800 rounded-lg p-3">
          <div className="text-slate-500 mb-1">Memory Delta</div>
          <div className="text-slate-200">
            E {receipt.memory_delta.episodic} · S {receipt.memory_delta.semantic} · P {receipt.memory_delta.procedural}
          </div>
        </div>
        <div className="border border-slate-800 rounded-lg p-3">
          <div className="text-slate-500 mb-1">Pool Share</div>
          <div className="text-slate-200">{receipt.wallet_delta.bloom.toFixed(2)} BLOOM</div>
        </div>
      </div>

      {hasCacheProof && (
        <div className="mt-3 border border-emerald-800/40 rounded-lg p-3 bg-emerald-950/10 text-xs">
          <div className="text-emerald-300 font-bold mb-1">Cache-Hit Proof</div>
          <p className="text-slate-300">
            Pattern <span className="font-mono text-emerald-300">{receipt.reflex_pattern}</span> matched in{" "}
            {receipt.reflex_latency_ms.toFixed(1)}ms. Previous System-2 average {receipt.comparison_s2_avg_ms.toFixed(1)}ms.
          </p>
        </div>
      )}

      {reasoningProof && (
        <div className="mt-3 border border-sky-800/40 rounded-lg p-3 bg-sky-950/10 text-xs">
          <div className="flex items-center justify-between gap-3 mb-1">
            <div className="text-sky-300 font-bold">Verified Reasoning Graph</div>
            <div className={reasoningProof.verified ? "text-emerald-300" : "text-amber-300"}>
              {reasoningProof.status || (reasoningProof.verified ? "ACCEPTED" : "UNAVAILABLE")}
            </div>
          </div>
          <p className="text-slate-300 break-all">
            Root <span className="font-mono text-sky-300">{reasoningProof.vrg_root || "pending"}</span>
          </p>
          <p className="text-slate-400 mt-1">
            Surviving branches {reasoningProof.surviving_branches}/{reasoningProof.branch_count}
            {reasoningProof.receipt_id ? ` | receipt ${reasoningProof.receipt_id}` : ""}
          </p>
          {reasoningProof.detail && (
            <p className="text-slate-500 mt-1">{reasoningProof.detail}</p>
          )}
        </div>
      )}
    </div>
  );
}

export default function TerminalMission() {
  const { data: potential } = useSeedPotential();
  const { data: terminalState } = useTerminalState();
  const { submitMission, receipt, loading, error, defaultPermissionEnvelope } = useMissionPlanner();
  const { events: missionAlerts, connected: missionAlertStreamConnected } = useTerminalStream(
    LIVE_MISSION_ALERT_TOPICS,
    20,
  );
  const [input, setInput] = useState("");
  const [stage, setStage] = useState<MissionStage>("draft");
  const envelope = defaultPermissionEnvelope;
  const activeAlerts = missionAlerts
    .filter((event) => event.severity === "warning" || event.severity === "critical")
    .slice(0, 3);

  const beginReview = () => {
    if (!input.trim()) {
      return;
    }
    setStage("review");
  };

  const approveAndExecute = async () => {
    setStage("executing");
    const result = await submitMission({
      description: input.trim(),
      source: "terminal",
      permissionEnvelope: envelope,
      proofMode: "verified",
    });
    if (result) {
      setStage("completed");
      setInput("");
      return;
    }
    setStage("review");
  };

  return (
    <div className="p-4 max-w-4xl mx-auto">
      <div className="flex items-start justify-between mb-4 gap-4">
        <div>
          <h2 className="text-lg font-bold text-slate-100">Mission</h2>
          <p className="text-xs text-slate-500">Submit once. Approve once. Receive one finished result with proof.</p>
        </div>
        <div className="text-right text-xs text-slate-500">
          <div>{potential.tier}</div>
          <div className="font-mono">{terminalState.state}</div>
        </div>
      </div>

      <div className="border border-slate-700/50 rounded-lg p-4 mb-4">
        <label className="text-xs text-slate-400 block mb-2">Mission composer</label>
        <textarea
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="Describe the mission you want completed..."
          rows={4}
          className="w-full bg-slate-900/70 border border-slate-700 rounded px-3 py-2 text-sm text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-teal-500 resize-y"
        />
        <div className="flex flex-wrap items-center justify-between gap-3 mt-3">
          <p className="text-[10px] text-slate-600">
            Intent → Contract → Orchestrate → Receipt → Constitutional Tick → Memory
          </p>
          <div className="flex items-center gap-2">
            {stage !== "draft" && (
              <button
                className="px-3 py-2 border border-slate-700 text-slate-300 text-xs rounded hover:border-slate-500 transition-colors"
                onClick={() => setStage("draft")}
              >
                Edit
              </button>
            )}
            <button
              className="px-4 py-2 bg-teal-600 hover:bg-teal-500 disabled:bg-slate-700 disabled:text-slate-400 text-white text-xs rounded font-bold transition-colors"
              disabled={!input.trim() || loading}
              onClick={beginReview}
            >
              Review Envelope
            </button>
          </div>
        </div>
      </div>

      {(stage === "review" || stage === "executing") && (
        <div className="border border-slate-700/50 rounded-lg p-4 mb-4 bg-slate-900/40">
          <div className="flex items-start justify-between gap-3 mb-3">
            <div>
              <h3 className="text-sm font-bold text-slate-100">Permission Envelope</h3>
              <p className="text-xs text-slate-500">Escalation only on boundary crossing.</p>
            </div>
            <span className="text-[10px] px-2 py-1 rounded bg-slate-800 text-slate-300">
              {stage === "executing" ? "EXECUTING" : "REVIEW"}
            </span>
          </div>
          {activeAlerts.length > 0 && (
            <div className="mb-3 rounded-lg border border-amber-700/40 bg-amber-950/20 p-3">
              <div className="flex items-center justify-between gap-3 mb-2">
                <div className="text-xs font-bold text-amber-300 uppercase tracking-wider">
                  Live Boundary Alerts
                </div>
                <div className={`text-[10px] uppercase tracking-wider ${missionAlertStreamConnected ? "text-emerald-400" : "text-amber-400"}`}>
                  {missionAlertStreamConnected ? "streaming" : "connecting"}
                </div>
              </div>
              <div className="space-y-2">
                {activeAlerts.map((event) => (
                  <div
                    key={event.event_hash}
                    className={`rounded border px-3 py-2 text-xs ${
                      event.severity === "critical"
                        ? "border-red-700/40 bg-red-950/30 text-red-200"
                        : "border-amber-700/40 bg-amber-950/20 text-amber-100"
                    }`}
                  >
                    <div className="font-mono text-[10px] uppercase tracking-wider text-slate-400">
                      {event.topic}
                    </div>
                    <div className="mt-1">{summarizeAlert(event.topic, event.payload ?? {})}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
          <pre className="text-xs text-slate-300 bg-slate-950/60 border border-slate-800 rounded-lg p-3 overflow-x-auto">
            {prettyEnvelope(envelope)}
          </pre>
          <div className="flex flex-wrap items-center justify-between gap-3 mt-3">
            <div className="text-xs text-slate-500">
              Route target: {executionLabel(receipt?.execution_path ?? terminalState.execution_path)}
            </div>
            <button
              className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 disabled:text-slate-400 text-white text-xs rounded font-bold transition-colors"
              disabled={loading}
              onClick={approveAndExecute}
            >
              {loading ? "Executing..." : "Approve & Execute"}
            </button>
          </div>
          {error && <p className="text-xs text-red-400 mt-3">{error}</p>}
        </div>
      )}

      {receipt && stage === "completed" && <ReceiptView receipt={receipt} />}
    </div>
  );
}
