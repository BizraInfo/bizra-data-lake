"use client";

import { useEffect, useState } from "react";
import {
  type TerminalStreamEvent,
  useCriticalEventAcknowledger,
  useTerminalStream,
} from "@/hooks/use-sovereign-api";

type EventSeverity = "info" | "notice" | "warning" | "critical";
type EventCategory =
  | "mission"
  | "economy"
  | "reflex"
  | "constitutional"
  | "auth"
  | "system";

interface TimelineEvent {
  id: string;
  timestamp: string;
  category: EventCategory;
  topic: string;
  summary: string;
  severity: EventSeverity;
  mission_id?: string;
  receipt_id?: string;
  hash_chain_ref?: string;
  prev_hash?: string;
  payload?: Record<string, unknown>;
}

const SEVERITY_STYLES: Record<EventSeverity, string> = {
  info: "border-l-slate-500 bg-slate-900/30",
  notice: "border-l-teal-400 bg-teal-900/20",
  warning: "border-l-amber-400 bg-amber-900/20",
  critical: "border-l-red-500 bg-red-900/30 ring-1 ring-red-500/30",
};

const SEVERITY_BADGE: Record<EventSeverity, string> = {
  info: "bg-slate-700 text-slate-300",
  notice: "bg-teal-800 text-teal-200",
  warning: "bg-amber-800 text-amber-200",
  critical: "bg-red-800 text-red-100",
};

const CATEGORY_LABELS: Record<EventCategory, string> = {
  mission: "Mission",
  economy: "Economy",
  reflex: "Reflex",
  constitutional: "Constitutional",
  auth: "Auth",
  system: "System",
};

const TOPIC_SEVERITY: Record<string, EventSeverity> = {
  "mission.created": "info",
  "mission.executed": "info",
  "mission.verified": "info",
  "mission.failed": "warning",
  "economy.seed_minted": "notice",
  "economy.zakat": "info",
  "economy.bloom_accrued": "info",
  "economy.asabiyyah": "info",
  "reflex.compiled": "notice",
  "ihsan.breach": "critical",
  "invariant.violation": "critical",
  "auth.boundary.crossed": "warning",
  "critical.acknowledged": "notice",
  "receipt.generated": "info",
  "receipt.verified": "info",
  "tick.completed": "info",
};

const LIVE_TIMELINE_TOPICS = [
  "mission.*",
  "receipt.generated",
  "receipt.verified",
  "reflex.compiled",
  "tick.completed",
  "ihsan.breach",
  "invariant.violation",
  "auth.boundary.crossed",
  "critical.acknowledged",
  "economy.*",
];

const ACKNOWLEDGED_CRITICALS_KEY = "bizra-terminal-acknowledged-criticals";

function canonicalTopic(topic: string): string {
  if (topic === "policy.invariant.violation") {
    return "invariant.violation";
  }
  return topic;
}

function classifyTopic(topic: string): EventCategory {
  if (topic.startsWith("mission.") || topic.startsWith("receipt.")) return "mission";
  if (topic.startsWith("economy.")) return "economy";
  if (topic.startsWith("reflex.")) return "reflex";
  if (
    topic.startsWith("ihsan.") ||
    topic.startsWith("invariant.") ||
    topic.startsWith("critical.") ||
    topic.startsWith("tick.")
  ) {
    return "constitutional";
  }
  if (topic.startsWith("auth.")) return "auth";
  return "system";
}

function severityForTopic(topic: string): EventSeverity {
  return TOPIC_SEVERITY[topic] ?? "info";
}

function formatTimestamp(value: string): string {
  const parsed = Date.parse(value);
  if (!Number.isFinite(parsed)) {
    return value;
  }
  return new Date(parsed).toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatDate(value: string): string {
  const parsed = Date.parse(value);
  if (!Number.isFinite(parsed)) {
    return "Unknown";
  }
  return new Date(parsed).toLocaleDateString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
  });
}

function timestampValue(value: string): number {
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function payloadString(
  payload: Record<string, unknown>,
  key: string,
  fallback = "",
): string {
  const value = payload[key];
  return typeof value === "string" ? value : fallback;
}

function payloadNumber(
  payload: Record<string, unknown>,
  key: string,
  fallback = 0,
): number {
  const value = payload[key];
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function summarizeEvent(topic: string, payload: Record<string, unknown>, event: TerminalStreamEvent): string {
  if (topic === "mission.created") {
    return `Mission created | ${event.mission_id || payloadString(payload, "mission_id", "pending")}`;
  }
  if (topic === "receipt.generated") {
    return (
      `Receipt generated | ${payloadString(payload, "status", "COMPLETE")}` +
      ` | Ihsan ${payloadNumber(payload, "ihsan_score").toFixed(2)}` +
      ` | SNR ${payloadNumber(payload, "snr_score").toFixed(2)}`
    );
  }
  if (topic === "receipt.verified") {
    return `Receipt verified | ${event.receipt_id || payloadString(payload, "receipt_id", "pending")}`;
  }
  if (topic === "mission.executed") {
    return (
      `Mission executed | ${payloadString(payload, "status", "COMPLETE")}` +
      ` | ${payloadString(payload, "execution_path", "SYSTEM_2_NOVEL")}`
    );
  }
  if (topic === "mission.verified") {
    const verified = payload.verified === true ? "verified" : "audit";
    const root = payloadString(payload, "vrg_root", "");
    return (
      `Mission verified | ${payloadString(payload, "proof_status", "UNKNOWN")}` +
      ` | ${verified}` +
      ` | ${payloadNumber(payload, "surviving_branches")}/${payloadNumber(payload, "branch_count")} branches` +
      (root ? ` | ${root.slice(0, 12)}` : "")
    );
  }
  if (topic === "mission.failed") {
    return `Mission failed | ${payloadString(payload, "status", "FAILED")}`;
  }
  if (topic === "tick.completed") {
    return (
      `Constitutional tick completed | scored ${payloadNumber(payload, "scored")}` +
      ` | minted ${payloadNumber(payload, "minted").toFixed(2)}` +
      ` | reflexes ${payloadNumber(payload, "reflexes")}`
    );
  }
  if (topic === "reflex.compiled") {
    const name = payloadString(payload, "name");
    if (name) {
      return `REFLEX COMPILED | "${name}" | avg Ihsan ${payloadNumber(payload, "avg_ihsan").toFixed(2)}`;
    }
    return `REFLEX COMPILED | cache size ${payloadNumber(payload, "count")}`;
  }
  if (topic === "auth.boundary.crossed") {
    return `Auth boundary crossed | ${payloadString(payload, "reason", "review required")}`;
  }
  if (topic === "critical.acknowledged") {
    return (
      `Critical acknowledged | ${payloadString(payload, "acknowledged_topic", "critical")}` +
      ` | receipt ${event.receipt_id || payloadString(payload, "receipt_id", "pending")}`
    );
  }
  if (topic === "ihsan.breach") {
    return `Ihsan breach | ${payloadNumber(payload, "rejected_count")} receipts rejected`;
  }
  if (topic === "invariant.violation") {
    return (
      `Invariant violation | ${payloadString(payload, "metric", "unknown")}` +
      ` > ${payloadNumber(payload, "threshold").toFixed(2)}`
    );
  }
  if (topic === "economy.seed_minted") {
    return (
      `SEED minted | ${payloadNumber(payload, "minted").toFixed(2)}` +
      ` | scored ${payloadNumber(payload, "scored")}`
    );
  }
  if (topic === "economy.zakat") {
    return `Zakat recorded | ${payloadNumber(payload, "zakat_pool").toFixed(2)}`;
  }
  if (topic === "economy.bloom_accrued") {
    return `BLOOM accrued | scored ${payloadNumber(payload, "scored")}`;
  }
  if (topic === "economy.asabiyyah") {
    return (
      `Asabiyyah updated | ${payloadNumber(payload, "asabiyyah").toFixed(2)}` +
      ` | Gini ${payloadNumber(payload, "gini").toFixed(2)}`
    );
  }
  return topic;
}

function buildTimelineEvent(event: TerminalStreamEvent): TimelineEvent {
  const topic = canonicalTopic(event.topic);
  const payload = event.payload ?? {};
  return {
    id: event.event_hash,
    timestamp: event.timestamp,
    category: classifyTopic(topic),
    topic,
    summary: summarizeEvent(topic, payload, event),
    severity: event.severity ?? severityForTopic(topic),
    mission_id: event.mission_id || undefined,
    receipt_id: event.receipt_id || undefined,
    hash_chain_ref: event.event_hash || undefined,
    prev_hash: event.prev_hash || undefined,
    payload,
  };
}

function loadAcknowledgedCriticals(): Set<string> {
  if (typeof window === "undefined") {
    return new Set();
  }

  try {
    const raw = window.sessionStorage.getItem(ACKNOWLEDGED_CRITICALS_KEY);
    if (!raw) {
      return new Set();
    }
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed)
      ? new Set(parsed.filter((value): value is string => typeof value === "string"))
      : new Set();
  } catch {
    return new Set();
  }
}

function SeverityBadge({ severity }: { severity: EventSeverity }) {
  return (
    <span className={`text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded font-bold ${SEVERITY_BADGE[severity]}`}>
      {severity}
    </span>
  );
}

function CategoryFilter({
  active,
  onToggle,
}: {
  active: Set<EventCategory>;
  onToggle: (category: EventCategory) => void;
}) {
  const categories: EventCategory[] = [
    "mission",
    "economy",
    "reflex",
    "constitutional",
    "auth",
    "system",
  ];

  return (
    <div className="flex flex-wrap gap-1.5 mb-4">
      {categories.map((category) => (
        <button
          key={category}
          onClick={() => onToggle(category)}
          className={`text-xs px-2.5 py-1 rounded-full border transition-all ${
            active.has(category)
              ? "border-teal-500 bg-teal-900/40 text-teal-200"
              : "border-slate-600 bg-slate-800/40 text-slate-500 hover:border-slate-500"
          }`}
        >
          {CATEGORY_LABELS[category]}
        </button>
      ))}
    </div>
  );
}

function HashChainLink({ prev, current }: { prev?: string; current?: string }) {
  if (!current) {
    return null;
  }

  return (
    <div className="flex items-center gap-1.5 text-[10px] text-slate-500 font-mono mt-1.5">
      {prev && (
        <>
          <span title={prev}>{prev.slice(0, 10)}...</span>
          <span className="text-teal-600">-&gt;</span>
        </>
      )}
      <span className="text-teal-400" title={current}>
        {current.slice(0, 10)}...
      </span>
    </div>
  );
}

function StickyAlert({
  event,
  onAcknowledge,
  acknowledging,
}: {
  event: TimelineEvent;
  onAcknowledge: (event: TimelineEvent) => void;
  acknowledging: boolean;
}) {
  return (
    <div className="bg-red-950/60 border border-red-500/50 rounded-lg p-3 mb-3 flex items-start justify-between gap-3">
      <div>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-red-400 text-sm font-bold">CRITICAL</span>
          <span className="text-xs text-red-300">{formatTimestamp(event.timestamp)}</span>
        </div>
        <p className="text-sm text-red-200">{event.summary}</p>
        <p className="text-xs text-red-400 mt-1">{event.topic}</p>
      </div>
      <button
        onClick={() => onAcknowledge(event)}
        disabled={acknowledging}
        className="text-xs px-2 py-1 rounded border border-red-500/50 text-red-300 hover:bg-red-900/50 transition-colors flex-shrink-0 disabled:opacity-60 disabled:cursor-not-allowed"
      >
        {acknowledging ? "Recording..." : "Acknowledge"}
      </button>
    </div>
  );
}

function EventRow({ event }: { event: TimelineEvent }) {
  const [expanded, setExpanded] = useState(event.severity !== "info");

  return (
    <div
      className={`border-l-2 pl-3 py-2 mb-1 rounded-r cursor-pointer hover:bg-white/5 transition-colors ${SEVERITY_STYLES[event.severity]}`}
      onClick={() => setExpanded((value) => !value)}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs text-slate-400 font-mono">{formatTimestamp(event.timestamp)}</span>
            <SeverityBadge severity={event.severity} />
            <span className="text-[10px] text-slate-500 uppercase tracking-wider">
              {CATEGORY_LABELS[event.category]}
            </span>
            {event.mission_id && (
              <span className="text-[10px] text-teal-500 font-mono bg-teal-900/30 px-1.5 rounded">
                {event.mission_id}
              </span>
            )}
          </div>
          <p className="text-sm text-slate-200 mt-0.5 truncate">{event.summary}</p>
        </div>
        <span className="text-xs text-slate-600 flex-shrink-0">{expanded ? "v" : ">"}</span>
      </div>

      {expanded && (
        <div className="mt-2 pl-6 text-xs space-y-1">
          <div className="text-slate-500">
            Topic: <span className="text-slate-300 font-mono">{event.topic}</span>
          </div>
          {event.receipt_id && (
            <div className="text-slate-500">
              Receipt: <span className="text-teal-400 font-mono">{event.receipt_id}</span>
            </div>
          )}
          <HashChainLink prev={event.prev_hash} current={event.hash_chain_ref} />
          {event.payload && (
            <pre className="text-slate-500 bg-slate-900/50 p-2 rounded text-[10px] mt-1 overflow-x-auto">
              {JSON.stringify(event.payload, null, 2)}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

export default function TerminalTimeline() {
  const [activeCategories, setActiveCategories] = useState<Set<EventCategory>>(
    new Set(["mission", "economy", "reflex", "constitutional", "auth", "system"]),
  );
  const [acknowledgedCriticals, setAcknowledgedCriticals] = useState<Set<string>>(
    () => loadAcknowledgedCriticals(),
  );
  const { events: liveStreamEvents, connected } = useTerminalStream(LIVE_TIMELINE_TOPICS, 100);
  const {
    acknowledge,
    acknowledgingId,
    error: acknowledgmentError,
    clearError: clearAcknowledgmentError,
  } = useCriticalEventAcknowledger();

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    window.sessionStorage.setItem(
      ACKNOWLEDGED_CRITICALS_KEY,
      JSON.stringify(Array.from(acknowledgedCriticals)),
    );
  }, [acknowledgedCriticals]);

  const events = liveStreamEvents
    .map(buildTimelineEvent)
    .sort((left, right) => timestampValue(right.timestamp) - timestampValue(left.timestamp));

  const filtered = events.filter((event) => activeCategories.has(event.category)).slice(0, 100);
  const proofAcknowledgedCriticals = new Set(
    events
      .filter((event) => event.topic === "critical.acknowledged")
      .map((event) => payloadString(event.payload ?? {}, "acknowledged_event_hash"))
      .filter((value) => value.length > 0),
  );

  const grouped = new Map<string, TimelineEvent[]>();
  const ungrouped: TimelineEvent[] = [];
  for (const event of filtered) {
    if (event.mission_id) {
      const group = grouped.get(event.mission_id) ?? [];
      group.push(event);
      grouped.set(event.mission_id, group);
    } else {
      ungrouped.push(event);
    }
  }

  const stickyCriticals = filtered.filter(
    (event) =>
      event.severity === "critical" &&
      !acknowledgedCriticals.has(event.id) &&
      !proofAcknowledgedCriticals.has(event.id),
  );

  const dateGroups = new Map<string, TimelineEvent[]>();
  for (const event of ungrouped) {
    const date = formatDate(event.timestamp);
    const group = dateGroups.get(date) ?? [];
    group.push(event);
    dateGroups.set(date, group);
  }

  return (
    <div className="p-4 max-w-3xl mx-auto">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-bold text-slate-100">Timeline</h2>
          <p className="text-xs text-slate-500">
            Living narrative - receipts, reflexes, and constitutional ticks
          </p>
          {stickyCriticals.length > 0 && (
            <p className="text-[10px] text-red-400 mt-1">
              {stickyCriticals.length} critical event
              {stickyCriticals.length === 1 ? "" : "s"} awaiting acknowledgment
            </p>
          )}
        </div>
        <div className="text-right">
          <div className={`text-[10px] font-bold uppercase tracking-wider ${connected ? "text-emerald-400" : "text-amber-400"}`}>
            {connected ? "Live Event Spine" : "Connecting Event Spine"}
          </div>
          <div className="text-xs text-slate-600">
            {filtered.length} events | {grouped.size} missions
          </div>
        </div>
      </div>

      <CategoryFilter
        active={activeCategories}
        onToggle={(category) => {
          clearAcknowledgmentError();
          setActiveCategories((previous) => {
            const next = new Set(previous);
            if (next.has(category)) {
              next.delete(category);
            } else {
              next.add(category);
            }
            return next;
          });
        }}
      />

      {acknowledgmentError && (
        <div className="mb-3 rounded-lg border border-red-700/40 bg-red-950/20 px-3 py-2 text-xs text-red-200">
          {acknowledgmentError}
        </div>
      )}

      {stickyCriticals.map((event) => (
        <StickyAlert
          key={event.id}
          event={event}
          acknowledging={acknowledgingId === event.id}
          onAcknowledge={async (criticalEvent) => {
            const receipt = await acknowledge({
              eventHash: criticalEvent.id,
              topic: criticalEvent.topic,
              summary: criticalEvent.summary,
              missionId: criticalEvent.mission_id,
              receiptId: criticalEvent.receipt_id,
            });
            if (receipt) {
              setAcknowledgedCriticals(
                (previous) => new Set([...previous, criticalEvent.id]),
              );
            }
          }}
        />
      ))}

      {Array.from(grouped.entries()).map(([missionId, missionEvents]) => (
        <div key={missionId} className="mb-4 border border-slate-700/50 rounded-lg overflow-hidden">
          <div className="bg-slate-800/60 px-3 py-1.5 flex items-center justify-between">
            <span className="text-xs text-teal-400 font-mono font-bold">{missionId}</span>
            <span className="text-[10px] text-slate-500">{missionEvents.length} events</span>
          </div>
          <div className="px-2 py-1">
            {missionEvents.map((event) => (
              <EventRow key={event.id} event={event} />
            ))}
          </div>
        </div>
      ))}

      {Array.from(dateGroups.entries()).map(([date, dateEvents]) => (
        <div key={date} className="mb-3">
          <div className="text-[10px] text-slate-600 uppercase tracking-wider mb-1 px-1">
            {date}
          </div>
          {dateEvents.map((event) => (
            <EventRow key={event.id} event={event} />
          ))}
        </div>
      ))}

      {filtered.length === 0 && (
        <div className="text-center py-12 text-slate-600">
          <p className="text-sm">Awaiting event spine history.</p>
        </div>
      )}

      <div className="text-center mt-6 text-[10px] text-slate-700">
        Streaming from /v1/stream with event-history replay and receipt-chain hashes.
      </div>
    </div>
  );
}
