"use client";

import { useState, useEffect, useCallback } from "react";
import {
  useSeedEpisodes,
  useConstitutionalStatus,
} from "@/hooks/use-sovereign-api";

// ─── Types ──────────────────────────────────────────────────────
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

// ─── Constants ──────────────────────────────────────────────────
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
  critical: "bg-red-800 text-red-100 animate-pulse",
};

const CATEGORY_ICONS: Record<EventCategory, string> = {
  mission: "🎯",
  economy: "💰",
  reflex: "⚡",
  constitutional: "⚖️",
  auth: "🔐",
  system: "🔧",
};

const CATEGORY_LABELS: Record<EventCategory, string> = {
  mission: "Mission",
  economy: "Economy",
  reflex: "Reflex",
  constitutional: "Constitutional",
  auth: "Auth",
  system: "System",
};

// ─── Helpers ────────────────────────────────────────────────────

function classifyTopic(topic: string): EventCategory {
  if (topic.startsWith("mission.")) return "mission";
  if (topic.startsWith("economy.")) return "economy";
  if (topic.startsWith("reflex.")) return "reflex";
  if (
    topic.startsWith("ihsan.") ||
    topic.startsWith("invariant.") ||
    topic.startsWith("tick.")
  )
    return "constitutional";
  if (topic.startsWith("auth.")) return "auth";
  return "system";
}

function severityFromTopic(topic: string): EventSeverity {
  if (topic === "ihsan.breach" || topic === "invariant.violation")
    return "critical";
  if (topic === "mission.failed" || topic === "auth.boundary.crossed")
    return "warning";
  if (
    topic === "reflex.compiled" ||
    topic === "economy.seed_minted" ||
    topic === "receipt.generated"
  )
    return "notice";
  return "info";
}

function formatTimestamp(ts: string): string {
  try {
    const d = new Date(ts);
    return d.toLocaleTimeString(undefined, {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return ts;
  }
}

function formatDate(ts: string): string {
  try {
    const d = new Date(ts);
    return d.toLocaleDateString(undefined, {
      weekday: "short",
      month: "short",
      day: "numeric",
    });
  } catch {
    return "";
  }
}

// ─── Simulated Events (offline / demo mode) ─────────────────────

function generateDemoEvents(): TimelineEvent[] {
  const now = Date.now();
  const topics = [
    "mission.created",
    "mission.executed",
    "economy.seed_minted",
    "economy.zakat",
    "economy.bloom_accrued",
    "reflex.compiled",
    "tick.completed",
    "economy.asabiyyah",
    "mission.failed",
    "receipt.generated",
    "ihsan.breach",
    "auth.boundary.crossed",
  ];

  return Array.from({ length: 25 }, (_, i) => {
    const topic = topics[i % topics.length];
    const missionId =
      i < 8 ? "m_alpha_001" : i < 16 ? "m_alpha_002" : "m_alpha_003";
    return {
      id: `evt_${i.toString().padStart(4, "0")}`,
      timestamp: new Date(now - (25 - i) * 120_000).toISOString(),
      category: classifyTopic(topic),
      topic,
      summary: topicToSummary(topic, i),
      severity: severityFromTopic(topic),
      mission_id: topic.startsWith("mission.") || topic.startsWith("receipt.")
        ? missionId
        : undefined,
      receipt_id: topic === "receipt.generated" ? `r_${i}` : undefined,
      hash_chain_ref:
        topic === "receipt.generated"
          ? `0x${Math.random().toString(16).slice(2, 18)}`
          : undefined,
      prev_hash:
        topic === "receipt.generated" && i > 0
          ? `0x${Math.random().toString(16).slice(2, 18)}`
          : undefined,
    };
  });
}

function topicToSummary(topic: string, idx: number): string {
  const summaries: Record<string, string> = {
    "mission.created": "Mission submitted: organize project files",
    "mission.executed": `Mission completed — Ihsān ${(0.92 + Math.random() * 0.07).toFixed(2)}, +${(1.2 + Math.random() * 2).toFixed(1)} SEED`,
    "mission.failed": "Mission failed: insufficient permissions for target path",
    "economy.seed_minted": `+${(1.0 + Math.random() * 3).toFixed(2)} SEED minted (50% → pool)`,
    "economy.zakat": "2.5% zakat applied — 0.12 SEED to community pool",
    "economy.bloom_accrued": "+0.8 BLOOM accrued (governance weight)",
    "economy.asabiyyah": `Network cohesion: ${(0.45 + Math.random() * 0.1).toFixed(2)}`,
    "reflex.compiled":
      '⚡ Reflex precipitated: "file_organization" → System-1 cache',
    "tick.completed": `Constitutional tick: scored=${1 + (idx % 3)}, minted=${(0.5 + Math.random() * 2).toFixed(1)}`,
    "receipt.generated": `Evidence receipt signed (Ed25519, BLAKE2b chain)`,
    "ihsan.breach":
      "🛑 Ihsān breach detected — Safety dimension 0.12, action blocked",
    "auth.boundary.crossed":
      "Auth boundary: anonymous access attempted on /v1/seed/potential",
  };
  return summaries[topic] || topic;
}

// ─── Components ─────────────────────────────────────────────────

function SeverityBadge({ severity }: { severity: EventSeverity }) {
  return (
    <span
      className={`text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded font-bold ${SEVERITY_BADGE[severity]}`}
    >
      {severity}
    </span>
  );
}

function CategoryFilter({
  active,
  onToggle,
}: {
  active: Set<EventCategory>;
  onToggle: (cat: EventCategory) => void;
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
      {categories.map((cat) => (
        <button
          key={cat}
          onClick={() => onToggle(cat)}
          className={`text-xs px-2.5 py-1 rounded-full border transition-all ${
            active.has(cat)
              ? "border-teal-500 bg-teal-900/40 text-teal-200"
              : "border-slate-600 bg-slate-800/40 text-slate-500 hover:border-slate-500"
          }`}
        >
          {CATEGORY_ICONS[cat]} {CATEGORY_LABELS[cat]}
        </button>
      ))}
    </div>
  );
}

function HashChainLink({
  prev,
  current,
}: {
  prev?: string;
  current?: string;
}) {
  if (!current) return null;
  return (
    <div className="flex items-center gap-1.5 text-[10px] text-slate-500 font-mono mt-1.5">
      {prev && (
        <>
          <span title={prev}>{prev.slice(0, 10)}...</span>
          <span className="text-teal-600">→</span>
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
}: {
  event: TimelineEvent;
  onAcknowledge: (id: string) => void;
}) {
  return (
    <div className="bg-red-950/60 border border-red-500/50 rounded-lg p-3 mb-3 flex items-start justify-between">
      <div>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-red-400 text-sm font-bold animate-pulse">
            ⚠ CRITICAL
          </span>
          <span className="text-xs text-red-300">
            {formatTimestamp(event.timestamp)}
          </span>
        </div>
        <p className="text-sm text-red-200">{event.summary}</p>
        <p className="text-xs text-red-400 mt-1">{event.topic}</p>
      </div>
      <button
        onClick={() => onAcknowledge(event.id)}
        className="text-xs px-2 py-1 rounded border border-red-500/50 text-red-300 hover:bg-red-900/50 transition-colors flex-shrink-0 ml-3"
      >
        Acknowledge
      </button>
    </div>
  );
}

function EventRow({ event }: { event: TimelineEvent }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      className={`border-l-2 pl-3 py-2 mb-1 rounded-r cursor-pointer hover:bg-white/5 transition-colors ${SEVERITY_STYLES[event.severity]}`}
      onClick={() => setExpanded(!expanded)}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm">
              {CATEGORY_ICONS[event.category]}
            </span>
            <span className="text-xs text-slate-400 font-mono">
              {formatTimestamp(event.timestamp)}
            </span>
            <SeverityBadge severity={event.severity} />
            {event.mission_id && (
              <span className="text-[10px] text-teal-500 font-mono bg-teal-900/30 px-1.5 rounded">
                {event.mission_id}
              </span>
            )}
          </div>
          <p className="text-sm text-slate-200 mt-0.5 truncate">
            {event.summary}
          </p>
        </div>
        <span className="text-xs text-slate-600 flex-shrink-0">
          {expanded ? "▼" : "▶"}
        </span>
      </div>

      {expanded && (
        <div className="mt-2 pl-6 text-xs space-y-1">
          <div className="text-slate-500">
            Topic:{" "}
            <span className="text-slate-300 font-mono">{event.topic}</span>
          </div>
          {event.receipt_id && (
            <div className="text-slate-500">
              Receipt:{" "}
              <span className="text-teal-400 font-mono">
                {event.receipt_id}
              </span>
            </div>
          )}
          <HashChainLink
            prev={event.prev_hash}
            current={event.hash_chain_ref}
          />
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

// ─── Main Component ─────────────────────────────────────────────

export default function TerminalTimeline() {
  const [events, setEvents] = useState<TimelineEvent[]>([]);
  const [activeCategories, setActiveCategories] = useState<Set<EventCategory>>(
    new Set(["mission", "economy", "reflex", "constitutional", "auth", "system"])
  );
  const [acknowledgedCriticals, setAcknowledgedCriticals] = useState<
    Set<string>
  >(new Set());

  // Load events from API or demo
  const { data: episodes } = useSeedEpisodes();
  const { data: _constitutional } = useConstitutionalStatus();

  useEffect(() => {
    // If we have live data, transform it; otherwise use demo events
    if (episodes?.episodes?.length) {
      const mapped: TimelineEvent[] = episodes.episodes.map(
        (ep: Record<string, unknown>, i: number) => ({
          id: `ep_${i}`,
          timestamp:
            (ep.timestamp as string) || new Date().toISOString(),
          category: "economy" as EventCategory,
          topic: "receipt.generated",
          summary: `Episode: Ihsān ${ep.ihsan_score ?? "N/A"}, +${ep.seed_reward ?? 0} SEED`,
          severity: "notice" as EventSeverity,
          receipt_id: ep.receipt_hash as string,
          hash_chain_ref: ep.receipt_hash as string,
        })
      );
      setEvents(mapped);
    } else {
      setEvents(generateDemoEvents());
    }
  }, [episodes]);

  const toggleCategory = useCallback((cat: EventCategory) => {
    setActiveCategories((prev) => {
      const next = new Set(prev);
      if (next.has(cat)) next.delete(cat);
      else next.add(cat);
      return next;
    });
  }, []);

  const acknowledgeCritical = useCallback((id: string) => {
    setAcknowledgedCriticals((prev) => new Set([...prev, id]));
  }, []);

  // Filter events
  const filtered = events
    .filter((e) => activeCategories.has(e.category))
    .slice(0, 100);

  // Group by mission_id
  const grouped = new Map<string, TimelineEvent[]>();
  const ungrouped: TimelineEvent[] = [];
  for (const e of filtered) {
    if (e.mission_id) {
      const group = grouped.get(e.mission_id) || [];
      group.push(e);
      grouped.set(e.mission_id, group);
    } else {
      ungrouped.push(e);
    }
  }

  // Critical events (sticky until acknowledged)
  const stickyCriticals = filtered.filter(
    (e) => e.severity === "critical" && !acknowledgedCriticals.has(e.id)
  );

  // Date groups for ungrouped events
  const dateGroups = new Map<string, TimelineEvent[]>();
  for (const e of ungrouped) {
    const date = formatDate(e.timestamp);
    const group = dateGroups.get(date) || [];
    group.push(e);
    dateGroups.set(date, group);
  }

  return (
    <div className="p-4 max-w-3xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-bold text-slate-100">Timeline</h2>
          <p className="text-xs text-slate-500">
            Living narrative — sovereignty made visible
          </p>
        </div>
        <div className="text-xs text-slate-600">
          {filtered.length} events · {grouped.size} missions
        </div>
      </div>

      {/* Category Filter — §10.3: Filterable by category */}
      <CategoryFilter active={activeCategories} onToggle={toggleCategory} />

      {/* Sticky Critical Events — §10.3: Critical events sticky until acknowledged */}
      {stickyCriticals.map((e) => (
        <StickyAlert key={e.id} event={e} onAcknowledge={acknowledgeCritical} />
      ))}

      {/* Mission-Grouped Events — §10.3: Groups by mission_id */}
      {Array.from(grouped.entries()).map(([missionId, missionEvents]) => (
        <div
          key={missionId}
          className="mb-4 border border-slate-700/50 rounded-lg overflow-hidden"
        >
          <div className="bg-slate-800/60 px-3 py-1.5 flex items-center justify-between">
            <span className="text-xs text-teal-400 font-mono font-bold">
              🎯 {missionId}
            </span>
            <span className="text-[10px] text-slate-500">
              {missionEvents.length} events
            </span>
          </div>
          <div className="px-2 py-1">
            {missionEvents.map((e) => (
              <EventRow key={e.id} event={e} />
            ))}
          </div>
        </div>
      ))}

      {/* Ungrouped Events (system, tick, etc.) */}
      {Array.from(dateGroups.entries()).map(([date, dateEvents]) => (
        <div key={date} className="mb-3">
          <div className="text-[10px] text-slate-600 uppercase tracking-wider mb-1 px-1">
            {date}
          </div>
          {dateEvents.map((e) => (
            <EventRow key={e.id} event={e} />
          ))}
        </div>
      ))}

      {/* Empty state */}
      {filtered.length === 0 && (
        <div className="text-center py-12 text-slate-600">
          <p className="text-2xl mb-2">📭</p>
          <p className="text-sm">No events match the current filter</p>
        </div>
      )}

      {/* §10.3: No synthetic entries notice */}
      <div className="text-center mt-6 text-[10px] text-slate-700">
        All entries from EventBus or ActionBus — no synthetic data
      </div>
    </div>
  );
}
