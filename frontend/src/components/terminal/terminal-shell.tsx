"use client";

import { useState, useEffect, useCallback, lazy, Suspense } from "react";
import {
  useMemoryProfile,
  useSeedPotential,
  useSovereignHealth,
} from "@/hooks/use-sovereign-api";
import {
  TERMINAL_THEME,
  TERMINAL_VIEW_META,
  type TerminalViewId,
} from "./terminal-manifest";

// ─── Lazy-load views ────────────────────────────────────────────
const TerminalDashboard = lazy(() => import("./terminal-dashboard"));
const TerminalMission = lazy(() => import("./terminal-mission"));
const TerminalTimeline = lazy(() => import("./terminal-timeline"));
const TerminalMemory = lazy(() => import("./terminal-memory"));
const TerminalSkills = lazy(() => import("./terminal-skills"));
const TerminalNetwork = lazy(() => import("./terminal-network"));
const TerminalSettings = lazy(() => import("./terminal-settings"));

// ─── Types ──────────────────────────────────────────────────────

interface ViewDef {
  id: TerminalViewId;
  label: string;
  shortcut: string;
  emoji: string;
  accentHex: string;
  description: string;
  component: React.LazyExoticComponent<React.ComponentType<unknown>>;
}

// ─── Constants ──────────────────────────────────────────────────

const VIEW_COMPONENTS: Record<
  TerminalViewId,
  React.LazyExoticComponent<React.ComponentType<unknown>>
> = {
  dashboard: TerminalDashboard,
  mission: TerminalMission,
  timeline: TerminalTimeline,
  memory: TerminalMemory,
  skills: TerminalSkills,
  network: TerminalNetwork,
  settings: TerminalSettings,
};

const VIEWS: ViewDef[] = TERMINAL_VIEW_META.map((view) => ({
  ...view,
  component: VIEW_COMPONENTS[view.id],
}));

// ─── Loading Fallback ───────────────────────────────────────────

function ViewLoader() {
  return (
    <div className="flex items-center justify-center py-20">
      <div className="flex flex-col items-center gap-3">
        <div
          className="w-6 h-6 border-2 border-t-transparent rounded-full animate-spin"
          style={{ borderColor: TERMINAL_THEME.gold, borderTopColor: "transparent" }}
        />
        <span className="text-xs text-slate-500">Loading sovereign view...</span>
      </div>
    </div>
  );
}

// ─── Status Bar ─────────────────────────────────────────────────

function metricTone(value: number, good: number, warn: number): string {
  if (value >= good) {
    return "text-emerald-400";
  }
  if (value >= warn) {
    return "text-amber-400";
  }
  return "text-red-400";
}

function MetricPill({
  label,
  value,
  className,
}: {
  label: string;
  value: string;
  className?: string;
}) {
  return (
    <div
      className="rounded-full border px-3 py-1 text-[10px]"
      style={{ borderColor: TERMINAL_THEME.line, background: "rgba(255,255,255,0.03)" }}
    >
      <span className="text-slate-500">{label}</span>
      <span className={`ml-2 font-medium ${className ?? "text-slate-200"}`}>{value}</span>
    </div>
  );
}

function StatusBar() {
  const { data: health } = useSovereignHealth();
  const { data: potential } = useSeedPotential();
  const { data: memory } = useMemoryProfile();
  const [now, setNow] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), 30000);
    return () => clearInterval(timer);
  }, []);

  const isLive = health.live_status === "LIVE" || health.running === true;
  const ihsan = health.ihsan_score ?? potential.reward_ema ?? 0;
  const snr = health.snr_score ?? 0;
  const gini = health.gini ?? 0;
  const seed = health.wallet_snapshot?.seed ?? memory.briefing.wallet_snapshot.seed ?? 0;
  const bloom = health.wallet_snapshot?.bloom ?? memory.briefing.wallet_snapshot.bloom ?? 0;
  const tier = potential?.tier ?? "—";
  const heartbeat = health.tick_interval_s ?? 60;
  const lastMissionSummary =
    health.last_mission_summary ||
    memory.briefing.last_mission_summary ||
    "No mission receipt has been recorded yet.";
  const nextAction =
    memory.briefing.next_action_suggestion ||
    "Review the mission envelope, then execute the next bounded task.";

  return (
    <div
      className="border-b px-4 py-3"
      style={{
        borderColor: TERMINAL_THEME.line,
        background:
          "radial-gradient(circle at top left, rgba(201,169,98,0.12), transparent 28%), " +
          "linear-gradient(180deg, rgba(3,8,16,0.98), rgba(8,18,31,0.94))",
      }}
    >
      <div className="flex flex-col gap-3 xl:flex-row xl:items-start xl:justify-between">
        <div className="flex flex-wrap items-center gap-2">
          <div
            className="rounded-full border px-3 py-1.5"
            style={{
              borderColor: `${TERMINAL_THEME.gold}50`,
              background: `${TERMINAL_THEME.gold}12`,
            }}
          >
            <div className="flex items-center gap-2">
              <span
                className="text-[11px] font-semibold tracking-[0.35em]"
                style={{ color: TERMINAL_THEME.goldBright }}
              >
                BIZRA
              </span>
              <span className="text-[10px] text-slate-500">
                sovereign mission terminal
              </span>
            </div>
          </div>
          <MetricPill
            label="status"
            value={isLive ? "LIVE" : "OFFLINE"}
            className={isLive ? "text-emerald-400" : "text-red-400"}
          />
          <MetricPill
            label="Ihsan"
            value={ihsan.toFixed(2)}
            className={metricTone(ihsan, 0.95, 0.85)}
          />
          <MetricPill
            label="SNR"
            value={snr.toFixed(2)}
            className={metricTone(snr, 0.85, 0.75)}
          />
          <MetricPill
            label="Gini"
            value={gini.toFixed(2)}
            className={gini <= 0.35 ? "text-emerald-400" : "text-red-400"}
          />
          <MetricPill label="SEED" value={seed.toFixed(1)} className="text-amber-300" />
          <MetricPill label="BLOOM" value={bloom.toFixed(2)} className="text-sky-300" />
          <MetricPill label="tier" value={tier} className="text-slate-200" />
        </div>

        <div className="max-w-2xl xl:text-right">
          <div className="text-sm text-slate-100">{lastMissionSummary}</div>
          <div className="mt-1 text-xs text-slate-500">{nextAction}</div>
        </div>
      </div>

      <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-[10px]">
        <div className="flex flex-wrap items-center gap-2 text-slate-500">
          <span>Intent → Contract → Orchestrate → Receipt → Tick → Memory</span>
          <span className="text-slate-700">·</span>
          <span>heartbeat {heartbeat}s</span>
          <span className="text-slate-700">·</span>
          <span>{memory.briefing.active_project || "no active project"}</span>
        </div>
        <div className="flex items-center gap-2 text-slate-500">
          <span>{now.toLocaleDateString()}</span>
          <span className="text-slate-700">·</span>
          <span>
            {now.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })}
          </span>
        </div>
      </div>
    </div>
  );
}

// ─── Navigation ─────────────────────────────────────────────────

function NavBar({
  active,
  onSelect,
}: {
  active: TerminalViewId;
  onSelect: (id: TerminalViewId) => void;
}) {
  return (
    <nav
      className="flex items-center gap-1 overflow-x-auto border-b px-2 py-2"
      style={{
        borderColor: TERMINAL_THEME.line,
        background: "rgba(3,8,16,0.92)",
      }}
    >
      {VIEWS.map((v) => (
        <button
          key={v.id}
          onClick={() => onSelect(v.id)}
          className="flex items-center gap-2 whitespace-nowrap rounded-md border px-3 py-1.5 text-xs transition-all"
          style={{
            borderColor:
              active === v.id ? `${v.accentHex}55` : "rgba(255,255,255,0.06)",
            background:
              active === v.id ? `${v.accentHex}14` : "rgba(255,255,255,0.02)",
            color: active === v.id ? TERMINAL_THEME.text : TERMINAL_THEME.textDim,
          }}
          title={v.description}
        >
          <span>{v.emoji}</span>
          <span className="hidden md:inline">{v.label}</span>
          <span className="md:hidden">{v.shortcut}</span>
          <kbd
            className="ml-1 rounded px-1 text-[9px]"
            style={{
              background: "rgba(255,255,255,0.05)",
              color: active === v.id ? v.accentHex : "#64748B",
            }}
          >
            {v.shortcut}
          </kbd>
        </button>
      ))}
    </nav>
  );
}

// ─── Main Shell ─────────────────────────────────────────────────

export default function TerminalShell() {
  const [activeView, setActiveView] = useState<TerminalViewId>("dashboard");

  // Keyboard shortcuts (1-7)
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    // Don't capture when typing in inputs
    const tag = (e.target as HTMLElement)?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (e.ctrlKey || e.altKey || e.metaKey) return;

    const idx = parseInt(e.key) - 1;
    if (idx >= 0 && idx < VIEWS.length) {
      setActiveView(VIEWS[idx].id);
    }
  }, []);

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  const ActiveComponent = VIEWS.find((v) => v.id === activeView)?.component ?? TerminalDashboard;

  return (
    <div
      className="min-h-screen flex flex-col text-slate-100"
      style={{
        background:
          "radial-gradient(circle at top left, rgba(201,169,98,0.08), transparent 24%), " +
          "radial-gradient(circle at top right, rgba(59,130,246,0.08), transparent 24%), " +
          "linear-gradient(180deg, #030810, #08121F 35%, #030810 100%)",
      }}
    >
      {/* Status bar */}
      <StatusBar />

      {/* Navigation */}
      <NavBar active={activeView} onSelect={setActiveView} />

      {/* View content */}
      <main className="flex-1 overflow-y-auto">
        <Suspense fallback={<ViewLoader />}>
          <ActiveComponent />
        </Suspense>
      </main>

      {/* Footer */}
      <footer
        className="flex items-center justify-between border-t px-4 py-2 text-[9px] text-slate-600"
        style={{ borderColor: TERMINAL_THEME.line, background: "rgba(3,8,16,0.92)" }}
      >
        <span>Keys 1-7 switch views · Every visible state derives from receipts, memory, or the event spine</span>
        <span style={{ color: TERMINAL_THEME.goldDeep }}>
          One mission, one proof, remembered forever
        </span>
      </footer>
    </div>
  );
}
