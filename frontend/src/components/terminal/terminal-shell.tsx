"use client";

import { useState, useEffect, useCallback, lazy, Suspense } from "react";
import { useSovereignHealth, useSeedPotential, useTokenBalance } from "@/hooks/use-sovereign-api";
import { TERMINAL_VIEW_META } from "./terminal-manifest";

// ─── Lazy-load views ────────────────────────────────────────────
const TerminalDashboard = lazy(() => import("./terminal-dashboard"));
const TerminalMission = lazy(() => import("./terminal-mission"));
const TerminalTimeline = lazy(() => import("./terminal-timeline"));
const TerminalMemory = lazy(() => import("./terminal-memory"));
const TerminalSkills = lazy(() => import("./terminal-skills"));
const TerminalNetwork = lazy(() => import("./terminal-network"));
const TerminalSettings = lazy(() => import("./terminal-settings"));
const OperatorCockpit = lazy(() => import("../OperatorCockpit"));

// ─── Types ──────────────────────────────────────────────────────

type ViewId = "dashboard" | "mission" | "timeline" | "memory" | "skills" | "network" | "settings" | "cockpit";

interface ViewDef {
  id: ViewId;
  label: string;
  shortcut: string;
  emoji: string;
  component: React.LazyExoticComponent<() => JSX.Element>;
}

// ─── Constants (canonical view metadata from terminal-manifest) ──

const LAZY_COMPONENTS: Record<ViewId, React.LazyExoticComponent<() => JSX.Element>> = {
  dashboard: TerminalDashboard,
  mission: TerminalMission,
  timeline: TerminalTimeline,
  memory: TerminalMemory,
  skills: TerminalSkills,
  network: TerminalNetwork,
  settings: TerminalSettings,
  cockpit: OperatorCockpit,
};

const VIEWS: ViewDef[] = TERMINAL_VIEW_META.map((meta) => ({
  id: meta.id as ViewId,
  label: meta.label,
  shortcut: meta.shortcut,
  emoji: meta.emoji,
  component: LAZY_COMPONENTS[meta.id as ViewId],
}));

// ─── Loading Fallback ───────────────────────────────────────────

function ViewLoader() {
  return (
    <div className="flex items-center justify-center py-20">
      <div className="flex flex-col items-center gap-3">
        <div className="w-6 h-6 border-2 border-teal-500 border-t-transparent rounded-full animate-spin" />
        <span className="text-xs text-slate-500">Loading view...</span>
      </div>
    </div>
  );
}

// ─── Status Bar ─────────────────────────────────────────────────

function StatusBar() {
  const { data: health } = useSovereignHealth();
  const { data: potential } = useSeedPotential();
  const { data: balance } = useTokenBalance();
  const [now, setNow] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), 30000);
    return () => clearInterval(timer);
  }, []);

  const isLive = !!health;
  const ihsan = potential?.sovereignty_score ?? 0;
  const seed = balance?.seed ?? 0;
  const tier = potential?.tier ?? "—";

  return (
    <div className="flex items-center justify-between px-4 py-1.5 bg-slate-900/80 border-b border-slate-800/50 text-[10px]">
      {/* Left: Status */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1">
          <span className={`w-1.5 h-1.5 rounded-full ${isLive ? "bg-emerald-500 animate-pulse" : "bg-red-500"}`} />
          <span className={isLive ? "text-emerald-400" : "text-red-400"}>
            {isLive ? "LIVE" : "OFFLINE"}
          </span>
        </div>

        <span className="text-slate-700">|</span>

        <span className={ihsan >= 0.95 ? "text-emerald-400" : ihsan >= 0.85 ? "text-amber-400" : "text-red-400"}>
          Ihsān {ihsan.toFixed(2)}
        </span>

        <span className="text-slate-700">|</span>

        <span className="text-amber-400">{seed.toFixed(1)} SEED</span>

        <span className="text-slate-700">|</span>

        <span className="text-slate-400">{tier}</span>
      </div>

      {/* Center: DEMA */}
      <span className="text-slate-600 hidden sm:block">
        💜 DEMA · بذرة · v3.0.0-GENESIS
      </span>

      {/* Right: Time */}
      <div className="flex items-center gap-2 text-slate-500">
        <span>♥ 60s</span>
        <span className="text-slate-700">|</span>
        <span>
          {now.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })}
        </span>
      </div>
    </div>
  );
}

// ─── Navigation ─────────────────────────────────────────────────

function NavBar({
  active,
  onSelect,
}: {
  active: ViewId;
  onSelect: (id: ViewId) => void;
}) {
  return (
    <nav className="flex items-center gap-0.5 px-2 py-1 bg-slate-950/60 border-b border-slate-800/30 overflow-x-auto">
      {VIEWS.map((v) => (
        <button
          key={v.id}
          onClick={() => onSelect(v.id)}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs transition-all whitespace-nowrap ${
            active === v.id
              ? "bg-slate-800 text-slate-100 font-medium"
              : "text-slate-500 hover:text-slate-300 hover:bg-slate-800/50"
          }`}
        >
          <span>{v.emoji}</span>
          <span className="hidden sm:inline">{v.label}</span>
          <kbd
            className={`ml-1 text-[9px] px-1 rounded ${
              active === v.id
                ? "bg-slate-700 text-teal-400"
                : "bg-slate-800/50 text-slate-600"
            }`}
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
  const [activeView, setActiveView] = useState<ViewId>("dashboard");

  // Keyboard shortcuts (1-7)
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    // Don't capture when typing in inputs
    const tag = (e.target as HTMLElement)?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

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
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
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
      <footer className="px-4 py-1 bg-slate-950 border-t border-slate-800/30 flex items-center justify-between text-[9px] text-slate-700">
        <span>Keys 1-7: switch views · Every action receipted · All data local</span>
        <span>One mission, one proof, remembered forever</span>
      </footer>
    </div>
  );
}
