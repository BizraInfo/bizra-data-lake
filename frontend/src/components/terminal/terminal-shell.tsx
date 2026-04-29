"use client";

import { useState, useEffect, useCallback, lazy, Suspense } from "react";
import { useSovereignHealth, useSeedPotential, useTokenBalance, useChainLatest } from "@/hooks/use-sovereign-api";
import { TERMINAL_VIEW_META } from "./terminal-manifest";

// ─── Lazy-load views ────────────────────────────────────────────
const TerminalDashboard = lazy(() => import("./terminal-dashboard"));
const TerminalNode0 = lazy(() => import("./terminal-node0"));
const TerminalMission = lazy(() => import("./terminal-mission"));
const TerminalTimeline = lazy(() => import("./terminal-timeline"));
const TerminalMemory = lazy(() => import("./terminal-memory"));
const TerminalSkills = lazy(() => import("./terminal-skills"));
const TerminalNetwork = lazy(() => import("./terminal-network"));
const TerminalSettings = lazy(() => import("./terminal-settings"));
const OperatorCockpit = lazy(() => import("../OperatorCockpit"));
const TerminalGoal = lazy(() => import("./terminal-goal"));

// ─── Types ──────────────────────────────────────────────────────

type ViewId = "node0" | "dashboard" | "mission" | "timeline" | "memory" | "skills" | "network" | "settings" | "cockpit" | "goal";

interface ViewDef {
  id: ViewId;
  label: string;
  shortcut: string;
  emoji: string;
  component: React.LazyExoticComponent<() => JSX.Element>;
}

// ─── Constants (canonical view metadata from terminal-manifest) ──

const LAZY_COMPONENTS: Record<ViewId, React.LazyExoticComponent<() => JSX.Element>> = {
  node0: TerminalNode0,
  dashboard: TerminalDashboard,
  mission: TerminalMission,
  timeline: TerminalTimeline,
  memory: TerminalMemory,
  skills: TerminalSkills,
  network: TerminalNetwork,
  settings: TerminalSettings,
  cockpit: OperatorCockpit,
  goal: TerminalGoal,
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
  const { data: health, error: healthError, loading: healthLoading } = useSovereignHealth();
  const { data: potential } = useSeedPotential();
  const { data: balance } = useTokenBalance();
  // Node0 Closure Sprint row 6 — trust_surface uses /v1/chain/latest as the
  // authoritative snapshot for both CHAIN and RECEIPT. This avoids cross-poll
  // drift and lets the UI distinguish genesis from receipt-detail failure.
  const {
    data: chainLatest,
    error: chainLatestError,
    loading: chainLatestLoading,
  } = useChainLatest();
  const [now, setNow] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), 30000);
    return () => clearInterval(timer);
  }, []);

  const isLive =
    !healthLoading && !healthError && (health.running || health.status === "healthy");
  const ihsan = potential?.sovereignty_score ?? 0;
  const seed = balance?.seed ?? 0;
  const tier = potential?.tier ?? "—";

  // CHAIN / RECEIPT truth surface comes from one authoritative snapshot.
  const chainSnapshotReady = !chainLatestLoading && !chainLatestError;
  const chainLive = chainSnapshotReady && chainLatest.head.length === 64;
  const chainHeadShort = chainLive ? chainLatest.head.slice(0, 8) : "—";
  const chainLength = chainSnapshotReady ? chainLatest.length : 0;

  // Row 6 enrichment — RECEIPT cell from /v1/chain/latest.
  // Honest absence taxonomy:
  // - chainLatestError => gateway unreachable
  // - chainLatest.length===0 or empty head => genesis / no receipts yet
  // - latestReceiptError => head exists but detail lookup failed
  // - latestReceipt===null with no error => no receipt detail available
  const latest = chainLatest.latestReceipt;
  const latestReceiptError = chainLatest.latestReceiptError;
  const chainAtGenesis =
    chainSnapshotReady &&
    (chainLatest.length === 0 ||
      !chainLatest.head ||
      chainLatest.head === "0".repeat(64));
  const receiptLookupFailed =
    chainSnapshotReady && !chainAtGenesis && latestReceiptError !== null;
  const receiptLive =
    chainSnapshotReady && !receiptLookupFailed && latest !== null;
  const receiptKind = receiptLive ? latest?.kind || "—" : "—";
  const receiptTimeLabel = (() => {
    if (!receiptLive || !latest || latest.timestamp === null) return "—";
    try {
      return new Date(latest.timestamp * 1000).toLocaleTimeString(undefined, {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });
    } catch {
      return "—";
    }
  })();

  // Row 6 enrichment — IHSĀN band from SovereignHealth.
  // Authoritative ihsan_score from /v1/health. Classify per the
  // two-band Ihsān policy (mission_nervous_system.py): >= 0.95 = ideal,
  // 0.85 <= score < 0.95 = warn, < 0.85 = halt.
  // When health is unavailable, band is "—" (honest absence).
  const authoritativeHealth = !healthLoading && !healthError;
  const healthIhsan =
    authoritativeHealth && typeof health.ihsan_score === "number"
      ? health.ihsan_score
      : null;
  const ihsanBand = (() => {
    if (healthIhsan === null) return { label: "—", color: "text-slate-600" };
    if (healthIhsan >= 0.95) return { label: "ideal", color: "text-emerald-400" };
    if (healthIhsan >= 0.85) return { label: "warn", color: "text-amber-400" };
    return { label: "halt", color: "text-red-400" };
  })();

  // Row 6 enrichment — Gini from SovereignHealth (ADL §14 threshold 0.35).
  const gini =
    authoritativeHealth && typeof health.gini === "number" ? health.gini : null;
  const giniColor =
    gini === null
      ? "text-slate-600"
      : gini <= 0.35
      ? "text-emerald-400"
      : "text-red-400";

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

        {/* IHSĀN band — authoritative health.ihsan_score classified into
            two-band policy: >= 0.95 ideal, >= 0.85 warn, < 0.85 halt.
            Matches backend halt trigger at mission_nervous_system.py. */}
        <span
          className={ihsanBand.color}
          data-testid="ihsan-band"
          title={
            healthIhsan !== null
              ? `Runtime Ihsān: ${healthIhsan.toFixed(3)} (band: ${ihsanBand.label})`
              : "Runtime Ihsān unavailable"
          }
        >
          IHSĀN:{ihsanBand.label}
        </span>

        <span className="text-slate-700">|</span>

        {/* CHAIN — authoritative head from Rust cognition-gateway via
            /v1/chain proxy. Shows "—" honestly when gateway is down. */}
         <span
           className={chainLive ? "text-cyan-400" : "text-slate-600"}
           data-testid="chain-status"
           title={
             chainLive
              ? `Chain head: ${chainLatest.head} (length: ${chainLatest.length})`
              : chainLatestLoading
              ? "Chain loading — awaiting authoritative snapshot"
              : chainLatestError
              ? "Chain unavailable — cognition-gateway unreachable"
              : chainAtGenesis
              ? "Chain at genesis — no receipts yet"
              : "Chain snapshot unavailable"
           }
         >
          CHAIN#{chainLength} {chainHeadShort}
        </span>

        <span className="text-slate-700">|</span>

        {/* RECEIPT — latest receipt kind + timestamp from /v1/chain/latest.
            Honest "—" when chain is at genesis or detail unavailable. */}
         <span
           className={receiptLive ? "text-violet-400" : "text-slate-600"}
           data-testid="receipt-status"
           title={
             receiptLive && latest !== null
               ? `Latest receipt: ${latest.kind} (id: ${latest.id.slice(0, 16)}…)`
              : chainLatestLoading
              ? "Receipt detail loading — awaiting authoritative snapshot"
              : chainLatestError
              ? "Receipt detail unavailable — gateway unreachable"
              : receiptLookupFailed
              ? "Receipt detail unavailable — upstream lookup failed"
              : chainAtGenesis
              ? "No receipts yet — chain at genesis"
              : "Receipt detail unavailable"
           }
         >
          RECEIPT:{receiptKind} {receiptTimeLabel}
        </span>

        <span className="text-slate-700">|</span>

        {/* GINI — ADL §14 justice invariant, authoritative from /v1/health. */}
        <span
          className={giniColor}
          data-testid="gini-status"
          title={
            gini !== null
              ? `ADL Gini: ${gini.toFixed(3)} (threshold: 0.35)`
              : "Gini unavailable"
          }
        >
          GINI:{gini !== null ? gini.toFixed(2) : "—"}
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
  const [activeView, setActiveView] = useState<ViewId>("node0");

  // Keyboard shortcuts use the manifest's explicit shortcut field.
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    // Don't capture when typing in inputs
    const tag = (e.target as HTMLElement)?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

    const view = VIEWS.find((candidate) => candidate.shortcut === e.key);
    if (view) {
      setActiveView(view.id);
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
        <span>Keys 0-9: switch views · Every action receipted · All data local</span>
        <span>One mission, one proof, remembered forever</span>
      </footer>
    </div>
  );
}
