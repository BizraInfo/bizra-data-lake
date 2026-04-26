"use client";

import { useSyncExternalStore, useEffect, useCallback } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useMissionStore } from "@/lib/mission-store";
import { useDEMAStore } from "@/lib/store";
import { TrustStrip } from "@/components/dema/trust-strip";
import { MissionComposer } from "@/components/dema/surfaces/mission-composer";
import { GateLadder } from "@/components/dema/surfaces/gate-ladder";
import { OrganizePreview } from "@/components/dema/surfaces/organize-preview";
import { ReceiptReveal } from "@/components/dema/surfaces/receipt-reveal";
import { MemoryConstellation } from "@/components/dema/surfaces/memory-constellation";
import { RejectRemediation } from "@/components/dema/surfaces/reject-remediation";
import { cn } from "@/lib/utils";
import {
  Sparkles,
  Brain,
  Database,
  BarChart3,
  History,
  Server,
  User,
  X,
} from "lucide-react";

const emptySubscribe = () => () => {};
function useMounted() {
  return useSyncExternalStore(
    emptySubscribe,
    () => true,
    () => false
  );
}

// ─── Stage Progress Indicator ──────────────────────────
function StageProgress() {
  const { currentStage, stageTransitions } = useMissionStore();
  const stages = [
    { key: "intent", label: "Intent", short: "1" },
    { key: "admissibility", label: "Gates", short: "2" },
    { key: "action", label: "Action", short: "3" },
    { key: "confirmation", label: "Confirm", short: "4" },
    { key: "receipt", label: "Sealed", short: "5" },
  ] as const;

  const stageKeys = stages.map((s) => s.key);
  const currentIndex = stageKeys.indexOf(currentStage as typeof stageKeys[number]);

  return (
    <div className="flex items-center gap-1">
      {stages.map((stage, i) => {
        const isActive = stage.key === currentStage;
        const isCompleted = currentIndex > i;
        const isBlocked = currentStage === "blocked" && i === 1;
        return (
          <div key={stage.key} className="flex items-center">
            <div className="flex flex-col items-center gap-1">
              <div
                className={cn(
                  "w-7 h-7 rounded-full flex items-center justify-center text-xs font-medium transition-all duration-500",
                  isActive && !isBlocked && "bg-trust text-trust-foreground shadow-[0_0_12px_oklch(0.78_0.14_75/30%)]",
                  isCompleted && "bg-success/20 text-success border border-success/40",
                  isBlocked && "bg-destructive/20 text-destructive border border-destructive/40",
                  !isActive && !isCompleted && !isBlocked && "bg-muted text-muted-foreground border border-border"
                )}
              >
                {isCompleted ? (
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                ) : isBlocked ? (
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="18" y1="6" x2="6" y2="18" />
                    <line x1="6" y1="6" x2="18" y2="18" />
                  </svg>
                ) : (
                  stage.short
                )}
              </div>
              <span
                className={cn(
                  "text-[10px] font-medium tracking-wide transition-colors duration-300 hidden sm:block",
                  isActive ? "text-trust" : isCompleted ? "text-success" : "text-muted-foreground"
                )}
              >
                {stage.label}
              </span>
            </div>
            {i < stages.length - 1 && (
              <div
                className={cn(
                  "w-6 sm:w-8 h-px mt-[-14px] sm:mt-[-14px] transition-colors duration-500",
                  currentIndex > i ? "bg-success/40" : "bg-border"
                )}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

// ─── Welcome Screen (Idle State) ───────────────────────
function WelcomeScreen() {
  const { trustState, receipts, resources } = useDEMAStore();
  const { missionHistory } = useMissionStore();

  const quickActions = [
    {
      label: "Organize Files",
      desc: "Arrange and manage your workspace",
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
        </svg>
      ),
    },
    {
      label: "Research",
      desc: "Deep analysis on any topic",
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="11" cy="11" r="8" />
          <path d="m21 21-4.3-4.3" />
        </svg>
      ),
    },
    {
      label: "Analyze",
      desc: "Break down complex data",
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 12h-4l-3 9L9 3l-3 9H3" />
        </svg>
      ),
    },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="flex flex-col items-center justify-center min-h-[60vh] px-4"
    >
      {/* Greeting */}
      <div className="text-center mb-10">
        <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-trust/10 mb-5">
          <Sparkles className="w-7 h-7 text-trust" />
        </div>
        <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight mb-2">
          Welcome back, {trustState.principalName}
        </h1>
        <p className="text-muted-foreground text-sm max-w-md mx-auto leading-relaxed">
          Dema is your sovereign operator face. Express an intent, and the right
          lawful surface will appear.
        </p>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-3 gap-3 mb-8 w-full max-w-sm">
        <div className="bg-card border border-border rounded-xl p-3 text-center">
          <div className="text-lg font-semibold text-foreground">{receipts.length}</div>
          <div className="text-[10px] text-muted-foreground uppercase tracking-wider mt-0.5">Receipts</div>
        </div>
        <div className="bg-card border border-border rounded-xl p-3 text-center">
          <div className="text-lg font-semibold text-foreground">{resources.length}</div>
          <div className="text-[10px] text-muted-foreground uppercase tracking-wider mt-0.5">Resources</div>
        </div>
        <div className="bg-card border border-border rounded-xl p-3 text-center">
          <div className="text-lg font-semibold text-foreground">{missionHistory.length}</div>
          <div className="text-[10px] text-muted-foreground uppercase tracking-wider mt-0.5">Missions</div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 w-full max-w-lg">
        {quickActions.map((action) => (
          <button
            key={action.label}
            className="group bg-card border border-border rounded-xl p-4 text-left hover:border-trust/30 hover:bg-trust/5 transition-all duration-200 cursor-pointer"
          >
            <div className="text-muted-foreground group-hover:text-trust transition-colors mb-2">
              {action.icon}
            </div>
            <div className="text-sm font-medium text-foreground">{action.label}</div>
            <div className="text-xs text-muted-foreground mt-0.5">{action.desc}</div>
          </button>
        ))}
      </div>

      {/* Core Principle */}
      <div className="mt-8 text-center">
        <p className="text-xs text-muted-foreground/60 tracking-wide">
          Every action produces a receipt. Every receipt binds to your trust chain.
        </p>
      </div>
    </motion.div>
  );
}

// ─── Main App Shell ────────────────────────────────────
function AppShell() {
  const {
    currentStage,
    memoryViewOpen,
    toggleMemoryView,
    getStageProgress,
  } = useMissionStore();
  const {
    isOnboarded,
    completeOnboarding,
    activationStatus,
    activationError,
  } = useDEMAStore();
  const mounted = useMounted();
  const activationPending = activationStatus === "activating";

  const progress = getStageProgress();
  const showStageProgress = currentStage !== "idle";

  // Keyboard shortcut: Escape to toggle memory
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape" && memoryViewOpen) {
        toggleMemoryView();
      }
    },
    [memoryViewOpen, toggleMemoryView]
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  if (!mounted) {
    return (
      <div className="h-screen flex items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-trust/10 flex items-center justify-center">
            <svg className="h-4 w-4 text-trust animate-spin" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
          </div>
          <span className="text-sm text-muted-foreground">Initializing DEMA...</span>
        </div>
      </div>
    );
  }

  if (!isOnboarded) {
    // Inline onboarding for first-time users
    return (
      <div className="h-screen flex items-center justify-center bg-background px-4">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          className="max-w-md w-full"
        >
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-trust/10 mb-4">
              <Sparkles className="w-8 h-8 text-trust" />
            </div>
            <h1 className="text-2xl font-semibold tracking-tight mb-2">DEMA</h1>
            <p className="text-sm text-muted-foreground leading-relaxed">
              Your sovereign operator face. One face, many lawful surfaces.
              Principal activation is only complete after the cognition gateway
              returns an authoritative envelope.
            </p>
          </div>
          <div className="space-y-3 mb-8">
            {[
              "Express intent, receive the right surface",
              "Every action produces a verifiable receipt",
              "Constitutional gates protect integrity",
              "Your data stays local, your trust grows",
            ].map((item, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -12 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3 + i * 0.1, duration: 0.3 }}
                className="flex items-center gap-3 bg-card border border-border rounded-lg px-4 py-3"
              >
                <div className="w-5 h-5 rounded-full bg-success/20 flex items-center justify-center flex-shrink-0">
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" className="text-success">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                </div>
                <span className="text-sm text-foreground">{item}</span>
              </motion.div>
            ))}
          </div>
          {activationError && (
            <div className="mb-4 rounded-lg border border-destructive/20 bg-destructive/5 px-4 py-3 text-sm text-destructive">
              {activationError}
            </div>
          )}
          <button
            onClick={() => {
              void completeOnboarding("Operator");
            }}
            disabled={activationPending}
            className="w-full h-11 rounded-lg bg-trust text-trust-foreground font-medium text-sm hover:opacity-90 transition-opacity cursor-pointer disabled:cursor-not-allowed disabled:opacity-60"
          >
            {activationPending ? "Activating..." : "Request activation"}
          </button>
        </motion.div>
      </div>
    );
  }

  // Determine which surface to show
  const renderSurface = () => {
    switch (currentStage) {
      case "idle":
        return <WelcomeScreen />;
      case "intent":
        return <MissionComposer />;
      case "admissibility":
        return <GateLadder />;
      case "action":
      case "confirmation":
        return <OrganizePreview />;
      case "receipt":
        return <ReceiptReveal />;
      case "blocked":
        return <RejectRemediation />;
      default:
        return <WelcomeScreen />;
    }
  };

  return (
    <div className="h-screen flex flex-col bg-background overflow-hidden">
      {/* Trust Strip — Always visible */}
      <TrustStrip />

      {/* Stage Progress Bar */}
      {showStageProgress && (
        <motion.div
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          className="flex items-center justify-center px-4 py-2.5 border-b border-border bg-card/50 backdrop-blur-sm"
        >
          <StageProgress />
        </motion.div>
      )}

      {/* Main Content Area */}
      <main className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden relative">
        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={currentStage}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.25, ease: "easeOut" }}
            className="max-w-2xl mx-auto px-4 py-6 sm:py-8"
          >
            {renderSurface()}
          </motion.div>
        </AnimatePresence>
      </main>

      {/* Bottom Bar */}
      <div className="border-t border-border bg-card/80 backdrop-blur-sm px-4 py-2 flex items-center justify-between safe-area-bottom">
        {/* Status indicator */}
        <div className="flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-success dema-pulse" />
          <span className="text-[11px] text-muted-foreground font-medium">
            {currentStage === "idle" ? "Ready" : progress.label}
          </span>
        </div>

        {/* Memory toggle */}
        <button
          onClick={toggleMemoryView}
          className={cn(
            "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all cursor-pointer",
            memoryViewOpen
              ? "bg-trust/10 text-trust border border-trust/20"
              : "text-muted-foreground hover:text-foreground hover:bg-muted"
          )}
        >
          {memoryViewOpen ? (
            <X className="w-3.5 h-3.5" />
          ) : (
            <Brain className="w-3.5 h-3.5" />
          )}
          <span>Memory</span>
        </button>

        {/* Version */}
        <span className="text-[10px] text-muted-foreground/50 font-mono">
          v0.2.2
        </span>
      </div>

      {/* Memory Constellation Overlay */}
      <AnimatePresence>
        {memoryViewOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-50 bg-background/95 backdrop-blur-md"
          >
            <MemoryConstellation />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function Home() {
  return <AppShell />;
}
