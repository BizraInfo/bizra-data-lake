"use client";

import { useDEMAStore } from "@/lib/store";
import { cn } from "@/lib/utils";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Progress } from "@/components/ui/progress";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Fingerprint,
  ScrollText,
  Target,
  FileCheck2,
  SearchCheck,
  Gavel,
  Users,
  CheckCircle2,
  Circle,
  Clock,
  AlertTriangle,
  Shield,
  Lock,
  ArrowRight,
  Zap,
  Brain,
  Eye,
  Landmark,
  Cpu,
  GitCompareArrows,
  BookOpen,
} from "lucide-react";
import type { ADKAgentState } from "@/lib/types";

// ─── Primitives Config ───────────────────────────────────────────

const PRIMITIVES = [
  {
    name: "AgentIdentity",
    arabic: "هوية",
    english: "Identity",
    description: "Genesis-bound identity with charter hash",
    color: "text-trust",
    border: "border-l-trust",
    icon: Fingerprint,
  },
  {
    name: "Charter",
    arabic: "ميثاق",
    english: "Covenant",
    description: "Immutable covenant, hashed on creation",
    color: "text-receipt",
    border: "border-l-receipt",
    icon: ScrollText,
  },
  {
    name: "Mission",
    arabic: "مهمة",
    english: "Intent Envelope",
    description: "Budget-bounded intent envelope",
    color: "text-manifest",
    border: "border-l-manifest",
    icon: Target,
  },
  {
    name: "Receipt",
    arabic: "إيصال",
    english: "Proof Chain",
    description: "Tree node, cryptographically chained",
    color: "text-action",
    border: "border-l-action",
    icon: FileCheck2,
  },
  {
    name: "Evidence",
    arabic: "دليل",
    english: "Citation Proof",
    description: "Content-hashed, source-verified citations",
    color: "text-success",
    border: "border-l-success",
    icon: SearchCheck,
  },
  {
    name: "Verdict",
    arabic: "حكم",
    english: "Binary Pass/Block",
    description: "Exhaustive binary pass/block with reasons",
    color: "text-warning",
    border: "border-l-warning",
    icon: Gavel,
  },
  {
    name: "Council",
    arabic: "مجلس",
    english: "Topology",
    description: "PAT-7 and SAT-5 hardcoded topologies",
    color: "text-gap",
    border: "border-l-gap",
    icon: Users,
  },
];

// ─── Lifecycle Steps Config ──────────────────────────────────────

const LIFECYCLE_STEPS = [
  {
    step: "NIYYAH",
    arabic: "نية",
    english: "Intent",
    description: "Parse and validate the mission question. Establish scope and boundaries.",
    icon: Brain,
  },
  {
    step: "BAYYINAH",
    arabic: "بيان",
    english: "Evidence",
    description: "Gather evidence from verified sources. Content-hash every citation.",
    icon: SearchCheck,
  },
  {
    step: "HADD",
    arabic: "حد",
    english: "Boundary",
    description: "Check all evidence against constitutional boundaries and dignity limits.",
    icon: Shield,
  },
  {
    step: "AMANAH",
    arabic: "أمانة",
    english: "Trust",
    description: "Validate trust delegation chain. Obtain guardian approvals.",
    icon: Landmark,
  },
  {
    step: "THAMARA",
    arabic: "ثمرة",
    english: "Fruit",
    description: "Synthesize evidence into answer. Calculate Ihsan score.",
    icon: Zap,
  },
  {
    step: "IISAL",
    arabic: "إيصال",
    english: "Delivery",
    description: "Seal receipt chain. Cryptographically bind all outputs.",
    icon: FileCheck2,
  },
  {
    step: "RETROSPECTIVE",
    arabic: "مراجعة",
    english: "Finalization",
    description: "Final review: dignity check, Ihsan validation, mission seal.",
    icon: Eye,
  },
];

// ─── State Colors ────────────────────────────────────────────────

const STATE_COLORS: Record<ADKAgentState, { bg: string; text: string; dot: string }> = {
  draft: { bg: "bg-muted", text: "text-muted-foreground", dot: "bg-muted-foreground" },
  chartered: { bg: "bg-sky-500/15", text: "text-sky-600 dark:text-sky-400", dot: "bg-sky-500" },
  wired: { bg: "bg-amber-500/15", text: "text-amber-600 dark:text-amber-400", dot: "bg-amber-500" },
  exercised: { bg: "bg-emerald-500/15", text: "text-emerald-600 dark:text-emerald-400", dot: "bg-emerald-500" },
  sealed: { bg: "bg-success/15", text: "text-success", dot: "bg-success" },
  frozen: { bg: "bg-purple-500/15", text: "text-purple-600 dark:text-purple-400", dot: "bg-purple-500" },
};

// ─── SAT-5 Ghost Slots ──────────────────────────────────────────

const SAT_5_SLOTS = [
  { name: "Oracle-S", description: "Sovereign oracle for high-stakes queries" },
  { name: "Sentinel", description: "Constitutional boundary guardian" },
  { name: "Ledger", description: "Immutable receipt chain steward" },
  { name: "Conductor", description: "Multi-agent mission orchestrator" },
  { name: "Ambassador", description: "External system interface (wrapped)" },
];

// ═══════════════════════════════════════════════════════════════════
// ADK Factory Screen
// ═══════════════════════════════════════════════════════════════════

export function AdkFactoryScreen() {
  const {
    adkAgents,
    adkMissions,
    adkLifecycleTraces,
    adkMigrationCheckpoints,
    adkTestSuites,
    adkSchemaVersions,
  } = useDEMAStore();

  const completedTrace = adkLifecycleTraces[0];

  // Test suite totals
  const totalCurrent = adkTestSuites.reduce((s, t) => s + t.current, 0);
  const totalTarget = adkTestSuites.reduce((s, t) => s + t.target, 0);
  const totalPassing = adkTestSuites.reduce((s, t) => s + t.passing, 0);

  return (
    <div className="h-full overflow-y-auto dema-scrollbar p-6 space-y-8">
      {/* ─── Section 1: Header Bar ──────────────────────────── */}
      <section className="space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div className="space-y-1">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-receipt/10 flex items-center justify-center">
                <Cpu className="h-4.5 w-4.5 text-receipt" />
              </div>
              <div>
                <h1 className="text-lg font-bold tracking-tight">
                  BIZRA-ADK <span className="text-muted-foreground font-normal text-sm">v0.2.2</span>
                </h1>
                <p className="text-xs text-muted-foreground">
                  Agent Development Kit — Internal Factory
                </p>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2 flex-wrap">
            <Badge variant="outline" className="text-[10px] border-warning/40 text-warning font-mono">
              DRAFT — Pre-implementation
            </Badge>
            <Badge variant="outline" className="text-[10px] border-receipt/40 text-receipt font-mono">
              Codename: Bayyinah | Mizan
            </Badge>
            <Badge variant="outline" className="text-[10px] border-destructive/50 text-destructive font-mono">
              Constitution: HARD
            </Badge>
          </div>
        </div>
        <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
          <Lock className="h-3 w-3" />
          <span>Hard mode: no batching, no opt-out, no escape valve</span>
        </div>
        <Separator />
      </section>

      {/* ─── Section 2: 7 Primitives Grid ───────────────────── */}
      <section className="space-y-3">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold">7 Primitives</h2>
          <div className="flex-1 h-px bizra-separator" />
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
          {PRIMITIVES.map((p) => {
            const Icon = p.icon;
            return (
              <Card
                key={p.name}
                className={cn(
                  "border border-border/60 bg-card/50",
                  "border-l-2",
                  p.border,
                  "hover:border-border transition-colors"
                )}
              >
                <CardContent className="p-3 flex items-start gap-3">
                  <div className={cn("w-8 h-8 rounded-md flex items-center justify-center shrink-0", p.color, "bg-current/5")}>
                    <Icon className="h-4 w-4" />
                  </div>
                  <div className="min-w-0">
                    <div className="flex items-center gap-1.5">
                      <span className="text-xs font-semibold">{p.name}</span>
                      <span className="text-[10px] text-muted-foreground" dir="rtl">{p.arabic}</span>
                    </div>
                    <p className="text-[10px] text-muted-foreground mt-0.5">{p.english}</p>
                    <p className="text-[10px] text-muted-foreground/70 mt-0.5">{p.description}</p>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </section>

      {/* ─── Section 3: 7 Lifecycle Steps Pipeline ───────────── */}
      <section className="space-y-3">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold">Lifecycle Pipeline</h2>
          <Badge variant="outline" className="text-[9px] font-mono">
            {completedTrace ? "Mission: " + completedTrace.missionId.split("-").slice(0, 2).join("-") : "No trace"}
          </Badge>
          <div className="flex-1 h-px bizra-separator" />
        </div>
        <Card className="border-border/60 bg-card/50">
          <CardContent className="p-4">
            <div className="flex items-center gap-0 overflow-x-auto pb-2">
              {LIFECYCLE_STEPS.map((s, i) => {
                const traceStep = completedTrace?.steps.find((ts) => ts.step === s.step);
                const status = traceStep?.status || "pending";
                const Icon = s.icon;
                const isCompleted = status === "completed";
                const isActive = status === "active";
                const isPending = status === "pending";

                return (
                  <div key={s.step} className="flex items-center shrink-0">
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <div className="flex flex-col items-center gap-1.5 w-[72px] cursor-default">
                          <div
                            className={cn(
                              "w-9 h-9 rounded-full flex items-center justify-center border-2 transition-all",
                              isCompleted && "bg-receipt/20 border-receipt text-receipt",
                              isActive && "bg-receipt/20 border-receipt text-receipt adk-seal-glow",
                              isPending && "bg-muted/50 border-muted-foreground/30 text-muted-foreground"
                            )}
                          >
                            {isCompleted ? (
                              <CheckCircle2 className="h-4 w-4" />
                            ) : isActive ? (
                              <Icon className="h-4 w-4 animate-pulse" />
                            ) : (
                              <Circle className="h-4 w-4" />
                            )}
                          </div>
                          <div className="text-center">
                            <p className={cn(
                              "text-[10px] font-semibold truncate max-w-[68px]",
                              isCompleted && "text-receipt",
                              isActive && "text-receipt",
                              isPending && "text-muted-foreground"
                            )}>
                              {s.step.slice(0, 6)}
                            </p>
                            <p className="text-[8px] text-muted-foreground">{s.english}</p>
                          </div>
                          {traceStep?.duration && (
                            <p className="text-[8px] text-muted-foreground/60 font-mono">
                              {(traceStep.duration / 1000).toFixed(1)}s
                            </p>
                          )}
                        </div>
                      </TooltipTrigger>
                      <TooltipContent side="bottom" className="max-w-[200px] text-[11px]">
                        <p className="font-semibold">{s.step} — {s.english}</p>
                        <p className="text-muted-foreground mt-1">{s.description}</p>
                        {traceStep?.output && (
                          <p className="mt-1 text-muted-foreground/80 italic">"{traceStep.output}"</p>
                        )}
                      </TooltipContent>
                    </Tooltip>
                    {i < LIFECYCLE_STEPS.length - 1 && (
                      <div className={cn(
                        "w-6 h-0.5 mx-0.5",
                        isCompleted ? "bg-receipt/40" : "bg-muted-foreground/20"
                      )} />
                    )}
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      </section>

      {/* ─── Section 4: Council Topology (PAT-7 + SAT-5) ────── */}
      <section className="space-y-4">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold">Council Topology</h2>
          <div className="flex-1 h-px bizra-separator" />
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          {/* PAT-7 Council */}
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="text-[10px] font-mono border-receipt/40 text-receipt">
                PAT-7
              </Badge>
              <span className="text-xs text-muted-foreground">Process Agent Topology</span>
              <span className="text-[10px] text-muted-foreground/60">
                {adkAgents.filter((a) => a.council === "PAT-7").length} agents
              </span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              {adkAgents.map((agent) => {
                const sc = STATE_COLORS[agent.state];
                const testPct = agent.testCount > 0 ? (agent.testsPassing / agent.testCount) * 100 : 0;
                const hasFails = agent.testsPassing < agent.testCount;

                return (
                  <Card
                    key={agent.id}
                    className={cn(
                      "border border-border/60 bg-card/50 hover:border-border transition-colors",
                      agent.frozen && "border-purple-500/20"
                    )}
                  >
                    <CardContent className="p-3 space-y-2">
                      {/* Header */}
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <div className={cn("w-2 h-2 rounded-full shrink-0", sc.dot, agent.state === "exercised" && "adk-seal-glow")} />
                          <span className="text-xs font-semibold">{agent.name}</span>
                        </div>
                        <Badge variant="outline" className={cn("text-[9px] font-mono border-0", sc.bg, sc.text)}>
                          {agent.state}
                        </Badge>
                      </div>

                      {/* Model */}
                      <p className="text-[10px] text-muted-foreground font-mono">{agent.model}</p>

                      {/* Stats row */}
                      <div className="flex items-center gap-3 text-[10px]">
                        <span className="text-muted-foreground">
                          <span className="font-mono font-medium text-foreground">{agent.locCount}</span> LOC
                        </span>
                        <span className="text-muted-foreground">
                          <span className={cn("font-mono font-medium", hasFails ? "text-warning" : "text-success")}>
                            {agent.testsPassing}
                          </span>/{agent.testCount} tests
                        </span>
                        {agent.lastLoopProofAt && (
                          <span className="text-muted-foreground flex items-center gap-0.5">
                            <CheckCircle2 className="h-2.5 w-2.5 text-success" />
                            loop
                          </span>
                        )}
                      </div>

                      {/* Test bar */}
                      {agent.testCount > 0 && (
                        <div className="w-full h-1 rounded-full bg-muted overflow-hidden">
                          <div
                            className={cn(
                              "h-full rounded-full transition-all",
                              testPct === 100 ? "bg-success" : hasFails ? "bg-warning" : "bg-muted-foreground"
                            )}
                            style={{ width: `${testPct}%` }}
                          />
                        </div>
                      )}

                      {/* Charter excerpt */}
                      <p className="text-[9px] text-muted-foreground/70 italic line-clamp-2">
                        &ldquo;{agent.charterText.slice(0, 80)}{agent.charterText.length > 80 ? "..." : ""}&rdquo;
                      </p>

                      {/* Tools */}
                      {agent.tools.length > 0 && (
                        <div className="flex flex-wrap gap-1">
                          {agent.tools.map((t) => (
                            <Badge
                              key={t.id}
                              variant="outline"
                              className="text-[8px] font-mono px-1.5 py-0 border-border/40"
                            >
                              {t.source === "local_corpus" ? (
                                <BookOpen className="h-2.5 w-2.5 mr-0.5" />
                              ) : (
                                <GitCompareArrows className="h-2.5 w-2.5 mr-0.5" />
                              )}
                              {t.name}
                            </Badge>
                          ))}
                        </div>
                      )}
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </div>

          {/* SAT-5 Council */}
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="text-[10px] font-mono border-manifest/40 text-manifest">
                SAT-5
              </Badge>
              <span className="text-xs text-muted-foreground">Sovereign Agent Topology</span>
              <span className="text-[10px] text-muted-foreground/60">5 agents</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              {SAT_5_SLOTS.map((slot) => (
                <Card
                  key={slot.name}
                  className="border border-dashed border-muted-foreground/20 bg-muted/20 opacity-60"
                >
                  <CardContent className="p-3 space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-muted-foreground/30" />
                        <span className="text-xs font-semibold text-muted-foreground">{slot.name}</span>
                      </div>
                      <Badge variant="outline" className="text-[9px] font-mono text-muted-foreground border-0 bg-muted/50">
                        Not yet wired
                      </Badge>
                    </div>
                    <p className="text-[10px] text-muted-foreground/60">{slot.description}</p>
                    <div className="w-full h-1 rounded-full bg-muted" />
                  </CardContent>
                </Card>
              ))}
              {/* Extra ghost card for visual balance */}
              <Card className="border border-dashed border-muted-foreground/20 bg-muted/20 opacity-40 sm:hidden xl:block">
                <CardContent className="p-3 flex items-center justify-center h-full">
                  <p className="text-[10px] text-muted-foreground/40 font-mono">SAT-5 Reserved</p>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </section>

      {/* ─── Section 5: Migration Roadmap ─────────────────────── */}
      <section className="space-y-3">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold">Migration Roadmap</h2>
          <Badge variant="outline" className="text-[9px] font-mono">
            {adkMigrationCheckpoints.filter((p) => p.status === "completed").length}/{adkMigrationCheckpoints.length} phases
          </Badge>
          <div className="flex-1 h-px bizra-separator" />
        </div>

        <div className="relative space-y-0">
          {adkMigrationCheckpoints.map((checkpoint, i) => {
            const isCompleted = checkpoint.status === "completed";
            const isInProgress = checkpoint.status === "in_progress";
            const isPending = checkpoint.status === "pending";
            const isLast = i === adkMigrationCheckpoints.length - 1;

            return (
              <div key={checkpoint.phase} className="relative flex gap-4">
                {/* Timeline line + dot */}
                <div className="flex flex-col items-center shrink-0 w-6">
                  <div
                    className={cn(
                      "w-3 h-3 rounded-full border-2 shrink-0 z-10",
                      isCompleted && "bg-receipt border-receipt",
                      isInProgress && "bg-warning border-warning adk-seal-glow",
                      isPending && "bg-transparent border-muted-foreground/30"
                    )}
                  />
                  {!isLast && (
                    <div className={cn(
                      "w-0.5 flex-1 min-h-[40px]",
                      isCompleted && "bg-receipt/30",
                      isInProgress && "bg-warning/20",
                      isPending && "bg-muted-foreground/15"
                    )} />
                  )}
                </div>

                {/* Phase card */}
                <Card
                  className={cn(
                    "mb-4 border bg-card/50 transition-colors",
                    isCompleted && "border-receipt/20",
                    isInProgress && "border-warning/30",
                    isPending && "border-border/40"
                  )}
                >
                  <CardContent className="p-3 space-y-2">
                    <div className="flex items-center justify-between flex-wrap gap-2">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-bold font-mono">
                          Phase {checkpoint.phase}
                        </span>
                        <span className="text-xs font-semibold">{checkpoint.title}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge
                          variant="outline"
                          className={cn(
                            "text-[9px] font-mono border-0",
                            isCompleted && "bg-receipt/15 text-receipt",
                            isInProgress && "bg-warning/15 text-warning",
                            isPending && "bg-muted text-muted-foreground"
                          )}
                        >
                          {isCompleted ? (
                            <><CheckCircle2 className="h-2.5 w-2.5 mr-0.5" /> Done</>
                          ) : isInProgress ? (
                            <><Clock className="h-2.5 w-2.5 mr-0.5" /> In Progress</>
                          ) : (
                            <><Circle className="h-2.5 w-2.5 mr-0.5" /> Pending</>
                          )}
                        </Badge>
                        <span className="text-[9px] text-muted-foreground font-mono">
                          {checkpoint.focusedDays}d · {checkpoint.calendarWeeks}
                        </span>
                      </div>
                    </div>

                    <p className="text-[10px] text-muted-foreground">{checkpoint.description}</p>

                    {/* Progress bar for in-progress phase */}
                    {isInProgress && (
                      <div className="space-y-1">
                        <div className="w-full h-1.5 rounded-full bg-muted overflow-hidden">
                          <div className="h-full rounded-full bg-warning adk-phase-progress" />
                        </div>
                      </div>
                    )}

                    {/* Gate */}
                    <div className="flex items-start gap-1.5 text-[10px]">
                      <Shield className="h-3 w-3 text-manifest shrink-0 mt-0.5" />
                      <span className="text-muted-foreground">
                        <span className="font-medium text-foreground/80">Gate:</span> {checkpoint.gate}
                      </span>
                    </div>

                    {/* Kill condition */}
                    {checkpoint.killCondition && (
                      <div className="flex items-start gap-1.5 text-[10px] rounded-md bg-destructive/5 border border-destructive/15 p-2">
                        <AlertTriangle className="h-3 w-3 text-destructive shrink-0 mt-0.5" />
                        <span className="text-destructive/90">
                          <span className="font-medium">Kill condition:</span> {checkpoint.killCondition}
                        </span>
                      </div>
                    )}

                    {/* Deliverables */}
                    <div className="flex flex-wrap gap-1.5 pt-1">
                      {checkpoint.deliverables.map((d) => (
                        <div
                          key={d.label}
                          className={cn(
                            "flex items-center gap-1 text-[9px] px-1.5 py-0.5 rounded",
                            d.done
                              ? "bg-receipt/10 text-receipt"
                              : "bg-muted/50 text-muted-foreground"
                          )}
                        >
                          {d.done ? (
                            <CheckCircle2 className="h-2.5 w-2.5" />
                          ) : (
                            <Circle className="h-2.5 w-2.5" />
                          )}
                          {d.label}
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            );
          })}
        </div>
      </section>

      {/* ─── Section 6: Test Coverage Dashboard ───────────────── */}
      <section className="space-y-3">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold">Test Coverage</h2>
          <Badge variant="outline" className="text-[9px] font-mono">
            {totalCurrent}/{totalTarget} ({Math.round((totalCurrent / totalTarget) * 100)}%)
          </Badge>
          <div className="flex-1 h-px bizra-separator" />
        </div>

        {/* Total bar */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between text-[10px]">
            <span className="text-muted-foreground">Total Passing</span>
            <span className="font-mono">
              <span className={cn("font-semibold", totalPassing === totalCurrent ? "text-success" : "text-warning")}>
                {totalPassing}
              </span>
              <span className="text-muted-foreground">/{totalCurrent}</span>
            </span>
          </div>
          <Progress value={(totalCurrent / totalTarget) * 100} className="h-2" />
        </div>

        {/* Categories */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
          {adkTestSuites.map((suite) => {
            const pct = suite.target > 0 ? (suite.current / suite.target) * 100 : 0;
            const allPassing = suite.current > 0 && suite.passing === suite.current;
            const isAdversarial = suite.category === "adversarial";

            return (
              <Card
                key={suite.category}
                className={cn(
                  "border bg-card/50",
                  isAdversarial && "border-destructive/30"
                )}
              >
                <CardContent className="p-3 space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-semibold font-mono uppercase">
                        {suite.category.replace("_", " ")}
                      </span>
                      {isAdversarial && (
                        <Badge variant="outline" className="text-[8px] text-destructive border-destructive/30 font-mono">
                          PROOF-OF-HARDNESS
                        </Badge>
                      )}
                    </div>
                    <span className="text-[10px] font-mono">
                      <span className={cn("font-medium", allPassing && suite.current > 0 ? "text-success" : suite.current > 0 ? "text-warning" : "text-muted-foreground")}>
                        {suite.current}
                      </span>
                      <span className="text-muted-foreground">/{suite.target}</span>
                    </span>
                  </div>
                  <div className={cn(
                    "w-full h-1.5 rounded-full bg-muted overflow-hidden",
                    isAdversarial && "bg-destructive/10"
                  )}>
                    <div
                      className={cn(
                        "h-full rounded-full transition-all",
                        pct === 100 ? "bg-success" : suite.current > 0 ? "bg-warning" : "bg-muted-foreground/30",
                        isAdversarial && pct > 0 && "bg-destructive"
                      )}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  {suite.passing < suite.current && suite.current > 0 && (
                    <p className="text-[9px] text-warning">
                      {suite.current - suite.passing} failing
                    </p>
                  )}
                  {suite.current === 0 && (
                    <p className="text-[9px] text-muted-foreground/60 italic">Not yet started</p>
                  )}
                </CardContent>
              </Card>
            );
          })}
        </div>
      </section>

      {/* ─── Section 7: Schema Sovereignty ────────────────────── */}
      <section className="space-y-3">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold">Schema Sovereignty</h2>
          <Badge variant="outline" className="text-[9px] font-mono border-success/40 text-success">
            DRIFT GATE: ACTIVE
          </Badge>
          <div className="flex-1 h-px bizra-separator" />
        </div>

        <Card className="border-border/60 bg-card/50">
          <CardContent className="p-4 space-y-3">
            <div className="flex items-center gap-2 text-[10px] text-success">
              <CheckCircle2 className="h-3.5 w-3.5" />
              <span>No drift detected — all mirrors in sync</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {adkSchemaVersions.map((sv) => {
                const isCanonical = sv.language === "CDDL";
                return (
                  <div
                    key={sv.language}
                    className={cn(
                      "rounded-lg border p-3 space-y-2",
                      isCanonical
                        ? "border-receipt/30 bg-receipt/5"
                        : "border-border/40 bg-muted/30"
                    )}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold font-mono">{sv.language}</span>
                      {isCanonical && (
                        <Badge className="text-[8px] bg-receipt/15 text-receipt border-0">
                          CANONICAL
                        </Badge>
                      )}
                    </div>
                    <p className="text-[10px] text-muted-foreground font-mono truncate">{sv.path}</p>
                    <div className="flex items-center gap-3 text-[9px] text-muted-foreground">
                      <span>v{sv.version}</span>
                      {sv.lastBumpedAt && (
                        <span>{new Date(sv.lastBumpedAt).toLocaleDateString()}</span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      </section>

      {/* ─── Active Missions ──────────────────────────────────── */}
      <section className="space-y-3">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold">Active Missions</h2>
          <Badge variant="outline" className="text-[9px] font-mono">
            {adkMissions.filter((m) => m.status !== "sealed").length} active
          </Badge>
          <div className="flex-1 h-px bizra-separator" />
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
          {adkMissions.map((mission) => {
            const isSealed = mission.status === "sealed";
            const isExecuting = mission.status === "executing";
            const statusColor = isSealed
              ? "bg-receipt/15 text-receipt"
              : isExecuting
                ? "bg-warning/15 text-warning"
                : "bg-muted text-muted-foreground";

            return (
              <Card
                key={mission.id}
                className={cn(
                  "border bg-card/50",
                  isSealed && "border-receipt/20 adk-seal-glow"
                )}
              >
                <CardContent className="p-3 space-y-2">
                  <div className="flex items-center justify-between">
                    <Badge variant="outline" className={cn("text-[9px] font-mono border-0", statusColor)}>
                      {mission.status.toUpperCase()}
                    </Badge>
                    {isSealed && (
                      <span className="text-[9px] text-success font-mono">Ihsan 0.97</span>
                    )}
                  </div>
                  <p className="text-[11px] font-medium line-clamp-2">{mission.question}</p>
                  <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
                    <ArrowRight className="h-2.5 w-2.5" />
                    <span>{mission.targetAgentName}</span>
                    <span className="text-muted-foreground/50">·</span>
                    <span className="font-mono">{mission.budget.maxTokens}tok</span>
                    <span className="text-muted-foreground/50">·</span>
                    <span className="font-mono">{mission.budget.maxWallSeconds}s</span>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </section>

      {/* Bottom spacer */}
      <div className="h-4" />
    </div>
  );
}
