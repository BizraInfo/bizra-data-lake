"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useMissionStore } from "@/lib/mission-store";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Shield, Check, X, Loader2, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";
import type { GateEvaluation, GateStatus } from "@/lib/types";

const GATE_ICONS: Record<string, React.ElementType> = {
  ZANN_ZERO: Shield,
  CLAIM_MUST_BIND: Shield,
  RIBA_ZERO: Shield,
  NO_SHADOW_STATE: Shield,
  IHSAN_FLOOR: Shield,
};

function GateStatusIndicator({ status }: { status: GateStatus }) {
  return (
    <div className="flex items-center justify-center w-8 h-8 shrink-0">
      <AnimatePresence mode="wait">
        {status === "pending" && (
          <motion.div
            key="pending"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            className="w-2.5 h-2.5 rounded-full bg-muted-foreground/30"
          />
        )}
        {status === "evaluating" && (
          <motion.div
            key="evaluating"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
          >
            <Loader2 className="h-5 w-5 text-warning animate-spin" />
          </motion.div>
        )}
        {status === "passed" && (
          <motion.div
            key="passed"
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            transition={{ type: "spring", stiffness: 300, damping: 20 }}
            className="relative"
          >
            <div className="absolute inset-0 rounded-full bg-success/20 animate-ping" />
            <div className="relative w-5 h-5 rounded-full bg-success/20 flex items-center justify-center">
              <Check className="h-3 w-3 text-success" strokeWidth={3} />
            </div>
          </motion.div>
        )}
        {status === "blocked" && (
          <motion.div
            key="blocked"
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1, x: [0, -3, 3, -2, 2, 0] }}
            exit={{ opacity: 0, scale: 0.8 }}
            transition={{
              opacity: { duration: 0.2 },
              x: { duration: 0.4, delay: 0.1 },
            }}
            className="w-5 h-5 rounded-full bg-destructive/20 flex items-center justify-center"
          >
            <X className="h-3 w-3 text-destructive" strokeWidth={3} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function GateRow({ gate, index }: { gate: GateEvaluation; index: number }) {
  const Icon = GATE_ICONS[gate.id] || Shield;
  const isPassed = gate.status === "passed";
  const isBlocked = gate.status === "blocked";
  const isEvaluating = gate.status === "evaluating";

  return (
    <motion.div
      initial={{ opacity: 0, x: -12 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3, delay: index * 0.08 }}
      className={cn(
        "relative flex items-start gap-4 p-4 rounded-lg border transition-colors",
        isPassed && "border-success/20 bg-success/[0.03]",
        isBlocked && "border-destructive/20 bg-destructive/[0.03]",
        isEvaluating && "border-warning/20 bg-warning/[0.03]",
        gate.status === "pending" && "border-border/40 bg-transparent"
      )}
    >
      {/* Vertical connector line */}
      {index < 4 && (
        <div className="absolute left-[27px] top-[52px] bottom-[-12px] w-px bg-border/40" />
      )}

      {/* Status indicator */}
      <GateStatusIndicator status={gate.status} />

      {/* Gate info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <Icon className={cn("h-3.5 w-3.5 shrink-0", isPassed ? "text-success/60" : isBlocked ? "text-destructive/60" : isEvaluating ? "text-warning/60" : "text-muted-foreground/40")} />
          <span className="text-xs font-mono font-semibold tracking-wider">
            {gate.id}
          </span>
        </div>
        <p className="text-sm text-muted-foreground leading-relaxed">
          {gate.description}
        </p>

        {/* Detail message */}
        <AnimatePresence>
          {gate.detail && (
            <motion.p
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.2 }}
              className={cn(
                "text-xs mt-1.5 leading-relaxed",
                isPassed && "text-success/70",
                isBlocked && "text-destructive/70"
              )}
            >
              {gate.detail}
            </motion.p>
          )}
        </AnimatePresence>
      </div>

      {/* Status badge */}
      <div className="shrink-0 pt-0.5">
        <AnimatePresence mode="wait">
          <motion.div
            key={gate.status}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
          >
            {gate.status === "pending" && (
              <Badge variant="outline" className="text-[10px] px-1.5 py-0 h-5 font-normal text-muted-foreground/50 border-border/40">
                Pending
              </Badge>
            )}
            {gate.status === "evaluating" && (
              <Badge className="text-[10px] px-1.5 py-0 h-5 font-normal bg-warning/10 text-warning border-warning/20">
                <Loader2 className="h-2.5 w-2.5 mr-1 animate-spin" />
                Checking
              </Badge>
            )}
            {gate.status === "passed" && (
              <Badge className="text-[10px] px-1.5 py-0 h-5 font-normal bg-success/10 text-success border-success/20">
                <Check className="h-2.5 w-2.5 mr-1" />
                Passed
              </Badge>
            )}
            {gate.status === "blocked" && (
              <Badge variant="destructive" className="text-[10px] px-1.5 py-0 h-5 font-normal bg-destructive/10 text-destructive border-destructive/20">
                <AlertTriangle className="h-2.5 w-2.5 mr-1" />
                Blocked
              </Badge>
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </motion.div>
  );
}

export function GateLadder() {
  const activeMission = useMissionStore((s) => s.activeMission);
  const isProcessing = useMissionStore((s) => s.isProcessing);
  const evaluateGates = useMissionStore((s) => s.evaluateGates);
  const evaluateGatesReal = useMissionStore((s) => s.evaluateGatesReal);
  const currentStage = useMissionStore((s) => s.currentStage);

  // Option A session 2 — "Evaluate Gates" routes to the real cognition
  // gateway only when the mission matches a wired path (organize + fs
  // path). Otherwise the local simulation still runs.
  const intent = activeMission?.intent?.trim() ?? "";
  const routeToGateway =
    activeMission?.missionType === "organize" &&
    (intent.startsWith("/") || intent.startsWith("~"));
  const handleEvaluate = () =>
    routeToGateway ? evaluateGatesReal() : evaluateGates();

  const gates = activeMission?.gates ?? [];
  const hasEvaluated = gates.some((g) => g.status !== "pending");
  const allPassed = gates.length > 0 && gates.every((g) => g.status === "passed");
  const anyBlocked = gates.some((g) => g.status === "blocked");
  const passedCount = gates.filter((g) => g.status === "passed").length;
  const isDone = hasEvaluated && !isProcessing;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="w-full max-w-2xl mx-auto px-4 py-8 sm:py-12"
    >
      {/* Header */}
      <div className="text-center mb-8">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="inline-flex items-center justify-center w-12 h-12 rounded-2xl bg-trust-muted border border-border/40 mb-4"
        >
          <Shield className="h-5 w-5 text-trust" />
        </motion.div>
        <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight mb-2">
          Constitutional Gate Evaluation
        </h1>
        <p className="text-sm text-muted-foreground max-w-md mx-auto leading-relaxed">
          Each constitutional invariant is checked sequentially. All gates must
          pass before the mission proceeds.
        </p>
      </div>

      {/* Gate List */}
      <Card className="border-border/60 bg-card/80 backdrop-blur-sm">
        <CardContent className="pt-6">
          <div className="space-y-3">
            {gates.map((gate, index) => (
              <GateRow key={gate.id} gate={gate} index={index} />
            ))}
          </div>

          {/* Overall Verdict */}
          <AnimatePresence>
            {isDone && (
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.3, delay: 0.15 }}
                className="mt-6 pt-5 border-t border-border/40"
              >
                <div
                  className={cn(
                    "flex items-center gap-3 p-4 rounded-lg border",
                    allPassed
                      ? "border-success/20 bg-success/[0.04]"
                      : anyBlocked
                        ? "border-destructive/20 bg-destructive/[0.04]"
                        : "border-border/40 bg-muted/20"
                  )}
                >
                  <div
                    className={cn(
                      "w-8 h-8 rounded-full flex items-center justify-center shrink-0",
                      allPassed
                        ? "bg-success/10"
                        : anyBlocked
                          ? "bg-destructive/10"
                          : "bg-muted/30"
                    )}
                  >
                    {allPassed ? (
                      <Check className="h-4 w-4 text-success" strokeWidth={2.5} />
                    ) : anyBlocked ? (
                      <AlertTriangle className="h-4 w-4 text-destructive" />
                    ) : (
                      <Shield className="h-4 w-4 text-muted-foreground/50" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium">
                      {allPassed
                        ? "All Gates Passed"
                        : anyBlocked
                          ? "Gate Blocked"
                          : "Evaluation Incomplete"}
                    </p>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      {allPassed
                        ? `${passedCount}/${gates.length} invariants verified — mission may proceed`
                        : anyBlocked
                          ? `Mission halted: constitutional violation detected`
                          : `${passedCount}/${gates.length} gates evaluated`}
                    </p>
                  </div>
                  {allPassed && (
                    <Badge className="text-[10px] bg-success/10 text-success border-success/20 shrink-0">
                      Admissible
                    </Badge>
                  )}
                  {anyBlocked && (
                    <Badge variant="destructive" className="text-[10px] bg-destructive/10 text-destructive border-destructive/20 shrink-0">
                      Inadmissible
                    </Badge>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Progress bar during evaluation */}
          <AnimatePresence>
            {isProcessing && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="mt-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-muted-foreground">Evaluating invariants...</span>
                  <span className="text-xs text-muted-foreground font-mono">
                    {passedCount}/{gates.length}
                  </span>
                </div>
                <div className="h-1 w-full bg-muted/30 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-trust rounded-full"
                    initial={{ width: 0 }}
                    animate={{
                      width: `${(passedCount / gates.length) * 100}%`,
                    }}
                    transition={{ duration: 0.4, ease: "easeOut" }}
                  />
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Action Button */}
          {!hasEvaluated && (
            <div className="mt-6">
              <Button
                onClick={handleEvaluate}
                disabled={isProcessing}
                className={cn(
                  "w-full h-11 text-sm font-medium",
                  "bg-trust hover:bg-trust/90 text-trust-foreground shadow-[0_0_20px_oklch(0.78_0.14_75_/15%)]"
                )}
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    {routeToGateway ? "Calling cognition gateway..." : "Evaluating..."}
                  </>
                ) : (
                  <>
                    <Shield className="h-4 w-4 mr-2" />
                    {routeToGateway ? "Evaluate via Gateway" : "Evaluate Gates"}
                  </>
                )}
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}
