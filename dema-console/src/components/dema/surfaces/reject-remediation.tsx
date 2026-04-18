'use client';

import { useMemo } from 'react';
import { motion } from 'framer-motion';
import { useMissionStore } from '@/lib/mission-store';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
  AlertTriangle,
  ArrowLeft,
  X,
  ShieldAlert,
  ShieldCheck,
  XCircle,
  CheckCircle2,
  Info,
  ArrowRight,
  Lightbulb,
  ChevronRight,
  Zap,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import type { GateEvaluation } from '@/lib/types';

// ─── Gate Detail Card ───────────────────────────────────────────

function GateDetailCard({
  gate,
  isBlocked,
}: {
  gate: GateEvaluation;
  isBlocked: boolean;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, x: isBlocked ? -8 : 8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.35, delay: isBlocked ? 0.15 : 0, ease: 'easeOut' }}
    >
      <Card
        className={cn(
          'border transition-all',
          isBlocked
            ? 'border-destructive/30 bg-destructive/5 shadow-sm shadow-destructive/5'
            : 'border-success/15 bg-success/5'
        )}
      >
        <CardContent className="p-4">
          <div className="flex items-start gap-3">
            {/* Status Icon */}
            <div
              className={cn(
                'w-8 h-8 rounded-lg flex items-center justify-center shrink-0',
                isBlocked ? 'bg-destructive/10' : 'bg-success/10'
              )}
            >
              {isBlocked ? (
                <ShieldAlert className="h-4 w-4 text-destructive" />
              ) : (
                <ShieldCheck className="h-4 w-4 text-success" />
              )}
            </div>

            {/* Content */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span
                  className={cn(
                    'text-xs font-semibold',
                    isBlocked ? 'text-destructive' : 'text-success'
                  )}
                >
                  {gate.id}
                </span>
                <Badge
                  variant="outline"
                  className={cn(
                    'text-[9px] px-1 py-0 border-0',
                    isBlocked
                      ? 'bg-destructive/10 text-destructive'
                      : 'bg-success/10 text-success'
                  )}
                >
                  {isBlocked ? 'BLOCKED' : 'PASSED'}
                </Badge>
              </div>

              <p className="text-[11px] text-muted-foreground mt-0.5 leading-relaxed">
                {gate.description}
              </p>

              {/* Blocking Detail */}
              {isBlocked && gate.detail && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  transition={{ duration: 0.3, delay: 0.25 }}
                  className="mt-2 p-2.5 rounded-md bg-destructive/5 border border-destructive/15"
                >
                  <div className="flex items-start gap-2">
                    <XCircle className="h-3.5 w-3.5 text-destructive shrink-0 mt-0.5" />
                    <div>
                      <p className="text-[10px] font-medium text-destructive">
                        Blocking Reason
                      </p>
                      <p className="text-[11px] text-destructive/80 mt-0.5 leading-relaxed">
                        {gate.detail}
                      </p>
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Passed detail */}
              {!isBlocked && gate.detail && (
                <div className="mt-1.5 flex items-center gap-1.5">
                  <CheckCircle2 className="h-3 w-3 text-success/60" />
                  <span className="text-[10px] text-muted-foreground">{gate.detail}</span>
                </div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}

// ─── Blocked Indicator with Shake ───────────────────────────────

function BlockedIndicator() {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
    >
      <motion.div
        animate={{ x: [0, -3, 3, -3, 3, 0] }}
        transition={{ duration: 0.5, delay: 0.3 }}
        className="w-12 h-12 rounded-xl bg-destructive/10 flex items-center justify-center mx-auto"
      >
        <ShieldAlert className="h-6 w-6 text-destructive" />
      </motion.div>
    </motion.div>
  );
}

// ─── Main Component ─────────────────────────────────────────────

export function RejectRemediation() {
  const {
    activeMission,
    retreatToIntent,
    cancelMission,
  } = useMissionStore();

  const { gates } = activeMission ?? { gates: [] as GateEvaluation[] };

  const blockedGates = useMemo(
    () => gates.filter((g) => g.status === 'blocked'),
    [gates]
  );
  const passedGates = useMemo(
    () => gates.filter((g) => g.status === 'passed'),
    [gates]
  );

  if (!activeMission) return null;

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-3.5 border-b border-destructive/20">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-destructive/10 flex items-center justify-center">
            <AlertTriangle className="h-4 w-4 text-destructive" />
          </div>
          <div>
            <h1 className="text-sm font-semibold">Mission Blocked</h1>
            <p className="text-[11px] text-muted-foreground">
              One or more constitutional gates were not satisfied
            </p>
          </div>
        </div>

        <Badge
          variant="outline"
          className="text-[10px] px-2 py-0.5 gap-1 border-destructive/30 bg-destructive/5 text-destructive"
        >
          <XCircle className="h-3 w-3" />
          {blockedGates.length} blocked
        </Badge>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto dema-scrollbar px-6 py-6">
        <div className="space-y-6 max-w-2xl mx-auto">
          {/* Blocked Indicator */}
          <div className="flex flex-col items-center gap-3 py-2">
            <BlockedIndicator />
            <div className="text-center">
              <p className="text-sm font-medium text-destructive">
                Mission &ldquo;{activeMission.intent}&rdquo; could not proceed
              </p>
              <p className="text-[11px] text-muted-foreground mt-1">
                Constitutional invariants protect the system — this is by design.
              </p>
            </div>
          </div>

          {/* Summary Bar */}
          <Card className="border-border/40 bg-card/40">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="h-4 w-4 text-success" />
                    <span className="text-xs text-success font-medium">{passedGates.length} passed</span>
                  </div>
                  <Separator orientation="vertical" className="h-4 bg-border/40" />
                  <div className="flex items-center gap-2">
                    <XCircle className="h-4 w-4 text-destructive" />
                    <span className="text-xs text-destructive font-medium">{blockedGates.length} blocked</span>
                  </div>
                </div>
                <Badge variant="outline" className="text-[10px] px-1.5 py-0 border-border/30">
                  {gates.length} gates evaluated
                </Badge>
              </div>
              {/* Visual bar */}
              <div className="flex h-1 rounded-full overflow-hidden mt-3 bg-muted/30">
                <div
                  className="bg-success transition-all"
                  style={{ width: `${(passedGates.length / gates.length) * 100}%` }}
                />
                <div
                  className="bg-destructive transition-all"
                  style={{ width: `${(blockedGates.length / gates.length) * 100}%` }}
                />
              </div>
            </CardContent>
          </Card>

          {/* Blocked Gates — Red accent, shown first */}
          {blockedGates.length > 0 && (
            <div>
              <div className="flex items-center gap-2 mb-3">
                <ShieldAlert className="h-3.5 w-3.5 text-destructive" />
                <h2 className="text-xs font-semibold uppercase tracking-wider text-destructive">
                  Blocked Gates
                </h2>
              </div>
              <div className="space-y-2">
                {blockedGates.map((gate) => (
                  <GateDetailCard key={gate.id} gate={gate} isBlocked={true} />
                ))}
              </div>
            </div>
          )}

          {/* Passed Gates — Muted */}
          {passedGates.length > 0 && (
            <>
              <Separator className="bg-border/30" />

              <div>
                <div className="flex items-center gap-2 mb-3">
                  <ShieldCheck className="h-3.5 w-3.5 text-success/60" />
                  <h2 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Passed Gates
                  </h2>
                  <ChevronRight className="h-3 w-3 text-muted-foreground/40" />
                  <span className="text-[10px] text-muted-foreground/60">collapsed</span>
                </div>
                <div className="space-y-2">
                  {passedGates.map((gate) => (
                    <GateDetailCard key={gate.id} gate={gate} isBlocked={false} />
                  ))}
                </div>
              </div>
            </>
          )}

          {/* Helpful Guidance */}
          <Card className="border-trust/20 bg-trust/5">
            <CardContent className="p-4">
              <div className="flex items-start gap-3">
                <div className="w-7 h-7 rounded-md bg-trust/10 flex items-center justify-center shrink-0">
                  <Lightbulb className="h-3.5 w-3.5 text-trust" />
                </div>
                <div>
                  <p className="text-[11px] font-medium text-trust">How to proceed</p>
                  <div className="mt-1.5 space-y-1.5">
                    <div className="flex items-start gap-2">
                      <ArrowRight className="h-3 w-3 text-trust/60 shrink-0 mt-0.5" />
                      <p className="text-[11px] text-muted-foreground leading-relaxed">
                        <strong className="text-foreground">Revise Mission</strong> — Return to the intent stage, adjust your parameters, and try again. The blocked gate will be re-evaluated with updated context.
                      </p>
                    </div>
                    <div className="flex items-start gap-2">
                      <ArrowRight className="h-3 w-3 text-trust/60 shrink-0 mt-0.5" />
                      <p className="text-[11px] text-muted-foreground leading-relaxed">
                        <strong className="text-foreground">Cancel</strong> — Dismiss this mission entirely. Your trust chain remains unaffected. You can start a new mission at any time.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="border-t border-border/50 p-4 bg-background/80 backdrop-blur-sm">
        <div className="flex gap-3 max-w-2xl mx-auto">
          <Button
            variant="outline"
            onClick={retreatToIntent}
            className="flex-1 h-11 gap-2 font-medium"
          >
            <ArrowLeft className="h-4 w-4" />
            Revise Mission
          </Button>
          <Button
            variant="ghost"
            onClick={cancelMission}
            className="h-11 gap-2 px-6 text-muted-foreground hover:text-foreground"
          >
            Cancel
          </Button>
        </div>
        <p className="text-[10px] text-muted-foreground text-center mt-2">
          Constitutional gates exist to protect integrity — blocking is a feature, not a failure
        </p>
      </div>
    </div>
  );
}
