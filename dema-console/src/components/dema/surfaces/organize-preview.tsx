'use client';

import { useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useMissionStore } from '@/lib/mission-store';
import { useDEMAStore } from '@/lib/store';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import {
  Eye,
  PenTool,
  Compass,
  Cpu,
  CheckCircle,
  Shield,
  FileText,
  Server,
  Loader2,
  AlertTriangle,
  Clock,
  ArrowRight,
  Zap,
  ChevronRight,
  Layers,
  FolderOpen,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { timeAgo } from '@/lib/helpers/dema';
import type {
  MissionActionStep,
  MissionActionPlan,
  MissionType,
} from '@/lib/types';
import { MISSION_TYPE_LABELS } from '@/lib/types';

// ─── Step Type Icons ────────────────────────────────────────────

const STEP_TYPE_CONFIG: Record<
  MissionActionStep['type'],
  { icon: typeof Eye; color: string; bg: string; label: string }
> = {
  read: {
    icon: Eye,
    color: 'text-manifest',
    bg: 'bg-manifest/10',
    label: 'Read',
  },
  write: {
    icon: PenTool,
    color: 'text-action',
    bg: 'bg-action/10',
    label: 'Write',
  },
  navigate: {
    icon: Compass,
    color: 'text-receipt',
    bg: 'bg-receipt/10',
    label: 'Navigate',
  },
  compute: {
    icon: Cpu,
    color: 'text-trust',
    bg: 'bg-trust/10',
    label: 'Compute',
  },
  verify: {
    icon: CheckCircle,
    color: 'text-success',
    bg: 'bg-success/10',
    label: 'Verify',
  },
};

// ─── Mission Type Icons ─────────────────────────────────────────

const MISSION_TYPE_ICON: Record<MissionType, { icon: typeof Layers; color: string }> = {
  organize: { icon: FolderOpen, color: 'text-trust' },
  research: { icon: Eye, color: 'text-manifest' },
  analyze: { icon: Cpu, color: 'text-receipt' },
  create: { icon: PenTool, color: 'text-action' },
  communicate: { icon: Compass, color: 'text-warning' },
  monitor: { icon: Shield, color: 'text-success' },
};

// ─── Action Plan Generator ──────────────────────────────────────
//
// NO_SHADOW_STATE: no fabricated step counts, no invented resource
// names, no placeholder dryRun metrics. Only `organize` is wired to
// shipped kernel truth at T=0; other mission types are honestly
// labeled "not yet wired". When real gateway data exists for any
// type, the plan shape below is the structural envelope the kernel
// fills — no invented values leak from this file.

function generateSampleActionPlan(missionType: MissionType): MissionActionPlan {
  // Only `organize` is end-to-end wired via submitOrganize() →
  // gateway → sealed MissionExecuted receipt at T=0.
  if (missionType === 'organize') {
    return {
      steps: [
        { id: 's-01', label: 'Allowlist check',
          description: 'Verify the target path is registered & allowlisted',
          type: 'verify', status: 'pending' },
        { id: 's-02', label: 'Read target path',
          description: 'Read top-level entries of the allowlisted directory (read-only, no mutation)',
          type: 'read', status: 'pending' },
        { id: 's-03', label: 'Admissibility chain',
          description: 'Pass the claim through the 5 constitutional gates',
          type: 'verify', status: 'pending' },
        { id: 's-04', label: 'Seal MissionExecuted receipt',
          description: 'Bind the listing digest + mission hash into the canonical receipt chain',
          type: 'verify', status: 'pending' },
      ],
      estimatedDuration: '',
      resourcesRequired: [],
      dryRunAvailable: false,
      dryRunResult: null,
    };
  }

  // All other mission types are Horizon / not wired at T=0.
  // Render an honest single-step "not yet wired" placeholder that
  // does NOT fabricate any operational metric.
  return {
    steps: [
      { id: 's-01', label: 'Not yet wired at T=0',
        description: `Mission type "${missionType}" is Horizon. Only "organize" is wired to the kernel at first fire. Run \`dema organize <path>\` for the real flow.`,
        type: 'verify', status: 'pending' },
    ],
    estimatedDuration: '',
    resourcesRequired: [],
    dryRunAvailable: false,
    dryRunResult: null,
  };
}

// ─── Risk Level Badge ───────────────────────────────────────────

function RiskBadge({ level }: { level: 'low' | 'medium' | 'high' }) {
  const config = {
    low: { color: 'bg-success/10 text-success border-success/20', label: 'Low Risk' },
    medium: { color: 'bg-warning/10 text-warning border-warning/20', label: 'Medium Risk' },
    high: { color: 'bg-destructive/10 text-destructive border-destructive/20', label: 'High Risk' },
  };
  const c = config[level];
  return (
    <Badge variant="outline" className={cn('text-[10px] font-semibold', c.color)}>
      {c.label}
    </Badge>
  );
}

// ─── Step Card ──────────────────────────────────────────────────

function StepCard({
  step,
  index,
  isProcessing,
}: {
  step: MissionActionStep;
  index: number;
  isProcessing: boolean;
}) {
  const config = STEP_TYPE_CONFIG[step.type];
  const Icon = config.icon;

  const isActive = isProcessing && step.status === 'active';
  const isCompleted = isProcessing && step.status === 'completed';
  const isPending = !isProcessing || step.status === 'pending';

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.08, ease: 'easeOut' }}
    >
      <div
        className={cn(
          'flex items-start gap-3 p-3 rounded-lg border transition-all',
          isActive
            ? 'bg-trust/5 border-trust/20'
            : isCompleted
              ? 'bg-success/5 border-success/15 opacity-70'
              : 'bg-card/50 border-border/40 hover:border-border/60'
        )}
      >
        {/* Step Number + Icon */}
        <div className="flex flex-col items-center gap-1.5 shrink-0 pt-0.5">
          <div
            className={cn(
              'w-7 h-7 rounded-md flex items-center justify-center text-[10px] font-bold',
              config.bg,
              isActive
                ? 'ring-2 ring-trust/40'
                : isCompleted
                  ? 'ring-2 ring-success/30'
                  : ''
            )}
          >
            {isActive ? (
              <Loader2 className={cn('h-3.5 w-3.5 animate-spin', config.color)} />
            ) : isCompleted ? (
              <CheckCircle className="h-3.5 w-3.5 text-success" />
            ) : (
              <span className={cn(config.color)}>{index + 1}</span>
            )}
          </div>
          <Icon className={cn('h-3 w-3', config.color, isPending && 'opacity-50')} />
        </div>

        {/* Step Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span
              className={cn(
                'text-xs font-medium',
                isActive ? 'text-foreground' : isCompleted ? 'text-muted-foreground' : 'text-foreground'
              )}
            >
              {step.label}
            </span>
            <Badge variant="outline" className={cn('text-[9px] px-1 py-0', config.bg, config.color, 'border-0')}>
              {config.label}
            </Badge>
          </div>
          <p className="text-[11px] text-muted-foreground mt-0.5 leading-relaxed">
            {step.description}
          </p>
          {step.resource && (
            <div className="flex items-center gap-1 mt-1.5">
              <Server className="h-2.5 w-2.5 text-muted-foreground/60" />
              <span className="text-[10px] text-muted-foreground">{step.resource}</span>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}

// ─── Processing Overlay ─────────────────────────────────────────

function ProcessingOverlay({ missionType }: { missionType: MissionType }) {
  const TypeIcon = MISSION_TYPE_ICON[missionType].icon;
  const steps = useMemo(
    () => generateSampleActionPlan(missionType).steps,
    [missionType]
  );
  const currentStep = useMemo(() => {
    return Math.min(Math.floor(Date.now() / 800) % (steps.length + 1), steps.length);
  }, [steps.length]);

  return (
    <div className="flex flex-col items-center justify-center py-12 gap-4">
      <motion.div
        animate={{ scale: [1, 1.1, 1] }}
        transition={{ repeat: Infinity, duration: 2, ease: 'easeInOut' }}
        className="w-12 h-12 rounded-xl bg-trust/10 flex items-center justify-center"
      >
        <TypeIcon className="h-5 w-5 text-trust" />
      </motion.div>
      <div className="text-center">
        <p className="text-sm font-medium">Executing mission...</p>
        <p className="text-xs text-muted-foreground mt-1">
          Step {currentStep} of {steps.length}
        </p>
      </div>
      <Progress value={(currentStep / steps.length) * 100} className="w-48 h-1.5" />
    </div>
  );
}

// ─── Main Component ─────────────────────────────────────────────

export function OrganizePreview() {
  const { activeMission, isProcessing, confirmAction, retreatToIntent, advanceToAction } =
    useMissionStore();
  const { resources } = useDEMAStore();

  // Auto-generate action plan when stage is "action" and no plan exists
  useEffect(() => {
    if (activeMission?.stage === 'action' && !activeMission.actionPlan) {
      const plan = generateSampleActionPlan(activeMission.missionType);
      advanceToAction(plan);
    }
  }, [activeMission?.stage, activeMission?.actionPlan, activeMission?.missionType, advanceToAction]);

  const actionPlan = activeMission?.actionPlan;
  const missionType = activeMission?.missionType ?? 'organize';
  const typeConfig = MISSION_TYPE_ICON[missionType];
  const TypeIcon = typeConfig.icon;

  // Resolve resources that will be used
  const requiredResources = useMemo(() => {
    if (!actionPlan) return [];
    return actionPlan.resourcesRequired
      .map((name) => resources.find((r) => r.name === name))
      .filter(Boolean);
  }, [actionPlan, resources]);

  if (!activeMission) return null;

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-3.5 border-b border-border/50">
        <div className="flex items-center gap-3">
          <div className={cn('w-8 h-8 rounded-lg bg-trust/10 flex items-center justify-center')}>
            <Zap className={cn('h-4 w-4 text-trust')} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-sm font-semibold">Action Plan</h1>
              <Badge
                variant="outline"
                className={cn(
                  'text-[10px] px-1.5 py-0 gap-1 border-0',
                  STEP_TYPE_CONFIG[actionPlan?.steps[0]?.type ?? 'compute'].bg,
                  STEP_TYPE_CONFIG[actionPlan?.steps[0]?.type ?? 'compute'].color
                )}
              >
                <TypeIcon className="h-2.5 w-2.5" />
                {MISSION_TYPE_LABELS[missionType]}
              </Badge>
            </div>
            <p className="text-[11px] text-muted-foreground mt-0.5">
              {activeMission.intent}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-[10px] px-1.5 py-0 gap-1">
            <Clock className="h-2.5 w-2.5 text-muted-foreground" />
            {actionPlan?.estimatedDuration ?? 'Calculating...'}
          </Badge>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto dema-scrollbar px-6 py-4">
        <AnimatePresence mode="wait">
          {isProcessing ? (
            <motion.div
              key="processing"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="h-full flex items-center justify-center"
            >
              <ProcessingOverlay missionType={missionType} />
            </motion.div>
          ) : (
            <motion.div
              key="content"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="space-y-5 max-w-2xl mx-auto"
            >
              {/* Action Steps */}
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <ChevronRight className="h-3.5 w-3.5 text-trust" />
                  <h2 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Execution Steps
                  </h2>
                  <span className="text-[10px] text-muted-foreground">
                    {actionPlan?.steps.length ?? 0} operations
                  </span>
                </div>

                <div className="space-y-2">
                  {actionPlan?.steps.map((step, i) => (
                    <StepCard key={step.id} step={step} index={i} isProcessing={false} />
                  ))}
                </div>
              </div>

              <Separator className="bg-border/30" />

              {/* Dry-Run Results */}
              {actionPlan?.dryRunResult && (
                <div>
                  <div className="flex items-center gap-2 mb-3">
                    <Shield className="h-3.5 w-3.5 text-receipt" />
                    <h2 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      Dry-Run Analysis
                    </h2>
                  </div>

                  <Card className="border-border/40 bg-card/50">
                    <CardContent className="p-4">
                      {/* Stats Grid */}
                      <div className="grid grid-cols-3 gap-3 mb-3">
                        <div className="text-center p-2 rounded-md bg-background/50">
                          <div className="text-lg font-semibold text-foreground tabular-nums">
                            {actionPlan.dryRunResult.filesAffected}
                          </div>
                          <div className="text-[10px] text-muted-foreground">Files Affected</div>
                        </div>
                        <div className="text-center p-2 rounded-md bg-background/50">
                          <div className="text-lg font-semibold text-foreground tabular-nums">
                            {actionPlan.dryRunResult.operationsPlanned}
                          </div>
                          <div className="text-[10px] text-muted-foreground">Operations</div>
                        </div>
                        <div className="text-center p-2 rounded-md bg-background/50">
                          <RiskBadge level={actionPlan.dryRunResult.riskLevel} />
                        </div>
                      </div>

                      {/* Warnings */}
                      {actionPlan.dryRunResult.warnings.length > 0 && (
                        <div className="space-y-1.5 pt-2 border-t border-border/30">
                          {actionPlan.dryRunResult.warnings.map((w, i) => (
                            <div key={i} className="flex items-start gap-2">
                              <AlertTriangle className="h-3 w-3 text-warning shrink-0 mt-0.5" />
                              <span className="text-[11px] text-muted-foreground leading-relaxed">
                                {w}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </div>
              )}

              {/* Resource Summary */}
              {requiredResources.length > 0 && (
                <>
                  <Separator className="bg-border/30" />

                  <div>
                    <div className="flex items-center gap-2 mb-3">
                      <Server className="h-3.5 w-3.5 text-manifest" />
                      <h2 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                        Resources Required
                      </h2>
                      <span className="text-[10px] text-muted-foreground">
                        {requiredResources.length} connected
                      </span>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                      {requiredResources.map((res) => {
                        if (!res) return null;
                        return (
                          <motion.div
                            key={res.id}
                            initial={{ opacity: 0, x: -8 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ duration: 0.2, delay: 0.1 }}
                          >
                            <div className="flex items-center gap-2.5 p-2.5 rounded-lg border border-border/40 bg-card/50">
                              <div className="w-6 h-6 rounded-md bg-manifest/10 flex items-center justify-center">
                                <FileText className="h-3 w-3 text-manifest" />
                              </div>
                              <div className="flex-1 min-w-0">
                                <p className="text-[11px] font-medium truncate">{res.name}</p>
                                <p className="text-[10px] text-muted-foreground">
                                  {res.type} · {res.status}
                                </p>
                              </div>
                              <div className="w-1.5 h-1.5 rounded-full bg-success" />
                            </div>
                          </motion.div>
                        );
                      })}
                    </div>
                  </div>
                </>
              )}

              {/* Gate Summary */}
              <Separator className="bg-border/30" />

              <div>
                <div className="flex items-center gap-2 mb-3">
                  <Shield className="h-3.5 w-3.5 text-trust" />
                  <h2 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Constitutional Gates
                  </h2>
                </div>

                <div className="flex flex-wrap gap-1.5">
                  {activeMission.gates.map((gate) => (
                    <Badge
                      key={gate.id}
                      variant="outline"
                      className={cn(
                        'text-[10px] px-2 py-0.5',
                        gate.status === 'passed'
                          ? 'bg-success/5 text-success border-success/20'
                          : 'bg-muted text-muted-foreground border-border/40'
                      )}
                    >
                      <CheckCircle className="h-2.5 w-2.5 mr-1" />
                      {gate.id}
                    </Badge>
                  ))}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Action Buttons */}
      {!isProcessing && (
        <div className="border-t border-border/50 p-4 bg-background/80 backdrop-blur-sm">
          <div className="flex gap-3 max-w-2xl mx-auto">
            <Button
              onClick={confirmAction}
              className="flex-1 h-11 gap-2 bg-trust hover:bg-trust/90 text-trust-foreground font-medium"
            >
              <Zap className="h-4 w-4" />
              Approve & Execute
              <ArrowRight className="h-3.5 w-3.5 ml-1" />
            </Button>
            <Button
              variant="outline"
              onClick={retreatToIntent}
              className="h-11 gap-2 px-6"
            >
              Revise
            </Button>
          </div>
          <p className="text-[10px] text-muted-foreground text-center mt-2">
            All operations produce immutable receipts bound to your trust chain
          </p>
        </div>
      )}
    </div>
  );
}
