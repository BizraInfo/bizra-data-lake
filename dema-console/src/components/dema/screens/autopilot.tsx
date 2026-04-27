"use client";

import { useState, useMemo } from "react";
import { useDEMAStore } from "@/lib/store";
import { cn } from "@/lib/utils";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Brain,
  TrendingUp,
  TrendingDown,
  Minus,
  Activity,
  RefreshCw,
  Target,
  Zap,
  AlertTriangle,
  CheckCircle2,
  ArrowUp,
  ArrowDown,
  Clock,
  Cpu,
  Shield,
  Gauge,
  RotateCcw,
  ChevronRight,
  Info,
  Sparkles,
  BarChart3,
  Timer,
  ArrowRightLeft,
  Lightbulb,
} from "lucide-react";
import { timeAgo } from "@/lib/helpers/dema";
import type {
  SystemMetric,
  MetricTrend,
  OptimizationCycle,
  OptimizationAction,
  OptimizationCycleStatus,
  EvolutionProjection,
} from "@/lib/types";

// ═══════════════════════════════════════════════════════════════
// Constants & Helpers
// ═══════════════════════════════════════════════════════════════

type MetricCategory = SystemMetric["category"];

const CATEGORY_CONFIG: Record<
  MetricCategory,
  { label: string; color: string; bg: string; border: string; badgeVariant: "default" | "secondary" | "outline" | "destructive" }
> = {
  performance: {
    label: "Performance",
    color: "text-trust",
    bg: "bg-trust/8",
    border: "border-trust/20",
    badgeVariant: "default",
  },
  reliability: {
    label: "Reliability",
    color: "text-receipt",
    bg: "bg-receipt/8",
    border: "border-receipt/20",
    badgeVariant: "secondary",
  },
  security: {
    label: "Security",
    color: "text-destructive",
    bg: "bg-destructive/8",
    border: "border-destructive/20",
    badgeVariant: "destructive",
  },
  efficiency: {
    label: "Efficiency",
    color: "text-manifest",
    bg: "bg-manifest/8",
    border: "border-manifest/20",
    badgeVariant: "secondary",
  },
  quality: {
    label: "Quality",
    color: "text-action",
    bg: "bg-action/8",
    border: "border-action/20",
    badgeVariant: "outline",
  },
};

const CATEGORY_ICONS: Record<MetricCategory, React.ElementType> = {
  performance: Gauge,
  reliability: Shield,
  security: Shield,
  efficiency: Cpu,
  quality: Target,
};

function trendIcon(trend: MetricTrend) {
  switch (trend) {
    case "improving":
      return { Icon: TrendingUp, color: "text-success" };
    case "degrading":
      return { Icon: TrendingDown, color: "text-destructive" };
    case "volatile":
      return { Icon: Activity, color: "text-warning" };
    case "stable":
    default:
      return { Icon: Minus, color: "text-muted-foreground" };
  }
}

function cycleStatusConfig(status: OptimizationCycleStatus) {
  switch (status) {
    case "completed":
      return { label: "Completed", color: "text-success", dot: "bg-success", badgeVariant: "outline" as const };
    case "idle":
      return { label: "Idle", color: "text-muted-foreground", dot: "bg-muted-foreground", badgeVariant: "outline" as const };
    case "optimizing":
    case "analyzing":
    case "applying":
      return { label: "Optimizing", color: "text-warning", dot: "bg-warning", badgeVariant: "outline" as const };
    case "scanning":
      return { label: "Scanning", color: "text-trust", dot: "bg-trust", badgeVariant: "outline" as const };
    case "validating":
      return { label: "Validating", color: "text-receipt", dot: "bg-receipt", badgeVariant: "outline" as const };
    case "rollback":
      return { label: "Rollback", color: "text-destructive", dot: "bg-destructive", badgeVariant: "destructive" as const };
    default:
      return { label: status, color: "text-muted-foreground", dot: "bg-muted-foreground", badgeVariant: "outline" as const };
  }
}

function formatDuration(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSec / 60);
  const seconds = totalSec % 60;
  return `${minutes}m ${seconds}s`;
}

function impactColor(impact: "low" | "medium" | "high") {
  switch (impact) {
    case "high":
      return "text-success";
    case "medium":
      return "text-trust";
    case "low":
      return "text-muted-foreground";
  }
}

function riskColor(risk: "low" | "medium" | "high") {
  switch (risk) {
    case "high":
      return "text-destructive";
    case "medium":
      return "text-warning";
    case "low":
      return "text-success";
  }
}

function actionStatusConfig(status: OptimizationAction["status"]) {
  switch (status) {
    case "applied":
      return { label: "Applied", color: "text-success", dot: "bg-success" };
    case "pending":
      return { label: "Pending", color: "text-muted-foreground", dot: "bg-muted-foreground" };
    case "failed":
      return { label: "Failed", color: "text-destructive", dot: "bg-destructive" };
    case "reverted":
      return { label: "Reverted", color: "text-warning", dot: "bg-warning" };
    default:
      return { label: status, color: "text-muted-foreground", dot: "bg-muted-foreground" };
  }
}

// ═══════════════════════════════════════════════════════════════
// SVG Sparkline Component
// ═══════════════════════════════════════════════════════════════

function Sparkline({
  data,
  trend,
  className,
}: {
  data: number[];
  trend: MetricTrend;
  className?: string;
}) {
  if (data.length < 2) return null;

  const width = 120;
  const height = 32;
  const padding = 2;

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;

  const points = data.map((value, i) => {
    const x = padding + (i / (data.length - 1)) * (width - padding * 2);
    const y = height - padding - ((value - min) / range) * (height - padding * 2);
    return { x, y };
  });

  const pathD = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`)
    .join(" ");

  const areaD = `${pathD} L ${(points[points.length - 1]?.x ?? 0).toFixed(1)} ${height} L ${padding} ${height} Z`;

  const trendStroke = (() => {
    switch (trend) {
      case "improving":
        return "var(--color-success)";
      case "degrading":
        return "var(--color-destructive)";
      case "volatile":
        return "var(--color-warning)";
      case "stable":
      default:
        return "var(--color-muted-foreground)";
    }
  })();

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className={cn("w-full h-8", className)}
      preserveAspectRatio="none"
    >
      <defs>
        <linearGradient id={`spark-grad-${trend}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={trendStroke} stopOpacity="0.2" />
          <stop offset="100%" stopColor={trendStroke} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={areaD} fill={`url(#spark-grad-${trend})`} />
      <path
        d={pathD}
        fill="none"
        stroke={trendStroke}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle
        cx={points[points.length - 1]?.x.toFixed(1)}
        cy={points[points.length - 1]?.y.toFixed(1)}
        r="2.5"
        fill={trendStroke}
      />
    </svg>
  );
}

// ═══════════════════════════════════════════════════════════════
// Tab 1: System Metrics
// ═══════════════════════════════════════════════════════════════

function MetricCard({ metric }: { metric: SystemMetric }) {
  const catConfig = CATEGORY_CONFIG[metric.category];
  const CatIcon = CATEGORY_ICONS[metric.category];
  const { Icon: TrendIcon, color: trendColor } = trendIcon(metric.trend);
  const progressPercent = metric.target > 0 ? Math.min((metric.value / metric.target) * 100, 100) : 0;
  // For metrics where lower is better (like latency, error rate), we invert progress
  const isInverse = metric.name === "Receipt Verification Time" || metric.name === "Error Rate";
  const displayProgress = isInverse ? Math.min((metric.target / metric.value) * 100, 100) : progressPercent;
  const sparkData = metric.history.map((h) => h.value);

  return (
    <Card className="border-border/40 bg-card/60 hover:border-border/70 transition-colors group">
      <CardContent className="p-4 space-y-3">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-2 min-w-0">
            <div className={cn("p-1.5 rounded-md shrink-0", catConfig.bg)}>
              <CatIcon className={cn("h-3.5 w-3.5", catConfig.color)} />
            </div>
            <div className="min-w-0">
              <p className="text-xs font-medium truncate">{metric.name}</p>
              <Badge variant={catConfig.badgeVariant} className={cn("text-[9px] px-1.5 py-0 mt-0.5", catConfig.color)}>
                {catConfig.label}
              </Badge>
            </div>
          </div>
          <div className="flex items-center gap-1 shrink-0">
            <TrendIcon className={cn("h-3.5 w-3.5", trendColor)} />
          </div>
        </div>

        {/* Value + Target */}
        <div className="flex items-baseline gap-2">
          <span className="text-xl font-bold font-mono tracking-tight">
            {metric.value}
          </span>
          <span className="text-[11px] text-muted-foreground">{metric.unit}</span>
          <div className="ml-auto text-[10px] text-muted-foreground">
            Target: <span className="font-mono text-foreground/70">{metric.target}{metric.unit}</span>
          </div>
        </div>

        {/* Sparkline */}
        <Sparkline data={sparkData} trend={metric.trend} className="opacity-70 group-hover:opacity-100 transition-opacity" />

        {/* Progress bar */}
        <div className="space-y-1">
          <div className="flex justify-between text-[10px]">
            <span className="text-muted-foreground">
              {isInverse ? "Inverted" : "Progress"}
            </span>
            <span className={cn("font-mono", displayProgress >= 100 ? "text-success" : "text-muted-foreground")}>
              {displayProgress.toFixed(0)}%
            </span>
          </div>
          <Progress value={displayProgress} className="h-1.5" />
        </div>
      </CardContent>
    </Card>
  );
}

function SystemMetricsTab() {
  const { systemMetrics } = useDEMAStore();
  const [activeCategory, setActiveCategory] = useState<MetricCategory | "all">("all");

  const categories: Array<{ key: MetricCategory | "all"; label: string }> = [
    { key: "all", label: "All" },
    { key: "performance", label: "Performance" },
    { key: "reliability", label: "Reliability" },
    { key: "security", label: "Security" },
    { key: "efficiency", label: "Efficiency" },
    { key: "quality", label: "Quality" },
  ];

  const filteredMetrics = useMemo(
    () =>
      activeCategory === "all"
        ? systemMetrics
        : systemMetrics.filter((m) => m.category === activeCategory),
    [systemMetrics, activeCategory]
  );

  const improvingCount = systemMetrics.filter((m) => m.trend === "improving").length;
  const degradingCount = systemMetrics.filter((m) => m.trend === "degrading").length;

  return (
    <div className="space-y-4">
      {/* Summary header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <Activity className="h-4 w-4 text-muted-foreground" />
            <span className="text-xs text-muted-foreground">
              {systemMetrics.length} metrics tracked
            </span>
          </div>
          <Separator orientation="vertical" className="h-4" />
          <div className="flex items-center gap-3">
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex items-center gap-1 text-xs text-success">
                  <TrendingUp className="h-3 w-3" />
                  <span>{improvingCount}</span>
                </div>
              </TooltipTrigger>
              <TooltipContent>Improving</TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex items-center gap-1 text-xs text-destructive">
                  <TrendingDown className="h-3 w-3" />
                  <span>{degradingCount}</span>
                </div>
              </TooltipTrigger>
              <TooltipContent>Degrading</TooltipContent>
            </Tooltip>
          </div>
        </div>
      </div>

      {/* Category filter */}
      <div className="flex flex-wrap gap-1.5">
        {categories.map((cat) => {
          const isActive = activeCategory === cat.key;
          const catConf = cat.key !== "all" ? CATEGORY_CONFIG[cat.key] : null;
          return (
            <Button
              key={cat.key}
              variant={isActive ? "default" : "outline"}
              size="sm"
              className={cn(
                "h-7 text-[11px] px-2.5",
                isActive && catConf && catConf.bg
              )}
              onClick={() => setActiveCategory(cat.key)}
            >
              {cat.label}
              {cat.key !== "all" && (
                <span className="ml-1 opacity-60">
                  {systemMetrics.filter((m) => m.category === cat.key).length}
                </span>
              )}
            </Button>
          );
        })}
      </div>

      {/* Metric cards grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {filteredMetrics.map((metric) => (
          <MetricCard key={metric.id} metric={metric} />
        ))}
      </div>

      {filteredMetrics.length === 0 && (
        <div className="text-center py-12 text-muted-foreground">
          <Gauge className="h-8 w-8 mx-auto mb-3 opacity-30" />
          <p className="text-sm">No metrics in this category</p>
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// Tab 2: Optimization Cycles
// ═══════════════════════════════════════════════════════════════

function OptimizationActionItem({ action }: { action: OptimizationAction }) {
  const actionConf = actionStatusConfig(action.status);

  return (
    <div className="flex items-start gap-3 p-3 rounded-lg bg-muted/20 border border-border/30">
      <div className="mt-0.5 shrink-0">
        <div className={cn("w-2 h-2 rounded-full", actionConf.dot)} />
      </div>
      <div className="min-w-0 flex-1 space-y-1.5">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-[10px] font-mono text-muted-foreground bg-muted px-1.5 py-0.5 rounded">
            {action.target}
          </span>
          <ChevronRight className="h-2.5 w-2.5 text-muted-foreground/50" />
          <span className="text-xs font-medium">{action.action.replace(/_/g, " ")}</span>
          <Badge variant="outline" className={cn("text-[9px] px-1.5 py-0", actionConf.color)}>
            {actionConf.label}
          </Badge>
        </div>
        <p className="text-[11px] text-muted-foreground leading-relaxed">
          {action.description}
        </p>
        <div className="flex items-center gap-3 text-[10px]">
          <span className={cn("flex items-center gap-1", impactColor(action.impact))}>
            <Zap className="h-2.5 w-2.5" />
            Impact: {action.impact}
          </span>
          <span className={cn("flex items-center gap-1", riskColor(action.risk))}>
            <Shield className="h-2.5 w-2.5" />
            Risk: {action.risk}
          </span>
        </div>
        {action.result && (
          <div className="flex items-start gap-1.5 p-2 rounded bg-success/5 border border-success/10">
            <CheckCircle2 className="h-3 w-3 text-success mt-0.5 shrink-0" />
            <span className="text-[10px] text-success font-mono">{action.result}</span>
          </div>
        )}
      </div>
    </div>
  );
}

function MetricsComparison({
  before,
  after,
}: {
  before: Record<string, number>;
  after: Record<string, number>;
}) {
  const keys = Object.keys(before);

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
      {keys.map((key) => {
        const beforeVal = before[key];
        const afterVal = after[key];
        const improvement = ((afterVal - beforeVal) / Math.abs(beforeVal || 1)) * 100;
        const isInverse = key.toLowerCase().includes("time") || key.toLowerCase().includes("overhead");
        const isImproved = isInverse ? improvement < 0 : improvement > 0;

        return (
          <div
            key={key}
            className="flex items-center justify-between p-2.5 rounded-lg bg-muted/20 border border-border/20"
          >
            <span className="text-[10px] text-muted-foreground capitalize font-mono">
              {key.replace(/([A-Z])/g, " $1").trim()}
            </span>
            <div className="flex items-center gap-2">
              <span className="text-[10px] text-muted-foreground font-mono line-through">
                {beforeVal}
              </span>
              <ArrowRightLeft className="h-3 w-3 text-muted-foreground/40" />
              <span className={cn("text-[11px] font-mono font-medium", isImproved ? "text-success" : "text-destructive")}>
                {afterVal}
              </span>
              <span className={cn("text-[9px] font-mono", isImproved ? "text-success" : "text-destructive")}>
                ({improvement > 0 ? "+" : ""}{improvement.toFixed(0)}%)
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function CycleCard({ cycle }: { cycle: OptimizationCycle }) {
  const statusConf = cycleStatusConfig(cycle.status);

  return (
    <Card className="border-border/40 bg-card/60">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <div className={cn("w-2 h-2 rounded-full", statusConf.dot, cycle.status === "optimizing" && "dema-pulse")} />
              <span className="text-xs font-mono text-muted-foreground">
                Cycle #{cycle.cycleNumber}
              </span>
            </div>
            <Badge variant={statusConf.badgeVariant} className={cn("text-[10px] px-2 py-0", statusConf.color)}>
              {statusConf.label}
            </Badge>
            {cycle.rollbackTriggered && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Badge variant="destructive" className="text-[10px] px-2 py-0">
                    <RotateCcw className="h-2.5 w-2.5 mr-1" />
                    Rolled Back
                  </Badge>
                </TooltipTrigger>
                <TooltipContent>This cycle was rolled back due to instability</TooltipContent>
              </Tooltip>
            )}
          </div>
          <div className="flex items-center gap-3">
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <Timer className="h-3 w-3" />
                  <span className="font-mono">{formatDuration(cycle.duration)}</span>
                </div>
              </TooltipTrigger>
              <TooltipContent>Cycle duration</TooltipContent>
            </Tooltip>
            <div className={cn(
              "flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium font-mono",
              cycle.improvement > 0
                ? "bg-success/10 text-success"
                : "bg-destructive/10 text-destructive"
            )}>
              {cycle.improvement > 0 ? (
                <ArrowUp className="h-3 w-3" />
              ) : (
                <ArrowDown className="h-3 w-3" />
              )}
              {cycle.improvement > 0 ? "+" : ""}
              {cycle.improvement.toFixed(1)}%
            </div>
          </div>
        </div>
        <div className="flex items-center gap-3 text-[10px] text-muted-foreground mt-1.5">
          <span className="flex items-center gap-1">
            <Clock className="h-2.5 w-2.5" />
            Started {timeAgo(cycle.startedAt)}
          </span>
          {cycle.completedAt && (
            <span className="flex items-center gap-1">
              <CheckCircle2 className="h-2.5 w-2.5 text-success" />
              Completed {timeAgo(cycle.completedAt)}
            </span>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Optimization actions */}
        <div>
          <div className="flex items-center gap-1.5 mb-2.5">
            <Zap className="h-3 w-3 text-trust" />
            <span className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider">
              Optimizations ({cycle.optimizations.length})
            </span>
          </div>
          <div className="space-y-2">
            {cycle.optimizations.map((opt) => (
              <OptimizationActionItem key={opt.id} action={opt} />
            ))}
          </div>
        </div>

        <Separator className="opacity-40" />

        {/* Metrics before/after */}
        <div>
          <div className="flex items-center gap-1.5 mb-2.5">
            <BarChart3 className="h-3 w-3 text-receipt" />
            <span className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider">
              Metrics Comparison
            </span>
          </div>
          <MetricsComparison before={cycle.metricsBefore} after={cycle.metricsAfter} />
        </div>
      </CardContent>
    </Card>
  );
}

function OptimizationCyclesTab() {
  const { optimizationCycles } = useDEMAStore();

  const totalCycles = optimizationCycles.length;
  const avgImprovement =
    totalCycles > 0
      ? optimizationCycles.reduce((sum, c) => sum + c.improvement, 0) / totalCycles
      : 0;
  const lastCycle = optimizationCycles[0];
  const lastCycleConf = lastCycle ? cycleStatusConfig(lastCycle.status) : null;

  return (
    <div className="space-y-4">
      {/* Summary header */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <Card className="border-border/40 bg-card/60">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <RefreshCw className="h-4 w-4 text-trust" />
              <span className="text-[11px] text-muted-foreground uppercase tracking-wider">Total Cycles</span>
            </div>
            <span className="text-2xl font-bold font-mono tracking-tight">{totalCycles}</span>
          </CardContent>
        </Card>
        <Card className="border-border/40 bg-card/60">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="h-4 w-4 text-success" />
              <span className="text-[11px] text-muted-foreground uppercase tracking-wider">Avg Improvement</span>
            </div>
            <div className="flex items-baseline gap-1">
              <span className="text-2xl font-bold font-mono tracking-tight text-success">
                +{avgImprovement.toFixed(1)}%
              </span>
            </div>
          </CardContent>
        </Card>
        <Card className="border-border/40 bg-card/60">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <Activity className="h-4 w-4 text-muted-foreground" />
              <span className="text-[11px] text-muted-foreground uppercase tracking-wider">Last Cycle</span>
            </div>
            <div className="flex items-center gap-2">
              {lastCycleConf && (
                <>
                  <div className={cn("w-2 h-2 rounded-full", lastCycleConf.dot)} />
                  <span className={cn("text-sm font-medium", lastCycleConf.color)}>
                    {lastCycleConf.label}
                  </span>
                </>
              )}
            </div>
            {lastCycle && (
              <p className="text-[10px] text-muted-foreground mt-1">
                #{lastCycle.cycleNumber} &middot; {timeAgo(lastCycle.startedAt)}
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Cycle list */}
      <div className="space-y-3">
        {optimizationCycles.map((cycle) => (
          <CycleCard key={cycle.id} cycle={cycle} />
        ))}
      </div>

      {optimizationCycles.length === 0 && (
        <div className="text-center py-12 text-muted-foreground">
          <RefreshCw className="h-8 w-8 mx-auto mb-3 opacity-30" />
          <p className="text-sm">No optimization cycles recorded yet</p>
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// Tab 3: Evolution Projection
// ═══════════════════════════════════════════════════════════════

function PredictionRow({
  prediction,
}: {
  prediction: EvolutionProjection["predictions"][number];
}) {
  const DirectionIcon = prediction.direction === "up" ? ArrowUp : prediction.direction === "down" ? ArrowDown : Minus;
  const directionColor =
    prediction.direction === "up"
      ? "text-success"
      : prediction.direction === "down"
        ? "text-destructive"
        : "text-muted-foreground";
  const change = prediction.projectedValue - prediction.currentValue;
  const changePercent =
    prediction.currentValue !== 0
      ? ((change / Math.abs(prediction.currentValue)) * 100).toFixed(1)
      : "0";
  const numericChangePercent = Number(changePercent);

  const barWidth = Math.abs(numericChangePercent) > 100 ? 100 : Math.abs(numericChangePercent);

  return (
    <div className="flex items-center gap-3 py-3 px-4 rounded-lg hover:bg-accent/20 transition-colors">
      <div className="flex items-center gap-1 shrink-0 w-5 justify-center">
        <DirectionIcon className={cn("h-3.5 w-3.5", directionColor)} />
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-xs font-medium truncate">{prediction.metric}</p>
        <div className="mt-1.5 h-1.5 rounded-full bg-muted overflow-hidden max-w-[180px]">
          <div
            className={cn(
              "h-full rounded-full transition-all duration-500",
              prediction.direction === "up" ? "bg-success" : prediction.direction === "down" ? "bg-destructive" : "bg-muted-foreground"
            )}
            style={{ width: `${Math.max(barWidth, 2)}%` }}
          />
        </div>
      </div>
      <div className="text-right shrink-0">
        <span className="text-xs font-mono text-muted-foreground">
          {prediction.currentValue}
        </span>
        <ChevronRight className="h-3 w-3 text-muted-foreground/40 inline mx-1" />
        <span className={cn("text-xs font-mono font-medium", directionColor)}>
          {prediction.projectedValue}
        </span>
      </div>
      <div className={cn("text-[10px] font-mono shrink-0 w-14 text-right", directionColor)}>
        {change > 0 ? "+" : ""}{changePercent}%
      </div>
    </div>
  );
}

function EvolutionProjectionTab() {
  const { evolutionProjection } = useDEMAStore();
  const [activeHorizon, setActiveHorizon] = useState<EvolutionProjection["horizon"]>(
    evolutionProjection.horizon
  );

  const horizons: Array<{ key: EvolutionProjection["horizon"]; label: string }> = [
    { key: "1h", label: "1h" },
    { key: "6h", label: "6h" },
    { key: "24h", label: "24h" },
    { key: "7d", label: "7d" },
    { key: "30d", label: "30d" },
  ];

  // Simulate confidence changes based on horizon
  const horizonConfidence: Record<string, number> = {
    "1h": 0.96,
    "6h": 0.91,
    "24h": 0.87,
    "7d": 0.68,
    "30d": 0.42,
  };

  const confidence = horizonConfidence[activeHorizon] ?? evolutionProjection.confidence;

  // Simulate different predictions per horizon
  const horizonPredictions = useMemo(() => {
    const base = evolutionProjection.predictions;
    const multiplier: Record<string, number> = { "1h": 0.2, "6h": 0.5, "24h": 1, "7d": 2.5, "30d": 5 };
    const m = multiplier[activeHorizon] ?? 1;
    return base.map((p) => ({
      ...p,
      projectedValue: Math.round((p.currentValue + (p.projectedValue - p.currentValue) * m) * 100) / 100,
      direction: p.direction,
    }));
  }, [evolutionProjection.predictions, activeHorizon]);

  return (
    <div className="space-y-4">
      {/* Main projection card */}
      <Card className="border-border/40 bg-card/60">
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between flex-wrap gap-3">
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-md bg-trust/8">
                <Brain className="h-4 w-4 text-trust" />
              </div>
              <div>
                <CardTitle className="text-sm">Evolution Projection</CardTitle>
                <p className="text-[10px] text-muted-foreground mt-0.5">
                  Updated {timeAgo(evolutionProjection.timestamp)}
                </p>
              </div>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-5">
          {/* Horizon selector */}
          <div>
            <div className="flex items-center gap-1.5 mb-2">
              <Clock className="h-3 w-3 text-muted-foreground" />
              <span className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">
                Projection Horizon
              </span>
            </div>
            <div className="flex gap-1">
              {horizons.map((h) => (
                <Button
                  key={h.key}
                  variant={activeHorizon === h.key ? "default" : "outline"}
                  size="sm"
                  className="h-7 text-[11px] px-3"
                  onClick={() => setActiveHorizon(h.key)}
                >
                  {h.label}
                </Button>
              ))}
            </div>
          </div>

          {/* Confidence */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-1.5">
                <Target className="h-3 w-3 text-muted-foreground" />
                <span className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">
                  Confidence Score
                </span>
              </div>
              <span className={cn(
                "text-sm font-bold font-mono",
                confidence >= 0.8 ? "text-success" : confidence >= 0.6 ? "text-trust" : confidence >= 0.4 ? "text-warning" : "text-destructive"
              )}>
                {(confidence * 100).toFixed(0)}%
              </span>
            </div>
            <Progress value={confidence * 100} className="h-2" />
            <p className="text-[10px] text-muted-foreground mt-1.5">
              {confidence >= 0.8
                ? "High confidence — projections are based on strong signal from recent optimization cycles."
                : confidence >= 0.6
                  ? "Moderate confidence — longer horizons introduce uncertainty from external factors."
                  : confidence >= 0.4
                    ? "Lower confidence — extended projections carry significant estimation uncertainty."
                    : "Low confidence — projections at this horizon are speculative."}
            </p>
          </div>

          <Separator className="opacity-40" />

          {/* Predictions */}
          <div>
            <div className="flex items-center gap-1.5 mb-2">
              <BarChart3 className="h-3 w-3 text-receipt" />
              <span className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">
                Projected Metrics
              </span>
            </div>
            <div className="divide-y divide-border/20 max-h-[320px] overflow-y-auto dema-scrollbar rounded-lg border border-border/20">
              {horizonPredictions.map((pred, i) => (
                <PredictionRow key={`${pred.metric}-${i}`} prediction={pred} />
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Recommendations + Risks */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Recommendations */}
        <Card className="border-border/40 bg-card/60">
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-md bg-success/8">
                <Lightbulb className="h-4 w-4 text-success" />
              </div>
              <CardTitle className="text-sm">Recommendations</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {evolutionProjection.recommendations.map((rec, i) => (
                <div key={i} className="flex items-start gap-3">
                  <div className="flex items-center justify-center w-5 h-5 rounded-full bg-success/10 text-success text-[10px] font-bold shrink-0 mt-0.5">
                    {i + 1}
                  </div>
                  <p className="text-xs text-muted-foreground leading-relaxed">{rec}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Risks */}
        <Card className="border-border/40 bg-card/60">
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-md bg-destructive/8">
                <AlertTriangle className="h-4 w-4 text-destructive" />
              </div>
              <CardTitle className="text-sm">Risks</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {evolutionProjection.risks.map((risk, i) => (
                <div key={i} className="flex items-start gap-3 p-3 rounded-lg bg-destructive/5 border border-destructive/10">
                  <AlertTriangle className="h-3.5 w-3.5 text-destructive mt-0.5 shrink-0" />
                  <p className="text-xs text-muted-foreground leading-relaxed">{risk}</p>
                </div>
              ))}
              {evolutionProjection.risks.length === 0 && (
                <div className="flex items-center gap-2 py-4 justify-center text-muted-foreground">
                  <CheckCircle2 className="h-4 w-4 text-success" />
                  <span className="text-xs">No risks identified</span>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// Main Autopilot Screen
// ═══════════════════════════════════════════════════════════════

export function AutopilotScreen() {
  return (
    <div className="space-y-4 p-6 max-w-6xl mx-auto dema-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold tracking-tight">Autopilot</h1>
            <Badge variant="outline" className="text-[10px] px-2 py-0 text-trust border-trust/30 bg-trust/5">
              <Sparkles className="h-3 w-3 mr-1" />
              Self-Optimization
            </Badge>
          </div>
          <p className="text-sm text-muted-foreground mt-1">
            Autonomous optimization engine. System metrics, optimization cycles, and evolution projections.
          </p>
        </div>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button variant="outline" size="sm" className="h-8 text-xs text-muted-foreground">
              <RefreshCw className="h-3.5 w-3.5 mr-1.5" />
              Refresh
            </Button>
          </TooltipTrigger>
          <TooltipContent>Refresh autopilot data</TooltipContent>
        </Tooltip>
      </div>

      {/* Tabs */}
      <Tabs defaultValue="metrics" className="space-y-4">
        <TabsList className="bg-muted/30">
          <TabsTrigger value="metrics" className="text-xs">
            <Activity className="h-3.5 w-3.5 mr-1.5" />
            System Metrics
          </TabsTrigger>
          <TabsTrigger value="cycles" className="text-xs">
            <RefreshCw className="h-3.5 w-3.5 mr-1.5" />
            Optimization Cycles
          </TabsTrigger>
          <TabsTrigger value="projection" className="text-xs">
            <Brain className="h-3.5 w-3.5 mr-1.5" />
            Evolution Projection
          </TabsTrigger>
        </TabsList>

        <TabsContent value="metrics">
          <ScrollArea className="max-h-[calc(100vh-240px)]">
            <SystemMetricsTab />
          </ScrollArea>
        </TabsContent>

        <TabsContent value="cycles">
          <ScrollArea className="max-h-[calc(100vh-240px)]">
            <OptimizationCyclesTab />
          </ScrollArea>
        </TabsContent>

        <TabsContent value="projection">
          <ScrollArea className="max-h-[calc(100vh-240px)]">
            <EvolutionProjectionTab />
          </ScrollArea>
        </TabsContent>
      </Tabs>
    </div>
  );
}
