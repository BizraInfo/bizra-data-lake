"use client";

import { useState, useMemo, useRef, useCallback } from "react";
import { useDEMAStore } from "@/lib/store";
import { cn } from "@/lib/utils";
import { timeAgo } from "@/lib/helpers/dema";
import type { SystemComponent, TelemetryLevel, TelemetryEvent } from "@/lib/types";

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
  Activity,
  Server,
  Cpu,
  MemoryStick,
  HardDrive,
  Wifi,
  Clock,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Info,
  AlertCircle,
  Zap,
  Eye,
  Radio,
  Gauge,
  TrendingUp,
  ArrowUpRight,
  ArrowDownRight,
  Minus,
  Shield,
  FileCheck,
  Brain,
  Scale,
  Trash2,
  ChevronDown,
  ChevronRight,
  Layers,
  RadioTower,
} from "lucide-react";

// ═══════════════════════════════════════════════════════════════
// Constants & Mappings
// ═══════════════════════════════════════════════════════════════

const COMPONENT_CONFIG: Record<
  SystemComponent,
  {
    label: string;
    icon: typeof Server;
    color: string;
    bg: string;
  }
> = {
  gateway: { label: "Gateway", icon: Server, color: "text-trust", bg: "bg-trust/10" },
  agent_runtime: { label: "Agent Runtime", icon: Cpu, color: "text-manifest", bg: "bg-manifest/10" },
  trust_engine: { label: "Trust Engine", icon: Shield, color: "text-success", bg: "bg-success/10" },
  receipt_chain: { label: "Receipt Chain", icon: FileCheck, color: "text-receipt", bg: "bg-receipt/10" },
  memory_store: { label: "Memory Store", icon: MemoryStick, color: "text-warning", bg: "bg-warning/10" },
  resource_registry: { label: "Resource Registry", icon: Server, color: "text-action", bg: "bg-action/10" },
  optimization_engine: { label: "Optimization Engine", icon: Brain, color: "text-manifest", bg: "bg-manifest/10" },
  governance_layer: { label: "Governance Layer", icon: Scale, color: "text-trust", bg: "bg-trust/10" },
};

const STATUS_CONFIG: Record<string, { dot: string; label: string; textColor: string }> = {
  healthy: { dot: "bg-success", label: "Healthy", textColor: "text-success" },
  degraded: { dot: "bg-warning dema-pulse", label: "Degraded", textColor: "text-warning" },
  down: { dot: "bg-destructive dema-pulse", label: "Down", textColor: "text-destructive" },
  unknown: { dot: "bg-muted-foreground", label: "Unknown", textColor: "text-muted-foreground" },
};

const LEVEL_CONFIG: Record<
  TelemetryLevel,
  { icon: typeof Info; color: string; bg: string; badge: "default" | "secondary" | "destructive" | "outline" }
> = {
  debug: { icon: Info, color: "text-muted-foreground", bg: "bg-muted", badge: "outline" },
  info: { icon: Info, color: "text-trust", bg: "bg-trust/10", badge: "outline" },
  warn: { icon: AlertTriangle, color: "text-warning", bg: "bg-warning/10", badge: "outline" },
  error: { icon: XCircle, color: "text-destructive", bg: "bg-destructive/10", badge: "destructive" },
  fatal: { icon: AlertCircle, color: "text-destructive", bg: "bg-destructive/10", badge: "destructive" },
};

const COMPONENT_FILTERS: { value: string; label: string }[] = [
  { value: "all", label: "All" },
  { value: "gateway", label: "Gateway" },
  { value: "agent_runtime", label: "Agent Runtime" },
  { value: "trust_engine", label: "Trust Engine" },
  { value: "receipt_chain", label: "Receipt Chain" },
  { value: "memory_store", label: "Memory Store" },
  { value: "resource_registry", label: "Resource Registry" },
  { value: "optimization_engine", label: "Optimization" },
  { value: "governance_layer", label: "Governance" },
];

const LEVEL_FILTERS: { value: string; label: string }[] = [
  { value: "all", label: "All" },
  { value: "debug", label: "Debug" },
  { value: "info", label: "Info" },
  { value: "warn", label: "Warn" },
  { value: "error", label: "Error" },
  { value: "fatal", label: "Fatal" },
];

// ═══════════════════════════════════════════════════════════════
// Utility Functions
// ═══════════════════════════════════════════════════════════════

function formatUptime(seconds: number): string {
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  if (d > 0) return `${d}d ${h}h`;
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}

function formatTimestampShort(ts: string): string {
  const d = new Date(ts);
  return d.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

// ═══════════════════════════════════════════════════════════════
// Tab 1: System Health
// ═══════════════════════════════════════════════════════════════

function HealthSummaryBar({ systemHealth }: { systemHealth: ReturnType<typeof useDEMAStore>["systemHealth"] }) {
  const healthy = systemHealth.filter((h) => h.status === "healthy").length;
  const degraded = systemHealth.filter((h) => h.status === "degraded").length;
  const down = systemHealth.filter((h) => h.status === "down").length;
  const unknown = systemHealth.filter((h) => h.status === "unknown").length;
  const total = systemHealth.length;

  const overallStatus =
    down > 0 ? "down" : degraded > 2 ? "degraded" : degraded > 0 ? "degraded" : "healthy";
  const statusCfg = STATUS_CONFIG[overallStatus];

  return (
    <Card className="border-border/50 bg-card/50">
      <CardContent className="p-4">
        <div className="flex flex-col sm:flex-row sm:items-center gap-3">
          <div className="flex items-center gap-3">
            <div
              className={cn(
                "w-3 h-3 rounded-full",
                statusCfg.dot
              )}
            />
            <div>
              <div className="text-sm font-semibold">
                System Health:{" "}
                <span className={statusCfg.textColor}>{statusCfg.label}</span>
              </div>
              <div className="text-xs text-muted-foreground">
                {healthy}/{total} healthy
              </div>
            </div>
          </div>

          <div className="flex-1">
            <div className="h-2.5 rounded-full bg-muted overflow-hidden flex">
              {healthy > 0 && (
                <div
                  className="bg-success h-full transition-all"
                  style={{ width: `${(healthy / total) * 100}%` }}
                />
              )}
              {degraded > 0 && (
                <div
                  className="bg-warning h-full transition-all"
                  style={{ width: `${(degraded / total) * 100}%` }}
                />
              )}
              {down > 0 && (
                <div
                  className="bg-destructive h-full transition-all"
                  style={{ width: `${(down / total) * 100}%` }}
                />
              )}
              {unknown > 0 && (
                <div
                  className="bg-muted-foreground h-full transition-all"
                  style={{ width: `${(unknown / total) * 100}%` }}
                />
              )}
            </div>
          </div>

          <div className="flex items-center gap-3 text-xs">
            <div className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full bg-success" />
              <span className="text-muted-foreground">{healthy} Healthy</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full bg-warning" />
              <span className="text-muted-foreground">{degraded} Degraded</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full bg-destructive" />
              <span className="text-muted-foreground">{down} Down</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function HealthCard({ health }: { health: ReturnType<typeof useDEMAStore>["systemHealth"][number] }) {
  const config = COMPONENT_CONFIG[health.component];
  const statusCfg = STATUS_CONFIG[health.status];
  const Icon = config.icon;

  return (
    <Card
      className={cn(
        "border-border/50 bg-card/50 transition-colors hover:border-border",
        health.status === "degraded" && "border-warning/30 hover:border-warning/50",
        health.status === "down" && "border-destructive/30 hover:border-destructive/50"
      )}
    >
      <CardContent className="p-4">
        {/* Header */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2.5">
            <div className={cn("p-1.5 rounded-md", config.bg)}>
              <Icon className={cn("h-4 w-4", config.color)} />
            </div>
            <div>
              <div className="text-xs font-semibold">{config.label}</div>
              <div className="text-[10px] text-muted-foreground">{health.component}</div>
            </div>
          </div>
          <Badge
            variant="outline"
            className={cn(
              "text-[10px] px-1.5 py-0 gap-1",
              health.status === "healthy" && "text-success border-success/30 bg-success/5",
              health.status === "degraded" && "text-warning border-warning/30 bg-warning/5",
              health.status === "down" && "text-destructive border-destructive/30 bg-destructive/5"
            )}
          >
            <div className={cn("w-1.5 h-1.5 rounded-full", statusCfg.dot)} />
            {statusCfg.label}
          </Badge>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-2 gap-x-4 gap-y-2">
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider">Uptime</div>
            <div className="dema-mono text-xs font-medium mt-0.5">
              {formatUptime(health.uptime)}
            </div>
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider">Latency</div>
            <div className="dema-mono text-xs font-medium mt-0.5">
              {health.latency}
              <span className="text-muted-foreground ml-0.5">ms</span>
            </div>
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider">Error Rate</div>
            <div className="dema-mono text-xs font-medium mt-0.5">
              {(health.errorRate * 100).toFixed(2)}
              <span className="text-muted-foreground ml-0.5">%</span>
            </div>
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider">Throughput</div>
            <div className="dema-mono text-xs font-medium mt-0.5">
              {health.throughput}
              <span className="text-muted-foreground ml-0.5">req/s</span>
            </div>
          </div>
        </div>

        {/* Last Check */}
        <Separator className="my-3" />
        <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
          <Clock className="h-3 w-3" />
          <span>Last check: {timeAgo(health.lastCheck)}</span>
        </div>
      </CardContent>
    </Card>
  );
}

function SystemHealthTab() {
  const { systemHealth } = useDEMAStore();

  return (
    <div className="space-y-4">
      <HealthSummaryBar systemHealth={systemHealth} />
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3">
        {systemHealth.map((h) => (
          <HealthCard key={h.component} health={h} />
        ))}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// Tab 2: Telemetry Stream
// ═══════════════════════════════════════════════════════════════

function TelemetryEventCard({
  event,
  expanded,
  onToggle,
}: {
  event: TelemetryEvent;
  expanded: boolean;
  onToggle: () => void;
}) {
  const levelCfg = LEVEL_CONFIG[event.level];
  const compCfg = COMPONENT_CONFIG[event.component];
  const LevelIcon = levelCfg.icon;
  const hasMetadata = event.metadata && Object.keys(event.metadata).length > 0;

  return (
    <div
      className={cn(
        "border-b border-border/30 last:border-0 transition-colors",
        event.level === "error" && "bg-destructive/[0.03]",
        event.level === "fatal" && "bg-destructive/[0.05]",
        "hover:bg-accent/20"
      )}
    >
      <div
        className="flex items-start gap-3 px-3 py-2.5 cursor-pointer"
        onClick={onToggle}
      >
        {/* Level Icon */}
        <div className={cn("p-1 rounded mt-0.5 shrink-0", levelCfg.bg)}>
          <LevelIcon
            className={cn(
              "h-3.5 w-3.5",
              levelCfg.color,
              (event.level === "fatal" || event.level === "error") && "dema-pulse"
            )}
          />
        </div>

        {/* Content */}
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <Badge
              variant={levelCfg.badge}
              className={cn(
                "text-[9px] px-1 py-0 h-4 font-mono uppercase",
                event.level === "error" && "bg-destructive/10 text-destructive border-destructive/20",
                event.level === "fatal" && "bg-destructive/10 text-destructive border-destructive/20"
              )}
            >
              {event.level}
            </Badge>
            <Badge
              variant="outline"
              className="text-[9px] px-1 py-0 h-4"
            >
              {compCfg.label}
            </Badge>
          </div>
          <p className="text-xs mt-1.5 leading-relaxed">{event.message}</p>

          {/* Footer */}
          <div className="flex items-center gap-3 mt-1.5 text-[10px] text-muted-foreground">
            <span className="dema-mono">{formatTimestampShort(event.timestamp)}</span>
            {event.traceId && (
              <>
                <span className="text-border">·</span>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span className="dema-mono text-trust/60 hover:text-trust cursor-help transition-colors">
                      {event.traceId.slice(0, 12)}…
                    </span>
                  </TooltipTrigger>
                  <TooltipContent side="top" className="font-mono text-[10px]">
                    {event.traceId}
                  </TooltipContent>
                </Tooltip>
              </>
            )}
            {hasMetadata && (
              <>
                <span className="text-border">·</span>
                <span className="text-muted-foreground/60">
                  {expanded ? "hide" : "show"} metadata
                </span>
              </>
            )}
          </div>
        </div>

        {/* Expand chevron */}
        {hasMetadata && (
          <div className="mt-1 shrink-0">
            {expanded ? (
              <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
            )}
          </div>
        )}
      </div>

      {/* Expandable Metadata */}
      {expanded && hasMetadata && (
        <div className="px-3 pb-2.5 pl-12">
          <div className="bg-muted/40 rounded-md border border-border/30 p-2.5">
            <div className="dema-mono text-[11px] leading-relaxed whitespace-pre-wrap text-muted-foreground">
              {JSON.stringify(event.metadata, null, 2)}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function TelemetryStreamTab() {
  const { telemetryEvents } = useDEMAStore();
  const [levelFilter, setLevelFilter] = useState("all");
  const [componentFilter, setComponentFilter] = useState("all");
  const [autoScroll, setAutoScroll] = useState(true);
  const [expandedEvents, setExpandedEvents] = useState<Set<string>>(new Set());
  const scrollRef = useRef<HTMLDivElement>(null);

  const filteredEvents = useMemo(() => {
    return telemetryEvents.filter((e) => {
      if (levelFilter !== "all" && e.level !== levelFilter) return false;
      if (componentFilter !== "all" && e.component !== componentFilter) return false;
      return true;
    });
  }, [telemetryEvents, levelFilter, componentFilter]);

  // Calculate event rate (events per minute from last event)
  const eventRate = useMemo(() => {
    if (telemetryEvents.length < 2) return 0;
    const newest = new Date(telemetryEvents[0].timestamp).getTime();
    const oldest = new Date(
      telemetryEvents[Math.min(telemetryEvents.length - 1, 20)].timestamp
    ).getTime();
    const minutes = (newest - oldest) / 60000;
    if (minutes < 0.5) return telemetryEvents.length * 2;
    return Math.round(
      (Math.min(telemetryEvents.length, 20) / minutes) * 60
    );
  }, [telemetryEvents]);

  const toggleExpanded = useCallback((id: string) => {
    setExpandedEvents((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }, []);

  return (
    <div className="space-y-3">
      {/* Controls Bar */}
      <Card className="border-border/50 bg-card/50">
        <CardContent className="p-3">
          <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-center justify-between">
            <div className="flex items-center gap-2 flex-wrap">
              {/* Level Filter */}
              <div className="flex items-center gap-1 bg-muted/50 rounded-md p-0.5">
                {LEVEL_FILTERS.map((f) => (
                  <Button
                    key={f.value}
                    variant="ghost"
                    size="sm"
                    className={cn(
                      "h-6 px-2 text-[10px] font-medium",
                      levelFilter === f.value &&
                        "bg-background shadow-sm text-foreground"
                    )}
                    onClick={() => setLevelFilter(f.value)}
                  >
                    {f.label}
                  </Button>
                ))}
              </div>

              {/* Component Filter */}
              <div className="flex items-center gap-1 bg-muted/50 rounded-md p-0.5 overflow-x-auto max-w-[320px]">
                {COMPONENT_FILTERS.map((f) => (
                  <Button
                    key={f.value}
                    variant="ghost"
                    size="sm"
                    className={cn(
                      "h-6 px-2 text-[10px] font-medium whitespace-nowrap shrink-0",
                      componentFilter === f.value &&
                        "bg-background shadow-sm text-foreground"
                    )}
                    onClick={() => setComponentFilter(f.value)}
                  >
                    {f.label}
                  </Button>
                ))}
              </div>
            </div>

            <div className="flex items-center gap-2 shrink-0">
              {/* Event Rate Indicator */}
              <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
                <Radio className="h-3 w-3" />
                <span className="dema-mono">{eventRate} evt/min</span>
              </div>

              <Separator orientation="vertical" className="h-4" />

              {/* Auto-scroll toggle */}
              <Button
                variant="ghost"
                size="sm"
                className={cn(
                  "h-6 px-2 text-[10px] gap-1",
                  autoScroll && "text-trust bg-trust/5"
                )}
                onClick={() => setAutoScroll(!autoScroll)}
              >
                <Eye className="h-3 w-3" />
                Auto-scroll
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Event Stream */}
      <Card className="border-border/50 bg-card/50">
        <CardHeader className="pb-2 pt-3 px-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <RadioTower className="h-4 w-4 text-trust" />
              <CardTitle className="text-sm font-medium">Event Stream</CardTitle>
              <Badge variant="outline" className="text-[10px] px-1.5 py-0 dema-mono">
                {filteredEvents.length}
              </Badge>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          <div
            ref={scrollRef}
            className="max-h-[480px] overflow-y-auto dema-scrollbar"
          >
            {filteredEvents.length === 0 ? (
              <div className="flex items-center justify-center py-12 text-sm text-muted-foreground">
                No events match the current filters.
              </div>
            ) : (
              <div>
                {filteredEvents.map((event) => (
                  <TelemetryEventCard
                    key={event.id}
                    event={event}
                    expanded={expandedEvents.has(event.id)}
                    onToggle={() => toggleExpanded(event.id)}
                  />
                ))}
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// Tab 3: Performance Monitor
// ═══════════════════════════════════════════════════════════════

function MetricCard({
  label,
  value,
  unit,
  icon: Icon,
  color,
  bg,
  trend,
  progress,
}: {
  label: string;
  value: string | number;
  unit: string;
  icon: typeof Cpu;
  color: string;
  bg: string;
  trend?: "up" | "down" | "stable";
  progress?: number;
}) {
  const trendIcon =
    trend === "up" ? (
      <ArrowUpRight className="h-3 w-3 text-success" />
    ) : trend === "down" ? (
      <ArrowDownRight className="h-3 w-3 text-destructive" />
    ) : (
      <Minus className="h-3 w-3 text-muted-foreground" />
    );

  return (
    <Card className="border-border/50 bg-card/50">
      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <div className={cn("p-1 rounded", bg)}>
              <Icon className={cn("h-3.5 w-3.5", color)} />
            </div>
            <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
              {label}
            </span>
          </div>
          {trend && trendIcon}
        </div>
        <div className="dema-mono text-xl font-bold tracking-tight">
          {value}
          <span className="text-xs text-muted-foreground ml-1 font-normal">
            {unit}
          </span>
        </div>
        {progress !== undefined && (
          <Progress value={progress} className="h-1 mt-2" />
        )}
      </CardContent>
    </Card>
  );
}

function LatencyDistributionCard({
  latest,
}: {
  latest: ReturnType<typeof useDEMAStore>["performanceSnapshots"][number];
}) {
  const maxLatency = Math.max(latest.p99Latency, latest.p95Latency, latest.p50Latency);

  const bars = [
    { label: "P50", value: latest.p50Latency, color: "bg-success" },
    { label: "P95", value: latest.p95Latency, color: "bg-warning" },
    { label: "P99", value: latest.p99Latency, color: "bg-destructive" },
  ];

  return (
    <Card className="border-border/50 bg-card/50">
      <CardHeader className="pb-2 pt-4 px-4">
        <div className="flex items-center gap-2">
          <Gauge className="h-4 w-4 text-manifest" />
          <CardTitle className="text-sm font-medium">Latency Distribution</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="px-4 pb-4 space-y-3">
        {bars.map((bar) => (
          <div key={bar.label}>
            <div className="flex items-center justify-between mb-1">
              <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
                {bar.label}
              </span>
              <span className="dema-mono text-xs font-medium">
                {bar.value}
                <span className="text-muted-foreground ml-0.5">ms</span>
              </span>
            </div>
            <div className="h-2 bg-muted rounded-full overflow-hidden">
              <div
                className={cn("h-full rounded-full transition-all duration-500", bar.color)}
                style={{ width: `${(bar.value / maxLatency) * 100}%` }}
              />
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

function PerformanceLineChart({
  snapshots,
}: {
  snapshots: ReturnType<typeof useDEMAStore>["performanceSnapshots"];
}) {
  const width = 700;
  const height = 220;
  const padding = { top: 20, right: 16, bottom: 28, left: 42 };
  const chartW = width - padding.left - padding.right;
  const chartH = height - padding.top - padding.bottom;

  // Use the last 30 snapshots
  const data = snapshots.slice(-30);

  // Compute scales
  const cpuMax = Math.ceil(Math.max(...data.map((d) => d.cpu), 40) / 10) * 10;
  const memMax = Math.ceil(Math.max(...data.map((d) => d.memory), 80) / 10) * 10;
  const reqMax = Math.ceil(Math.max(...data.map((d) => d.requestRate), 1200) / 200) * 200;

  const yMax = Math.max(cpuMax, memMax, (reqMax / 20)); // Normalize req to similar scale

  // Build polyline points
  function toPoints(
    accessor: (d: (typeof data)[number]) => number,
    maxVal: number
  ): string {
    return data
      .map((d, i) => {
        const x = padding.left + (i / Math.max(data.length - 1, 1)) * chartW;
        const y = padding.top + chartH - (accessor(d) / maxVal) * chartH;
        return `${x},${y}`;
      })
      .join(" ");
  }

  const cpuPoints = toPoints((d) => d.cpu, yMax);
  const memPoints = toPoints((d) => d.memory, yMax);
  const reqPoints = toPoints((d) => d.requestRate / 20, yMax); // Scale down request rate

  // Grid lines
  const gridLines = 4;
  const yTicks: { y: number; label: string }[] = [];
  for (let i = 0; i <= gridLines; i++) {
    const val = Math.round((yMax / gridLines) * i);
    const yPos = padding.top + chartH - (i / gridLines) * chartH;
    yTicks.push({ y: yPos, label: `${val}` });
  }

  // X-axis time labels
  const xLabels: { x: number; label: string }[] = [];
  const labelInterval = Math.max(Math.floor(data.length / 5), 1);
  for (let i = 0; i < data.length; i += labelInterval) {
    const xPos = padding.left + (i / Math.max(data.length - 1, 1)) * chartW;
    const d = new Date(data[i].timestamp);
    xLabels.push({
      x: xPos,
      label: d.toLocaleTimeString("en-US", {
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
      }),
    });
  }

  return (
    <Card className="border-border/50 bg-card/50">
      <CardHeader className="pb-2 pt-4 px-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <TrendingUp className="h-4 w-4 text-trust" />
            <CardTitle className="text-sm font-medium">
              Performance Over Time
            </CardTitle>
            <Badge variant="outline" className="text-[10px] px-1.5 py-0">
              Last 30 snapshots
            </Badge>
          </div>
        </div>
        {/* Legend */}
        <div className="flex items-center gap-4 mt-1">
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-[2px] bg-manifest rounded" />
            <span className="text-[10px] text-muted-foreground">CPU %</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-[2px] bg-success rounded" />
            <span className="text-[10px] text-muted-foreground">Memory %</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-[2px] bg-warning rounded" />
            <span className="text-[10px] text-muted-foreground">Req Rate (÷20)</span>
          </div>
        </div>
      </CardHeader>
      <CardContent className="px-4 pb-4">
        <div className="w-full overflow-x-auto">
          <svg
            viewBox={`0 0 ${width} ${height}`}
            className="w-full h-auto"
            preserveAspectRatio="xMidYMid meet"
          >
            {/* Background */}
            <rect
              x={padding.left}
              y={padding.top}
              width={chartW}
              height={chartH}
              fill="currentColor"
              className="text-muted/20"
              rx="2"
            />

            {/* Grid lines */}
            {yTicks.map((tick, i) => (
              <g key={i}>
                <line
                  x1={padding.left}
                  y1={tick.y}
                  x2={padding.left + chartW}
                  y2={tick.y}
                  stroke="currentColor"
                  className="text-border/40"
                  strokeWidth="0.5"
                  strokeDasharray={i === 0 ? "none" : "3,3"}
                />
                <text
                  x={padding.left - 6}
                  y={tick.y + 3}
                  textAnchor="end"
                  className="fill-muted-foreground"
                  fontSize="9"
                  fontFamily="ui-monospace, monospace"
                >
                  {tick.label}
                </text>
              </g>
            ))}

            {/* X-axis labels */}
            {xLabels.map((lbl, i) => (
              <text
                key={i}
                x={lbl.x}
                y={height - 4}
                textAnchor="middle"
                className="fill-muted-foreground"
                fontSize="8"
                fontFamily="ui-monospace, monospace"
              >
                {lbl.label}
              </text>
            ))}

            {/* CPU Line + Area */}
            <polyline
              points={cpuPoints}
              fill="none"
              stroke="currentColor"
              className="text-manifest"
              strokeWidth="1.5"
              strokeLinejoin="round"
              strokeLinecap="round"
            />

            {/* Memory Line */}
            <polyline
              points={memPoints}
              fill="none"
              stroke="currentColor"
              className="text-success"
              strokeWidth="1.5"
              strokeLinejoin="round"
              strokeLinecap="round"
            />

            {/* Request Rate Line */}
            <polyline
              points={reqPoints}
              fill="none"
              stroke="currentColor"
              className="text-warning"
              strokeWidth="1.5"
              strokeLinejoin="round"
              strokeLinecap="round"
              strokeDasharray="4,2"
            />

            {/* Current value dots */}
            {data.length > 0 && (
              <>
                <circle
                  cx={
                    padding.left +
                    ((data.length - 1) / Math.max(data.length - 1, 1)) * chartW
                  }
                  cy={
                    padding.top +
                    chartH -
                    (data[data.length - 1].cpu / yMax) * chartH
                  }
                  r="3"
                  className="fill-manifest stroke-background"
                  strokeWidth="1.5"
                />
                <circle
                  cx={
                    padding.left +
                    ((data.length - 1) / Math.max(data.length - 1, 1)) * chartW
                  }
                  cy={
                    padding.top +
                    chartH -
                    (data[data.length - 1].memory / yMax) * chartH
                  }
                  r="3"
                  className="fill-success stroke-background"
                  strokeWidth="1.5"
                />
                <circle
                  cx={
                    padding.left +
                    ((data.length - 1) / Math.max(data.length - 1, 1)) * chartW
                  }
                  cy={
                    padding.top +
                    chartH -
                    (data[data.length - 1].requestRate / 20 / yMax) * chartH
                  }
                  r="3"
                  className="fill-warning stroke-background"
                  strokeWidth="1.5"
                />
              </>
            )}
          </svg>
        </div>
      </CardContent>
    </Card>
  );
}

function PerformanceMonitorTab() {
  const { performanceSnapshots } = useDEMAStore();

  const latest = performanceSnapshots[performanceSnapshots.length - 1];
  if (!latest) {
    return (
      <Card className="border-border/50 bg-card/50">
        <CardContent className="flex items-center justify-center py-12 text-muted-foreground text-sm">
          No performance data available.
        </CardContent>
      </Card>
    );
  }

  // Compute trend for CPU (compare last vs 5 snapshots ago)
  const computeTrend = (
    accessor: (s: (typeof performanceSnapshots)[number]) => number
  ): "up" | "down" | "stable" => {
    if (performanceSnapshots.length < 5) return "stable";
    const curr = accessor(latest);
    const prev = accessor(
      performanceSnapshots[performanceSnapshots.length - 5]
    );
    const diff = curr - prev;
    if (Math.abs(diff) < 1) return "stable";
    return diff > 0 ? "up" : "down";
  };

  return (
    <div className="space-y-4">
      {/* Resource Metrics Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <MetricCard
          label="CPU Usage"
          value={latest.cpu}
          unit="%"
          icon={Cpu}
          color="text-manifest"
          bg="bg-manifest/10"
          trend={computeTrend((s) => s.cpu)}
          progress={latest.cpu}
        />
        <MetricCard
          label="Memory Usage"
          value={latest.memory}
          unit="%"
          icon={MemoryStick}
          color="text-success"
          bg="bg-success/10"
          trend={computeTrend((s) => s.memory)}
          progress={latest.memory}
        />
        <MetricCard
          label="Disk I/O"
          value={latest.diskIo}
          unit="MB/s"
          icon={HardDrive}
          color="text-action"
          bg="bg-action/10"
          trend={computeTrend((s) => s.diskIo)}
        />
        <MetricCard
          label="Network I/O"
          value={latest.networkIo}
          unit="MB/s"
          icon={Wifi}
          color="text-trust"
          bg="bg-trust/10"
          trend={computeTrend((s) => s.networkIo)}
        />
      </div>

      {/* Connections, Request Rate, Error Rate, Latency */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <div className="grid grid-cols-2 gap-3">
          <MetricCard
            label="Active Connections"
            value={latest.activeConnections}
            unit="conn"
            icon={Layers}
            color="text-receipt"
            bg="bg-receipt/10"
            trend={computeTrend((s) => s.activeConnections)}
          />
          <MetricCard
            label="Request Rate"
            value={latest.requestRate}
            unit="req/s"
            icon={Zap}
            color="text-warning"
            bg="bg-warning/10"
            trend={computeTrend((s) => s.requestRate)}
          />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <MetricCard
            label="Error Rate"
            value={latest.errorRate}
            unit="%"
            icon={AlertTriangle}
            color="text-destructive"
            bg="bg-destructive/10"
            trend={computeTrend((s) => s.errorRate)}
          />
          <MetricCard
            label="P95 Latency"
            value={latest.p95Latency}
            unit="ms"
            icon={Gauge}
            color="text-trust"
            bg="bg-trust/10"
            trend={computeTrend((s) => s.p95Latency)}
          />
        </div>
      </div>

      {/* Latency Distribution */}
      <LatencyDistributionCard latest={latest} />

      {/* Line Chart */}
      <PerformanceLineChart snapshots={performanceSnapshots} />
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// Main Export
// ═══════════════════════════════════════════════════════════════

export function OperationsScreen() {
  const { systemHealth } = useDEMAStore();

  // Count warnings/errors for the header
  const degradedCount = systemHealth.filter((h) => h.status === "degraded").length;
  const downCount = systemHealth.filter((h) => h.status === "down").length;

  return (
    <div className="space-y-4 p-6 max-w-7xl mx-auto dema-fade-in">
      {/* Header */}
      <div>
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-trust/10">
            <Activity className="h-5 w-5 text-trust" />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight">
              Operations Console
            </h1>
            <p className="text-sm text-muted-foreground mt-0.5">
              Real-time system health, telemetry, and performance monitoring.
            </p>
          </div>
        </div>

        {/* Status Alerts */}
        {(degradedCount > 0 || downCount > 0) && (
          <div className="flex items-center gap-3 mt-3">
            {downCount > 0 && (
              <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md bg-destructive/10 border border-destructive/20 text-destructive text-xs">
                <XCircle className="h-3.5 w-3.5" />
                <span className="font-medium">
                  {downCount} component{downCount > 1 ? "s" : ""} down
                </span>
              </div>
            )}
            {degradedCount > 0 && (
              <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md bg-warning/10 border border-warning/20 text-warning text-xs">
                <AlertTriangle className="h-3.5 w-3.5" />
                <span className="font-medium">
                  {degradedCount} component{degradedCount > 1 ? "s" : ""} degraded
                </span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Tabs */}
      <Tabs defaultValue="system-health" className="space-y-4">
        <TabsList>
          <TabsTrigger value="system-health" className="gap-1.5">
            <CheckCircle2 className="h-3.5 w-3.5" />
            System Health
          </TabsTrigger>
          <TabsTrigger value="telemetry" className="gap-1.5">
            <RadioTower className="h-3.5 w-3.5" />
            Telemetry Stream
          </TabsTrigger>
          <TabsTrigger value="performance" className="gap-1.5">
            <Gauge className="h-3.5 w-3.5" />
            Performance Monitor
          </TabsTrigger>
        </TabsList>

        <TabsContent value="system-health">
          <SystemHealthTab />
        </TabsContent>

        <TabsContent value="telemetry">
          <TelemetryStreamTab />
        </TabsContent>

        <TabsContent value="performance">
          <PerformanceMonitorTab />
        </TabsContent>
      </Tabs>
    </div>
  );
}
