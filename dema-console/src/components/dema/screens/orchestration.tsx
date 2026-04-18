"use client";

import { useState, useMemo } from "react";
import { useDEMAStore } from "@/lib/store";
import { cn } from "@/lib/utils";
import { timeAgo, formatId } from "@/lib/helpers/dema";
import type {
  Agent,
  AgentTask,
  AgentStatus,
  AgentRole,
  AgentCapability,
  OrchestrationEvent,
} from "@/lib/types";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Bot,
  Cpu,
  CheckCircle2,
  XCircle,
  Clock,
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  Crown,
  Activity,
  ArrowUpDown,
  Network,
  ListOrdered,
  Radio,
  Shield,
  Search,
  Brain,
  Code2,
  Globe,
  Terminal,
  Monitor,
  Eye,
  Wrench,
  Lock,
  BarChart3,
  Zap,
  UserCheck,
  Timer,
  Filter,
} from "lucide-react";

// ═══════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════

function formatUptime(seconds: number): string {
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${d}d ${h}h ${m}m`;
}

function agentStatusConfig(status: AgentStatus) {
  switch (status) {
    case "active":
      return { dot: "bg-success", label: "Active", text: "text-success" };
    case "busy":
      return { dot: "bg-warning", label: "Busy", text: "text-warning" };
    case "idle":
      return { dot: "bg-muted-foreground", label: "Idle", text: "text-muted-foreground" };
    case "error":
      return { dot: "bg-destructive", label: "Error", text: "text-destructive" };
    case "sleeping":
      return { dot: "bg-muted-foreground/60", label: "Sleeping", text: "text-muted-foreground" };
    case "terminated":
      return { dot: "bg-destructive/60", label: "Terminated", text: "text-muted-foreground" };
    default:
      return { dot: "bg-muted-foreground", label: status, text: "text-muted-foreground" };
  }
}

function roleBadgeConfig(role: AgentRole) {
  switch (role) {
    case "coordinator":
      return { label: "Coordinator", className: "bg-trust/15 text-trust border-trust/25" };
    case "researcher":
      return { label: "Researcher", className: "bg-receipt/15 text-receipt border-receipt/25" };
    case "executor":
      return { label: "Executor", className: "bg-action/15 text-action border-action/25" };
    case "verifier":
      return { label: "Verifier", className: "bg-success/15 text-success border-success/25" };
    case "observer":
      return { label: "Observer", className: "bg-manifest/15 text-manifest border-manifest/25" };
    case "optimizer":
      return { label: "Optimizer", className: "bg-warning/15 text-warning border-warning/25" };
    case "guardian":
      return { label: "Guardian", className: "bg-destructive/15 text-destructive border-destructive/25" };
    default:
      return { label: role, className: "bg-muted text-muted-foreground border-border" };
  }
}

const ROLE_ICONS: Record<AgentRole, typeof Bot> = {
  coordinator: Crown,
  researcher: Brain,
  executor: Code2,
  verifier: Shield,
  observer: Eye,
  optimizer: Wrench,
  guardian: Lock,
};

const CAPABILITY_ICONS: Record<AgentCapability, typeof Bot> = {
  reasoning: Brain,
  code_gen: Code2,
  web_search: Search,
  file_io: Monitor,
  browser_auto: Globe,
  system_exec: Terminal,
  graph_analysis: Network,
  crypto_verify: Lock,
  telemetry: BarChart3,
  memory_mgmt: Cpu,
};

function priorityConfig(priority: AgentTask["priority"]) {
  switch (priority) {
    case "critical":
      return { label: "Critical", className: "bg-destructive/15 text-destructive border-destructive/25" };
    case "high":
      return { label: "High", className: "bg-warning/15 text-warning border-warning/25" };
    case "medium":
      return { label: "Medium", className: "bg-trust/15 text-trust border-trust/25" };
    case "low":
      return { label: "Low", className: "bg-muted text-muted-foreground border-border" };
  }
}

function taskStatusConfig(status: AgentTask["status"]) {
  switch (status) {
    case "executing":
      return { dot: "bg-trust dema-pulse", label: "Executing", text: "text-trust" };
    case "assigned":
      return { dot: "bg-receipt", label: "Assigned", text: "text-receipt" };
    case "queued":
      return { dot: "bg-muted-foreground", label: "Queued", text: "text-muted-foreground" };
    case "completed":
      return { dot: "bg-success", label: "Completed", text: "text-success" };
    case "failed":
      return { dot: "bg-destructive", label: "Failed", text: "text-destructive" };
    case "cancelled":
      return { dot: "bg-muted-foreground/50", label: "Cancelled", text: "text-muted-foreground" };
  }
}

const SEVERITY_CONFIG: Record<
  OrchestrationEvent["severity"],
  { icon: typeof CheckCircle2; color: string; bg: string }
> = {
  success: { icon: CheckCircle2, color: "text-success", bg: "bg-success/10" },
  info: { icon: Radio, color: "text-trust", bg: "bg-trust/10" },
  warning: { icon: AlertTriangle, color: "text-warning", bg: "bg-warning/10" },
  error: { icon: XCircle, color: "text-destructive", bg: "bg-destructive/10" },
};

function eventTypeLabel(type: OrchestrationEvent["type"]) {
  switch (type) {
    case "agent_spawn": return "Spawn";
    case "agent_status": return "Status";
    case "task_assigned": return "Assigned";
    case "task_completed": return "Completed";
    case "task_failed": return "Failed";
    case "coordination": return "Coordination";
    case "handoff": return "Handoff";
    case "termination": return "Termination";
    case "heartbeat": return "Heartbeat";
    default: return type;
  }
}

const TASK_STATUS_ORDER: AgentTask["status"][] = [
  "executing",
  "assigned",
  "queued",
  "completed",
  "failed",
  "cancelled",
];

// ═══════════════════════════════════════════════════════════════
// Sub-Components: Tab 1 — Agent Swarm
// ═══════════════════════════════════════════════════════════════

function SwarmHeader({
  agents,
  onFilterChange,
  filter,
  search,
  onSearchChange,
}: {
  agents: Agent[];
  onFilterChange: (v: string) => void;
  filter: string;
  search: string;
  onSearchChange: (v: string) => void;
}) {
  const activeCount = agents.filter((a) => a.status === "active").length;
  const busyCount = agents.filter((a) => a.status === "busy").length;
  const idleCount = agents.filter((a) => a.status === "idle").length;

  return (
    <div className="space-y-3">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Bot className="h-4.5 w-4.5 text-trust" />
            <h3 className="text-base font-semibold tracking-tight">
              Agent Swarm
            </h3>
          </div>
          <Badge variant="outline" className="font-mono text-xs px-2 py-0.5">
            {agents.length}
          </Badge>
        </div>

        <div className="flex items-center gap-2">
          <div className="relative flex-1 sm:w-48">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
            <Input
              placeholder="Search agents..."
              value={search}
              onChange={(e) => onSearchChange(e.target.value)}
              className="h-8 pl-8 text-xs"
            />
          </div>
          <Select value={filter} onValueChange={onFilterChange}>
            <SelectTrigger className="h-8 w-32 text-xs">
              <Filter className="h-3 w-3 mr-1" />
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Status</SelectItem>
              <SelectItem value="active">Active</SelectItem>
              <SelectItem value="busy">Busy</SelectItem>
              <SelectItem value="idle">Idle</SelectItem>
              <SelectItem value="error">Error</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="flex items-center gap-2 flex-wrap">
        <Badge
          variant="outline"
          className={cn(
            "text-[11px] border-success/30 text-success bg-success/5",
            activeCount === 0 && "opacity-50"
          )}
        >
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-success mr-1.5" />
          {activeCount} active
        </Badge>
        <Badge
          variant="outline"
          className={cn(
            "text-[11px] border-warning/30 text-warning bg-warning/5",
            busyCount === 0 && "opacity-50"
          )}
        >
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-warning mr-1.5" />
          {busyCount} busy
        </Badge>
        <Badge
          variant="outline"
          className={cn(
            "text-[11px] border-muted-foreground/30 text-muted-foreground bg-muted",
            idleCount === 0 && "opacity-50"
          )}
        >
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-muted-foreground mr-1.5" />
          {idleCount} idle
        </Badge>
      </div>
    </div>
  );
}

function AgentCard({ agent, agents }: { agent: Agent; agents: Agent[] }) {
  const statusCfg = agentStatusConfig(agent.status);
  const roleCfg = roleBadgeConfig(agent.role);
  const RoleIcon = ROLE_ICONS[agent.role] || Bot;
  const isCoordinator = agent.role === "coordinator";
  const parentAgent = agent.parentId
    ? agents.find((a) => a.id === agent.parentId)
    : null;
  const childAgents = agent.children
    ? agents.filter((a) => agent.children!.includes(a.id))
    : [];

  return (
    <Card
      className={cn(
        "border-border/50 bg-card/50 hover:bg-card/80 transition-all duration-200 group",
        isCoordinator && "ring-2 ring-trust/30 border-trust/20 shadow-sm shadow-trust/5"
      )}
    >
      <CardContent className="p-4 space-y-3.5">
        {/* Header row */}
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2.5 min-w-0">
            <div
              className={cn(
                "shrink-0 p-1.5 rounded-md transition-colors",
                isCoordinator ? "bg-trust/10" : "bg-muted/60"
              )}
            >
              <RoleIcon
                className={cn(
                  "h-4 w-4",
                  isCoordinator ? "text-trust" : "text-muted-foreground"
                )}
              />
            </div>
            <div className="min-w-0">
              <div className="flex items-center gap-1.5">
                <span className="text-sm font-semibold truncate">
                  {agent.name}
                </span>
                {isCoordinator && (
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Crown className="h-3 w-3 text-trust shrink-0" />
                    </TooltipTrigger>
                    <TooltipContent className="text-xs">
                      Coordinator Agent
                    </TooltipContent>
                  </Tooltip>
                )}
              </div>
              <div className="flex items-center gap-2 mt-0.5">
                <Badge
                  variant="outline"
                  className={cn("text-[10px] px-1.5 py-0", roleCfg.className)}
                >
                  {roleCfg.label}
                </Badge>
                <span className="text-[10px] font-mono text-muted-foreground">
                  {formatId(agent.id)}
                </span>
              </div>
            </div>
          </div>

          <Tooltip>
            <TooltipTrigger asChild>
              <div className="flex items-center gap-1.5 shrink-0 mt-0.5">
                <span
                  className={cn(
                    "w-2 h-2 rounded-full shrink-0",
                    statusCfg.dot
                  )}
                />
                <span className={cn("text-[11px] font-medium", statusCfg.text)}>
                  {statusCfg.label}
                </span>
              </div>
            </TooltipTrigger>
            <TooltipContent className="text-xs">
              Last activity: {timeAgo(agent.lastActivity)}
            </TooltipContent>
          </Tooltip>
        </div>

        {/* Trust score */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="text-[11px] text-muted-foreground font-medium">
              Trust Score
            </span>
            <span
              className={cn(
                "text-[11px] font-mono font-semibold",
                agent.trustScore >= 95
                  ? "text-success"
                  : agent.trustScore >= 85
                    ? "text-trust"
                    : agent.trustScore >= 70
                      ? "text-warning"
                      : "text-destructive"
              )}
            >
              {agent.trustScore}
            </span>
          </div>
          <Progress
            value={agent.trustScore}
            className={cn(
              "h-1.5",
              agent.trustScore >= 95
                ? "[&>div]:bg-success"
                : agent.trustScore >= 85
                  ? "[&>div]:bg-trust"
                  : agent.trustScore >= 70
                    ? "[&>div]:bg-warning"
                    : "[&>div]:bg-destructive"
            )}
          />
        </div>

        {/* Capabilities */}
        <div className="flex flex-wrap gap-1">
          {agent.capabilities.map((cap) => {
            const CapIcon = CAPABILITY_ICONS[cap] || Zap;
            return (
              <Tooltip key={cap}>
                <TooltipTrigger asChild>
                  <div className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md bg-muted/50 border border-border/30 hover:bg-muted/80 transition-colors cursor-default">
                    <CapIcon className="h-3 w-3 text-muted-foreground" />
                    <span className="text-[10px] text-muted-foreground hidden sm:inline">
                      {cap.replace(/_/g, " ")}
                    </span>
                  </div>
                </TooltipTrigger>
                <TooltipContent className="text-xs capitalize">
                  {cap.replace(/_/g, " ")}
                </TooltipContent>
              </Tooltip>
            );
          })}
        </div>

        <Separator className="opacity-50" />

        {/* Stats row */}
        <div className="grid grid-cols-3 gap-2">
          <div className="text-center">
            <div className="text-sm font-bold font-mono text-success">
              {agent.tasksCompleted.toLocaleString()}
            </div>
            <div className="text-[10px] text-muted-foreground">Completed</div>
          </div>
          <div className="text-center">
            <div className="text-sm font-bold font-mono text-destructive">
              {agent.tasksFailed}
            </div>
            <div className="text-[10px] text-muted-foreground">Failed</div>
          </div>
          <div className="text-center">
            <div className="text-sm font-bold font-mono text-foreground">
              {formatUptime(agent.uptime)}
            </div>
            <div className="text-[10px] text-muted-foreground">Uptime</div>
          </div>
        </div>

        {/* Parent / Child relationships */}
        {(parentAgent || childAgents.length > 0) && (
          <div className="space-y-1.5 pt-1">
            {parentAgent && (
              <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
                <ArrowUpDown className="h-3 w-3 rotate-90" />
                <span>
                  Parent:{" "}
                  <span className="font-medium text-trust">
                    {parentAgent.name}
                  </span>
                </span>
              </div>
            )}
            {childAgents.length > 0 && (
              <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
                <ArrowUpDown className="h-3 w-3" />
                <span>
                  Children:{" "}
                  {childAgents.map((c) => (
                    <span
                      key={c.id}
                      className="font-medium text-manifest mr-1.5"
                    >
                      {c.name}
                    </span>
                  ))}
                </span>
              </div>
            )}
          </div>
        )}

        {/* Model metadata */}
        <div className="flex items-center justify-between text-[10px] text-muted-foreground pt-0.5">
          <span>
            v{(agent.metadata.version as string) || "—"}
          </span>
          <span className="font-mono">
            {(agent.metadata.model as string) || "—"}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}

function AgentSwarmTab() {
  const { agents } = useDEMAStore();
  const [filter, setFilter] = useState("all");
  const [search, setSearch] = useState("");

  const filteredAgents = useMemo(() => {
    let list = agents;
    if (filter !== "all") {
      list = list.filter((a) => a.status === filter);
    }
    if (search.trim()) {
      const q = search.toLowerCase();
      list = list.filter(
        (a) =>
          a.name.toLowerCase().includes(q) ||
          a.role.toLowerCase().includes(q) ||
          a.capabilities.some((c) => c.toLowerCase().includes(q))
      );
    }
    return list;
  }, [agents, filter, search]);

  return (
    <div className="space-y-4">
      <SwarmHeader
        agents={agents}
        filter={filter}
        onFilterChange={setFilter}
        search={search}
        onSearchChange={setSearch}
      />
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
        {filteredAgents.map((agent) => (
          <AgentCard key={agent.id} agent={agent} agents={agents} />
        ))}
        {filteredAgents.length === 0 && (
          <div className="col-span-full flex flex-col items-center justify-center py-12 text-muted-foreground">
            <Bot className="h-8 w-8 mb-2 opacity-40" />
            <p className="text-sm">No agents match your filter.</p>
          </div>
        )}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// Sub-Components: Tab 2 — Task Queue
// ═══════════════════════════════════════════════════════════════

function TaskQueueHeader({ tasks }: { tasks: AgentTask[] }) {
  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const t of tasks) {
      c[t.status] = (c[t.status] || 0) + 1;
    }
    return c;
  }, [tasks]);

  const summary = [
    {
      label: "Executing",
      count: counts["executing"] || 0,
      color: "text-trust",
      bg: "bg-trust/5",
      border: "border-trust/20",
    },
    {
      label: "Assigned",
      count: counts["assigned"] || 0,
      color: "text-receipt",
      bg: "bg-receipt/5",
      border: "border-receipt/20",
    },
    {
      label: "Queued",
      count: counts["queued"] || 0,
      color: "text-muted-foreground",
      bg: "bg-muted/50",
      border: "border-border",
    },
    {
      label: "Completed",
      count: counts["completed"] || 0,
      color: "text-success",
      bg: "bg-success/5",
      border: "border-success/20",
    },
    {
      label: "Failed",
      count: counts["failed"] || 0,
      color: "text-destructive",
      bg: "bg-destructive/5",
      border: "border-destructive/20",
    },
  ];

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <ListOrdered className="h-4.5 w-4.5 text-receipt" />
        <h3 className="text-base font-semibold tracking-tight">Task Queue</h3>
        <Badge variant="outline" className="font-mono text-xs px-2 py-0.5">
          {tasks.length}
        </Badge>
      </div>

      <div className="flex items-center gap-2 flex-wrap">
        {summary.map((s) => (
          <Badge
            key={s.label}
            variant="outline"
            className={cn(
              "text-[11px] border px-2 py-0.5",
              s.bg,
              s.border,
              s.color,
              s.count === 0 && "opacity-50"
            )}
          >
            {s.count} {s.label}
          </Badge>
        ))}
      </div>
    </div>
  );
}

function TaskItem({
  task,
  agents,
}: {
  task: AgentTask;
  agents: Agent[];
}) {
  const statusCfg = taskStatusConfig(task.status);
  const priorityCfg = priorityConfig(task.priority);
  const agent = agents.find((a) => a.id === task.agentId);
  const [expanded, setExpanded] = useState(false);

  return (
    <Card className="border-border/50 bg-card/50 hover:bg-card/80 transition-colors">
      <CardContent className="p-3">
        <div className="flex items-start gap-3">
          {/* Status dot */}
          <div className="mt-1 shrink-0">
            <span className={cn("block w-2 h-2 rounded-full", statusCfg.dot)} />
          </div>

          <div className="min-w-0 flex-1">
            {/* Title row */}
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                <div className="text-xs font-medium leading-snug">
                  {task.title}
                </div>
                <p className="text-[11px] text-muted-foreground mt-0.5 line-clamp-1">
                  {task.description}
                </p>
              </div>
              <Badge
                variant="outline"
                className={cn(
                  "text-[10px] px-1.5 py-0 shrink-0",
                  priorityCfg.className
                )}
              >
                {priorityCfg.label}
              </Badge>
            </div>

            {/* Metadata row */}
            <div className="flex items-center gap-2 mt-2 flex-wrap">
              {agent && (
                <Badge
                  variant="outline"
                  className={cn(
                    "text-[10px] px-1.5 py-0",
                    roleBadgeConfig(agent.role).className
                  )}
                >
                  {agent.name}
                </Badge>
              )}
              <Badge
                variant="outline"
                className={cn("text-[10px] px-1.5 py-0", statusCfg.text)}
              >
                {statusCfg.label}
              </Badge>
              <span className="text-[10px] font-mono text-muted-foreground">
                {formatId(task.id)}
              </span>
            </div>

            {/* Timestamps */}
            <div className="flex items-center gap-3 mt-1.5 text-[10px] text-muted-foreground">
              {task.assignedAt && (
                <span className="flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  Assigned {timeAgo(task.assignedAt)}
                </span>
              )}
              {task.startedAt && (
                <span className="flex items-center gap-1">
                  <Timer className="h-3 w-3" />
                  Started {timeAgo(task.startedAt)}
                </span>
              )}
              {task.completedAt && (
                <span className="flex items-center gap-1 text-success">
                  <CheckCircle2 className="h-3 w-3" />
                  Completed {timeAgo(task.completedAt)}
                </span>
              )}
            </div>

            {/* Result preview for completed */}
            {task.status === "completed" && task.result && (
              <div className="mt-2 p-2 rounded-md bg-success/5 border border-success/15">
                <div className="flex items-center gap-1 mb-0.5">
                  <CheckCircle2 className="h-3 w-3 text-success" />
                  <span className="text-[10px] font-medium text-success">
                    Result
                  </span>
                </div>
                <p className="text-[11px] text-success/80 leading-relaxed">
                  {task.result}
                </p>
              </div>
            )}

            {/* Error message for failed */}
            {task.status === "failed" && task.error && (
              <div className="mt-2 p-2 rounded-md bg-destructive/5 border border-destructive/15">
                <div className="flex items-center gap-1 mb-0.5">
                  <XCircle className="h-3 w-3 text-destructive" />
                  <span className="text-[10px] font-medium text-destructive">
                    Error
                  </span>
                </div>
                <p className="text-[11px] text-destructive/80 leading-relaxed">
                  {task.error}
                </p>
              </div>
            )}

            {/* Expandable section for other statuses */}
            {task.status !== "completed" && task.status !== "failed" && (
              <button
                onClick={() => setExpanded(!expanded)}
                className="flex items-center gap-1 mt-2 text-[10px] text-muted-foreground hover:text-foreground transition-colors"
              >
                {expanded ? (
                  <ChevronDown className="h-3 w-3" />
                ) : (
                  <ChevronRight className="h-3 w-3" />
                )}
                Details
              </button>
            )}

            {expanded && (
              <div className="mt-2 p-2 rounded-md bg-muted/30 border border-border/30 space-y-1">
                <div className="text-[10px] text-muted-foreground">
                  <span className="font-medium">Agent ID:</span>{" "}
                  <span className="font-mono">{task.agentId}</span>
                </div>
                {task.description && (
                  <p className="text-[11px] text-muted-foreground leading-relaxed">
                    {task.description}
                  </p>
                )}
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function TaskQueueTab() {
  const { agents, agentTasks } = useDEMAStore();

  const groupedTasks = useMemo(() => {
    const groups: Record<string, AgentTask[]> = {};
    for (const status of TASK_STATUS_ORDER) {
      const tasks = agentTasks.filter((t) => t.status === status);
      if (tasks.length > 0) {
        groups[status] = tasks;
      }
    }
    return groups;
  }, [agentTasks]);

  const statusGroupLabels: Record<string, { label: string; color: string }> = {
    executing: { label: "Executing", color: "text-trust" },
    assigned: { label: "Assigned", color: "text-receipt" },
    queued: { label: "Queued", color: "text-muted-foreground" },
    completed: { label: "Completed", color: "text-success" },
    failed: { label: "Failed", color: "text-destructive" },
    cancelled: { label: "Cancelled", color: "text-muted-foreground" },
  };

  return (
    <div className="space-y-4">
      <TaskQueueHeader tasks={agentTasks} />

      <div className="space-y-5 max-h-[calc(100vh-14rem)] overflow-y-auto dema-scrollbar pr-1">
        {Object.entries(groupedTasks).map(([status, tasks]) => {
          const group = statusGroupLabels[status];
          if (!group) return null;
          return (
            <div key={status} className="space-y-2">
              <div className="flex items-center gap-2">
                <span
                  className={cn("text-xs font-semibold", group.color)}
                >
                  {group.label}
                </span>
                <Badge
                  variant="outline"
                  className="text-[10px] px-1.5 py-0 font-mono"
                >
                  {tasks.length}
                </Badge>
                <Separator className="flex-1 opacity-30" />
              </div>
              <div className="space-y-2">
                {tasks.map((task) => (
                  <TaskItem key={task.id} task={task} agents={agents} />
                ))}
              </div>
            </div>
          );
        })}

        {agentTasks.length === 0 && (
          <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
            <ListOrdered className="h-8 w-8 mb-2 opacity-40" />
            <p className="text-sm">No tasks in queue.</p>
          </div>
        )}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// Sub-Components: Tab 3 — Event Stream
// ═══════════════════════════════════════════════════════════════

function EventStreamItem({
  event,
  agents,
}: {
  event: OrchestrationEvent;
  agents: Agent[];
}) {
  const sevCfg = SEVERITY_CONFIG[event.severity];
  const SevIcon = sevCfg.icon;
  const [expanded, setExpanded] = useState(false);
  const agent = event.agentId
    ? agents.find((a) => a.id === event.agentId)
    : null;

  const metadataEntries = Object.entries(event.metadata);

  return (
    <div
      className={cn(
        "flex items-start gap-3 p-3 rounded-lg border border-border/30 hover:bg-accent/20 transition-colors cursor-default group",
        event.severity === "error" && "bg-destructive/5 border-destructive/15",
        event.severity === "warning" && "bg-warning/5 border-warning/10",
        event.severity === "success" && "bg-success/5 border-success/10"
      )}
    >
      {/* Severity icon */}
      <div
        className={cn(
          "shrink-0 p-1.5 rounded-md mt-0.5",
          sevCfg.bg
        )}
      >
        <SevIcon className={cn("h-3.5 w-3.5", sevCfg.color)} />
      </div>

      <div className="min-w-0 flex-1">
        {/* Top row: event type + time */}
        <div className="flex items-center gap-2 flex-wrap">
          <Badge
            variant="outline"
            className="text-[10px] px-1.5 py-0 font-mono text-muted-foreground border-border/50"
          >
            {eventTypeLabel(event.type)}
          </Badge>
          {agent && (
            <Badge
              variant="outline"
              className={cn(
                "text-[10px] px-1.5 py-0",
                roleBadgeConfig(agent.role).className
              )}
            >
              {agent.name}
            </Badge>
          )}
          <span className="text-[10px] text-muted-foreground ml-auto shrink-0">
            {timeAgo(event.timestamp)}
          </span>
        </div>

        {/* Message */}
        <p className="text-xs leading-relaxed mt-1 text-foreground/90">
          {event.message}
        </p>

        {/* Expandable metadata */}
        {metadataEntries.length > 0 && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="flex items-center gap-1 mt-1.5 text-[10px] text-muted-foreground hover:text-foreground transition-colors"
          >
            {expanded ? (
              <ChevronDown className="h-3 w-3" />
            ) : (
              <ChevronRight className="h-3 w-3" />
            )}
            Metadata ({metadataEntries.length})
          </button>
        )}

        {expanded && (
          <div className="mt-2 p-2 rounded-md bg-muted/30 border border-border/30">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-1">
              {metadataEntries.map(([key, value]) => (
                <div
                  key={key}
                  className="flex items-start gap-1.5 text-[10px]"
                >
                  <span className="font-mono text-muted-foreground shrink-0">
                    {key}:
                  </span>
                  <span className="font-mono text-foreground break-all">
                    {typeof value === "object"
                      ? JSON.stringify(value)
                      : String(value)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function EventStreamTab() {
  const { orchestrationEvents, agents } = useDEMAStore();
  const [severityFilter, setSeverityFilter] = useState("all");

  const filteredEvents = useMemo(() => {
    if (severityFilter === "all") return orchestrationEvents;
    return orchestrationEvents.filter((e) => e.severity === severityFilter);
  }, [orchestrationEvents, severityFilter]);

  const severityCounts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const e of orchestrationEvents) {
      c[e.severity] = (c[e.severity] || 0) + 1;
    }
    return c;
  }, [orchestrationEvents]);

  return (
    <div className="space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Activity className="h-4.5 w-4.5 text-warning" />
          <h3 className="text-base font-semibold tracking-tight">
            Event Stream
          </h3>
          <Badge variant="outline" className="font-mono text-xs px-2 py-0.5">
            {orchestrationEvents.length}
          </Badge>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          <Select value={severityFilter} onValueChange={setSeverityFilter}>
            <SelectTrigger className="h-8 w-36 text-xs">
              <Filter className="h-3 w-3 mr-1" />
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Severity</SelectItem>
              <SelectItem value="info">Info</SelectItem>
              <SelectItem value="success">Success</SelectItem>
              <SelectItem value="warning">Warning</SelectItem>
              <SelectItem value="error">Error</SelectItem>
            </SelectContent>
          </Select>

          <div className="flex items-center gap-1.5">
            <Badge
              variant="outline"
              className={cn(
                "text-[10px] px-1.5 py-0",
                "bg-success/5 border-success/20 text-success",
                (severityCounts["success"] || 0) === 0 && "opacity-50"
              )}
            >
              {(severityCounts["success"] || 0)} ok
            </Badge>
            <Badge
              variant="outline"
              className={cn(
                "text-[10px] px-1.5 py-0",
                "bg-warning/5 border-warning/20 text-warning",
                (severityCounts["warning"] || 0) === 0 && "opacity-50"
              )}
            >
              {(severityCounts["warning"] || 0)} warn
            </Badge>
            <Badge
              variant="outline"
              className={cn(
                "text-[10px] px-1.5 py-0",
                "bg-destructive/5 border-destructive/20 text-destructive",
                (severityCounts["error"] || 0) === 0 && "opacity-50"
              )}
            >
              {(severityCounts["error"] || 0)} err
            </Badge>
          </div>
        </div>
      </div>

      <div className="space-y-2 max-h-[calc(100vh-14rem)] overflow-y-auto dema-scrollbar pr-1">
        {filteredEvents.map((event) => (
          <EventStreamItem key={event.id} event={event} agents={agents} />
        ))}
        {filteredEvents.length === 0 && (
          <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
            <Radio className="h-8 w-8 mb-2 opacity-40" />
            <p className="text-sm">No events match the filter.</p>
          </div>
        )}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// Main Orchestration Screen
// ═══════════════════════════════════════════════════════════════

export function OrchestrationScreen() {
  const { agents, agentTasks, orchestrationEvents } = useDEMAStore();

  const summaryStats = useMemo(() => {
    const active = agents.filter((a) => a.status === "active").length;
    const busy = agents.filter((a) => a.status === "busy").length;
    const executing = agentTasks.filter((t) => t.status === "executing").length;
    const recentEvents = orchestrationEvents.filter(
      (e) =>
        Date.now() - new Date(e.timestamp).getTime() < 60000
    ).length;

    return { active, busy, executing, recentEvents };
  }, [agents, agentTasks, orchestrationEvents]);

  return (
    <div className="space-y-4 p-4 sm:p-6 max-w-7xl mx-auto dema-fade-in">
      {/* Page header */}
      <div className="space-y-1">
        <div className="flex items-center gap-3">
          <div className="p-1.5 rounded-md bg-trust/10">
            <Network className="h-5 w-5 text-trust" />
          </div>
          <div>
            <h1 className="text-xl sm:text-2xl font-bold tracking-tight">
              Agent Orchestration Hub
            </h1>
            <p className="text-xs sm:text-sm text-muted-foreground mt-0.5">
              Multi-agent coordination, task management, and event streaming
              for the BIZRA orchestration layer.
            </p>
          </div>
        </div>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {[
          {
            label: "Active Agents",
            value: summaryStats.active,
            sub: `${agents.length} total`,
            icon: Bot,
            color: "text-success",
            bg: "bg-success/5",
          },
          {
            label: "Busy Agents",
            value: summaryStats.busy,
            sub: `${agents.filter((a) => a.status === "idle").length} idle`,
            icon: Activity,
            color: "text-warning",
            bg: "bg-warning/5",
          },
          {
            label: "Executing Tasks",
            value: summaryStats.executing,
            sub: `${agentTasks.filter((t) => t.status === "queued").length} queued`,
            icon: Zap,
            color: "text-trust",
            bg: "bg-trust/5",
          },
          {
            label: "Recent Events",
            value: summaryStats.recentEvents,
            sub: `last 60s`,
            icon: Radio,
            color: "text-receipt",
            bg: "bg-receipt/5",
          },
        ].map((stat) => {
          const Icon = stat.icon;
          return (
            <Card
              key={stat.label}
              className="border-border/50 bg-card/50"
            >
              <CardContent className="p-4">
                <div className="flex items-start justify-between mb-2">
                  <div className={cn("p-1.5 rounded-md", stat.bg)}>
                    <Icon className={cn("h-4 w-4", stat.color)} />
                  </div>
                  <Badge
                    variant="outline"
                    className="text-[10px] px-1.5 py-0"
                  >
                    {stat.sub}
                  </Badge>
                </div>
                <div className="text-2xl font-bold font-mono tracking-tight">
                  {stat.value}
                </div>
                <div className="text-[11px] text-muted-foreground mt-0.5">
                  {stat.label}
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Tabs */}
      <Tabs defaultValue="swarm" className="w-full">
        <TabsList className="grid w-full grid-cols-3 h-9">
          <TabsTrigger value="swarm" className="text-xs gap-1.5">
            <Bot className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Agent Swarm</span>
            <span className="sm:hidden">Swarm</span>
          </TabsTrigger>
          <TabsTrigger value="tasks" className="text-xs gap-1.5">
            <ListOrdered className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Task Queue</span>
            <span className="sm:hidden">Tasks</span>
          </TabsTrigger>
          <TabsTrigger value="events" className="text-xs gap-1.5">
            <Radio className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Event Stream</span>
            <span className="sm:hidden">Events</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="swarm" className="mt-4">
          <AgentSwarmTab />
        </TabsContent>

        <TabsContent value="tasks" className="mt-4">
          <TaskQueueTab />
        </TabsContent>

        <TabsContent value="events" className="mt-4">
          <EventStreamTab />
        </TabsContent>
      </Tabs>
    </div>
  );
}
