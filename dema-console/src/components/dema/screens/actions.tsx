"use client";

import { useState } from "react";
import { useDEMAStore } from "@/lib/store";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Progress } from "@/components/ui/progress";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Globe,
  Monitor,
  Terminal,
  Search,
  Zap,
  Shield,
  ShieldAlert,
  Play,
  Square,
  RotateCcw,
  Clock,
  CheckCircle2,
  XCircle,
  AlertCircle,
  Loader2,
  Eye,
  KeyRound,
  Settings,
  ArrowRight,
  Code,
  FileText,
  Trash2,
} from "lucide-react";
import { timeAgo, actionStatusColor } from "@/lib/helpers/dema";
import { cn } from "@/lib/utils";
import type { ActionStatus } from "@/lib/types";

const MODE_INFO: Record<string, { label: string; description: string; icon: React.ElementType; color: string; bg: string }> = {
  browser: {
    label: "Browser Operator",
    description: "Navigate, interact, and extract from web pages with explicit permission.",
    icon: Globe,
    color: "text-manifest",
    bg: "bg-manifest/5",
  },
  computer: {
    label: "Computer Operator",
    description: "Read files, execute terminal commands, and manage local applications.",
    icon: Monitor,
    color: "text-action",
    bg: "bg-action/5",
  },
  code: {
    label: "Code Mode",
    description: "Repo-wide context awareness, multi-file edits, tests, and git actions.",
    icon: Code,
    color: "text-success",
    bg: "bg-success/5",
  },
  research: {
    label: "Research Mode",
    description: "Deep cited research with source-backed analysis and library saving.",
    icon: Search,
    color: "text-trust",
    bg: "bg-trust/5",
  },
};

const STATUS_ICONS: Record<string, React.ElementType> = {
  completed: CheckCircle2,
  executing: Loader2,
  pending: Clock,
  approved: Shield,
  failed: XCircle,
  denied: ShieldAlert,
  stopped: AlertCircle,
};

function StatusIcon({ status }: { status: ActionStatus }) {
  const Icon = STATUS_ICONS[status] || Clock;
  return (
    <Icon
      className={cn(
        "h-4 w-4",
        actionStatusColor(status),
        status === "executing" && "animate-spin"
      )}
    />
  );
}

function ActionLogItem({ action }: { action: ReturnType<typeof useDEMAStore.getState>["actionLog"][0] }) {
  const modeInfo = MODE_INFO[action.mode];
  const Icon = modeInfo?.icon || Zap;

  return (
    <div className="flex items-start gap-3 px-3 py-2.5 rounded-lg hover:bg-accent/20 transition-colors">
      <StatusIcon status={action.status} />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2 flex-wrap">
          <Icon className={cn("h-3.5 w-3.5", modeInfo?.color)} />
          <span className="text-xs font-medium capitalize">{action.mode}</span>
          <span className="text-[10px] text-muted-foreground">→</span>
          <span className="text-xs font-mono text-muted-foreground">{action.action}</span>
          <Badge
            variant="outline"
            className={cn("text-[10px] px-1.5 py-0", actionStatusColor(action.status))}
          >
            {action.status}
          </Badge>
          <Badge variant="outline" className="text-[10px] px-1.5 py-0">
            {action.permission}
          </Badge>
        </div>
        {action.description && (
          <p className="text-[11px] text-muted-foreground mt-1 line-clamp-1">
            {action.description}
          </p>
        )}
        <div className="flex items-center gap-3 mt-1.5 text-[10px] text-muted-foreground">
          <span>{timeAgo(action.createdAt)}</span>
          {action.completedAt && (
            <span>Completed in {Math.round((new Date(action.completedAt).getTime() - new Date(action.createdAt).getTime()) / 1000)}s</span>
          )}
          {action.evidence && (
            <span className="flex items-center gap-0.5">
              <FileText className="h-2.5 w-2.5" />
              Evidence
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

function BrowserOperatorPanel() {
  const { browserSession, startBrowserSession, stopBrowserSession } = useDEMAStore();
  const [url, setUrl] = useState("");

  const handleStart = () => {
    if (!url.trim()) return;
    startBrowserSession(url.trim());
    setUrl("");
  };

  return (
    <div className="space-y-4">
      <Card className="border-border/30">
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <Globe className="h-4 w-4 text-manifest" />
            <CardTitle className="text-sm">Browser Session</CardTitle>
          </div>
          <CardDescription className="text-xs">
            Explicit permission required. Actions are logged and produce receipts.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {!browserSession ? (
            <div className="space-y-3">
              <div className="flex gap-2">
                <Input
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="https://example.com"
                  className="text-sm font-mono"
                  onKeyDown={(e) => e.key === "Enter" && handleStart()}
                />
                <Button size="sm" onClick={handleStart} disabled={!url.trim()} className="h-9 text-xs">
                  <Play className="h-3.5 w-3.5 mr-1" />
                  Start
                </Button>
              </div>
              <div className="flex items-center gap-2 text-[10px] text-muted-foreground px-1">
                <ShieldAlert className="h-3 w-3 text-warning" />
                <span>Browser operator requires explicit permission for each action</span>
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-xs">
                  <div className="w-2 h-2 rounded-full bg-success dema-pulse" />
                  <span className="font-mono truncate max-w-[300px]">{browserSession.url}</span>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={stopBrowserSession}
                  className="h-7 text-xs text-destructive"
                >
                  <Square className="h-3 w-3 mr-1" />
                  Stop
                </Button>
              </div>

              {/* Simulated browser view */}
              <div className="rounded-lg border border-border/50 bg-muted/20 p-8 text-center">
                <Monitor className="h-8 w-8 text-muted-foreground/30 mx-auto mb-3" />
                <p className="text-xs text-muted-foreground">Browser viewport</p>
                <p className="text-[10px] text-muted-foreground/60 mt-1">
                  Session: {browserSession.id}
                </p>
              </div>

              <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
                <Eye className="h-3 w-3" />
                <span>All actions are visible in the action log</span>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Permission model */}
      <Card className="border-border/30">
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <KeyRound className="h-4 w-4 text-warning" />
            <CardTitle className="text-sm">Permission Model</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="space-y-2">
          {[
            { label: "Auto-approve safe reads", desc: "Page navigation, text extraction", value: true },
            { label: "Explicit for writes", desc: "Form fills, clicks, submissions", value: false },
            { label: "Explicit for downloads", desc: "File downloads, data exports", value: false },
            { label: "Session sandbox", desc: "One-task sandbox per session", value: true },
          ].map((item) => (
            <div key={item.label} className="flex items-center justify-between py-1.5">
              <div>
                <span className="text-xs">{item.label}</span>
                <p className="text-[10px] text-muted-foreground">{item.desc}</p>
              </div>
              <div className={cn(
                "w-2 h-2 rounded-full",
                item.value ? "bg-success" : "bg-warning"
              )} />
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}

function ComputerOperatorPanel() {
  return (
    <div className="space-y-4">
      <Card className="border-border/30">
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <Monitor className="h-4 w-4 text-action" />
            <CardTitle className="text-sm">Computer Operator</CardTitle>
          </div>
          <CardDescription className="text-xs">
            Local files, terminal, and app launching with bounded access.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="rounded-lg border border-border/50 bg-muted/20 p-6 text-center">
            <Terminal className="h-8 w-8 text-muted-foreground/30 mx-auto mb-3" />
            <p className="text-xs text-muted-foreground">Computer operator interface</p>
            <p className="text-[10px] text-muted-foreground/60 mt-1">
              Requires explicit permission for destructive actions
            </p>
          </div>

          <div className="mt-4 space-y-2">
            {[
              { icon: FileText, label: "Read local files", status: "auto", color: "text-success" },
              { icon: Terminal, label: "Execute terminal commands", status: "explicit", color: "text-warning" },
              { icon: Monitor, label: "Launch applications", status: "explicit", color: "text-warning" },
              { icon: Code, label: "Edit files", status: "explicit", color: "text-warning" },
            ].map((item) => {
              const Icon = item.icon;
              return (
                <div key={item.label} className="flex items-center justify-between py-2 px-2 rounded-md hover:bg-accent/20 transition-colors">
                  <div className="flex items-center gap-2">
                    <Icon className={cn("h-3.5 w-3.5", item.color)} />
                    <span className="text-xs">{item.label}</span>
                  </div>
                  <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                    {item.status}
                  </Badge>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export function ActionsScreen() {
  const { actionLog, clearCompletedActions } = useDEMAStore();

  const completedCount = actionLog.filter((a) => a.status === "completed").length;
  const pendingCount = actionLog.filter((a) => a.status === "pending").length;

  return (
    <div className="space-y-4 p-6 max-w-6xl mx-auto dema-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Actions</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Operator modes with explicit permission model. Browser, computer, code, and research actions.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-xs">
            {completedCount} completed · {pendingCount} pending
          </Badge>
          {completedCount > 0 && (
            <Button
              variant="ghost"
              size="sm"
              className="h-8 text-xs text-muted-foreground"
              onClick={clearCompletedActions}
            >
              <Trash2 className="h-3 w-3 mr-1" />
              Clear done
            </Button>
          )}
        </div>
      </div>

      <Tabs defaultValue="operators" className="space-y-4">
        <TabsList className="bg-muted/30">
          <TabsTrigger value="operators" className="text-xs">
            <Zap className="h-3.5 w-3.5 mr-1.5" />
            Operator Modes
          </TabsTrigger>
          <TabsTrigger value="log" className="text-xs">
            <FileText className="h-3.5 w-3.5 mr-1.5" />
            Action Log
          </TabsTrigger>
        </TabsList>

        <TabsContent value="operators" className="space-y-4">
          {/* Mode Cards */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {Object.entries(MODE_INFO).map(([key, mode]) => {
              const Icon = mode.icon;
              return (
                <Card key={key} className="border-border/30 hover:border-border/60 transition-colors cursor-pointer">
                  <CardContent className="p-4">
                    <div className={cn("p-2 rounded-md w-fit", mode.bg)}>
                      <Icon className={cn("h-5 w-5", mode.color)} />
                    </div>
                    <h3 className="text-sm font-medium mt-3">{mode.label}</h3>
                    <p className="text-[11px] text-muted-foreground mt-1 leading-relaxed">
                      {mode.description}
                    </p>
                  </CardContent>
                </Card>
              );
            })}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <BrowserOperatorPanel />
            <ComputerOperatorPanel />
          </div>
        </TabsContent>

        <TabsContent value="log">
          <Card className="border-border/30">
            <CardContent className="p-0">
              <div className="divide-y divide-border/20 max-h-[500px] overflow-y-auto dema-scrollbar">
                {actionLog.map((action) => (
                  <ActionLogItem key={action.id} action={action} />
                ))}
              </div>
              {actionLog.length === 0 && (
                <div className="text-center py-12 text-muted-foreground">
                  <FileText className="h-8 w-8 mx-auto mb-3 opacity-30" />
                  <p className="text-sm">No actions recorded yet</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
