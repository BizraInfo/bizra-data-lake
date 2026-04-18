"use client";

import { useDEMAStore } from "@/lib/store";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import {
  Shield,
  FileCheck,
  Package,
  Box,
  ArrowRight,
  Zap,
  Activity,
  TrendingUp,
  ExternalLink,
  MessageSquare,
  Brain,
} from "lucide-react";
import {
  timeAgo,
  receiptStatusDot,
  urgencyColor,
} from "@/lib/helpers/dema";
import { cn } from "@/lib/utils";

function StatsGrid() {
  const { trustState, receipts, manifests, resources } = useDEMAStore();

  const verifiedReceipts = receipts.filter((r) => r.status === "verified").length;
  const activeManifests = manifests.filter((m) => m.status === "active").length;
  const activeResources = resources.filter((r) => r.status === "active").length;

  const stats = [
    {
      label: "Trust Score",
      value: `${trustState.score}/${trustState.maxScore}`,
      sub: trustState.level,
      icon: Shield,
      color: "text-trust",
      bg: "bg-trust/5",
      progress: (trustState.score / trustState.maxScore) * 100,
    },
    {
      label: "Verified Receipts",
      value: `${verifiedReceipts}/${receipts.length}`,
      sub: `${receipts.filter((r) => r.status === "pending").length} pending`,
      icon: FileCheck,
      color: "text-success",
      bg: "bg-success/5",
      progress: receipts.length > 0 ? (verifiedReceipts / receipts.length) * 100 : 0,
    },
    {
      label: "Active Manifests",
      value: activeManifests.toString(),
      sub: `${manifests.filter((m) => m.status === "draft").length} draft`,
      icon: Package,
      color: "text-manifest",
      bg: "bg-manifest/5",
      progress: (activeManifests / Math.max(manifests.length, 1)) * 100,
    },
    {
      label: "Resources",
      value: activeResources.toString(),
      sub: `${resources.length} registered`,
      icon: Box,
      color: "text-action",
      bg: "bg-action/5",
      progress: (activeResources / Math.max(resources.length, 1)) * 100,
    },
  ];

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      {stats.map((stat) => {
        const Icon = stat.icon;
        return (
          <Card key={stat.label} className="border-border/50 bg-card/50">
            <CardContent className="p-4">
              <div className="flex items-start justify-between mb-3">
                <div className={cn("p-1.5 rounded-md", stat.bg)}>
                  <Icon className={cn("h-4 w-4", stat.color)} />
                </div>
                <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                  {stat.sub}
                </Badge>
              </div>
              <div className="text-2xl font-bold font-mono tracking-tight">
                {stat.value}
              </div>
              <div className="text-xs text-muted-foreground mt-1">{stat.label}</div>
              <Progress value={stat.progress} className="h-1 mt-2" />
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}

function StateGapPanel() {
  const { stateGap } = useDEMAStore();

  return (
    <Card className="border-border/50 bg-card/50">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ArrowRight className="h-4 w-4 text-gap" />
            <CardTitle className="text-sm font-medium">Current → Ideal Gap</CardTitle>
          </div>
          <Badge variant="outline" className={cn("text-[10px]", urgencyColor(stateGap.urgency))}>
            {stateGap.urgency}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="relative">
          <div className="flex items-center justify-between text-xs text-muted-foreground mb-2">
            <span>Current</span>
            <span className="font-mono font-medium text-foreground">{stateGap.gapPercent}%</span>
            <span>Ideal</span>
          </div>
          <div className="h-2 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-warning to-success rounded-full transition-all duration-700"
              style={{ width: `${stateGap.gapPercent}%` }}
            />
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="p-3 rounded-lg bg-muted/30 border border-border/30">
            <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1.5">
              Current State
            </div>
            <p className="text-xs leading-relaxed">{stateGap.current}</p>
          </div>
          <div className="p-3 rounded-lg bg-trust/5 border border-trust/10">
            <div className="text-[10px] font-medium text-trust uppercase tracking-wider mb-1.5">
              Ideal State
            </div>
            <p className="text-xs leading-relaxed text-trust-foreground">{stateGap.ideal}</p>
          </div>
        </div>

        <div className="p-3 rounded-lg bg-warning/5 border border-warning/10">
          <div className="flex items-center gap-1.5 mb-1.5">
            <Zap className="h-3.5 w-3.5 text-warning" />
            <span className="text-[10px] font-medium text-warning uppercase tracking-wider">
              Next Admissible Action
            </span>
          </div>
          <p className="text-xs leading-relaxed">{stateGap.nextAction}</p>
        </div>
      </CardContent>
    </Card>
  );
}

function RecentActivity() {
  const { receipts, actionLog, setScreen } = useDEMAStore();

  const recentItems = [
    ...receipts.slice(0, 3).map((r) => ({
      id: r.id,
      title: r.title,
      description: r.description,
      time: r.issuedAt,
      dot: receiptStatusDot(r.status),
    })),
    ...actionLog.slice(0, 3).map((a) => ({
      id: a.id,
      title: `${a.mode}: ${a.action}`,
      description: a.description,
      time: a.createdAt,
      dot: a.status === "completed" ? "bg-success" : "bg-warning",
    })),
  ]
    .sort((a, b) => new Date(b.time).getTime() - new Date(a.time).getTime())
    .slice(0, 6);

  return (
    <Card className="border-border/50 bg-card/50">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-muted-foreground" />
            <CardTitle className="text-sm font-medium">Recent Activity</CardTitle>
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="text-xs text-muted-foreground h-7"
            onClick={() => setScreen("receipts")}
          >
            View all
            <ExternalLink className="h-3 w-3 ml-1" />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div className="divide-y divide-border/30">
          {recentItems.map((item) => (
            <div
              key={item.id}
              className="flex items-start gap-3 px-4 py-3 hover:bg-accent/20 transition-colors cursor-default"
            >
              <div className="mt-1.5 shrink-0">
                <div className={cn("w-1.5 h-1.5 rounded-full", item.dot)} />
              </div>
              <div className="min-w-0 flex-1">
                <div className="text-xs font-medium truncate">{item.title}</div>
                {item.description && (
                  <div className="text-[11px] text-muted-foreground truncate mt-0.5">
                    {item.description}
                  </div>
                )}
              </div>
              <div className="text-[10px] text-muted-foreground shrink-0 mt-0.5">
                {timeAgo(item.time)}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

function ActiveManifests() {
  const { manifests, setScreen } = useDEMAStore();

  return (
    <Card className="border-border/50 bg-card/50">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Package className="h-4 w-4 text-manifest" />
            <CardTitle className="text-sm font-medium">Active Manifests</CardTitle>
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="text-xs text-muted-foreground h-7"
            onClick={() => setScreen("receipts")}
          >
            View all
            <ExternalLink className="h-3 w-3 ml-1" />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {manifests.slice(0, 3).map((manifest) => (
          <div
            key={manifest.id}
            className="flex items-start gap-3 p-3 rounded-lg border border-border/30 hover:bg-accent/20 transition-colors cursor-default"
          >
            <div className="p-1.5 rounded-md bg-manifest/5 shrink-0 mt-0.5">
              <Package className="h-3.5 w-3.5 text-manifest" />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="text-xs font-medium truncate">{manifest.title}</span>
                <Badge variant="outline" className="text-[10px] px-1.5 py-0 shrink-0">
                  {manifest.status}
                </Badge>
              </div>
              {manifest.description && (
                <p className="text-[11px] text-muted-foreground mt-1 line-clamp-2">
                  {manifest.description}
                </p>
              )}
              <div className="flex items-center gap-3 mt-2 text-[10px] text-muted-foreground">
                <span>{manifest.artifactCount} artifacts</span>
                <span>{timeAgo(manifest.updatedAt)}</span>
              </div>
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

function QuickActions() {
  const { setScreen } = useDEMAStore();

  const actions = [
    {
      label: "Ask DEMA",
      description: "Research, code, or reason",
      icon: MessageSquare,
      screen: "ask" as const,
      color: "text-trust",
      bg: "bg-trust/5",
    },
    {
      label: "New Action",
      description: "Browser or computer operator",
      icon: Zap,
      screen: "actions" as const,
      color: "text-action",
      bg: "bg-action/5",
    },
    {
      label: "View Memory",
      description: "Trust, knowledge, preferences",
      icon: Brain,
      screen: "settings" as const,
      color: "text-manifest",
      bg: "bg-manifest/5",
    },
  ];

  return (
    <Card className="border-border/50 bg-card/50">
      <CardHeader className="pb-3">
        <div className="flex items-center gap-2">
          <TrendingUp className="h-4 w-4 text-muted-foreground" />
          <CardTitle className="text-sm font-medium">Quick Actions</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="space-y-2">
        {actions.map((action) => {
          const Icon = action.icon;
          return (
            <Button
              key={action.label}
              variant="ghost"
              className="w-full h-auto p-3 justify-start gap-3 hover:bg-accent/30"
              onClick={() => setScreen(action.screen)}
            >
              <div className={cn("p-2 rounded-md", action.bg)}>
                <Icon className={cn("h-4 w-4", action.color)} />
              </div>
              <div className="text-left">
                <div className="text-xs font-medium">{action.label}</div>
                <div className="text-[10px] text-muted-foreground">{action.description}</div>
              </div>
            </Button>
          );
        })}
      </CardContent>
    </Card>
  );
}

export function DashboardScreen() {
  return (
    <div className="space-y-4 p-6 max-w-7xl mx-auto dema-fade-in">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Lawful state overview. Trust score, receipt chain, manifest status, and current → ideal gap.
        </p>
      </div>

      <StatsGrid />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <StateGapPanel />
        <ActiveManifests />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2">
          <RecentActivity />
        </div>
        <QuickActions />
      </div>
    </div>
  );
}
