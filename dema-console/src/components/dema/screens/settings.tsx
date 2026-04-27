"use client";

import { useState } from "react";
import { useDEMAStore } from "@/lib/store";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Settings,
  Shield,
  Brain,
  SlidersHorizontal,
  Layers,
  Bookmark,
  Database,
  Moon,
  Sun,
  Monitor,
  Lock,
  Eye,
  Trash2,
  Plus,
  Tag,
  ChevronRight,
  Activity,
  Clock,
  KeyRound,
  AlertTriangle,
} from "lucide-react";
import { timeAgo, memoryCategoryIcon, memoryCategoryColor, confidenceBarColor } from "@/lib/helpers/dema";
import {
  formatOptionalText,
  formatTrustLevel,
  formatTrustScore,
  trustScoreProgress,
} from "@/lib/activation-state";
import { cn } from "@/lib/utils";
import type { MemoryCategory } from "@/lib/types";

const CATEGORIES: { id: MemoryCategory; label: string; icon: React.ElementType; description: string }[] = [
  { id: "preference", label: "Preferences", icon: SlidersHorizontal, description: "User preferences and configuration" },
  { id: "context", label: "Context", icon: Layers, description: "Situational context and current state" },
  { id: "knowledge", label: "Knowledge", icon: Brain, description: "Verified knowledge and canon entries" },
  { id: "poi", label: "Points of Interest", icon: Bookmark, description: "Notable references and citations" },
];

function MemoryCard({ entry }: { entry: ReturnType<typeof useDEMAStore.getState>["memoryEntries"][0] }) {
  const catInfo = CATEGORIES.find((c) => c.id === entry.category);
  const CatIcon = catInfo?.icon || Database;

  return (
    <Card className="border-border/30 hover:border-border/60 transition-colors">
      <CardContent className="p-4">
        <div className="flex items-start gap-3">
          <div className={cn("p-1.5 rounded-md shrink-0 mt-0.5", memoryCategoryColor(entry.category))}>
            <CatIcon className="h-3.5 w-3.5" />
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-medium">{entry.title}</span>
              <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                {entry.category}
              </Badge>
            </div>
            <p className="text-[11px] text-muted-foreground line-clamp-2 leading-relaxed">
              {entry.content}
            </p>
            <div className="flex items-center gap-4 mt-3">
              <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
                <span>Confidence</span>
                <div className="w-12 h-1 bg-muted rounded-full overflow-hidden">
                  <div className={cn("h-full rounded-full", confidenceBarColor(entry.confidence))} style={{ width: `${entry.confidence * 100}%` }} />
                </div>
                <span className="font-mono">{Math.round(entry.confidence * 100)}%</span>
              </div>
              <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
                <span>Relevance</span>
                <span className="font-mono">{Math.round(entry.relevance * 100)}%</span>
              </div>
              {entry.source && (
                <div className="text-[10px] text-muted-foreground">
                  via {entry.source}
                </div>
              )}
            </div>
            {entry.tags.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-2">
                {entry.tags.map((tag) => (
                  <span key={tag} className="px-1.5 py-0.5 text-[10px] rounded bg-muted/50 text-muted-foreground">
                    {tag}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function PermissionsPanel() {
  const [permissions, setPermissions] = useState({
    browserAutoRead: true,
    browserExplicitWrite: true,
    computerAutoRead: true,
    computerExplicitWrite: true,
    codeExplicitEdit: true,
    researchAutoCite: true,
    stopAnytime: true,
    auditLog: true,
    receiptEveryAction: true,
  });

  return (
    <Card className="border-border/30">
      <CardHeader className="pb-3">
        <div className="flex items-center gap-2">
          <KeyRound className="h-4 w-4 text-warning" />
          <CardTitle className="text-sm">Action Permissions</CardTitle>
        </div>
        <CardDescription className="text-xs">
          Control which actions require explicit approval versus auto-approval.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {[
          { key: "browserAutoRead" as const, label: "Browser: Auto-approve reads", desc: "Page navigation and text extraction" },
          { key: "browserExplicitWrite" as const, label: "Browser: Explicit for writes", desc: "Form fills, clicks, submissions" },
          { key: "computerAutoRead" as const, label: "Computer: Auto-approve reads", desc: "File reads and system queries" },
          { key: "computerExplicitWrite" as const, label: "Computer: Explicit for writes", desc: "File edits and terminal commands" },
          { key: "codeExplicitEdit" as const, label: "Code: Explicit for edits", desc: "Multi-file edits and git actions" },
          { key: "researchAutoCite" as const, label: "Research: Auto-cite sources", desc: "Always include citations in research" },
          { key: "stopAnytime" as const, label: "Stop anytime", desc: "Allow immediate session termination" },
          { key: "auditLog" as const, label: "Audit logging", desc: "Log all actions for review" },
          { key: "receiptEveryAction" as const, label: "Receipt for every action", desc: "Generate receipts for all operations" },
        ].map((item) => (
          <div key={item.key} className="flex items-center justify-between py-2">
            <div>
              <span className="text-xs">{item.label}</span>
              <p className="text-[10px] text-muted-foreground">{item.desc}</p>
            </div>
            <Switch
              checked={permissions[item.key]}
              onCheckedChange={(checked) =>
                setPermissions((prev) => ({ ...prev, [item.key]: checked }))
              }
              className="scale-75"
            />
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

function MemoryPanel() {
  const { memoryEntries } = useDEMAStore();
  const [categoryFilter, setCategoryFilter] = useState<string>("all");

  const filtered = categoryFilter === "all"
    ? memoryEntries
    : memoryEntries.filter((e) => e.category === categoryFilter);

  const categoryCounts = memoryEntries.reduce(
    (acc, e) => {
      acc[e.category] = (acc[e.category] || 0) + 1;
      return acc;
    },
    {} as Record<string, number>
  );

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2">
        <Badge
          variant="outline"
          className={cn("text-[11px] px-2.5 py-1 cursor-pointer hover:bg-accent/50 transition-colors", categoryFilter === "all" && "bg-accent")}
          onClick={() => setCategoryFilter("all")}
        >
          All ({memoryEntries.length})
        </Badge>
        {CATEGORIES.map((cat) => {
          const Icon = cat.icon;
          return (
            <Badge
              key={cat.id}
              variant="outline"
              className={cn("text-[11px] px-2.5 py-1 cursor-pointer hover:bg-accent/50 transition-colors", categoryFilter === cat.id && "bg-accent")}
              onClick={() => setCategoryFilter(cat.id)}
            >
              <Icon className="h-3 w-3 mr-1" />
              {cat.label} ({categoryCounts[cat.id] || 0})
            </Badge>
          );
        })}
      </div>

      <div className="space-y-2">
        {filtered.map((entry) => (
          <MemoryCard key={entry.id} entry={entry} />
        ))}
      </div>

      {filtered.length === 0 && (
        <div className="text-center py-12 text-muted-foreground">
          <Brain className="h-8 w-8 mx-auto mb-3 opacity-30" />
          <p className="text-sm">No memory entries in this category</p>
        </div>
      )}
    </div>
  );
}

function TrustSettings() {
  const { trustState } = useDEMAStore();

  return (
    <Card className="border-border/30">
      <CardHeader className="pb-3">
        <div className="flex items-center gap-2">
          <Shield className="h-4 w-4 text-trust" />
          <CardTitle className="text-sm">Trust Configuration</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <Label className="text-xs text-muted-foreground">Principal</Label>
            <p className="text-sm font-medium mt-1">{formatOptionalText(trustState.principalName)}</p>
          </div>
          <div>
            <Label className="text-xs text-muted-foreground">Trust Level</Label>
            <Badge variant="outline" className="mt-1.5 text-xs">
              {formatTrustLevel(trustState.level)}
            </Badge>
          </div>
          <div>
            <Label className="text-xs text-muted-foreground">Trust Score</Label>
            <div className="flex items-center gap-2 mt-1.5">
              <Progress value={trustScoreProgress(trustState.score, trustState.maxScore)} className="h-1.5 flex-1" />
              <span className="text-xs font-mono">{formatTrustScore(trustState.score, trustState.maxScore)}</span>
            </div>
          </div>
          <div>
            <Label className="text-xs text-muted-foreground">Session</Label>
            <p className="text-xs font-mono mt-1.5">{trustState.sessionId ? trustState.sessionId.slice(0, 16) : "—"}</p>
          </div>
          <div>
            <Label className="text-xs text-muted-foreground">Chain Head</Label>
            <p className="text-xs font-mono mt-1.5">{trustState.chainHead ? trustState.chainHead.slice(0, 16) : "—"}</p>
          </div>
          <div>
            <Label className="text-xs text-muted-foreground">Activation Receipt</Label>
            <p className="text-xs font-mono mt-1.5">
              {trustState.activationReceiptId ? trustState.activationReceiptId.slice(0, 16) : "—"}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function SettingsScreen() {
  const [theme, setTheme] = useState<"dark" | "light" | "system">("dark");

  return (
    <div className="space-y-4 p-6 max-w-5xl mx-auto dema-fade-in">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Settings</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Permissions, memory management, trust configuration, and preferences.
        </p>
      </div>

      <Tabs defaultValue="permissions" className="space-y-4">
        <TabsList className="bg-muted/30">
          <TabsTrigger value="permissions" className="text-xs">
            <Shield className="h-3.5 w-3.5 mr-1.5" />
            Permissions
          </TabsTrigger>
          <TabsTrigger value="memory" className="text-xs">
            <Brain className="h-3.5 w-3.5 mr-1.5" />
            Memory
          </TabsTrigger>
          <TabsTrigger value="trust" className="text-xs">
            <Activity className="h-3.5 w-3.5 mr-1.5" />
            Trust
          </TabsTrigger>
          <TabsTrigger value="appearance" className="text-xs">
            <Settings className="h-3.5 w-3.5 mr-1.5" />
            Appearance
          </TabsTrigger>
        </TabsList>

        <TabsContent value="permissions">
          <PermissionsPanel />
        </TabsContent>

        <TabsContent value="memory">
          <MemoryPanel />
        </TabsContent>

        <TabsContent value="trust" className="space-y-4">
          <TrustSettings />
          <Card className="border-border/30">
            <CardHeader className="pb-3">
              <div className="flex items-center gap-2">
                <Lock className="h-4 w-4 text-destructive" />
                <CardTitle className="text-sm">Boundary: Core vs Face</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-xs">
                <div className="flex items-start gap-2 text-success">
                  <Eye className="h-3.5 w-3.5 mt-0.5 shrink-0" />
                  <span>Read trust state from core</span>
                </div>
                <div className="flex items-start gap-2 text-success">
                  <Eye className="h-3.5 w-3.5 mt-0.5 shrink-0" />
                  <span>Render manifests from core</span>
                </div>
                <div className="flex items-start gap-2 text-success">
                  <Eye className="h-3.5 w-3.5 mt-0.5 shrink-0" />
                  <span>Present receipts from core</span>
                </div>
                <Separator />
                <div className="flex items-start gap-2 text-destructive">
                  <AlertTriangle className="h-3.5 w-3.5 mt-0.5 shrink-0" />
                  <span>Does not duplicate mission law</span>
                </div>
                <div className="flex items-start gap-2 text-destructive">
                  <AlertTriangle className="h-3.5 w-3.5 mt-0.5 shrink-0" />
                  <span>Does not invent receipts</span>
                </div>
                <div className="flex items-start gap-2 text-destructive">
                  <AlertTriangle className="h-3.5 w-3.5 mt-0.5 shrink-0" />
                  <span>Does not mutate chain truth outside approved contracts</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="appearance">
          <Card className="border-border/30">
            <CardHeader className="pb-3">
              <div className="flex items-center gap-2">
                <Settings className="h-4 w-4 text-muted-foreground" />
                <CardTitle className="text-sm">Theme</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-3">
                {[
                  { value: "light", label: "Light", icon: Sun },
                  { value: "dark", label: "Dark", icon: Moon },
                  { value: "system", label: "System", icon: Monitor },
                ].map((t) => {
                  const Icon = t.icon;
                  return (
                    <Button
                      key={t.value}
                      variant="outline"
                      className={cn(
                        "h-16 flex-col gap-1.5",
                        theme === t.value && "border-trust bg-trust/5"
                      )}
                      onClick={() => {
                        setTheme(t.value as "dark" | "light" | "system");
                        document.documentElement.classList.toggle("dark", t.value !== "light");
                      }}
                    >
                      <Icon className={cn("h-4 w-4", theme === t.value ? "text-trust" : "text-muted-foreground")} />
                      <span className="text-xs">{t.label}</span>
                    </Button>
                  );
                })}
              </div>
            </CardContent>
          </Card>

          <Card className="border-border/30 mt-4">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">About DEMA</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-xs text-muted-foreground">
              <p><strong>DEMA</strong> — The one visible face of BIZRA.</p>
              <p>Unifies web, CLI, and desktop operator surfaces. Consumes lawful runtime truth from the core BIZRA substrate. Does not duplicate constitutional logic or create shadow state.</p>
              <Separator className="my-3" />
              <p className="font-mono text-[10px]">Version 0.1.0 · Phase R1</p>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
