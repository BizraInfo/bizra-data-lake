'use client';

import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useMissionStore } from '@/lib/mission-store';
import { useDEMAStore } from '@/lib/store';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import {
  User,
  FileCheck,
  FolderKanban,
  Target,
  Brain,
  Server,
  X,
  Shield,
  Star,
  Clock,
  CheckCircle,
  AlertCircle,
  Hourglass,
  FileText,
  Zap,
  Globe,
  KeyRound,
  Monitor,
  Terminal,
  Bookmark,
  Layers,
  SlidersHorizontal,
  ChevronRight,
  Sparkles,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { timeAgo, trustLevelColor, receiptStatusColor, receiptStatusDot } from '@/lib/helpers/dema';
import type {
  MemoryCategory,
  ResourceType,
  ManifestStatus,
  TrustLevel,
} from '@/lib/types';

// ─── Resource Type Icon Map ─────────────────────────────────────

const RESOURCE_ICON_MAP: Record<ResourceType, typeof Server> = {
  file: FileText,
  url: Globe,
  credential: KeyRound,
  service: Server,
  knowledge: Brain,
  browser: Monitor,
  terminal: Terminal,
};

// ─── Memory Category Config ─────────────────────────────────────

const MEMORY_CATEGORY_CONFIG: Record<
  MemoryCategory,
  { icon: typeof Brain; color: string; bg: string; label: string }
> = {
  preference: {
    icon: SlidersHorizontal,
    color: 'text-action',
    bg: 'bg-action/10',
    label: 'Preference',
  },
  context: {
    icon: Layers,
    color: 'text-manifest',
    bg: 'bg-manifest/10',
    label: 'Context',
  },
  knowledge: {
    icon: Brain,
    color: 'text-trust',
    bg: 'bg-trust/10',
    label: 'Knowledge',
  },
  poi: {
    icon: Bookmark,
    color: 'text-receipt',
    bg: 'bg-receipt/10',
    label: 'Point of Interest',
  },
};

// ─── Manifest Status Config ─────────────────────────────────────

const MANIFEST_STATUS_CONFIG: Record<
  ManifestStatus,
  { color: string; icon: typeof CheckCircle }
> = {
  active: { color: 'text-success', icon: CheckCircle },
  draft: { color: 'text-warning', icon: AlertCircle },
  completed: { color: 'text-trust', icon: CheckCircle },
  archived: { color: 'text-muted-foreground', icon: FileText },
};

// ─── Trust Level Description ────────────────────────────────────

const TRUST_LEVEL_DESCRIPTIONS: Record<TrustLevel, string> = {
  visitor: 'Limited access — observation mode',
  citizen: 'Standard access — mission-capable',
  operator: 'Extended access — system operations',
  admin: 'Full access — governance control',
};

// ─── Receipt Status Badge ───────────────────────────────────────

function ReceiptStatusBadge({ status }: { status: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <div className={cn('w-1.5 h-1.5 rounded-full', receiptStatusDot(status))} />
      <span className={cn('text-[10px] font-medium', receiptStatusColor(status))}>
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </span>
    </div>
  );
}

// ─── Compact Card Variants ──────────────────────────────────────

function CompactReceiptCard({ receipt }: { receipt: ReturnType<typeof useDEMAStore.getState>['receipts'][0] }) {
  return (
    <div className="flex items-start gap-2.5 p-3 rounded-lg border border-border/40 bg-card/40 hover:bg-card/70 transition-colors">
      <div className="w-6 h-6 rounded-md bg-receipt/10 flex items-center justify-center shrink-0 mt-0.5">
        <FileCheck className="h-3 w-3 text-receipt" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between gap-2">
          <p className="text-[11px] font-medium truncate">{receipt.title}</p>
          <ReceiptStatusBadge status={receipt.status} />
        </div>
        {receipt.description && (
          <p className="text-[10px] text-muted-foreground mt-0.5 line-clamp-1">
            {receipt.description}
          </p>
        )}
        <div className="flex items-center gap-2 mt-1.5">
          <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 border-border/30">
            {receipt.type}
          </Badge>
          <span className="text-[9px] text-muted-foreground">{timeAgo(receipt.issuedAt)}</span>
        </div>
      </div>
    </div>
  );
}

function CompactManifestCard({ manifest }: { manifest: ReturnType<typeof useDEMAStore.getState>['manifests'][0] }) {
  const config = MANIFEST_STATUS_CONFIG[manifest.status];
  const StatusIcon = config.icon;

  return (
    <div className="flex items-start gap-2.5 p-3 rounded-lg border border-border/40 bg-card/40 hover:bg-card/70 transition-colors">
      <div className="w-6 h-6 rounded-md bg-manifest/10 flex items-center justify-center shrink-0 mt-0.5">
        <FolderKanban className="h-3 w-3 text-manifest" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between gap-2">
          <p className="text-[11px] font-medium truncate">{manifest.title}</p>
          <div className={cn('flex items-center gap-1', config.color)}>
            <StatusIcon className="h-3 w-3" />
            <span className="text-[10px] font-medium">{manifest.status}</span>
          </div>
        </div>
        {manifest.description && (
          <p className="text-[10px] text-muted-foreground mt-0.5 line-clamp-1">
            {manifest.description}
          </p>
        )}
        <div className="flex items-center gap-2 mt-1.5">
          <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 border-border/30">
            {manifest.artifactCount} artifacts
          </Badge>
          <span className="text-[9px] text-muted-foreground">{timeAgo(manifest.updatedAt)}</span>
        </div>
      </div>
    </div>
  );
}

function CompactResourceCard({ resource }: { resource: ReturnType<typeof useDEMAStore.getState>['resources'][0] }) {
  const Icon = RESOURCE_ICON_MAP[resource.type] || Server;

  return (
    <div className="flex items-center gap-2.5 p-3 rounded-lg border border-border/40 bg-card/40 hover:bg-card/70 transition-colors">
      <div className="w-6 h-6 rounded-md bg-trust/10 flex items-center justify-center shrink-0">
        <Icon className="h-3 w-3 text-trust" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-[11px] font-medium truncate">{resource.name}</p>
        <div className="flex items-center gap-2 mt-0.5">
          <span className="text-[10px] text-muted-foreground">{resource.type}</span>
          {resource.path && (
            <>
              <span className="text-[10px] text-muted-foreground/40">·</span>
              <span className="text-[10px] text-muted-foreground truncate max-w-[160px]">
                {resource.path}
              </span>
            </>
          )}
        </div>
      </div>
      <div className={cn('w-1.5 h-1.5 rounded-full shrink-0', resource.status === 'active' ? 'bg-success' : 'bg-muted-foreground')} />
    </div>
  );
}

function CompactMemoryCard({ entry }: { entry: ReturnType<typeof useDEMAStore.getState>['memoryEntries'][0] }) {
  const config = MEMORY_CATEGORY_CONFIG[entry.category];
  const CategoryIcon = config.icon;

  return (
    <div className="flex items-start gap-2.5 p-3 rounded-lg border border-border/40 bg-card/40 hover:bg-card/70 transition-colors">
      <div className={cn('w-6 h-6 rounded-md flex items-center justify-center shrink-0 mt-0.5', config.bg)}>
        <CategoryIcon className={cn('h-3 w-3', config.color)} />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between gap-2">
          <p className="text-[11px] font-medium truncate">{entry.title}</p>
          <div className="flex items-center gap-1 shrink-0">
            <Badge variant="outline" className={cn('text-[9px] px-1 py-0 h-3.5 border-0', config.bg, config.color)}>
              {config.label}
            </Badge>
          </div>
        </div>
        <p className="text-[10px] text-muted-foreground mt-0.5 line-clamp-2 leading-relaxed">
          {entry.content}
        </p>
        <div className="flex items-center gap-3 mt-1.5">
          <div className="flex items-center gap-1">
            <Star className="h-2.5 w-2.5 text-trust" />
            <span className="text-[10px] text-muted-foreground">
              {Math.round(entry.confidence * 100)}% confidence
            </span>
          </div>
          {entry.tags.length > 0 && (
            <div className="flex items-center gap-1">
              <span className="text-[10px] text-muted-foreground/60">
                {entry.tags.slice(0, 2).join(', ')}
                {entry.tags.length > 2 && ` +${entry.tags.length - 2}`}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function CompactMissionCard({ mission }: { mission: ReturnType<typeof useMissionStore.getState>['missionHistory'][0] }) {
  const statusColor = mission.stage === 'receipt' ? 'text-success' : mission.stage === 'blocked' ? 'text-destructive' : 'text-trust';

  return (
    <div className="flex items-start gap-2.5 p-3 rounded-lg border border-border/40 bg-card/40 hover:bg-card/70 transition-colors">
      <div className="w-6 h-6 rounded-md bg-trust/10 flex items-center justify-center shrink-0 mt-0.5">
        <Target className="h-3 w-3 text-trust" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between gap-2">
          <p className="text-[11px] font-medium truncate">{mission.intent}</p>
          <span className={cn('text-[10px] font-medium shrink-0', statusColor)}>
            {mission.stage === 'receipt' ? 'Sealed' : mission.stage === 'blocked' ? 'Blocked' : mission.stage}
          </span>
        </div>
        <div className="flex items-center gap-2 mt-1">
          <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 border-border/30">
            {mission.missionType}
          </Badge>
          <Badge variant="outline" className={cn('text-[9px] px-1 py-0 h-3.5 border-0', 'bg-trust/10 text-trust')}>
            {mission.urgency}
          </Badge>
          <span className="text-[9px] text-muted-foreground">{timeAgo(mission.createdAt)}</span>
        </div>
        {mission.sealedReceipt && (
          <div className="flex items-center gap-1 mt-1.5">
            <Shield className="h-2.5 w-2.5 text-success" />
            <span className="text-[9px] text-muted-foreground">Receipt sealed</span>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Tab Content Components ─────────────────────────────────────

function ProfileTab() {
  const { trustState } = useDEMAStore();

  return (
    <div className="space-y-4">
      {/* Identity Card */}
      <Card className="border-border/40 bg-card/40">
        <CardContent className="p-4">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-xl bg-trust/10 flex items-center justify-center">
              <User className="h-5 w-5 text-trust" />
            </div>
            <div>
              <p className="text-sm font-semibold">{trustState.principalName}</p>
              <p className="text-[10px] text-muted-foreground dema-mono">{trustState.principalId}</p>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 rounded-lg bg-background/50 border border-border/30">
              <p className="text-[10px] text-muted-foreground mb-1">Trust Level</p>
              <div className="flex items-center gap-2">
                <Shield className={cn('h-4 w-4', trustLevelColor(trustState.level))} />
                <span className={cn('text-sm font-semibold', trustLevelColor(trustState.level))}>
                  {trustState.level.charAt(0).toUpperCase() + trustState.level.slice(1)}
                </span>
              </div>
              <p className="text-[9px] text-muted-foreground mt-1">
                {TRUST_LEVEL_DESCRIPTIONS[trustState.level]}
              </p>
            </div>

            <div className="p-3 rounded-lg bg-background/50 border border-border/30">
              <p className="text-[10px] text-muted-foreground mb-1">Trust Score</p>
              <div className="flex items-baseline gap-1">
                <span className="text-lg font-semibold tabular-nums">{trustState.score}</span>
                <span className="text-[10px] text-muted-foreground">/ {trustState.maxScore}</span>
              </div>
              <div className="w-full h-1 rounded-full bg-muted mt-1.5">
                <div
                  className="h-full rounded-full bg-trust transition-all"
                  style={{ width: `${(trustState.score / trustState.maxScore) * 100}%` }}
                />
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4 mt-3 pt-3 border-t border-border/30">
            <div className="flex items-center gap-1.5">
              <Clock className="h-3 w-3 text-muted-foreground" />
              <span className="text-[10px] text-muted-foreground">
                {trustState.lastVerified ? timeAgo(trustState.lastVerified) : 'Never'}
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className={cn('w-1.5 h-1.5 rounded-full', trustState.isActive ? 'bg-success dema-pulse' : 'bg-muted-foreground')} />
              <span className="text-[10px] text-muted-foreground">
                {trustState.isActive ? 'Session active' : 'Session inactive'}
              </span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function ReceiptsTab() {
  const { receipts } = useDEMAStore();
  const recentReceipts = receipts.slice(0, 5);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[10px] text-muted-foreground">Last 5 receipts</span>
        <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 border-border/30">
          {receipts.length} total
        </Badge>
      </div>
      {recentReceipts.map((r) => (
        <CompactReceiptCard key={r.id} receipt={r} />
      ))}
      {receipts.length === 0 && (
        <div className="flex items-center justify-center py-8 text-muted-foreground text-xs">
          No receipts yet
        </div>
      )}
    </div>
  );
}

function ManifestsTab() {
  const { manifests } = useDEMAStore();

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[10px] text-muted-foreground">Active manifests</span>
        <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 border-border/30">
          {manifests.length} total
        </Badge>
      </div>
      {manifests.map((m) => (
        <CompactManifestCard key={m.id} manifest={m} />
      ))}
      {manifests.length === 0 && (
        <div className="flex items-center justify-center py-8 text-muted-foreground text-xs">
          No manifests
        </div>
      )}
    </div>
  );
}

function MissionsTab() {
  const { missionHistory } = useMissionStore();

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[10px] text-muted-foreground">Mission history</span>
        <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 border-border/30">
          {missionHistory.length} total
        </Badge>
      </div>
      {missionHistory.map((m) => (
        <CompactMissionCard key={m.id} mission={m} />
      ))}
      {missionHistory.length === 0 && (
        <div className="flex items-center justify-center py-8 text-muted-foreground text-xs">
          No mission history
        </div>
      )}
    </div>
  );
}

function StateTab() {
  const { memoryEntries } = useDEMAStore();

  // Group by category
  const grouped = useMemo(() => {
    const groups: Record<string, typeof memoryEntries> = {};
    for (const entry of memoryEntries) {
      if (!groups[entry.category]) groups[entry.category] = [];
      groups[entry.category].push(entry);
    }
    return groups;
  }, [memoryEntries]);

  return (
    <div className="space-y-4">
      {Object.entries(grouped).map(([category, entries]) => {
        const config = MEMORY_CATEGORY_CONFIG[category as MemoryCategory];
        if (!config) return null;
        const CategoryIcon = config.icon;

        return (
          <div key={category}>
            <div className="flex items-center gap-2 mb-2">
              <div className={cn('w-5 h-5 rounded flex items-center justify-center', config.bg)}>
                <CategoryIcon className={cn('h-3 w-3', config.color)} />
              </div>
              <span className="text-[11px] font-medium">{config.label}</span>
              <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 border-border/30">
                {entries.length}
              </Badge>
            </div>
            <div className="space-y-1.5">
              {entries.map((entry) => (
                <CompactMemoryCard key={entry.id} entry={entry} />
              ))}
            </div>
          </div>
        );
      })}
      {memoryEntries.length === 0 && (
        <div className="flex items-center justify-center py-8 text-muted-foreground text-xs">
          No memory entries
        </div>
      )}
    </div>
  );
}

function ResourcesTab() {
  const { resources } = useDEMAStore();

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[10px] text-muted-foreground">Registered resources</span>
        <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 border-border/30">
          {resources.length} total
        </Badge>
      </div>
      {resources.map((r) => (
        <CompactResourceCard key={r.id} resource={r} />
      ))}
      {resources.length === 0 && (
        <div className="flex items-center justify-center py-8 text-muted-foreground text-xs">
          No resources registered
        </div>
      )}
    </div>
  );
}

// ─── Main Component ─────────────────────────────────────────────

export function MemoryConstellation() {
  const toggleMemoryView = useMissionStore((s) => s.toggleMemoryView);
  const [activeTab, setActiveTab] = useState('profile');

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-3 border-b border-border/50">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-manifest/10 flex items-center justify-center">
            <Sparkles className="h-4 w-4 text-manifest" />
          </div>
          <div>
            <h1 className="text-sm font-semibold">Memory Constellation</h1>
            <p className="text-[11px] text-muted-foreground">
              Layer 5 — Persistent memory navigator
            </p>
          </div>
        </div>

        <Button
          variant="ghost"
          size="sm"
          onClick={toggleMemoryView}
          className="h-7 w-7 p-0"
        >
          <X className="h-4 w-4 text-muted-foreground" />
        </Button>
      </div>

      {/* Tabs + Content */}
      <Tabs
        value={activeTab}
        onValueChange={setActiveTab}
        className="flex-1 flex flex-col min-h-0"
      >
        <div className="px-6 pt-3">
          <TabsList className="bg-muted/30 h-9 p-0.5 w-full">
            <TabsTrigger
              value="profile"
              className="text-[11px] gap-1.5 data-[state=active]:bg-card data-[state=active]:shadow-sm flex-1 h-7"
            >
              <User className="h-3 w-3" />
              <span className="hidden sm:inline">Profile</span>
            </TabsTrigger>
            <TabsTrigger
              value="receipts"
              className="text-[11px] gap-1.5 data-[state=active]:bg-card data-[state=active]:shadow-sm flex-1 h-7"
            >
              <FileCheck className="h-3 w-3" />
              <span className="hidden sm:inline">Receipts</span>
            </TabsTrigger>
            <TabsTrigger
              value="manifests"
              className="text-[11px] gap-1.5 data-[state=active]:bg-card data-[state=active]:shadow-sm flex-1 h-7"
            >
              <FolderKanban className="h-3 w-3" />
              <span className="hidden sm:inline">Manifests</span>
            </TabsTrigger>
            <TabsTrigger
              value="missions"
              className="text-[11px] gap-1.5 data-[state=active]:bg-card data-[state=active]:shadow-sm flex-1 h-7"
            >
              <Target className="h-3 w-3" />
              <span className="hidden sm:inline">Missions</span>
            </TabsTrigger>
            <TabsTrigger
              value="state"
              className="text-[11px] gap-1.5 data-[state=active]:bg-card data-[state=active]:shadow-sm flex-1 h-7"
            >
              <Brain className="h-3 w-3" />
              <span className="hidden sm:inline">State</span>
            </TabsTrigger>
            <TabsTrigger
              value="resources"
              className="text-[11px] gap-1.5 data-[state=active]:bg-card data-[state=active]:shadow-sm flex-1 h-7"
            >
              <Server className="h-3 w-3" />
              <span className="hidden sm:inline">Resources</span>
            </TabsTrigger>
          </TabsList>
        </div>

        <ScrollArea className="flex-1 px-6 py-4">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.2, ease: 'easeOut' }}
            >
              <div className="max-w-2xl mx-auto">
                <TabsContent value="profile" className="mt-0">
                  <ProfileTab />
                </TabsContent>
                <TabsContent value="receipts" className="mt-0">
                  <ReceiptsTab />
                </TabsContent>
                <TabsContent value="manifests" className="mt-0">
                  <ManifestsTab />
                </TabsContent>
                <TabsContent value="missions" className="mt-0">
                  <MissionsTab />
                </TabsContent>
                <TabsContent value="state" className="mt-0">
                  <StateTab />
                </TabsContent>
                <TabsContent value="resources" className="mt-0">
                  <ResourcesTab />
                </TabsContent>
              </div>
            </motion.div>
          </AnimatePresence>
        </ScrollArea>
      </Tabs>
    </div>
  );
}
