"use client";

import { useState } from "react";
import { useDEMAStore } from "@/lib/store";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  FileCheck,
  Package,
  CheckCircle2,
  Clock,
  AlertCircle,
  XCircle,
  ArrowRightLeft,
  Zap,
  Shield,
  Copy,
  ExternalLink,
  Filter,
} from "lucide-react";
import {
  timeAgo,
  formatTimestamp,
  formatId,
  receiptStatusColor,
  receiptStatusDot,
  receiptTypeIcon,
} from "@/lib/helpers/dema";
import { cn } from "@/lib/utils";
import type { Receipt, Manifest } from "@/lib/types";

function ReceiptIcon({ type }: { type: string }) {
  const iconName = receiptTypeIcon(type);
  const icons: Record<string, React.ElementType> = {
    "check-circle-2": CheckCircle2,
    zap: Zap,
    "shield-check": Shield,
    "arrow-right-left": ArrowRightLeft,
    "alert-circle": AlertCircle,
    "file-text": FileCheck,
  };
  const Icon = icons[iconName] || FileCheck;

  const colorMap: Record<string, string> = {
    completion: "text-success",
    action: "text-trust",
    verification: "text-manifest",
    delegation: "text-action",
    error: "text-destructive",
  };

  return <Icon className={cn("h-4 w-4", colorMap[type] || "text-muted-foreground")} />;
}

function ReceiptCard({ receipt, onClick }: { receipt: Receipt; onClick: () => void }) {
  return (
    <div
      className="flex items-start gap-3 p-4 rounded-lg border border-border/30 hover:bg-accent/20 transition-colors cursor-pointer"
      onClick={onClick}
    >
      <div className="mt-0.5 shrink-0">
        <ReceiptIcon type={receipt.type} />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs font-medium">{receipt.title}</span>
          <div className={cn("w-1.5 h-1.5 rounded-full", receiptStatusDot(receipt.status))} />
          <Badge variant="outline" className={cn("text-[10px] px-1.5 py-0", receiptStatusColor(receipt.status))}>
            {receipt.status}
          </Badge>
        </div>
        {receipt.description && (
          <p className="text-[11px] text-muted-foreground mt-1 line-clamp-2">
            {receipt.description}
          </p>
        )}
        <div className="flex items-center gap-3 mt-2 text-[10px] text-muted-foreground">
          <span className="font-mono">{formatId(receipt.id)}</span>
          <span>{timeAgo(receipt.issuedAt)}</span>
          {receipt.verifiedAt && (
            <span className="text-success">Verified {timeAgo(receipt.verifiedAt)}</span>
          )}
        </div>
      </div>
    </div>
  );
}

function ReceiptDetail({ receipt, onClose }: { receipt: Receipt; onClose: () => void }) {
  let evidenceData = null;
  try {
    evidenceData = receipt.evidence ? JSON.parse(receipt.evidence) : null;
  } catch {
    // ignore
  }

  return (
    <DialogContent className="max-w-lg">
      <DialogHeader>
        <DialogTitle className="flex items-center gap-2 text-base">
          <ReceiptIcon type={receipt.type} />
          {receipt.title}
        </DialogTitle>
      </DialogHeader>
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-3 text-xs">
          <div>
            <span className="text-muted-foreground">ID</span>
            <p className="font-mono mt-0.5">{receipt.id}</p>
          </div>
          <div>
            <span className="text-muted-foreground">Status</span>
            <p className="mt-0.5">
              <Badge variant="outline" className={cn("text-[10px]", receiptStatusColor(receipt.status))}>
                {receipt.status}
              </Badge>
            </p>
          </div>
          <div>
            <span className="text-muted-foreground">Type</span>
            <p className="mt-0.5 capitalize">{receipt.type}</p>
          </div>
          <div>
            <span className="text-muted-foreground">Mission</span>
            <p className="mt-0.5 font-mono">{receipt.missionId || "—"}</p>
          </div>
          <div>
            <span className="text-muted-foreground">Issued</span>
            <p className="mt-0.5">{formatTimestamp(receipt.issuedAt)}</p>
          </div>
          {receipt.verifiedAt && (
            <div>
              <span className="text-muted-foreground">Verified</span>
              <p className="mt-0.5">{formatTimestamp(receipt.verifiedAt)}</p>
            </div>
          )}
        </div>

        <Separator />

        {receipt.description && (
          <div>
            <span className="text-xs text-muted-foreground">Description</span>
            <p className="text-sm mt-1 leading-relaxed">{receipt.description}</p>
          </div>
        )}

        {evidenceData && (
          <div>
            <span className="text-xs text-muted-foreground">Evidence</span>
            <pre className="mt-1 p-3 rounded-lg bg-muted/30 border border-border/30 text-xs font-mono overflow-x-auto">
              {JSON.stringify(evidenceData, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </DialogContent>
  );
}

function ManifestCard({ manifest }: { manifest: Manifest }) {
  const statusColors: Record<string, string> = {
    active: "text-success bg-success/5 border-success/10",
    draft: "text-warning bg-warning/5 border-warning/10",
    completed: "text-manifest bg-manifest/5 border-manifest/10",
    archived: "text-muted-foreground bg-muted/30 border-border/30",
  };

  return (
    <Card className="border-border/30 hover:border-border/60 transition-colors">
      <CardContent className="p-4">
        <div className="flex items-start gap-3">
          <div className="p-1.5 rounded-md bg-manifest/5 shrink-0 mt-0.5">
            <Package className="h-4 w-4 text-manifest" />
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-sm font-medium">{manifest.title}</span>
              <Badge
                variant="outline"
                className={cn("text-[10px] px-1.5 py-0", statusColors[manifest.status])}
              >
                {manifest.status}
              </Badge>
            </div>
            {manifest.description && (
              <p className="text-xs text-muted-foreground mt-1.5 line-clamp-2">
                {manifest.description}
              </p>
            )}
            <div className="flex items-center gap-4 mt-3 text-[10px] text-muted-foreground">
              <span className="flex items-center gap-1">
                <FileCheck className="h-3 w-3" />
                {manifest.artifactCount} artifacts
              </span>
              <span>{timeAgo(manifest.createdAt)}</span>
              <span>Updated {timeAgo(manifest.updatedAt)}</span>
              {manifest.missionId && (
                <span className="font-mono">{formatId(manifest.missionId)}</span>
              )}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function ReceiptsScreen() {
  const { receipts, manifests } = useDEMAStore();
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [selectedReceipt, setSelectedReceipt] = useState<Receipt | null>(null);

  const filteredReceipts =
    statusFilter === "all"
      ? receipts
      : receipts.filter((r) => r.status === statusFilter);

  return (
    <div className="space-y-4 p-6 max-w-5xl mx-auto dema-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Receipts & Manifest</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Verification chain and artifact manifests. Every action produces a receipt.
          </p>
        </div>
        <Badge variant="outline" className="text-xs">
          {receipts.length} receipts · {manifests.length} manifests
        </Badge>
      </div>

      <Tabs defaultValue="receipts" className="space-y-4">
        <TabsList className="bg-muted/30">
          <TabsTrigger value="receipts" className="text-xs">
            <FileCheck className="h-3.5 w-3.5 mr-1.5" />
            Receipts
          </TabsTrigger>
          <TabsTrigger value="manifests" className="text-xs">
            <Package className="h-3.5 w-3.5 mr-1.5" />
            Manifests
          </TabsTrigger>
        </TabsList>

        <TabsContent value="receipts" className="space-y-3">
          <div className="flex items-center gap-2">
            <Filter className="h-3.5 w-3.5 text-muted-foreground" />
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger className="h-8 text-xs w-[140px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Status</SelectItem>
                <SelectItem value="verified">Verified</SelectItem>
                <SelectItem value="pending">Pending</SelectItem>
                <SelectItem value="expired">Expired</SelectItem>
                <SelectItem value="rejected">Rejected</SelectItem>
              </SelectContent>
            </Select>
            <span className="text-[10px] text-muted-foreground ml-auto">
              Showing {filteredReceipts.length} of {receipts.length}
            </span>
          </div>

          <div className="space-y-2">
            {filteredReceipts.map((receipt) => (
              <ReceiptCard
                key={receipt.id}
                receipt={receipt}
                onClick={() => setSelectedReceipt(receipt)}
              />
            ))}
          </div>
        </TabsContent>

        <TabsContent value="manifests" className="space-y-3">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {manifests.map((manifest) => (
              <ManifestCard key={manifest.id} manifest={manifest} />
            ))}
          </div>
        </TabsContent>
      </Tabs>

      {selectedReceipt && (
        <ReceiptDetail
          receipt={selectedReceipt}
          onClose={() => setSelectedReceipt(null)}
        />
      )}
    </div>
  );
}
