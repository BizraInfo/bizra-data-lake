"use client";

import { useState } from "react";
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
import { Switch } from "@/components/ui/switch";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Shield,
  ShieldCheck,
  ShieldAlert,
  Lock,
  Key,
  Fingerprint,
  Scale,
  FileCheck,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Eye,
  Hash,
  ChevronDown,
  ChevronUp,
  Clock,
  Activity,
  Ban,
  ShieldHalf,
  Gavel,
  EyeOff,
  Search,
  Crown,
  BrainCircuit,
  GraduationCap,
} from "lucide-react";
import { timeAgo } from "@/lib/helpers/dema";
import type {
  TrustAnchor,
  CryptoProof,
  GovernanceRule,
  GovernanceEvent,
  TrustAnchorType,
  ProofType,
  GovernanceAction,
} from "@/lib/types";

// ─── Shared Color Helpers ──────────────────────────────────────

function anchorTypeConfig(type: TrustAnchorType): {
  label: string;
  color: string;
  icon: React.ElementType;
  bg: string;
} {
  const map: Record<
    TrustAnchorType,
    { label: string; color: string; icon: React.ElementType; bg: string }
  > = {
    constitutional: {
      label: "Constitutional",
      color: "text-amber-600 dark:text-amber-400",
      icon: Crown,
      bg: "bg-amber-500/10 border-amber-500/20",
    },
    cryptographic: {
      label: "Cryptographic",
      color: "text-teal-600 dark:text-teal-400",
      icon: Lock,
      bg: "bg-teal-500/10 border-teal-500/20",
    },
    reputation: {
      label: "Reputation",
      color: "text-violet-600 dark:text-violet-400",
      icon: GraduationCap,
      bg: "bg-violet-500/10 border-violet-500/20",
    },
    behavioral: {
      label: "Behavioral",
      color: "text-green-600 dark:text-green-400",
      icon: BrainCircuit,
      bg: "bg-green-500/10 border-green-500/20",
    },
  };
  return map[type];
}

function proofTypeConfig(type: ProofType): {
  label: string;
  color: string;
  icon: React.ElementType;
} {
  const map: Record<
    ProofType,
    { label: string; color: string; icon: React.ElementType }
  > = {
    receipt_chain: {
      label: "Receipt Chain",
      color: "text-receipt",
      icon: FileCheck,
    },
    hash_verification: {
      label: "Hash Verification",
      color: "text-trust",
      icon: Hash,
    },
    signature: {
      label: "Signature",
      color: "text-amber-600 dark:text-amber-400",
      icon: Key,
    },
    merkle_proof: {
      label: "Merkle Proof",
      color: "text-teal-600 dark:text-teal-400",
      icon: ShieldCheck,
    },
    zero_knowledge: {
      label: "Zero Knowledge",
      color: "text-violet-600 dark:text-violet-400",
      icon: EyeOff,
    },
  };
  return map[type];
}

function severityConfig(severity: "low" | "medium" | "high" | "critical"): {
  color: string;
  bg: string;
  icon: React.ElementType;
  dot: string;
} {
  const map = {
    low: {
      color: "text-muted-foreground",
      bg: "bg-muted/50 border-muted-foreground/20",
      icon: Activity,
      dot: "bg-muted-foreground",
    },
    medium: {
      color: "text-blue-600 dark:text-blue-400",
      bg: "bg-blue-500/10 border-blue-500/20",
      icon: Eye,
      dot: "bg-blue-500",
    },
    high: {
      color: "text-amber-600 dark:text-amber-400",
      bg: "bg-amber-500/10 border-amber-500/20",
      icon: AlertTriangle,
      dot: "bg-amber-500",
    },
    critical: {
      color: "text-destructive",
      bg: "bg-destructive/10 border-destructive/20",
      icon: ShieldAlert,
      dot: "bg-destructive",
    },
  };
  return map[severity];
}

function actionConfig(action: GovernanceAction): {
  label: string;
  color: string;
  bg: string;
  icon: React.ElementType;
} {
  const map: Record<
    GovernanceAction,
    { label: string; color: string; bg: string; icon: React.ElementType }
  > = {
    allow: {
      label: "Allow",
      color: "text-success",
      bg: "bg-success/10 border-success/20",
      icon: CheckCircle2,
    },
    deny: {
      label: "Deny",
      color: "text-destructive",
      bg: "bg-destructive/10 border-destructive/20",
      icon: Ban,
    },
    escalate: {
      label: "Escalate",
      color: "text-amber-600 dark:text-amber-400",
      bg: "bg-amber-500/10 border-amber-500/20",
      icon: AlertTriangle,
    },
    quarantine: {
      label: "Quarantine",
      color: "text-orange-600 dark:text-orange-400",
      bg: "bg-orange-500/10 border-orange-500/20",
      icon: ShieldHalf,
    },
    audit: {
      label: "Audit",
      color: "text-manifest",
      bg: "bg-manifest/10 border-manifest/20",
      icon: Search,
    },
    revoke: {
      label: "Revoke",
      color: "text-violet-600 dark:text-violet-400",
      bg: "bg-violet-500/10 border-violet-500/20",
      icon: XCircle,
    },
  };
  return map[action];
}

function categoryConfig(
  category: GovernanceRule["category"]
): { label: string; color: string } {
  const map: Record<
    GovernanceRule["category"],
    { label: string; color: string }
  > = {
    boundary: { label: "Boundary", color: "text-trust" },
    permission: { label: "Permission", color: "text-warning" },
    integrity: { label: "Integrity", color: "text-receipt" },
    privacy: { label: "Privacy", color: "text-action" },
    performance: { label: "Performance", color: "text-gap" },
  };
  return map[category];
}

function truncateHash(hash: string, prefixLen = 10, suffixLen = 8): string {
  if (hash.length <= prefixLen + suffixLen + 3) return hash;
  return `${hash.slice(0, prefixLen)}...${hash.slice(-suffixLen)}`;
}

function successRate(verifications: number, failures: number): number {
  const total = verifications + failures;
  if (total === 0) return 100;
  return Math.round((verifications / total) * 100);
}

// ─── Trust Anchors Tab ─────────────────────────────────────────

function TrustAnchorCard({
  anchor,
}: {
  anchor: TrustAnchor;
}) {
  const config = anchorTypeConfig(anchor.type);
  const TypeIcon = config.icon;
  const rate = successRate(anchor.verifications, anchor.failures);

  return (
    <Card
      className={cn(
        "border-border/30 hover:border-border/60 transition-all duration-200",
        !anchor.active && "opacity-60"
      )}
    >
      <CardHeader className="p-4 pb-0">
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2.5 min-w-0">
            <div
              className={cn(
                "shrink-0 p-2 rounded-lg border",
                config.bg
              )}
            >
              <TypeIcon className={cn("h-4 w-4", config.color)} />
            </div>
            <div className="min-w-0">
              <CardTitle className="text-sm font-semibold leading-tight truncate">
                {anchor.name}
              </CardTitle>
              <div className="flex items-center gap-2 mt-1">
                <Badge
                  variant="outline"
                  className={cn(
                    "text-[10px] px-1.5 py-0 border",
                    config.bg,
                    config.color
                  )}
                >
                  {config.label}
                </Badge>
                <div className="flex items-center gap-1">
                  <div
                    className={cn(
                      "w-1.5 h-1.5 rounded-full",
                      anchor.active
                        ? "bg-success dema-pulse"
                        : "bg-muted-foreground"
                    )}
                  />
                  <span className="text-[10px] text-muted-foreground">
                    {anchor.active ? "Active" : "Inactive"}
                  </span>
                </div>
              </div>
            </div>
          </div>
          <Tooltip>
            <TooltipTrigger asChild>
              <Shield
                className={cn(
                  "h-4 w-4 shrink-0",
                  anchor.active ? "text-success" : "text-muted-foreground"
                )}
              />
            </TooltipTrigger>
            <TooltipContent>
              {anchor.active ? "Active trust anchor" : "Inactive anchor"}
            </TooltipContent>
          </Tooltip>
        </div>
      </CardHeader>

      <CardContent className="p-4 pt-3 space-y-3">
        {anchor.algorithm && (
          <div className="flex items-center gap-2">
            <Lock className="h-3 w-3 text-muted-foreground shrink-0" />
            <span className="text-xs text-muted-foreground">Algorithm</span>
            <span className="text-xs font-mono font-medium ml-auto">
              {anchor.algorithm}
            </span>
          </div>
        )}

        {anchor.publicKey && (
          <div className="flex items-center gap-2">
            <Key className="h-3 w-3 text-muted-foreground shrink-0" />
            <span className="text-xs text-muted-foreground">Public Key</span>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="text-[10px] font-mono text-foreground/70 ml-auto cursor-help select-all">
                  {truncateHash(anchor.publicKey, 10, 6)}
                </span>
              </TooltipTrigger>
              <TooltipContent side="left" className="max-w-xs">
                <p className="font-mono text-[10px] break-all">
                  {anchor.publicKey}
                </p>
              </TooltipContent>
            </Tooltip>
          </div>
        )}

        <Separator className="opacity-50" />

        <div className="grid grid-cols-2 gap-3">
          <div>
            <span className="text-[10px] text-muted-foreground uppercase tracking-wider">
              Verifications
            </span>
            <p className="text-sm font-semibold text-success mt-0.5">
              {anchor.verifications.toLocaleString()}
            </p>
          </div>
          <div>
            <span className="text-[10px] text-muted-foreground uppercase tracking-wider">
              Failures
            </span>
            <p className="text-sm font-semibold mt-0.5">
              {anchor.failures > 0 ? (
                <span className="text-destructive">
                  {anchor.failures.toLocaleString()}
                </span>
              ) : (
                <span className="text-muted-foreground">0</span>
              )}
            </p>
          </div>
        </div>

        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-muted-foreground uppercase tracking-wider">
              Success Rate
            </span>
            <span
              className={cn(
                "text-xs font-bold",
                rate >= 99
                  ? "text-success"
                  : rate >= 95
                    ? "text-trust"
                    : rate >= 90
                      ? "text-warning"
                      : "text-destructive"
              )}
            >
              {rate}%
            </span>
          </div>
          <Progress value={rate} className="h-1.5" />
        </div>

        <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
          <Clock className="h-3 w-3" />
          <span>Last used {timeAgo(anchor.lastUsed)}</span>
        </div>
      </CardContent>
    </Card>
  );
}

function TrustAnchorsTab() {
  const { trustAnchors } = useDEMAStore();

  const activeCount = trustAnchors.filter((a) => a.active).length;
  const totalVerifications = trustAnchors.reduce(
    (sum, a) => sum + a.verifications,
    0
  );
  const totalFailures = trustAnchors.reduce(
    (sum, a) => sum + a.failures,
    0
  );

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-3">
        <Card className="border-border/30">
          <CardContent className="p-3 text-center">
            <div className="flex items-center justify-center gap-1.5 text-[10px] text-muted-foreground uppercase tracking-wider mb-1">
              <Shield className="h-3 w-3" />
              Anchors
            </div>
            <p className="text-lg font-bold">
              {activeCount}
              <span className="text-muted-foreground text-xs font-normal">
                /{trustAnchors.length}
              </span>
            </p>
          </CardContent>
        </Card>
        <Card className="border-border/30">
          <CardContent className="p-3 text-center">
            <div className="flex items-center justify-center gap-1.5 text-[10px] text-muted-foreground uppercase tracking-wider mb-1">
              <CheckCircle2 className="h-3 w-3" />
              Verified
            </div>
            <p className="text-lg font-bold text-success">
              {totalVerifications.toLocaleString()}
            </p>
          </CardContent>
        </Card>
        <Card className="border-border/30">
          <CardContent className="p-3 text-center">
            <div className="flex items-center justify-center gap-1.5 text-[10px] text-muted-foreground uppercase tracking-wider mb-1">
              <XCircle className="h-3 w-3" />
              Failures
            </div>
            <p
              className={cn(
                "text-lg font-bold",
                totalFailures > 0
                  ? "text-destructive"
                  : "text-muted-foreground"
              )}
            >
              {totalFailures}
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {trustAnchors.map((anchor) => (
          <TrustAnchorCard key={anchor.id} anchor={anchor} />
        ))}
      </div>
    </div>
  );
}

// ─── Crypto Proofs Tab ─────────────────────────────────────────

function CryptoProofCard({ proof }: { proof: CryptoProof }) {
  const [expanded, setExpanded] = useState(false);
  const config = proofTypeConfig(proof.type);
  const ProofIcon = config.icon;

  const { trustAnchors } = useDEMAStore();
  const anchor = trustAnchors.find((a) => a.id === proof.anchorId);

  return (
    <div className="rounded-lg border border-border/30 hover:border-border/60 transition-all duration-200">
      <div
        className="p-4 cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-start gap-3">
          <div className="p-1.5 rounded-md bg-muted/50 shrink-0 mt-0.5">
            <ProofIcon className={cn("h-4 w-4", config.color)} />
          </div>

          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2 flex-wrap">
              <Badge
                variant="outline"
                className={cn("text-[10px] px-1.5 py-0 border", {
                  "text-receipt border-receipt/20": proof.type === "receipt_chain",
                  "text-trust border-trust/20": proof.type === "hash_verification",
                  "text-amber-600 dark:text-amber-400 border-amber-500/20":
                    proof.type === "signature",
                  "text-teal-600 dark:text-teal-400 border-teal-500/20":
                    proof.type === "merkle_proof",
                  "text-violet-600 dark:text-violet-400 border-violet-500/20":
                    proof.type === "zero_knowledge",
                })}
              >
                {config.label}
              </Badge>
              {proof.verified ? (
                <div className="flex items-center gap-1">
                  <CheckCircle2 className="h-3 w-3 text-success" />
                  <span className="text-[10px] text-success font-medium">
                    Verified
                  </span>
                </div>
              ) : (
                <div className="flex items-center gap-1">
                  <XCircle className="h-3 w-3 text-muted-foreground" />
                  <span className="text-[10px] text-muted-foreground">
                    Unverified
                  </span>
                </div>
              )}
            </div>

            <p className="text-xs font-medium mt-1.5">{proof.subject}</p>

            <div className="flex items-center gap-2 mt-2">
              <Fingerprint className="h-3 w-3 text-muted-foreground shrink-0" />
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="text-[10px] font-mono text-foreground/60 truncate select-all">
                    {truncateHash(proof.hash, 12, 8)}
                  </span>
                </TooltipTrigger>
                <TooltipContent side="bottom" className="max-w-sm">
                  <p className="font-mono text-[10px] break-all">{proof.hash}</p>
                </TooltipContent>
              </Tooltip>
            </div>

            <div className="flex items-center gap-4 mt-2.5 text-[10px] text-muted-foreground">
              {proof.verifiedAt && (
                <span className="flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  {timeAgo(proof.verifiedAt)}
                </span>
              )}
              {proof.expiresAt && (
                <span className="flex items-center gap-1">
                  <Shield className="h-3 w-3" />
                  Expires {timeAgo(proof.expiresAt)}
                </span>
              )}
              {anchor && (
                <span className="text-[10px] text-muted-foreground ml-auto">
                  via {anchor.name}
                </span>
              )}
            </div>
          </div>

          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6 shrink-0 mt-0.5"
            onClick={(e) => {
              e.stopPropagation();
              setExpanded(!expanded);
            }}
          >
            {expanded ? (
              <ChevronUp className="h-3.5 w-3.5" />
            ) : (
              <ChevronDown className="h-3.5 w-3.5" />
            )}
          </Button>
        </div>
      </div>

      {expanded && Object.keys(proof.metadata).length > 0 && (
        <>
          <Separator className="opacity-50" />
          <div className="p-4 pt-3">
            <span className="text-[10px] text-muted-foreground uppercase tracking-wider">
              Proof Metadata
            </span>
            <pre className="mt-2 p-3 rounded-lg bg-muted/30 border border-border/20 text-[10px] font-mono overflow-x-auto dema-scrollbar">
              {JSON.stringify(proof.metadata, null, 2)}
            </pre>
            {proof.signature && (
              <div className="mt-2 flex items-center gap-2">
                <span className="text-[10px] text-muted-foreground">
                  Signature:
                </span>
                <span className="text-[10px] font-mono text-foreground/60 truncate select-all">
                  {proof.signature}
                </span>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function CryptoProofsTab() {
  const { cryptoProofs } = useDEMAStore();
  const verifiedCount = cryptoProofs.filter((p) => p.verified).length;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        <Card className="border-border/30">
          <CardContent className="p-3 text-center">
            <div className="flex items-center justify-center gap-1.5 text-[10px] text-muted-foreground uppercase tracking-wider mb-1">
              <Fingerprint className="h-3 w-3" />
              Total Proofs
            </div>
            <p className="text-lg font-bold">{cryptoProofs.length}</p>
          </CardContent>
        </Card>
        <Card className="border-border/30">
          <CardContent className="p-3 text-center">
            <div className="flex items-center justify-center gap-1.5 text-[10px] text-muted-foreground uppercase tracking-wider mb-1">
              <ShieldCheck className="h-3 w-3" />
              Verified
            </div>
            <p className="text-lg font-bold text-success">
              {verifiedCount}
              <span className="text-muted-foreground text-xs font-normal">
                /{cryptoProofs.length}
              </span>
            </p>
          </CardContent>
        </Card>
      </div>

      {cryptoProofs.length === 0 ? (
        <Alert>
          <Shield className="h-4 w-4" />
          <AlertDescription>
            No cryptographic proofs have been generated yet.
          </AlertDescription>
        </Alert>
      ) : (
        <div className="space-y-2 max-h-[600px] overflow-y-auto dema-scrollbar pr-1">
          {cryptoProofs.map((proof) => (
            <CryptoProofCard key={proof.id} proof={proof} />
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Governance Rules Tab ──────────────────────────────────────

function GovernanceRuleCard({ rule }: { rule: GovernanceRule }) {
  const sevConfig = severityConfig(rule.severity);
  const actConfig = actionConfig(rule.action);
  const catConfig = categoryConfig(rule.category);
  const SevIcon = sevConfig.icon;
  const ActIcon = actConfig.icon;

  return (
    <Card
      className={cn(
        "border-border/30 hover:border-border/60 transition-all duration-200",
        !rule.active && "opacity-50"
      )}
    >
      <CardContent className="p-4">
        <div className="flex items-start gap-3">
          <div
            className={cn(
              "shrink-0 p-2 rounded-lg border mt-0.5",
              sevConfig.bg
            )}
          >
            <SevIcon className={cn("h-4 w-4", sevConfig.color)} />
          </div>

          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-sm font-semibold">{rule.name}</span>
              <Badge
                variant="outline"
                className={cn("text-[10px] px-1.5 py-0", catConfig.color)}
              >
                {catConfig.label}
              </Badge>
              <Badge
                variant="outline"
                className={cn(
                  "text-[10px] px-1.5 py-0 border",
                  sevConfig.bg,
                  sevConfig.color
                )}
              >
                {rule.severity}
              </Badge>
            </div>

            <p className="text-xs text-muted-foreground mt-1.5 leading-relaxed">
              {rule.description}
            </p>

            <div className="flex items-center gap-2 mt-3 flex-wrap">
              <Badge
                variant="outline"
                className={cn(
                  "text-[10px] px-1.5 py-0 border",
                  actConfig.bg,
                  actConfig.color
                )}
              >
                <ActIcon className="h-2.5 w-2.5 mr-1" />
                {actConfig.label}
              </Badge>
              {rule.conditions.map((cond) => (
                <Badge
                  key={cond}
                  variant="secondary"
                  className="text-[9px] px-1.5 py-0 font-mono"
                >
                  {cond}
                </Badge>
              ))}
            </div>

            <Separator className="my-3 opacity-50" />

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4 text-[10px] text-muted-foreground">
                <span className="flex items-center gap-1">
                  <Gavel className="h-3 w-3" />
                  {rule.violations} violation{rule.violations !== 1 ? "s" : ""}
                </span>
                {rule.lastViolated && (
                  <span className="flex items-center gap-1">
                    <AlertTriangle className="h-3 w-3" />
                    Last {timeAgo(rule.lastViolated)}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[10px] text-muted-foreground">
                  {rule.active ? "Active" : "Inactive"}
                </span>
                <Switch
                  checked={rule.active}
                  className="scale-75 origin-right"
                />
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function GovernanceRulesTab() {
  const { governanceRules } = useDEMAStore();

  const activeRules = governanceRules.filter((r) => r.active).length;
  const criticalRules = governanceRules.filter(
    (r) => r.severity === "critical" && r.active
  ).length;
  const totalViolations = governanceRules.reduce(
    (sum, r) => sum + r.violations,
    0
  );

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-3">
        <Card className="border-border/30">
          <CardContent className="p-3 text-center">
            <div className="flex items-center justify-center gap-1.5 text-[10px] text-muted-foreground uppercase tracking-wider mb-1">
              <Scale className="h-3 w-3" />
              Active Rules
            </div>
            <p className="text-lg font-bold">
              {activeRules}
              <span className="text-muted-foreground text-xs font-normal">
                /{governanceRules.length}
              </span>
            </p>
          </CardContent>
        </Card>
        <Card className="border-border/30">
          <CardContent className="p-3 text-center">
            <div className="flex items-center justify-center gap-1.5 text-[10px] text-muted-foreground uppercase tracking-wider mb-1">
              <ShieldAlert className="h-3 w-3" />
              Critical
            </div>
            <p className="text-lg font-bold text-destructive">
              {criticalRules}
            </p>
          </CardContent>
        </Card>
        <Card className="border-border/30">
          <CardContent className="p-3 text-center">
            <div className="flex items-center justify-center gap-1.5 text-[10px] text-muted-foreground uppercase tracking-wider mb-1">
              <AlertTriangle className="h-3 w-3" />
              Violations
            </div>
            <p
              className={cn(
                "text-lg font-bold",
                totalViolations > 0
                  ? "text-warning"
                  : "text-muted-foreground"
              )}
            >
              {totalViolations}
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="space-y-3 max-h-[600px] overflow-y-auto dema-scrollbar pr-1">
        {governanceRules.map((rule) => (
          <GovernanceRuleCard key={rule.id} rule={rule} />
        ))}
      </div>
    </div>
  );
}

// ─── Violation Log Tab ─────────────────────────────────────────

function ViolationEventCard({ event }: { event: GovernanceEvent }) {
  const sevConfig = severityConfig(event.severity);
  const actConfig = actionConfig(event.action);
  const SevIcon = sevConfig.icon;
  const ActIcon = actConfig.icon;

  return (
    <div className="relative flex gap-4 group">
      {/* Timeline connector line */}
      <div className="flex flex-col items-center shrink-0">
        <div
          className={cn(
            "w-8 h-8 rounded-full border flex items-center justify-center",
            sevConfig.bg
          )}
        >
          <SevIcon className={cn("h-3.5 w-3.5", sevConfig.color)} />
        </div>
        <div className="w-px flex-1 bg-border/50 mt-2" />
      </div>

      {/* Content card */}
      <div className="pb-6 min-w-0 flex-1">
        <div className="rounded-lg border border-border/30 p-4 hover:border-border/60 transition-colors">
          <div className="flex items-center gap-2 flex-wrap mb-1.5">
            <Badge
              variant="outline"
              className={cn(
                "text-[10px] px-1.5 py-0 border",
                sevConfig.bg,
                sevConfig.color
              )}
            >
              {event.severity}
            </Badge>
            <Badge
              variant="outline"
              className={cn(
                "text-[10px] px-1.5 py-0 border",
                actConfig.bg,
                actConfig.color
              )}
            >
              <ActIcon className="h-2.5 w-2.5 mr-1" />
              {actConfig.label}
            </Badge>
            <span className="text-[10px] text-muted-foreground ml-auto flex items-center gap-1">
              <Clock className="h-3 w-3" />
              {timeAgo(event.timestamp)}
            </span>
          </div>

          <div className="space-y-1.5">
            <div className="flex items-center gap-2">
              <Scale className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
              <span className="text-xs font-semibold">{event.ruleName}</span>
            </div>
            <div className="flex items-center gap-2 pl-5.5 ml-0.5">
              <span className="text-[10px] text-muted-foreground">Subject:</span>
              <Badge
                variant="secondary"
                className="text-[10px] px-1.5 py-0"
              >
                {event.subject}
              </Badge>
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed pl-5.5 ml-0.5">
              {event.description}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function ViolationLogTab() {
  const { governanceEvents } = useDEMAStore();

  const sortedEvents = [...governanceEvents].sort(
    (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
  );

  const criticalCount = sortedEvents.filter(
    (e) => e.severity === "critical"
  ).length;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        <Card className="border-border/30">
          <CardContent className="p-3 text-center">
            <div className="flex items-center justify-center gap-1.5 text-[10px] text-muted-foreground uppercase tracking-wider mb-1">
              <Activity className="h-3 w-3" />
              Total Events
            </div>
            <p className="text-lg font-bold">{sortedEvents.length}</p>
          </CardContent>
        </Card>
        <Card className="border-border/30">
          <CardContent className="p-3 text-center">
            <div className="flex items-center justify-center gap-1.5 text-[10px] text-muted-foreground uppercase tracking-wider mb-1">
              <ShieldAlert className="h-3 w-3" />
              Critical
            </div>
            <p
              className={cn(
                "text-lg font-bold",
                criticalCount > 0
                  ? "text-destructive"
                  : "text-muted-foreground"
              )}
            >
              {criticalCount}
            </p>
          </CardContent>
        </Card>
      </div>

      {sortedEvents.length === 0 ? (
        <Alert>
          <ShieldCheck className="h-4 w-4" />
          <AlertDescription>
            No governance violations recorded. The system is fully compliant.
          </AlertDescription>
        </Alert>
      ) : (
        <div className="max-h-[600px] overflow-y-auto dema-scrollbar pr-1">
          {sortedEvents.map((event, index) => (
            <div key={event.id}>
              <ViolationEventCard event={event} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Main GovernanceScreen ─────────────────────────────────────

export function GovernanceScreen() {
  const { trustAnchors, cryptoProofs, governanceRules, governanceEvents } =
    useDEMAStore();

  const totalViolations = governanceEvents.filter(
    (e) => e.severity === "critical" || e.severity === "high"
  ).length;
  const allProofsVerified = cryptoProofs.every((p) => p.verified);
  const allRulesActive = governanceRules.every((r) => r.active);

  return (
    <div className="space-y-4 p-6 max-w-5xl mx-auto dema-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">
            Governance & Cryptographic Validation
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Constitutional governance layer — trust anchors, cryptographic
            proofs, governance rules, and violation history.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {allProofsVerified && allRulesActive && totalViolations === 0 ? (
            <Badge
              variant="outline"
              className="text-xs text-success border-success/30 bg-success/5"
            >
              <ShieldCheck className="h-3 w-3 mr-1" />
              All Clear
            </Badge>
          ) : (
            <Badge
              variant="outline"
              className={cn(
                "text-xs border",
                totalViolations > 0
                  ? "text-warning border-warning/30 bg-warning/5"
                  : "text-trust border-trust/30 bg-trust/5"
              )}
            >
              <Shield
                className={cn(
                  "h-3 w-3 mr-1",
                  totalViolations > 0 ? "" : ""
                )}
              />
              {totalViolations > 0
                ? `${totalViolations} active issue${totalViolations > 1 ? "s" : ""}`
                : "Monitoring"}
            </Badge>
          )}
        </div>
      </div>

      {/* Tabs */}
      <Tabs defaultValue="trust-anchors" className="space-y-4">
        <TabsList className="bg-muted/30">
          <TabsTrigger value="trust-anchors" className="text-xs">
            <Shield className="h-3.5 w-3.5 mr-1.5" />
            Trust Anchors
          </TabsTrigger>
          <TabsTrigger value="crypto-proofs" className="text-xs">
            <Fingerprint className="h-3.5 w-3.5 mr-1.5" />
            Crypto Proofs
          </TabsTrigger>
          <TabsTrigger value="governance-rules" className="text-xs">
            <Scale className="h-3.5 w-3.5 mr-1.5" />
            Governance Rules
          </TabsTrigger>
          <TabsTrigger value="violation-log" className="text-xs">
            <Gavel className="h-3.5 w-3.5 mr-1.5" />
            Violation Log
          </TabsTrigger>
        </TabsList>

        <TabsContent value="trust-anchors">
          <TrustAnchorsTab />
        </TabsContent>

        <TabsContent value="crypto-proofs">
          <CryptoProofsTab />
        </TabsContent>

        <TabsContent value="governance-rules">
          <GovernanceRulesTab />
        </TabsContent>

        <TabsContent value="violation-log">
          <ViolationLogTab />
        </TabsContent>
      </Tabs>
    </div>
  );
}
