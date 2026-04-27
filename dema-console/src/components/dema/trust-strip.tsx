"use client";

import { useDEMAStore } from "@/lib/store";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Shield,
  FileCheck,
  Activity,
  ArrowRight,
  Circle,
  Clock,
} from "lucide-react";
import { timeAgo, formatId, trustLevelColor } from "@/lib/helpers/dema";
import {
  formatOptionalText,
  formatTrustLevel,
  formatTrustScore,
  trustScoreProgress,
} from "@/lib/activation-state";
import { cn } from "@/lib/utils";

export function TrustStrip() {
  const { trustState, receipts, stateGap } = useDEMAStore();

  const latestReceipt = receipts[0];
  const verifiedCount = receipts.filter((r) => r.status === "verified").length;

  // NO_SHADOW_STATE: when no principal is activated, render an honest
  // activation prompt instead of blank score / session / receipt fields.
  // The face MUST NOT display scores, session ids, or receipt counts
  // unless the kernel has sealed a PrincipalActivation receipt.
  if (!trustState.isActive) {
    return (
      <div className="trust-glow border-b border-border bg-card/80 backdrop-blur-sm">
        <div className="flex items-center h-10 px-3 gap-2 text-xs">
          <Shield className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="font-medium text-muted-foreground">
            No principal activated
          </span>
          <span className="text-muted-foreground">
            — activate with
          </span>
          <code className="bg-accent/30 px-1.5 py-0.5 rounded font-mono text-[11px]">
            dema activate-principal
          </code>
        </div>
      </div>
    );
  }

  return (
    <div className="trust-glow border-b border-border bg-card/80 backdrop-blur-sm">
      <div className="flex items-center h-10 px-3 gap-1 text-xs overflow-x-auto dema-scrollbar">
        {/* Principal */}
        <Tooltip>
          <TooltipTrigger asChild>
            <div className="flex items-center gap-1.5 px-2 py-1 rounded-md hover:bg-accent/50 transition-colors cursor-default shrink-0">
              <Shield className="h-3.5 w-3.5 text-trust" />
                <span className="font-medium truncate max-w-[120px]">
                  {formatOptionalText(trustState.principalName)}
                </span>
                <Badge
                  variant="outline"
                  className={cn(
                    "text-[10px] px-1.5 py-0 h-4 border-current/20",
                    trustLevelColor(trustState.level ?? "")
                  )}
                >
                  {formatTrustLevel(trustState.level)}
                </Badge>
              </div>
            </TooltipTrigger>
            <TooltipContent side="bottom" className="text-xs">
              <div className="flex flex-col gap-1">
                <span>Session: {trustState.sessionId ? formatId(trustState.sessionId) : "—"}</span>
                <span>
                  Trust Score: {formatTrustScore(trustState.score, trustState.maxScore)}
                </span>
                <span>Verified: {trustState.lastVerified ? timeAgo(trustState.lastVerified) : "—"}</span>
                <span>Chain: {trustState.chainHead ? formatId(trustState.chainHead) : "—"}</span>
                <span>Mission: {trustState.missionId ? formatId(trustState.missionId) : "—"}</span>
                <span>
                  Activation receipt: {trustState.activationReceiptId ? formatId(trustState.activationReceiptId) : "—"}
                </span>
                <span>Final stage: {formatOptionalText(trustState.finalStage)}</span>
                {trustState.cacheWarning && <span>Cache warning: {trustState.cacheWarning}</span>}
              </div>
            </TooltipContent>
          </Tooltip>

        <div className="w-px h-4 bg-border shrink-0" />

        {/* Trust Score */}
        <Tooltip>
          <TooltipTrigger asChild>
            <div className="flex items-center gap-1.5 px-2 py-1 rounded-md hover:bg-accent/50 transition-colors cursor-default shrink-0">
              <Activity className="h-3.5 w-3.5 text-success" />
              <span className="text-muted-foreground">Score</span>
              <span className="font-mono font-medium">
                {formatTrustScore(trustState.score, trustState.maxScore)}
              </span>
              <Progress
                value={trustScoreProgress(trustState.score, trustState.maxScore)}
                className="h-1 w-12"
              />
            </div>
          </TooltipTrigger>
          <TooltipContent side="bottom" className="text-xs">
            {verifiedCount}/{receipts.length} receipts verified
          </TooltipContent>
        </Tooltip>

        <div className="w-px h-4 bg-border shrink-0" />

        {/* Latest Receipt */}
        <Tooltip>
          <TooltipTrigger asChild>
            <div className="flex items-center gap-1.5 px-2 py-1 rounded-md hover:bg-accent/50 transition-colors cursor-default shrink-0">
              <FileCheck
                className={cn(
                  "h-3.5 w-3.5",
                  latestReceipt?.status === "verified"
                    ? "text-success"
                    : latestReceipt?.status === "pending"
                      ? "text-warning"
                      : "text-muted-foreground"
                )}
              />
              <span className="text-muted-foreground">Receipt</span>
              <span className="font-medium truncate max-w-[180px]">
                {latestReceipt?.title ?? "None"}
              </span>
              {latestReceipt && (
                <span className="text-muted-foreground">
                  {timeAgo(latestReceipt.issuedAt)}
                </span>
              )}
            </div>
          </TooltipTrigger>
          <TooltipContent side="bottom" className="text-xs max-w-xs">
            {latestReceipt?.description ?? "No receipts yet"}
          </TooltipContent>
        </Tooltip>

        <div className="w-px h-4 bg-border shrink-0" />

        {/* State Gap */}
        <Tooltip>
          <TooltipTrigger asChild>
            <div className="flex items-center gap-1.5 px-2 py-1 rounded-md hover:bg-accent/50 transition-colors cursor-default shrink-0">
              <ArrowRight className="h-3.5 w-3.5 text-gap" />
              <span className="text-muted-foreground">Gap</span>
              <span className="font-mono font-medium">
                {typeof stateGap.gapPercent === "number" ? `${stateGap.gapPercent}%` : "—"}
              </span>
            </div>
          </TooltipTrigger>
          <TooltipContent side="bottom" className="text-xs max-w-xs">
            <div className="flex flex-col gap-1">
              <span className="font-medium">Current → Ideal State</span>
              <span className="text-muted-foreground">
                {stateGap.current ? `${stateGap.current.slice(0, 80)}...` : "Unavailable"}
              </span>
            </div>
          </TooltipContent>
        </Tooltip>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Next Action */}
        <Tooltip>
          <TooltipTrigger asChild>
            <div className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-trust/5 border border-trust/10 cursor-default shrink-0">
              <Circle className="h-2 w-2 fill-trust text-trust dema-pulse" />
              <span className="text-muted-foreground">Next:</span>
              <span className="font-medium text-trust-foreground truncate max-w-[260px]">
                {formatOptionalText(stateGap.nextAction)}
              </span>
            </div>
          </TooltipTrigger>
          <TooltipContent side="bottom" className="text-xs max-w-sm">
            <div className="flex flex-col gap-1">
              <span className="font-medium">Next Admissible Action</span>
              <span className="text-muted-foreground">
                {stateGap.nextAction ? `Urgency: ${stateGap.urgency}` : "Unavailable"}
              </span>
            </div>
          </TooltipContent>
        </Tooltip>

        {/* Live indicator */}
        <Tooltip>
          <TooltipTrigger asChild>
            <div className="flex items-center gap-1 px-1.5 shrink-0">
              <Clock className="h-3 w-3 text-muted-foreground" />
            </div>
          </TooltipTrigger>
          <TooltipContent side="bottom" className="text-xs">
            {trustState.lastVerified
              ? `Session active since ${timeAgo(trustState.lastVerified)}`
              : "Activation timestamp unavailable"}
          </TooltipContent>
        </Tooltip>
      </div>
    </div>
  );
}
