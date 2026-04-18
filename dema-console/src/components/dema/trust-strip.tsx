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
import { cn } from "@/lib/utils";

export function TrustStrip() {
  const { trustState, receipts, stateGap } = useDEMAStore();

  const latestReceipt = receipts[0];
  const verifiedCount = receipts.filter((r) => r.status === "verified").length;

  return (
    <div className="trust-glow border-b border-border bg-card/80 backdrop-blur-sm">
      <div className="flex items-center h-10 px-3 gap-1 text-xs overflow-x-auto dema-scrollbar">
        {/* Principal */}
        <Tooltip>
          <TooltipTrigger asChild>
            <div className="flex items-center gap-1.5 px-2 py-1 rounded-md hover:bg-accent/50 transition-colors cursor-default shrink-0">
              <Shield className="h-3.5 w-3.5 text-trust" />
              <span className="font-medium truncate max-w-[120px]">
                {trustState.principalName}
              </span>
              <Badge
                variant="outline"
                className={cn(
                  "text-[10px] px-1.5 py-0 h-4 border-current/20",
                  trustLevelColor(trustState.level)
                )}
              >
                {trustState.level}
              </Badge>
            </div>
          </TooltipTrigger>
          <TooltipContent side="bottom" className="text-xs">
            <div className="flex flex-col gap-1">
              <span>Session: {formatId(trustState.sessionId)}</span>
              <span>
                Trust Score: {trustState.score}/{trustState.maxScore}
              </span>
              {trustState.lastVerified && (
                <span>Verified: {timeAgo(trustState.lastVerified)}</span>
              )}
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
                {trustState.score}
              </span>
              <Progress
                value={(trustState.score / trustState.maxScore) * 100}
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
              <span className="font-mono font-medium">{stateGap.gapPercent}%</span>
            </div>
          </TooltipTrigger>
          <TooltipContent side="bottom" className="text-xs max-w-xs">
            <div className="flex flex-col gap-1">
              <span className="font-medium">Current → Ideal State</span>
              <span className="text-muted-foreground">
                {stateGap.current.slice(0, 80)}...
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
                {stateGap.nextAction}
              </span>
            </div>
          </TooltipTrigger>
          <TooltipContent side="bottom" className="text-xs max-w-sm">
            <div className="flex flex-col gap-1">
              <span className="font-medium">Next Admissible Action</span>
              <span className="text-muted-foreground">
                Urgency: {stateGap.urgency}
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
            Session active since {timeAgo(trustState.lastVerified ?? new Date())}
          </TooltipContent>
        </Tooltip>
      </div>
    </div>
  );
}
