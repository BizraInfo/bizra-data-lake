"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { useMissionStore } from "@/lib/mission-store";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { CheckCircle2, FileCheck, Copy, RotateCcw, Plus } from "lucide-react";
import { cn } from "@/lib/utils";

function EvidenceRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex items-center justify-between py-1.5">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="text-xs font-mono text-foreground">{value}</span>
    </div>
  );
}

function HashDisplay({ label, hash }: { label: string; hash: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(hash).catch(() => {});
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">{label}</span>
        <button
          onClick={handleCopy}
          className="text-muted-foreground/50 hover:text-muted-foreground transition-colors p-0.5 rounded"
        >
          <Copy className="h-3 w-3" />
        </button>
      </div>
      <p className="text-[11px] font-mono text-foreground/70 break-all leading-relaxed select-all">
        {hash.slice(0, 10)}...{hash.slice(-8)}
      </p>
      {copied && (
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="text-[10px] text-success"
        >
          Copied
        </motion.p>
      )}
    </div>
  );
}

export function ReceiptReveal() {
  const activeMission = useMissionStore((s) => s.activeMission);
  const resetToIdle = useMissionStore((s) => s.resetToIdle);
  const [replayClicked, setReplayClicked] = useState(false);

  const receipt = activeMission?.sealedReceipt;
  const evidence = receipt?.evidence
    ? (() => {
        try {
          return JSON.parse(receipt.evidence);
        } catch {
          return null;
        }
      })()
    : null;

  const handleReplay = () => {
    setReplayClicked(true);
    setTimeout(() => setReplayClicked(false), 3000);
  };

  const handleNewMission = () => {
    resetToIdle();
  };

  if (!receipt) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="w-full max-w-2xl mx-auto px-4 py-8 sm:py-12"
    >
      {/* Header */}
      <div className="text-center mb-8">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="inline-flex items-center justify-center w-12 h-12 rounded-2xl bg-success/10 border border-success/20 mb-4"
        >
          <CheckCircle2 className="h-5 w-5 text-success" />
        </motion.div>
        <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight mb-2">
          Mission Sealed
        </h1>
        <p className="text-sm text-muted-foreground max-w-md mx-auto leading-relaxed">
          This receipt is your proof of execution. Every claim binds to a
          verifiable chain — constitutional invariants verified.
        </p>
      </div>

      {/* Receipt Card */}
      <motion.div
        initial={{ opacity: 0, scale: 0.97 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, delay: 0.15, ease: "easeOut" }}
        className="relative"
      >
        {/* Seal glow effect */}
        <div className="absolute -inset-px rounded-xl bg-gradient-to-b from-success/20 via-success/5 to-transparent pointer-events-none" />
        <div className="absolute -inset-px rounded-xl opacity-30 animate-pulse pointer-events-none" style={{
          boxShadow: "0 0 40px oklch(0.68 0.14 150 / 15%), 0 0 80px oklch(0.68 0.14 150 / 5%)",
        }} />

        <Card className="relative border-success/20 bg-card/95 backdrop-blur-sm overflow-hidden">
          {/* Top success stripe */}
          <div className="h-0.5 w-full bg-gradient-to-r from-transparent via-success/60 to-transparent" />

          <CardContent className="pt-6 pb-6 space-y-5">
            {/* Receipt ID & Status Row */}
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-[10px] uppercase tracking-widest text-muted-foreground/60 mb-1">
                  Receipt
                </p>
                <p className="text-sm font-mono font-semibold tracking-tight">
                  {receipt.id}
                </p>
              </div>
              <Badge className="text-[10px] bg-success/10 text-success border-success/20 shrink-0">
                <CheckCircle2 className="h-3 w-3 mr-1" />
                Verified
              </Badge>
            </div>

            <Separator className="opacity-40" />

            {/* Mission Info */}
            <div className="space-y-3">
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0 flex-1">
                  <p className="text-[10px] uppercase tracking-widest text-muted-foreground/60 mb-1">
                    Mission
                  </p>
                  <p className="text-sm font-medium leading-relaxed truncate">
                    {activeMission?.intent || receipt.title}
                  </p>
                </div>
                <Badge variant="outline" className="text-[10px] px-2 py-0 h-5 font-mono shrink-0 border-border/40">
                  {activeMission?.missionType || receipt.type}
                </Badge>
              </div>

              <p className="text-xs text-muted-foreground leading-relaxed">
                {receipt.description}
              </p>
            </div>

            <Separator className="opacity-40" />

            {/* Cryptographic Data */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <FileCheck className="h-3.5 w-3.5 text-receipt" />
                <span className="text-[10px] uppercase tracking-widest text-muted-foreground/60">
                  Cryptographic Proof
                </span>
              </div>

              <div className="space-y-2.5">
                {evidence?.contentHash && (
                  <HashDisplay label="Content Hash" hash={evidence.contentHash} />
                )}
                {evidence?.parentHash && (
                  <HashDisplay label="Chain Head" hash={evidence.parentHash} />
                )}
                {evidence?.manifestId && (
                  <EvidenceRow label="Manifest ID" value={evidence.manifestId} />
                )}
              </div>
            </div>

            <Separator className="opacity-40" />

            {/* Evidence Summary */}
            {evidence && (
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <FileCheck className="h-3.5 w-3.5 text-manifest" />
                  <span className="text-[10px] uppercase tracking-widest text-muted-foreground/60">
                    Evidence Summary
                  </span>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-6 gap-y-0.5">
                  <EvidenceRow
                    label="Steps Executed"
                    value={evidence.stepsExecuted ?? 0}
                  />
                  <EvidenceRow
                    label="Resources Used"
                    value={evidence.resourcesUsed ?? 0}
                  />
                  <EvidenceRow
                    label="Gates Passed"
                    value={`${evidence.gates?.filter((g: { status: string }) => g.status === "passed").length ?? 0}/${evidence.gates?.length ?? 0}`}
                  />
                </div>
              </div>
            )}

            <Separator className="opacity-40" />

            {/* Timestamps */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <FileCheck className="h-3.5 w-3.5 text-muted-foreground/50" />
                <span className="text-[10px] uppercase tracking-widest text-muted-foreground/60">
                  Timeline
                </span>
              </div>
              <div className="space-y-0.5">
                <EvidenceRow
                  label="Issued"
                  value={new Date(receipt.issuedAt).toLocaleString()}
                />
                <EvidenceRow
                  label="Verified"
                  value={
                    receipt.verifiedAt
                      ? new Date(receipt.verifiedAt).toLocaleString()
                      : "—"
                  }
                />
                <EvidenceRow
                  label="Mission ID"
                  value={receipt.missionId ?? "—"}
                />
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Replay toast feedback */}
      {replayClicked && (
        <motion.div
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          className="fixed top-6 left-1/2 -translate-x-1/2 z-50 flex items-center gap-2 px-4 py-2.5 rounded-lg bg-card border border-border/60 shadow-lg text-sm"
        >
          <RotateCcw className="h-4 w-4 text-trust" />
          <span>Replay mode activated — reviewing mission steps...</span>
        </motion.div>
      )}

      {/* Actions */}
      <div className="flex flex-col sm:flex-row gap-3 mt-6">
        <Button
          variant="outline"
          onClick={handleReplay}
          className="flex-1 h-10 text-sm border-border/50 hover:bg-muted/30"
        >
          <RotateCcw className="h-4 w-4 mr-2" />
          Replay
        </Button>
        <Button
          onClick={handleNewMission}
          className="flex-1 h-10 text-sm bg-trust hover:bg-trust/90 text-trust-foreground"
        >
          <Plus className="h-4 w-4 mr-2" />
          New Mission
        </Button>
      </div>
    </motion.div>
  );
}
