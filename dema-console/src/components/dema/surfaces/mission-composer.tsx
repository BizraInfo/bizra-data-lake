"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { useMissionStore } from "@/lib/mission-store";
import { Card, CardContent } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Sparkles, Target, Zap, Shield, Eye } from "lucide-react";
import { cn } from "@/lib/utils";
import type {
  MissionType,
  MissionUrgency,
  MissionQuality,
  MissionScope,
} from "@/lib/types";

const MISSION_TYPES: { value: MissionType; label: string; icon: React.ElementType }[] = [
  { value: "organize", label: "Organize", icon: Target },
  { value: "research", label: "Research", icon: Eye },
  { value: "analyze", label: "Analyze", icon: Shield },
  { value: "create", label: "Create", icon: Sparkles },
  { value: "communicate", label: "Communicate", icon: Zap },
  { value: "monitor", label: "Monitor", icon: Eye },
];

const URGENCY_OPTIONS: { value: MissionUrgency; label: string; color: string }[] = [
  { value: "low", label: "Low", color: "text-success" },
  { value: "medium", label: "Med", color: "text-warning" },
  { value: "high", label: "High", color: "text-gap-foreground" },
  { value: "critical", label: "Crit", color: "text-destructive" },
];

const QUALITY_OPTIONS: { value: MissionQuality; label: string }[] = [
  { value: "draft", label: "Draft" },
  { value: "standard", label: "Standard" },
  { value: "precise", label: "Precise" },
];

const SCOPE_OPTIONS: { value: MissionScope; label: string }[] = [
  { value: "narrow", label: "Narrow" },
  { value: "normal", label: "Normal" },
  { value: "wide", label: "Wide" },
];

export function MissionComposer() {
  const beginMission = useMissionStore((s) => s.beginMission);
  const advanceToAdmissibility = useMissionStore((s) => s.advanceToAdmissibility);

  const [intent, setIntent] = useState("");
  const [currentState, setCurrentState] = useState("");
  const [desiredState, setDesiredState] = useState("");
  const [missionType, setMissionType] = useState<MissionType>("organize");
  const [urgency, setUrgency] = useState<MissionUrgency>("medium");
  const [quality, setQuality] = useState<MissionQuality>("standard");
  const [scope, setScope] = useState<MissionScope>("normal");

  const canBegin = intent.trim().length > 0;

  const handleSubmit = () => {
    if (!canBegin) return;
    beginMission({
      intent: intent.trim(),
      currentState: currentState.trim(),
      desiredState: desiredState.trim(),
      missionType,
      urgency,
      quality,
      scope,
    });
    advanceToAdmissibility();
  };

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
          className="inline-flex items-center justify-center w-12 h-12 rounded-2xl bg-trust/10 border border-trust/20 mb-4"
        >
          <Sparkles className="h-5 w-5 text-trust" />
        </motion.div>
        <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight mb-2">
          What would you like Dema to do?
        </h1>
        <p className="text-sm text-muted-foreground max-w-md mx-auto leading-relaxed">
          Describe your intent. Dema will evaluate it against constitutional
          invariants, plan the action, and seal every step with a receipt.
        </p>
      </div>

      {/* Composer Card */}
      <Card className="border-border/60 bg-card/80 backdrop-blur-sm">
        <CardContent className="space-y-6 pt-6">
          {/* Intent */}
          <div className="space-y-2">
            <Label
              htmlFor="intent"
              className="text-sm font-medium flex items-center gap-2"
            >
              <Target className="h-3.5 w-3.5 text-trust" />
              Intent
            </Label>
            <Textarea
              id="intent"
              value={intent}
              onChange={(e) => setIntent(e.target.value)}
              placeholder="What do you need accomplished? Be as specific as possible..."
              className="min-h-[120px] resize-none text-sm leading-relaxed placeholder:text-muted-foreground/50"
            />
          </div>

          {/* Current & Desired State — side by side on desktop */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label
                htmlFor="currentState"
                className="text-xs font-medium text-muted-foreground flex items-center gap-1.5"
              >
                <div className="w-1.5 h-1.5 rounded-full bg-muted-foreground/40" />
                Current State
              </Label>
              <Input
                id="currentState"
                value={currentState}
                onChange={(e) => setCurrentState(e.target.value)}
                placeholder="What exists now..."
                className="h-9 text-sm"
              />
            </div>
            <div className="space-y-2">
              <Label
                htmlFor="desiredState"
                className="text-xs font-medium text-muted-foreground flex items-center gap-1.5"
              >
                <div className="w-1.5 h-1.5 rounded-full bg-trust/60" />
                Desired State
              </Label>
              <Input
                id="desiredState"
                value={desiredState}
                onChange={(e) => setDesiredState(e.target.value)}
                placeholder="What should exist..."
                className="h-9 text-sm"
              />
            </div>
          </div>

          {/* Mission Type Selector */}
          <div className="space-y-3">
            <Label className="text-xs font-medium text-muted-foreground">
              Mission Type
            </Label>
            <RadioGroup
              value={missionType}
              onValueChange={(v) => setMissionType(v as MissionType)}
              className="grid grid-cols-2 sm:grid-cols-3 gap-2"
            >
              {MISSION_TYPES.map(({ value, label, icon: Icon }) => (
                <label
                  key={value}
                  htmlFor={`type-${value}`}
                  className={cn(
                    "flex items-center gap-2 px-3 py-2.5 rounded-lg border cursor-pointer transition-all text-sm",
                    missionType === value
                      ? "border-trust/40 bg-trust/5 text-trust-foreground"
                      : "border-border/50 bg-transparent text-muted-foreground hover:border-border hover:bg-muted/30"
                  )}
                >
                  <RadioGroupItem value={value} id={`type-${value}`} className="sr-only" />
                  <Icon className={cn("h-3.5 w-3.5", missionType === value ? "text-trust" : "text-muted-foreground/60")} />
                  {label}
                </label>
              ))}
            </RadioGroup>
          </div>

          {/* Compact Toggles */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {/* Urgency */}
            <div className="space-y-2">
              <Label className="text-xs font-medium text-muted-foreground flex items-center gap-1.5">
                <Zap className="h-3 w-3" />
                Urgency
              </Label>
              <div className="flex gap-1">
                {URGENCY_OPTIONS.map((opt) => (
                  <button
                    key={opt.value}
                    type="button"
                    onClick={() => setUrgency(opt.value)}
                    className={cn(
                      "flex-1 px-2 py-1.5 rounded-md text-xs font-medium transition-all border",
                      urgency === opt.value
                        ? "border-border bg-muted text-foreground"
                        : "border-transparent text-muted-foreground/50 hover:text-muted-foreground"
                    )}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Quality */}
            <div className="space-y-2">
              <Label className="text-xs font-medium text-muted-foreground flex items-center gap-1.5">
                <Shield className="h-3 w-3" />
                Quality
              </Label>
              <div className="flex gap-1">
                {QUALITY_OPTIONS.map((opt) => (
                  <button
                    key={opt.value}
                    type="button"
                    onClick={() => setQuality(opt.value)}
                    className={cn(
                      "flex-1 px-2 py-1.5 rounded-md text-xs font-medium transition-all border",
                      quality === opt.value
                        ? "border-border bg-muted text-foreground"
                        : "border-transparent text-muted-foreground/50 hover:text-muted-foreground"
                    )}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Scope */}
            <div className="space-y-2">
              <Label className="text-xs font-medium text-muted-foreground flex items-center gap-1.5">
                <Eye className="h-3 w-3" />
                Scope
              </Label>
              <div className="flex gap-1">
                {SCOPE_OPTIONS.map((opt) => (
                  <button
                    key={opt.value}
                    type="button"
                    onClick={() => setScope(opt.value)}
                    className={cn(
                      "flex-1 px-2 py-1.5 rounded-md text-xs font-medium transition-all border",
                      scope === opt.value
                        ? "border-border bg-muted text-foreground"
                        : "border-transparent text-muted-foreground/50 hover:text-muted-foreground"
                    )}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Submit */}
          <div className="pt-2">
            <Button
              onClick={handleSubmit}
              disabled={!canBegin}
              className={cn(
                "w-full h-11 text-sm font-medium transition-all",
                canBegin
                  ? "bg-trust hover:bg-trust/90 text-trust-foreground shadow-[0_0_20px_oklch(0.78_0.14_75_/15%)]"
                  : "opacity-40"
              )}
            >
              <Sparkles className="h-4 w-4 mr-2" />
              Begin Mission
            </Button>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
